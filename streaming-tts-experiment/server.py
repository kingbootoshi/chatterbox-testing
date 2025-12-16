"""
Chatterbox Turbo Streaming TTS Server (v0.5)

WebSocket server with:
- Pre-cached voice conditionals (instant load)
- Chunked audio streaming during generation
- State-passing streaming (HiFiGAN cache + finalize control)
- TRUE flow-level streaming with z_cache continuity (no chunk artifacts)
- MPS acceleration on Mac

Endpoints:
- /ws/tts        - Basic chunked streaming
- /ws/tts/crossfade - Overlap-add crossfade streaming
- /ws/tts/stateful  - HiFiGAN cache-based streaming
- /ws/tts/flow      - TRUE streaming with z_cache (recommended)
- /ws/tts/simple    - Batch generation, chunked delivery
"""
import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from tqdm import tqdm
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# Add parent directory to path for chatterbox imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals, punc_norm
from chatterbox.models.s3gen.const import S3GEN_SIL

# Configuration
SAMPLE_RATE = 24000
DEFAULT_CHUNK_DURATION_MS = 200
VOICE_PATH = Path(__file__).parent.parent / "voices" / "qb_voice.pt"
TOKENS_PER_CHUNK = 25  # ~1 second of audio per chunk (25 tokens/sec)

# Overlap-add crossfade settings
OVERLAP_TOKENS = 3  # Number of tokens to overlap between chunks
CROSSFADE_MS = 50  # Crossfade duration in milliseconds
CROSSFADE_SAMPLES = int(CROSSFADE_MS / 1000 * SAMPLE_RATE)  # ~1200 samples

# Global model instance
model: ChatterboxTurboTTS = None
generation_lock: asyncio.Lock = None


class S3GenStreamer:
    """
    Stateful streaming wrapper for S3Gen with z_cache continuity.

    This enables TRUE streaming by maintaining z_cache state between
    calls. The z_cache stores cached noise (z) for already-processed
    frames, ensuring chunks share a coherent latent trajectory without
    boundary artifacts.

    Usage:
        streamer = S3GenStreamer(model, tokens_per_chunk=25)
        for token in generated_tokens:
            streamer.append_token(token)
            if len(streamer.speech_tokens) >= tokens_per_chunk:
                audio = streamer.step(finalize=False)
                yield audio
        # Final chunk
        audio = streamer.step(finalize=True)
        yield audio
    """

    def __init__(
        self,
        tts_model: ChatterboxTurboTTS,
        tokens_per_chunk: int = 25,
    ):
        self.model = tts_model
        self.s3gen = tts_model.s3gen
        self.ref_dict = tts_model.conds.gen
        self.tokens_per_chunk = tokens_per_chunk
        # NOTE: lookahead is handled internally by inference_streaming (3 tokens = 6 mel frames)
        self.reset()

    def reset(self):
        """Reset state for new utterance."""
        self.speech_tokens = []
        self.z_cache = None
        self.cache_source = None
        self.total_mels_emitted = 0

    def append_token(self, token: torch.Tensor):
        """Append a token (shape [1, 1])."""
        self.speech_tokens.append(token)

    def _run_flow(self, finalize: bool):
        """Run flow on all accumulated tokens with z_cache."""
        speech_tokens = torch.cat(self.speech_tokens, dim=1).to(self.model.device)

        # Filter out invalid/EOS tokens (vocab size is 6561, valid range 0-6560)
        speech_tokens = speech_tokens[:, speech_tokens[0] < 6561]

        if speech_tokens.shape[1] == 0:
            return torch.zeros(1, 80, 0, device=self.model.device, dtype=self.s3gen.dtype)

        mels, self.z_cache = self.s3gen.flow_stream_step(
            speech_tokens=speech_tokens,
            ref_dict=self.ref_dict,
            z_cache=self.z_cache,
            n_cfm_timesteps=2,
            finalize=finalize,
        )
        return mels

    def step(self, finalize: bool = False) -> np.ndarray:
        """
        Run one streaming step.

        Args:
            finalize: True if this is the final chunk (include lookahead frames)

        Returns:
            New audio samples as np.int16 array
        """
        mels = self._run_flow(finalize=finalize)
        T_total = mels.shape[2]

        # NOTE: Lookahead is already handled in inference_streaming (crops h by 6 frames)
        # We emit ALL mels returned - no additional holding back needed
        usable_end = T_total

        start = self.total_mels_emitted
        if usable_end <= start:
            return np.zeros(0, dtype=np.int16)

        new_mels = mels[:, :, start:usable_end]
        self.total_mels_emitted = usable_end

        # HiFiGAN with cache_source for waveform continuity
        new_mels = new_mels.to(dtype=self.s3gen.dtype)
        if self.cache_source is None:
            self.cache_source = torch.zeros(1, 1, 0,
                                            device=self.s3gen.device,
                                            dtype=self.s3gen.dtype)

        # Truncate cache_source if it's larger than what current mel chunk needs
        # HiFiGAN upsamples: 8 * 5 * 3 * 4 (istft hop) = 480x
        cache = self.cache_source
        if cache is not None and cache.shape[2] > 0:
            expected_source_size = new_mels.shape[2] * 480
            if cache.shape[2] > expected_source_size:
                cache = cache[:, :, -expected_source_size:]

        wav, new_cache = self.s3gen.hift_inference(new_mels, cache_source=cache)

        # Only keep a reasonable tail for continuity (~100ms at 24kHz)
        CACHE_TAIL_SAMPLES = 2400
        if new_cache.shape[2] > CACHE_TAIL_SAMPLES:
            new_cache = new_cache[:, :, -CACHE_TAIL_SAMPLES:]
        self.cache_source = new_cache
        wav = wav.squeeze(0).detach().cpu().numpy()
        wav = np.clip(wav * 1.5, -1.0, 1.0)
        return (wav * 32767).astype(np.int16)


def setup_device():
    """Configure device for Mac M-series chips."""
    load_dotenv()

    if os.getenv("HF_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
        logger.info("HF token loaded from .env")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    map_location = torch.device(device)

    logger.info(f"Detected device: {device.upper()}")
    if device == "mps":
        logger.success("Using Metal Performance Shaders (GPU acceleration)")
    else:
        logger.warning("Using CPU (Metal not available)")

    # Patch torch.load for MPS compatibility
    torch_load_original = torch.load

    def patched_torch_load(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = map_location
        return torch_load_original(*args, **kwargs)

    torch.load = patched_torch_load

    return device, map_location


def crossfade_audio(audio1: np.ndarray, audio2: np.ndarray, crossfade_samples: int) -> np.ndarray:
    """
    Crossfade two audio arrays at the overlap region.

    Args:
        audio1: First audio chunk (we take all but last crossfade_samples)
        audio2: Second audio chunk (we blend the start)
        crossfade_samples: Number of samples to crossfade

    Returns:
        Blended audio array
    """
    if len(audio1) < crossfade_samples or len(audio2) < crossfade_samples:
        # Not enough samples to crossfade, just concatenate
        return np.concatenate([audio1, audio2])

    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, crossfade_samples).astype(np.float32)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples).astype(np.float32)

    # Get the regions to blend
    audio1_end = audio1[-crossfade_samples:].astype(np.float32)
    audio2_start = audio2[:crossfade_samples].astype(np.float32)

    # Blend
    blended = (audio1_end * fade_out + audio2_start * fade_in).astype(audio1.dtype)

    # Concatenate: audio1[:-crossfade] + blended + audio2[crossfade:]
    return np.concatenate([
        audio1[:-crossfade_samples],
        blended,
        audio2[crossfade_samples:]
    ])


def load_model(device: str, map_location) -> ChatterboxTurboTTS:
    """Load model with pre-cached voice conditionals."""
    t0 = time.perf_counter()
    logger.info("Loading Chatterbox Turbo model...")
    tts_model = ChatterboxTurboTTS.from_pretrained(device=device)
    logger.success(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    # Load pre-cached voice conditionals (instant!)
    t0 = time.perf_counter()
    if VOICE_PATH.exists():
        logger.info(f"Loading voice: {VOICE_PATH}")
        tts_model.conds = Conditionals.load(VOICE_PATH, map_location=map_location).to(device)
        logger.success(f"Voice loaded in {time.perf_counter() - t0:.2f}s")
    else:
        logger.error(f"Voice file not found: {VOICE_PATH}")
        raise FileNotFoundError(f"Voice file not found: {VOICE_PATH}")

    return tts_model


async def generate_streaming(
    text: str,
    tokens_per_chunk: int = TOKENS_PER_CHUNK,
) -> AsyncGenerator[tuple[bytes, dict], None]:
    """
    Generate audio with streaming - yields audio chunks as they're ready.

    Uses chunked token generation + S3Gen processing for lower latency.

    Yields:
        (audio_bytes, stats_dict) tuples
    """
    t_start = time.perf_counter()

    # Prepare text
    text = punc_norm(text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)

    # Setup logits processors
    logits_processors = LogitsProcessorList([
        TemperatureLogitsWarper(0.8),
        TopKLogitsWarper(1000),
        TopPLogitsWarper(0.95),
        RepetitionPenaltyLogitsProcessor(1.2),
    ])

    t3 = model.t3
    t3_cond = model.conds.t3

    # Initialize T3 generation
    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )

    # First forward pass
    llm_outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    hidden_states = llm_outputs[0]
    past_key_values = llm_outputs.past_key_values

    speech_logits = t3.speech_head(hidden_states[:, -1:])
    processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
    probs = F.softmax(processed_logits, dim=-1)
    current_token = torch.multinomial(probs, num_samples=1)

    all_tokens = [current_token]
    chunk_tokens = [current_token]

    t_first_token = time.perf_counter()
    chunk_count = 0
    total_audio_samples = 0

    # Generate tokens and process in chunks
    for i in range(1000):  # max_gen_len
        current_embed = t3.speech_emb(current_token)
        llm_outputs = t3.tfmr(
            inputs_embeds=current_embed,
            past_key_values=past_key_values,
            use_cache=True
        )
        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values
        speech_logits = t3.speech_head(hidden_states)

        input_ids = torch.cat(all_tokens, dim=1)
        processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])

        if torch.all(processed_logits == -float("inf")):
            break

        probs = F.softmax(processed_logits, dim=-1)
        current_token = torch.multinomial(probs, num_samples=1)

        all_tokens.append(current_token)
        chunk_tokens.append(current_token)

        # Check for EOS
        if torch.all(current_token == t3.hp.stop_speech_token):
            break

        # Process chunk when we have enough tokens
        if len(chunk_tokens) >= tokens_per_chunk:
            audio_bytes, samples = await process_token_chunk(chunk_tokens, chunk_count == 0)
            total_audio_samples += samples
            chunk_count += 1

            stats = {
                "chunk": chunk_count,
                "tokens": len(chunk_tokens),
                "t_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

            if chunk_count == 1:
                stats["t_first_chunk_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

            yield audio_bytes, stats
            chunk_tokens = []

    # Process remaining tokens
    if chunk_tokens:
        # Remove EOS if present
        if len(chunk_tokens) > 0 and torch.all(chunk_tokens[-1] == t3.hp.stop_speech_token):
            chunk_tokens = chunk_tokens[:-1]

        if chunk_tokens:
            audio_bytes, samples = await process_token_chunk(chunk_tokens, chunk_count == 0)
            total_audio_samples += samples
            chunk_count += 1

            stats = {
                "chunk": chunk_count,
                "tokens": len(chunk_tokens),
                "t_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 2),
                "final": True,
            }
            yield audio_bytes, stats

    # Final stats
    t_total = time.perf_counter() - t_start
    audio_duration = total_audio_samples / SAMPLE_RATE

    final_stats = {
        "type": "complete",
        "stats": {
            "t_total_ms": round(t_total * 1000, 2),
            "t_first_token_ms": round((t_first_token - t_start) * 1000, 2),
            "audio_duration_ms": round(audio_duration * 1000, 2),
            "realtime_factor": round(t_total / audio_duration, 3) if audio_duration > 0 else 0,
            "total_tokens": len(all_tokens),
            "chunks_sent": chunk_count,
        }
    }
    yield None, final_stats


async def generate_streaming_crossfade(
    text: str,
    tokens_per_chunk: int = TOKENS_PER_CHUNK,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> AsyncGenerator[tuple[bytes, dict], None]:
    """
    Generate audio with streaming and overlap-add crossfade.

    Generates overlapping token chunks, processes through S3Gen,
    then crossfades at boundaries for smooth audio.

    Yields:
        (audio_bytes, stats_dict) tuples
    """
    t_start = time.perf_counter()

    # Prepare text
    text = punc_norm(text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)

    # Setup logits processors
    logits_processors = LogitsProcessorList([
        TemperatureLogitsWarper(0.8),
        TopKLogitsWarper(1000),
        TopPLogitsWarper(0.95),
        RepetitionPenaltyLogitsProcessor(1.2),
    ])

    t3 = model.t3
    t3_cond = model.conds.t3

    # Initialize T3 generation
    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )

    # First forward pass
    llm_outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    hidden_states = llm_outputs[0]
    past_key_values = llm_outputs.past_key_values

    speech_logits = t3.speech_head(hidden_states[:, -1:])
    processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
    probs = F.softmax(processed_logits, dim=-1)
    current_token = torch.multinomial(probs, num_samples=1)

    all_tokens = [current_token]

    t_first_token = time.perf_counter()

    # Generate ALL tokens first (we need them for overlapping chunks)
    for i in range(1000):
        current_embed = t3.speech_emb(current_token)
        llm_outputs = t3.tfmr(
            inputs_embeds=current_embed,
            past_key_values=past_key_values,
            use_cache=True
        )
        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values
        speech_logits = t3.speech_head(hidden_states)

        input_ids = torch.cat(all_tokens, dim=1)
        processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])

        if torch.all(processed_logits == -float("inf")):
            break

        probs = F.softmax(processed_logits, dim=-1)
        current_token = torch.multinomial(probs, num_samples=1)

        all_tokens.append(current_token)

        if torch.all(current_token == t3.hp.stop_speech_token):
            break

    t_tokens_done = time.perf_counter()
    logger.debug(f"Generated {len(all_tokens)} tokens in {(t_tokens_done - t_start)*1000:.0f}ms")

    # Remove EOS if present
    if len(all_tokens) > 0 and torch.all(all_tokens[-1] == t3.hp.stop_speech_token):
        all_tokens = all_tokens[:-1]

    # Now create overlapping chunks
    total_tokens = len(all_tokens)
    chunk_indices = []  # List of (start, end) indices

    start = 0
    while start < total_tokens:
        end = min(start + tokens_per_chunk, total_tokens)
        chunk_indices.append((start, end))
        # Next chunk starts (tokens_per_chunk - overlap) tokens later
        start = end - overlap_tokens
        if start >= total_tokens:
            break
        # Avoid infinite loop if chunk is tiny
        if end == total_tokens:
            break

    logger.debug(f"Created {len(chunk_indices)} overlapping chunks")

    # Process chunks and apply crossfade
    previous_tail = None  # Only store the tail portion for crossfading
    chunk_count = 0
    total_audio_samples = 0

    for idx, (start_idx, end_idx) in enumerate(chunk_indices):
        chunk_tokens = all_tokens[start_idx:end_idx]

        # Process this chunk
        audio_array = await process_token_chunk_raw(chunk_tokens)

        is_last = (idx == len(chunk_indices) - 1)

        if previous_tail is not None:
            # Crossfade the start of this chunk with the previous tail
            crossfade_len = min(CROSSFADE_SAMPLES, len(previous_tail), len(audio_array))

            if crossfade_len > 0:
                fade_out = np.linspace(1.0, 0.0, crossfade_len).astype(np.float32)
                fade_in = np.linspace(0.0, 1.0, crossfade_len).astype(np.float32)

                # Blend the overlap region
                blended = (previous_tail[-crossfade_len:] * fade_out +
                          audio_array[:crossfade_len] * fade_in)

                # Build the audio to send: previous_tail[:-crossfade] + blended + audio[crossfade:]
                if len(previous_tail) > crossfade_len:
                    to_send = np.concatenate([
                        previous_tail[:-crossfade_len],
                        blended,
                        audio_array[crossfade_len:-CROSSFADE_SAMPLES] if not is_last else audio_array[crossfade_len:]
                    ])
                else:
                    to_send = np.concatenate([
                        blended,
                        audio_array[crossfade_len:-CROSSFADE_SAMPLES] if not is_last else audio_array[crossfade_len:]
                    ])
            else:
                # No crossfade possible, just concatenate
                to_send = audio_array[:-CROSSFADE_SAMPLES] if not is_last else audio_array

            # Store tail for next crossfade (if not last)
            if not is_last:
                previous_tail = audio_array[-CROSSFADE_SAMPLES:]
            else:
                previous_tail = None

        else:
            # First chunk - send most of it, hold back tail for crossfade
            if not is_last and len(audio_array) > CROSSFADE_SAMPLES:
                to_send = audio_array[:-CROSSFADE_SAMPLES]
                previous_tail = audio_array[-CROSSFADE_SAMPLES:]
            else:
                # Only one chunk or too short - send it all
                to_send = audio_array
                previous_tail = None

        # Send this chunk
        if len(to_send) > 0:
            audio_int16 = (np.clip(to_send, -1.0, 1.0) * 32767).astype(np.int16)
            total_audio_samples += len(audio_int16)
            chunk_count += 1

            stats = {
                "chunk": chunk_count,
                "tokens": end_idx - start_idx,
                "t_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

            if chunk_count == 1:
                stats["t_first_chunk_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

            yield audio_int16.tobytes(), stats

    # Final stats
    t_total = time.perf_counter() - t_start
    audio_duration = total_audio_samples / SAMPLE_RATE

    final_stats = {
        "type": "complete",
        "stats": {
            "t_total_ms": round(t_total * 1000, 2),
            "t_tokens_ms": round((t_tokens_done - t_start) * 1000, 2),
            "audio_duration_ms": round(audio_duration * 1000, 2),
            "realtime_factor": round(t_total / audio_duration, 3) if audio_duration > 0 else 0,
            "total_tokens": len(all_tokens),
            "chunks_sent": chunk_count,
            "overlap_tokens": overlap_tokens,
        }
    }
    yield None, final_stats


async def generate_streaming_stateful(
    text: str,
    tokens_per_chunk: int = TOKENS_PER_CHUNK,
) -> AsyncGenerator[tuple[bytes, dict], None]:
    """
    Generate audio with HiFiGAN cache-based streaming.

    The CFM (tokens→mel) doesn't support chunked inference, so we:
    1. Generate all tokens (T3)
    2. Run full CFM inference → get all mels
    3. Chunk the mels and stream through HiFiGAN with cache_source

    The cache_source mechanism in HiFiGAN maintains audio continuity.

    Yields:
        (audio_bytes, stats_dict) tuples
    """
    t_start = time.perf_counter()

    # Prepare text
    text = punc_norm(text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)

    # Setup logits processors
    logits_processors = LogitsProcessorList([
        TemperatureLogitsWarper(0.8),
        TopKLogitsWarper(1000),
        TopPLogitsWarper(0.95),
        RepetitionPenaltyLogitsProcessor(1.2),
    ])

    t3 = model.t3
    t3_cond = model.conds.t3

    # Initialize T3 generation
    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )

    # First forward pass
    llm_outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    hidden_states = llm_outputs[0]
    past_key_values = llm_outputs.past_key_values

    speech_logits = t3.speech_head(hidden_states[:, -1:])
    processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
    probs = F.softmax(processed_logits, dim=-1)
    current_token = torch.multinomial(probs, num_samples=1)

    all_tokens = [current_token]

    t_first_token = time.perf_counter()

    # Generate ALL tokens first
    for i in range(1000):
        current_embed = t3.speech_emb(current_token)
        llm_outputs = t3.tfmr(
            inputs_embeds=current_embed,
            past_key_values=past_key_values,
            use_cache=True
        )
        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values
        speech_logits = t3.speech_head(hidden_states)

        input_ids = torch.cat(all_tokens, dim=1)
        processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])

        if torch.all(processed_logits == -float("inf")):
            break

        probs = F.softmax(processed_logits, dim=-1)
        current_token = torch.multinomial(probs, num_samples=1)

        all_tokens.append(current_token)

        if torch.all(current_token == t3.hp.stop_speech_token):
            break

    t_tokens_done = time.perf_counter()
    logger.debug(f"Generated {len(all_tokens)} tokens in {(t_tokens_done - t_start)*1000:.0f}ms")

    # Remove EOS if present
    if len(all_tokens) > 0 and torch.all(all_tokens[-1] == t3.hp.stop_speech_token):
        all_tokens = all_tokens[:-1]

    # Concatenate all tokens and run full CFM inference
    speech_tokens = torch.cat(all_tokens, dim=1).squeeze(0)
    speech_tokens = speech_tokens[speech_tokens < 6561]  # Filter invalid
    speech_tokens = speech_tokens.to(model.device)

    # Add silence padding
    silence = torch.tensor([S3GEN_SIL]).long().to(model.device)
    speech_tokens = torch.cat([speech_tokens, silence])

    # Run full CFM inference (tokens -> mels)
    def _cfm_inference():
        mels = model.s3gen.flow_inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=2,
            finalize=True,
        )
        return mels.to(dtype=model.s3gen.dtype)

    all_mels = await asyncio.to_thread(_cfm_inference)
    t_cfm_done = time.perf_counter()
    logger.debug(f"CFM inference done in {(t_cfm_done - t_tokens_done)*1000:.0f}ms, mels shape: {all_mels.shape}")

    # Now chunk the mels and stream through HiFiGAN with cache
    # Mels are shape [1, 80, T] where T = 2 * num_tokens (roughly)
    # Each token = ~40ms of audio, so tokens_per_chunk tokens = tokens_per_chunk * 2 mel frames
    mel_frames_per_chunk = tokens_per_chunk * 2
    total_mel_frames = all_mels.shape[2]  # Time dimension is last

    cache_source = None
    chunk_count = 0
    total_audio_samples = 0

    start_frame = 0
    while start_frame < total_mel_frames:
        end_frame = min(start_frame + mel_frames_per_chunk, total_mel_frames)
        mel_chunk = all_mels[:, :, start_frame:end_frame]  # Slice time dimension

        def _hifigan_inference(mels, cache):
            # Truncate cache if it's larger than what current chunk needs
            # This can happen with variable-sized chunks (last chunk is smaller)
            if cache is not None and cache.shape[2] > 0:
                # Calculate expected source size: mel_frames * upsample_factor
                # HiFiGAN upsamples: 8 * 5 * 3 * 4 (istft hop) = 480x
                expected_source_size = mels.shape[2] * 480
                if cache.shape[2] > expected_source_size:
                    # Only keep the tail that fits
                    cache = cache[:, :, -expected_source_size:]

            wav, new_cache = model.s3gen.hift_inference(mels, cache_source=cache)

            # Only keep a reasonable tail for the next chunk's continuity
            # ~100ms worth of source signal should be enough for glitch-free transitions
            CACHE_TAIL_SAMPLES = 2400  # ~100ms at 24kHz
            if new_cache.shape[2] > CACHE_TAIL_SAMPLES:
                new_cache = new_cache[:, :, -CACHE_TAIL_SAMPLES:]

            return wav.squeeze(0).detach().cpu().numpy(), new_cache

        wav, cache_source = await asyncio.to_thread(_hifigan_inference, mel_chunk, cache_source)

        # Volume boost and clip
        wav = np.clip(wav * 1.5, -1.0, 1.0)

        # Convert to int16
        audio_int16 = (wav * 32767).astype(np.int16)
        total_audio_samples += len(audio_int16)
        chunk_count += 1

        stats = {
            "chunk": chunk_count,
            "mel_frames": end_frame - start_frame,
            "t_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 2),
        }

        if chunk_count == 1:
            stats["t_first_chunk_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

        yield audio_int16.tobytes(), stats

        start_frame = end_frame

    # Final stats
    t_total = time.perf_counter() - t_start
    audio_duration = total_audio_samples / SAMPLE_RATE

    final_stats = {
        "type": "complete",
        "stats": {
            "t_total_ms": round(t_total * 1000, 2),
            "t_tokens_ms": round((t_tokens_done - t_start) * 1000, 2),
            "t_cfm_ms": round((t_cfm_done - t_tokens_done) * 1000, 2),
            "audio_duration_ms": round(audio_duration * 1000, 2),
            "realtime_factor": round(t_total / audio_duration, 3) if audio_duration > 0 else 0,
            "total_tokens": len(all_tokens),
            "total_mel_frames": total_mel_frames,
            "chunks_sent": chunk_count,
        }
    }
    yield None, final_stats


async def process_token_chunk_raw(tokens: list) -> np.ndarray:
    """Process a chunk of tokens through S3Gen to get raw float audio."""
    speech_tokens = torch.cat(tokens, dim=1).squeeze(0)
    speech_tokens = speech_tokens[speech_tokens < 6561]
    speech_tokens = speech_tokens.to(model.device)

    silence = torch.tensor([S3GEN_SIL]).long().to(model.device)
    speech_tokens = torch.cat([speech_tokens, silence])

    def _inference():
        wav, _ = model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=2,
        )
        return wav.squeeze(0).detach().cpu().numpy()

    wav = await asyncio.to_thread(_inference)

    # Apply volume boost but return as float for crossfading
    wav = np.clip(wav * 1.5, -1.0, 1.0)
    return wav


async def process_token_chunk(tokens: list, is_first: bool) -> tuple[bytes, int]:
    """Process a chunk of tokens through S3Gen to get audio."""
    # Concatenate tokens
    speech_tokens = torch.cat(tokens, dim=1).squeeze(0)

    # Filter invalid tokens and add silence
    speech_tokens = speech_tokens[speech_tokens < 6561]
    speech_tokens = speech_tokens.to(model.device)

    # Add silence padding for better chunk boundaries
    silence = torch.tensor([S3GEN_SIL]).long().to(model.device)
    speech_tokens = torch.cat([speech_tokens, silence])

    # Run S3Gen inference
    def _inference():
        wav, _ = model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=2,
        )
        return wav.squeeze(0).detach().cpu().numpy()

    wav = await asyncio.to_thread(_inference)

    # Convert to PCM int16
    wav = np.clip(wav, -1.0, 1.0)

    # Apply volume boost
    wav = wav * 1.5
    wav = np.clip(wav, -1.0, 1.0)

    audio_int16 = (wav * 32767).astype(np.int16)

    return audio_int16.tobytes(), len(audio_int16)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global model, generation_lock

    device, map_location = setup_device()
    model = load_model(device, map_location)
    generation_lock = asyncio.Lock()

    logger.info("Server ready to accept connections")
    yield
    logger.info("Shutting down server...")


app = FastAPI(
    title="Chatterbox Turbo Streaming TTS",
    description="WebSocket-based streaming TTS server (v0.5 with flow-level streaming)",
    lifespan=lifespan,
)


@app.websocket("/ws/tts")
async def tts_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS.

    Streams audio chunks as they're generated (true streaming).
    """
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected")

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            text = request.get("text", "")
            tokens_per_chunk = request.get("tokens_per_chunk", TOKENS_PER_CHUNK)

            if not text:
                await websocket.send_json({"type": "error", "message": "No text provided"})
                continue

            logger.info(f"Client {client_id}: '{text[:50]}...' (tokens_per_chunk={tokens_per_chunk})")

            async with generation_lock:
                try:
                    async for audio_bytes, stats in generate_streaming(text, tokens_per_chunk):
                        if audio_bytes is not None:
                            await websocket.send_bytes(audio_bytes)
                            logger.debug(f"Sent chunk {stats.get('chunk', '?')}")
                        else:
                            # Final stats
                            await websocket.send_json(stats)
                            s = stats.get("stats", {})
                            logger.info(
                                f"Client {client_id} complete: "
                                f"total={s.get('t_total_ms', 0):.0f}ms, "
                                f"audio={s.get('audio_duration_ms', 0):.0f}ms, "
                                f"rtf={s.get('realtime_factor', 0):.2f}x"
                            )
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Client {client_id} error: {e}")


@app.websocket("/ws/tts/crossfade")
async def tts_crossfade_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS with overlap-add crossfade.

    Generates overlapping chunks and crossfades at boundaries for smooth audio.
    """
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected (crossfade mode)")

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            text = request.get("text", "")
            tokens_per_chunk = request.get("tokens_per_chunk", TOKENS_PER_CHUNK)
            overlap_tokens = request.get("overlap_tokens", OVERLAP_TOKENS)

            if not text:
                await websocket.send_json({"type": "error", "message": "No text provided"})
                continue

            logger.info(
                f"Client {client_id}: '{text[:50]}...' "
                f"(tokens={tokens_per_chunk}, overlap={overlap_tokens})"
            )

            async with generation_lock:
                try:
                    async for audio_bytes, stats in generate_streaming_crossfade(
                        text, tokens_per_chunk, overlap_tokens
                    ):
                        if audio_bytes is not None:
                            await websocket.send_bytes(audio_bytes)
                            logger.debug(f"Sent chunk {stats.get('chunk', '?')}")
                        else:
                            await websocket.send_json(stats)
                            s = stats.get("stats", {})
                            logger.info(
                                f"Client {client_id} complete (crossfade): "
                                f"total={s.get('t_total_ms', 0):.0f}ms, "
                                f"audio={s.get('audio_duration_ms', 0):.0f}ms, "
                                f"rtf={s.get('realtime_factor', 0):.2f}x"
                            )
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Client {client_id} error: {e}")


@app.websocket("/ws/tts/stateful")
async def tts_stateful_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS with proper state passing.

    Uses S3Gen's built-in streaming:
    - finalize=False for intermediate chunks (3-token lookahead)
    - HiFiGAN cache_source for seamless audio continuity
    """
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected (stateful mode)")

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            text = request.get("text", "")
            tokens_per_chunk = request.get("tokens_per_chunk", TOKENS_PER_CHUNK)

            if not text:
                await websocket.send_json({"type": "error", "message": "No text provided"})
                continue

            logger.info(
                f"Client {client_id}: '{text[:50]}...' "
                f"(tokens={tokens_per_chunk}, stateful)"
            )

            async with generation_lock:
                try:
                    async for audio_bytes, stats in generate_streaming_stateful(
                        text, tokens_per_chunk
                    ):
                        if audio_bytes is not None:
                            await websocket.send_bytes(audio_bytes)
                            logger.debug(f"Sent chunk {stats.get('chunk', '?')} (finalize={stats.get('finalize', False)})")
                        else:
                            await websocket.send_json(stats)
                            s = stats.get("stats", {})
                            logger.info(
                                f"Client {client_id} complete (stateful): "
                                f"total={s.get('t_total_ms', 0):.0f}ms, "
                                f"audio={s.get('audio_duration_ms', 0):.0f}ms, "
                                f"rtf={s.get('realtime_factor', 0):.2f}x"
                            )
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Client {client_id} error: {e}")


@app.websocket("/ws/tts/flow")
async def tts_flow_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for TRUE streaming TTS with z_cache continuity.

    This is the optimal streaming mode that:
    - Generates tokens one at a time (T3 autoregressive)
    - Runs CFM on accumulated tokens with z_cache for consistent latent trajectory
    - Uses HiFiGAN cache_source for seamless audio continuity

    Unlike /ws/tts/stateful which processes mels independently, this endpoint
    maintains flow-level state so all chunks share the same noise trajectory,
    eliminating boundary artifacts.
    """
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected (flow mode - true streaming)")

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            text = request.get("text", "")
            tokens_per_chunk = request.get("tokens_per_chunk", TOKENS_PER_CHUNK)

            if not text:
                await websocket.send_json({"type": "error", "message": "No text provided"})
                continue

            logger.info(f"Client {client_id} (flow): '{text[:50]}...' (tokens_per_chunk={tokens_per_chunk})")

            async with generation_lock:
                t_start = time.perf_counter()

                # Create streamer with z_cache support for noise continuity
                streamer = S3GenStreamer(model, tokens_per_chunk=tokens_per_chunk)

                # Prepare T3 generation
                text_normalized = punc_norm(text)
                text_tokens = model.tokenizer(text_normalized, return_tensors="pt", padding=True, truncation=True)
                text_tokens = text_tokens.input_ids.to(model.device)

                # Setup logits processors
                logits_processors = LogitsProcessorList([
                    TemperatureLogitsWarper(0.8),
                    TopKLogitsWarper(1000),
                    TopPLogitsWarper(0.95),
                    RepetitionPenaltyLogitsProcessor(1.2),
                ])

                t3 = model.t3
                t3_cond = model.conds.t3

                # Initialize T3
                speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
                embeds, _ = t3.prepare_input_embeds(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    speech_tokens=speech_start_token,
                    cfg_weight=0.0,
                )

                llm_outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
                hidden_states = llm_outputs[0]
                past_key_values = llm_outputs.past_key_values

                speech_logits = t3.speech_head(hidden_states[:, -1:])
                processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
                probs = F.softmax(processed_logits, dim=-1)
                current_token = torch.multinomial(probs, num_samples=1)

                all_tokens = [current_token]
                streamer.append_token(current_token)

                chunk_count = 0
                t_first_chunk = None
                last_emitted_count = 0  # Track tokens at last emission

                try:
                    # Generate tokens one at a time
                    for i in range(1000):
                        current_embed = t3.speech_emb(current_token)
                        llm_outputs = t3.tfmr(
                            inputs_embeds=current_embed,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                        hidden_states = llm_outputs[0]
                        past_key_values = llm_outputs.past_key_values
                        speech_logits = t3.speech_head(hidden_states)

                        input_ids = torch.cat(all_tokens, dim=1)
                        processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])

                        if torch.all(processed_logits == -float("inf")):
                            break

                        probs = F.softmax(processed_logits, dim=-1)
                        current_token = torch.multinomial(probs, num_samples=1)

                        all_tokens.append(current_token)
                        streamer.append_token(current_token)

                        # Check for EOS
                        if torch.all(current_token == t3.hp.stop_speech_token):
                            break

                        # Process chunk when enough NEW tokens accumulated since last emission
                        new_tokens = len(streamer.speech_tokens) - last_emitted_count
                        if new_tokens >= tokens_per_chunk:
                            audio = await asyncio.to_thread(streamer.step, False)
                            last_emitted_count = len(streamer.speech_tokens)

                            if len(audio) > 0:
                                chunk_count += 1
                                if t_first_chunk is None:
                                    t_first_chunk = time.perf_counter()
                                    logger.success(f"First chunk in {(t_first_chunk - t_start)*1000:.0f}ms")

                                await websocket.send_bytes(audio.tobytes())
                                logger.debug(f"Sent chunk {chunk_count} ({new_tokens} new tokens)")

                    # Final chunk with finalize=True
                    remaining_tokens = len(streamer.speech_tokens) - last_emitted_count
                    audio = await asyncio.to_thread(streamer.step, True)
                    if len(audio) > 0:
                        chunk_count += 1
                        await websocket.send_bytes(audio.tobytes())
                        logger.debug(f"Sent chunk {chunk_count} (final, {remaining_tokens} remaining tokens)")

                    # Stats
                    t_total = time.perf_counter() - t_start
                    total_samples = streamer.total_mels_emitted * 480  # approx samples
                    audio_duration = total_samples / SAMPLE_RATE

                    final_stats = {
                        "type": "complete",
                        "stats": {
                            "t_total_ms": round(t_total * 1000, 2),
                            "t_first_chunk_ms": round((t_first_chunk - t_start) * 1000, 2) if t_first_chunk else 0,
                            "audio_duration_ms": round(audio_duration * 1000, 2),
                            "realtime_factor": round(t_total / audio_duration, 3) if audio_duration > 0 else 0,
                            "total_tokens": len(all_tokens),
                            "chunks_sent": chunk_count,
                            "mode": "flow_streaming",
                        }
                    }
                    await websocket.send_json(final_stats)
                    logger.info(f"Client {client_id} complete (flow): rtf={final_stats['stats']['realtime_factor']:.2f}x, chunks={chunk_count}")

                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Client {client_id} error: {e}")
        import traceback
        traceback.print_exc()


@app.websocket("/ws/tts/simple")
async def tts_simple_websocket(websocket: WebSocket):
    """
    Simple WebSocket endpoint - generates full audio then streams chunks.

    Use this for comparison with the streaming endpoint.
    """
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected (simple mode)")

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            text = request.get("text", "")
            chunk_duration_ms = request.get("chunk_duration_ms", DEFAULT_CHUNK_DURATION_MS)

            if not text:
                await websocket.send_json({"type": "error", "message": "No text provided"})
                continue

            async with generation_lock:
                t_start = time.perf_counter()

                wav = await asyncio.to_thread(model.generate, text)
                wav = wav * 1.5
                wav = torch.clamp(wav, -1.0, 1.0)

                t_gen = time.perf_counter()

                # Convert to PCM
                audio = wav.squeeze().numpy()
                audio = np.clip(audio, -1.0, 1.0)
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

                # Chunk and stream
                samples_per_chunk = int((chunk_duration_ms / 1000) * SAMPLE_RATE)
                bytes_per_chunk = samples_per_chunk * 2

                chunks_sent = 0
                for i in range(0, len(audio_bytes), bytes_per_chunk):
                    chunk = audio_bytes[i:i + bytes_per_chunk]
                    await websocket.send_bytes(chunk)
                    chunks_sent += 1

                t_total = time.perf_counter()
                audio_duration = len(audio_int16) / SAMPLE_RATE

                stats = {
                    "type": "complete",
                    "stats": {
                        "t_generation_ms": round((t_gen - t_start) * 1000, 2),
                        "t_total_ms": round((t_total - t_start) * 1000, 2),
                        "audio_duration_ms": round(audio_duration * 1000, 2),
                        "realtime_factor": round((t_gen - t_start) / audio_duration, 3),
                        "chunks_sent": chunks_sent,
                    }
                }
                await websocket.send_json(stats)
                logger.info(f"Client {client_id} (simple): rtf={stats['stats']['realtime_factor']:.2f}x")

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": model.device if model else None,
        "voice_loaded": model.conds is not None if model else False,
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Chatterbox Turbo Streaming TTS Server v0.5 (with flow streaming)...")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8765,
        workers=1,
        log_level="info",
    )
