"""
Chatterbox Turbo Streaming TTS Server (v0.2)

WebSocket server with:
- Pre-cached voice conditionals (instant load)
- Chunked audio streaming during generation
- MPS acceleration on Mac
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

# Global model instance
model: ChatterboxTurboTTS = None
generation_lock: asyncio.Lock = None


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
    description="WebSocket-based streaming TTS server (v0.2 with chunked generation)",
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

    logger.info("Starting Chatterbox Turbo Streaming TTS Server v0.2...")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8765,
        workers=1,
        log_level="info",
    )
