"""
Chatterbox Turbo Streaming TTS Client (v0.5)

Test client that connects to the WebSocket server, sends text,
receives audio chunks in real-time, and plays them as they arrive.

Supports multiple modes:
- Streaming: Chunks arrive during generation
- Flow: TRUE streaming with flow_cache continuity (RECOMMENDED - no chunk artifacts)
- Stateful: HiFiGAN cache + finalize control for seamless audio
- Crossfade: Overlap-add crossfade for smooth transitions
- Simple: Full generation then stream
"""
import argparse
import asyncio
import json
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import websockets
from loguru import logger

# Configuration
SAMPLE_RATE = 24000
DEFAULT_TOKENS_PER_CHUNK = 25
DEFAULT_SERVER_URL = "ws://localhost:8765/ws/tts"


class StreamingAudioPlayer:
    """
    Real-time audio player with optional jitter buffer.

    Uses sentinel-only termination to ensure all buffered audio is played
    before stopping. Call stop() to signal end and wait for playback to finish.

    Jitter buffer mode:
    - Buffers audio until jitter_buffer_ms worth of audio is accumulated
    - Then starts playback, ensuring smooth continuous audio even if
      chunks arrive with variable timing
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, jitter_buffer_ms: int = 0):
        self.sample_rate = sample_rate
        self.jitter_buffer_ms = jitter_buffer_ms
        self.bytes_per_second = sample_rate * 2  # int16 = 2 bytes per sample
        self.jitter_buffer_bytes = int((jitter_buffer_ms / 1000) * self.bytes_per_second)

        self.audio_queue = queue.Queue()
        self.playback_thread = None
        self.first_chunk_time = None
        self.playback_started_time = None
        self.total_samples_played = 0

        # Jitter buffer state
        self.jitter_buffer = []
        self.jitter_buffer_size = 0
        self.jitter_buffer_ready = threading.Event()
        self.stream_ended = False

    def start(self):
        """Start the audio playback thread."""
        self.first_chunk_time = None
        self.playback_started_time = None
        self.total_samples_played = 0
        self.jitter_buffer = []
        self.jitter_buffer_size = 0
        self.jitter_buffer_ready.clear()
        self.stream_ended = False
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def stop(self):
        """
        Stop the audio playback gracefully.

        Pushes a sentinel to the queue and waits for the playback thread
        to consume all remaining audio before returning. This ensures
        no audio is lost at the end.
        """
        self.stream_ended = True
        # If jitter buffer hasn't been flushed yet, signal it's ready (stream ended)
        self.jitter_buffer_ready.set()
        # Push sentinel to tell the playback loop to exit when done
        self.audio_queue.put(None)
        # Wait for playback to finish (all audio consumed)
        if self.playback_thread:
            self.playback_thread.join(timeout=10.0)

    def add_chunk(self, audio_bytes: bytes):
        """Add an audio chunk to the playback queue (or jitter buffer)."""
        if self.first_chunk_time is None:
            self.first_chunk_time = time.perf_counter()

        if self.jitter_buffer_ms > 0 and not self.jitter_buffer_ready.is_set():
            # Accumulate in jitter buffer until threshold reached
            self.jitter_buffer.append(audio_bytes)
            self.jitter_buffer_size += len(audio_bytes)

            buffered_ms = (self.jitter_buffer_size / self.bytes_per_second) * 1000
            if self.jitter_buffer_size >= self.jitter_buffer_bytes:
                logger.debug(f"Jitter buffer full ({buffered_ms:.0f}ms), starting playback")
                # Flush buffer to queue
                for chunk in self.jitter_buffer:
                    self.audio_queue.put(chunk)
                self.jitter_buffer = []
                self.jitter_buffer_ready.set()
        else:
            # Normal mode or jitter buffer already flushed
            self.audio_queue.put(audio_bytes)

    def _playback_loop(self):
        """
        Main playback loop running in separate thread.

        Runs until it receives a None sentinel, ensuring all buffered
        audio is played before exiting.
        """
        try:
            # Wait for jitter buffer if enabled
            if self.jitter_buffer_ms > 0:
                self.jitter_buffer_ready.wait()
                # Flush any remaining buffer (in case stream ended early)
                for chunk in self.jitter_buffer:
                    self.audio_queue.put(chunk)
                self.jitter_buffer = []

            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
                blocksize=2048,
            ) as stream:
                while True:
                    # Block until we get a chunk (no timeout - wait forever)
                    chunk = self.audio_queue.get()

                    # Sentinel signals end of stream
                    if chunk is None:
                        break

                    if self.playback_started_time is None:
                        self.playback_started_time = time.perf_counter()

                    audio = np.frombuffer(chunk, dtype=np.int16)
                    self.total_samples_played += len(audio)
                    stream.write(audio)

        except Exception as e:
            logger.error(f"Playback error: {e}")


async def run_streaming_client(text: str, server_url: str, tokens_per_chunk: int):
    """Connect to streaming endpoint and play audio in real-time."""
    logger.info(f"Connecting to {server_url}...")

    player = StreamingAudioPlayer()
    t_request_start = time.perf_counter()

    try:
        async with websockets.connect(server_url) as ws:
            logger.success("Connected!")

            request = {
                "text": text,
                "tokens_per_chunk": tokens_per_chunk,
            }
            await ws.send(json.dumps(request))
            logger.info(f"Sent: '{text[:60]}...'")

            player.start()

            chunks_received = 0
            total_bytes = 0
            t_first_audio = None

            async for message in ws:
                if isinstance(message, bytes):
                    chunks_received += 1
                    total_bytes += len(message)
                    player.add_chunk(message)

                    if chunks_received == 1:
                        t_first_audio = time.perf_counter()
                        latency = (t_first_audio - t_request_start) * 1000
                        logger.success(f"First audio chunk in {latency:.0f}ms!")

                else:
                    data = json.loads(message)

                    if data.get("type") == "error":
                        logger.error(f"Server error: {data.get('message')}")
                        break

                    elif data.get("type") == "complete":
                        stats = data.get("stats", {})

                        logger.info("")
                        logger.info("=" * 55)
                        logger.info("GENERATION COMPLETE")
                        logger.info("=" * 55)
                        logger.info(f"  Text: '{text[:45]}...'")
                        logger.info(f"  Chunks received: {chunks_received}")
                        logger.info(f"  Total audio: {total_bytes / 1000:.1f} KB")
                        logger.info("")
                        logger.info("Server Metrics:")
                        logger.info(f"  Total time: {stats.get('t_total_ms', 0):.0f}ms")
                        logger.info(f"  Audio duration: {stats.get('audio_duration_ms', 0):.0f}ms")
                        logger.info(f"  Realtime factor: {stats.get('realtime_factor', 0):.2f}x")
                        logger.info(f"  Tokens generated: {stats.get('total_tokens', 0)}")
                        logger.info("")
                        logger.info("Client Metrics:")
                        if t_first_audio:
                            logger.info(f"  Time to first audio: {(t_first_audio - t_request_start)*1000:.0f}ms")
                        logger.info("=" * 55)
                        break

            logger.info("Waiting for playback to finish...")
            player.stop()

            t_end = time.perf_counter()
            logger.success(f"Total client session: {(t_end - t_request_start)*1000:.0f}ms")

    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"Connection closed: {e}")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise
    finally:
        player.stop()


async def run_crossfade_client(text: str, server_url: str, tokens_per_chunk: int, overlap_tokens: int):
    """Connect to crossfade endpoint with overlap-add smoothing."""
    url = server_url.replace("/ws/tts", "/ws/tts/crossfade")
    logger.info(f"Connecting to {url} (crossfade mode)...")

    player = StreamingAudioPlayer()
    t_request_start = time.perf_counter()

    try:
        async with websockets.connect(url) as ws:
            logger.success("Connected!")

            request = {
                "text": text,
                "tokens_per_chunk": tokens_per_chunk,
                "overlap_tokens": overlap_tokens,
            }
            await ws.send(json.dumps(request))
            logger.info(f"Sent: '{text[:60]}...' (tokens={tokens_per_chunk}, overlap={overlap_tokens})")

            player.start()

            chunks_received = 0
            total_bytes = 0
            t_first_audio = None

            async for message in ws:
                if isinstance(message, bytes):
                    chunks_received += 1
                    total_bytes += len(message)
                    player.add_chunk(message)

                    if chunks_received == 1:
                        t_first_audio = time.perf_counter()
                        latency = (t_first_audio - t_request_start) * 1000
                        logger.success(f"First audio chunk in {latency:.0f}ms!")

                else:
                    data = json.loads(message)

                    if data.get("type") == "error":
                        logger.error(f"Server error: {data.get('message')}")
                        break

                    elif data.get("type") == "complete":
                        stats = data.get("stats", {})

                        logger.info("")
                        logger.info("=" * 55)
                        logger.info("CROSSFADE MODE COMPLETE")
                        logger.info("=" * 55)
                        logger.info(f"  Text: '{text[:45]}...'")
                        logger.info(f"  Chunks received: {chunks_received}")
                        logger.info(f"  Total audio: {total_bytes / 1000:.1f} KB")
                        logger.info("")
                        logger.info("Server Metrics:")
                        logger.info(f"  Token gen time: {stats.get('t_tokens_ms', 0):.0f}ms")
                        logger.info(f"  Total time: {stats.get('t_total_ms', 0):.0f}ms")
                        logger.info(f"  Audio duration: {stats.get('audio_duration_ms', 0):.0f}ms")
                        logger.info(f"  Realtime factor: {stats.get('realtime_factor', 0):.2f}x")
                        logger.info(f"  Tokens: {stats.get('total_tokens', 0)}, Overlap: {stats.get('overlap_tokens', 0)}")
                        logger.info("")
                        logger.info("Client Metrics:")
                        if t_first_audio:
                            logger.info(f"  Time to first audio: {(t_first_audio - t_request_start)*1000:.0f}ms")
                        logger.info("=" * 55)
                        break

            logger.info("Waiting for playback to finish...")
            player.stop()

            t_end = time.perf_counter()
            logger.success(f"Total client session: {(t_end - t_request_start)*1000:.0f}ms")

    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"Connection closed: {e}")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise
    finally:
        player.stop()


async def run_stateful_client(text: str, server_url: str, tokens_per_chunk: int):
    """Connect to stateful endpoint with HiFiGAN cache + finalize control."""
    url = server_url.replace("/ws/tts", "/ws/tts/stateful")
    logger.info(f"Connecting to {url} (stateful mode)...")

    player = StreamingAudioPlayer()
    t_request_start = time.perf_counter()

    try:
        async with websockets.connect(url) as ws:
            logger.success("Connected!")

            request = {
                "text": text,
                "tokens_per_chunk": tokens_per_chunk,
            }
            await ws.send(json.dumps(request))
            logger.info(f"Sent: '{text[:60]}...' (tokens={tokens_per_chunk}, stateful)")

            player.start()

            chunks_received = 0
            total_bytes = 0
            t_first_audio = None

            async for message in ws:
                if isinstance(message, bytes):
                    chunks_received += 1
                    total_bytes += len(message)
                    player.add_chunk(message)

                    if chunks_received == 1:
                        t_first_audio = time.perf_counter()
                        latency = (t_first_audio - t_request_start) * 1000
                        logger.success(f"First audio chunk in {latency:.0f}ms!")

                else:
                    data = json.loads(message)

                    if data.get("type") == "error":
                        logger.error(f"Server error: {data.get('message')}")
                        break

                    elif data.get("type") == "complete":
                        stats = data.get("stats", {})

                        logger.info("")
                        logger.info("=" * 55)
                        logger.info("STATEFUL MODE COMPLETE")
                        logger.info("=" * 55)
                        logger.info(f"  Text: '{text[:45]}...'")
                        logger.info(f"  Chunks received: {chunks_received}")
                        logger.info(f"  Total audio: {total_bytes / 1000:.1f} KB")
                        logger.info("")
                        logger.info("Server Metrics:")
                        logger.info(f"  Token gen: {stats.get('t_tokens_ms', 0):.0f}ms")
                        logger.info(f"  CFM (mel): {stats.get('t_cfm_ms', 0):.0f}ms")
                        logger.info(f"  Total time: {stats.get('t_total_ms', 0):.0f}ms")
                        logger.info(f"  Audio duration: {stats.get('audio_duration_ms', 0):.0f}ms")
                        logger.info(f"  Realtime factor: {stats.get('realtime_factor', 0):.2f}x")
                        logger.info(f"  Tokens: {stats.get('total_tokens', 0)}, Mel frames: {stats.get('total_mel_frames', 0)}")
                        logger.info("")
                        logger.info("Client Metrics:")
                        if t_first_audio:
                            logger.info(f"  Time to first audio: {(t_first_audio - t_request_start)*1000:.0f}ms")
                        logger.info("=" * 55)
                        break

            logger.info("Waiting for playback to finish...")
            player.stop()

            t_end = time.perf_counter()
            logger.success(f"Total client session: {(t_end - t_request_start)*1000:.0f}ms")

    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"Connection closed: {e}")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise
    finally:
        player.stop()


async def run_flow_client(
    text: str,
    server_url: str,
    tokens_per_chunk: int,
    jitter_buffer_ms: int = 0,
    cfm_steps: int = 2,
):
    """
    Connect to flow endpoint - TRUE streaming with flow_cache continuity.

    This is the RECOMMENDED mode for streaming TTS. It maintains flow-level
    state between chunks, ensuring all chunks share the same latent trajectory
    without boundary artifacts.

    Args:
        text: Text to synthesize
        server_url: WebSocket server URL
        tokens_per_chunk: Tokens per audio chunk
        jitter_buffer_ms: Buffer this much audio before starting playback (0=disabled)
        cfm_steps: Number of CFM ODE solver steps (1=fastest, 2=default)
    """
    url = server_url.replace("/ws/tts", "/ws/tts/flow")

    jitter_info = f", jitter={jitter_buffer_ms}ms" if jitter_buffer_ms > 0 else ""
    cfm_info = f", cfm_steps={cfm_steps}" if cfm_steps != 2 else ""
    logger.info(f"Connecting to {url} (flow mode{jitter_info}{cfm_info})...")

    player = StreamingAudioPlayer(jitter_buffer_ms=jitter_buffer_ms)
    t_request_start = time.perf_counter()

    try:
        async with websockets.connect(url) as ws:
            logger.success("Connected!")

            request = {
                "text": text,
                "tokens_per_chunk": tokens_per_chunk,
                "n_cfm_timesteps": cfm_steps,
            }
            await ws.send(json.dumps(request))
            logger.info(f"Sent: '{text[:60]}...' (tokens={tokens_per_chunk}, cfm={cfm_steps})")

            player.start()

            chunks_received = 0
            total_bytes = 0
            t_first_audio = None
            t_playback_start = None

            async for message in ws:
                if isinstance(message, bytes):
                    chunks_received += 1
                    total_bytes += len(message)
                    player.add_chunk(message)

                    if chunks_received == 1:
                        t_first_audio = time.perf_counter()
                        latency = (t_first_audio - t_request_start) * 1000
                        logger.success(f"First audio chunk in {latency:.0f}ms!")

                    # Track when playback actually starts (jitter buffer may delay it)
                    if t_playback_start is None and player.playback_started_time is not None:
                        t_playback_start = player.playback_started_time

                else:
                    data = json.loads(message)

                    if data.get("type") == "error":
                        logger.error(f"Server error: {data.get('message')}")
                        break

                    elif data.get("type") == "complete":
                        stats = data.get("stats", {})

                        logger.info("")
                        logger.info("=" * 55)
                        logger.info("FLOW MODE COMPLETE (TRUE STREAMING)")
                        logger.info("=" * 55)
                        logger.info(f"  Text: '{text[:45]}...'")
                        logger.info(f"  Chunks received: {chunks_received}")
                        logger.info(f"  Total audio: {total_bytes / 1000:.1f} KB")
                        if jitter_buffer_ms > 0:
                            logger.info(f"  Jitter buffer: {jitter_buffer_ms}ms")
                        logger.info("")
                        logger.info("Server Metrics:")
                        logger.info(f"  First chunk: {stats.get('t_first_chunk_ms', 0):.0f}ms")
                        logger.info(f"  Total time: {stats.get('t_total_ms', 0):.0f}ms")
                        logger.info(f"  Audio duration: {stats.get('audio_duration_ms', 0):.0f}ms")
                        logger.info(f"  Realtime factor: {stats.get('realtime_factor', 0):.2f}x")
                        logger.info(f"  Tokens: {stats.get('total_tokens', 0)}")
                        logger.info(f"  CFM steps: {stats.get('n_cfm_timesteps', cfm_steps)}")
                        logger.info("")
                        logger.info("Client Metrics:")
                        if t_first_audio:
                            logger.info(f"  Time to first audio: {(t_first_audio - t_request_start)*1000:.0f}ms")
                        if jitter_buffer_ms > 0 and player.playback_started_time:
                            playback_delay = (player.playback_started_time - t_request_start) * 1000
                            logger.info(f"  Time to playback start: {playback_delay:.0f}ms")
                        logger.info("=" * 55)
                        break

            logger.info("Waiting for playback to finish...")
            player.stop()

            t_end = time.perf_counter()
            logger.success(f"Total client session: {(t_end - t_request_start)*1000:.0f}ms")

    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"Connection closed: {e}")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise
    finally:
        player.stop()


async def run_simple_client(text: str, server_url: str, chunk_duration_ms: int):
    """Connect to simple endpoint (full generation then stream)."""
    # Use simple endpoint
    url = server_url.replace("/ws/tts", "/ws/tts/simple")
    logger.info(f"Connecting to {url} (simple mode)...")

    player = StreamingAudioPlayer()
    t_request_start = time.perf_counter()

    try:
        async with websockets.connect(url) as ws:
            logger.success("Connected!")

            request = {
                "text": text,
                "chunk_duration_ms": chunk_duration_ms,
            }
            await ws.send(json.dumps(request))
            logger.info(f"Sent: '{text[:60]}...'")

            player.start()

            chunks_received = 0
            t_first_audio = None

            async for message in ws:
                if isinstance(message, bytes):
                    chunks_received += 1
                    player.add_chunk(message)

                    if chunks_received == 1:
                        t_first_audio = time.perf_counter()
                        latency = (t_first_audio - t_request_start) * 1000
                        logger.info(f"First audio after {latency:.0f}ms (after full generation)")

                else:
                    data = json.loads(message)
                    if data.get("type") == "complete":
                        stats = data.get("stats", {})
                        logger.info("")
                        logger.info("=" * 55)
                        logger.info("SIMPLE MODE COMPLETE")
                        logger.info("=" * 55)
                        logger.info(f"  Generation time: {stats.get('t_generation_ms', 0):.0f}ms")
                        logger.info(f"  Audio duration: {stats.get('audio_duration_ms', 0):.0f}ms")
                        logger.info(f"  Realtime factor: {stats.get('realtime_factor', 0):.2f}x")
                        logger.info("=" * 55)
                        break

            logger.info("Waiting for playback to finish...")
            player.stop()

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        player.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Chatterbox Turbo Streaming TTS Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Flow mode (TRUE streaming with flow_cache) - RECOMMENDED
  python client.py "Hello, this is a test." --flow

  # Flow mode with custom chunk size
  python client.py "Hello world" --flow --tokens 20

  # Streaming mode (chunks arrive during generation)
  python client.py "Hello, this is a test."

  # Stateful mode (HiFiGAN cache for seamless audio)
  python client.py "Hello world" --stateful

  # Crossfade mode (smooth chunk transitions)
  python client.py "Hello world" --crossfade

  # Simple mode (full generation then stream)
  python client.py "Hello world" --simple
        """,
    )
    parser.add_argument(
        "text",
        type=str,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=DEFAULT_SERVER_URL,
        help=f"WebSocket server URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_TOKENS_PER_CHUNK,
        help=f"Tokens per audio chunk (default: {DEFAULT_TOKENS_PER_CHUNK})",
    )
    parser.add_argument(
        "--flow",
        action="store_true",
        help="Use flow mode (RECOMMENDED - true streaming with flow_cache, no chunk artifacts)",
    )
    parser.add_argument(
        "--stateful",
        action="store_true",
        help="Use stateful mode (HiFiGAN cache + finalize control for seamless audio)",
    )
    parser.add_argument(
        "--crossfade",
        action="store_true",
        help="Use crossfade mode (overlap-add for smooth transitions)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=3,
        help="Overlap tokens for crossfade mode (default: 3)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple mode (full generation then stream)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=200,
        help="Chunk duration in ms for simple mode (default: 200)",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=0,
        help="Jitter buffer in ms - buffer audio before playback to avoid hiccups (default: 0, try 800)",
    )
    parser.add_argument(
        "--cfm-steps",
        type=int,
        default=2,
        help="CFM ODE solver steps - 1=fastest, 2=default quality (default: 2)",
    )

    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )

    if args.simple:
        asyncio.run(run_simple_client(args.text, args.server, args.chunk))
    elif args.flow:
        asyncio.run(run_flow_client(
            args.text,
            args.server,
            args.tokens,
            jitter_buffer_ms=args.jitter,
            cfm_steps=args.cfm_steps,
        ))
    elif args.stateful:
        asyncio.run(run_stateful_client(args.text, args.server, args.tokens))
    elif args.crossfade:
        asyncio.run(run_crossfade_client(args.text, args.server, args.tokens, args.overlap))
    else:
        asyncio.run(run_streaming_client(args.text, args.server, args.tokens))


if __name__ == "__main__":
    main()
