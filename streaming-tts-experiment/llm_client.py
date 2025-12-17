"""
LLM Streaming → TTS Client

Simulates LLM streaming and pipes text to the TTS server for real-time audio.

Usage:
    # Simulated LLM mode
    python llm_client.py --simulate "Your text to speak here."

    # With jitter buffer
    python llm_client.py --simulate --jitter 800 "Your text here."
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
DEFAULT_SERVER_URL = "ws://localhost:8765/ws/tts/llm-stream"


class StreamingAudioPlayer:
    """Real-time audio player with optional jitter buffer."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, jitter_buffer_ms: int = 0):
        self.sample_rate = sample_rate
        self.jitter_buffer_ms = jitter_buffer_ms
        self.bytes_per_second = sample_rate * 2
        self.jitter_buffer_bytes = int((jitter_buffer_ms / 1000) * self.bytes_per_second)

        self.audio_queue = queue.Queue()
        self.playback_thread = None
        self.first_chunk_time = None
        self.playback_started_time = None
        self.total_samples_played = 0

        self.jitter_buffer = []
        self.jitter_buffer_size = 0
        self.jitter_buffer_ready = threading.Event()
        self.stream_ended = False

    def start(self):
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
        self.stream_ended = True
        self.jitter_buffer_ready.set()
        self.audio_queue.put(None)
        if self.playback_thread:
            self.playback_thread.join(timeout=10.0)

    def add_chunk(self, audio_bytes: bytes):
        if self.first_chunk_time is None:
            self.first_chunk_time = time.perf_counter()

        if self.jitter_buffer_ms > 0 and not self.jitter_buffer_ready.is_set():
            self.jitter_buffer.append(audio_bytes)
            self.jitter_buffer_size += len(audio_bytes)

            if self.jitter_buffer_size >= self.jitter_buffer_bytes:
                buffered_ms = (self.jitter_buffer_size / self.bytes_per_second) * 1000
                logger.debug(f"Jitter buffer full ({buffered_ms:.0f}ms), starting playback")
                for chunk in self.jitter_buffer:
                    self.audio_queue.put(chunk)
                self.jitter_buffer = []
                self.jitter_buffer_ready.set()
        else:
            self.audio_queue.put(audio_bytes)

    def _playback_loop(self):
        try:
            if self.jitter_buffer_ms > 0:
                self.jitter_buffer_ready.wait()
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
                    chunk = self.audio_queue.get()
                    if chunk is None:
                        break
                    if self.playback_started_time is None:
                        self.playback_started_time = time.perf_counter()
                    audio = np.frombuffer(chunk, dtype=np.int16)
                    self.total_samples_played += len(audio)
                    stream.write(audio)
        except Exception as e:
            logger.error(f"Playback error: {e}")


async def simulate_llm_stream(text: str, words_per_chunk: int = 3, delay_ms: int = 50):
    """
    Simulate LLM streaming by yielding word chunks.

    Args:
        text: Full text to stream
        words_per_chunk: Words per chunk (simulates token batching)
        delay_ms: Delay between chunks (simulates LLM latency)

    Yields:
        Text chunks
    """
    words = text.split()
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i:i + words_per_chunk])
        yield chunk + " "
        await asyncio.sleep(delay_ms / 1000)


async def run_llm_stream_client(
    text: str,
    server_url: str = DEFAULT_SERVER_URL,
    jitter_buffer_ms: int = 0,
    words_per_chunk: int = 3,
    delay_ms: int = 50,
):
    """
    Run the LLM stream → TTS client.

    Args:
        text: Text to synthesize (will be streamed word by word)
        server_url: WebSocket server URL
        jitter_buffer_ms: Jitter buffer size
        words_per_chunk: Words per simulated LLM chunk
        delay_ms: Delay between chunks
    """
    logger.info(f"Connecting to {server_url}...")

    player = StreamingAudioPlayer(jitter_buffer_ms=jitter_buffer_ms)
    t_start = time.perf_counter()

    try:
        async with websockets.connect(server_url) as ws:
            logger.success("Connected!")

            # Send initial config
            config = {
                "type": "config",
                "tokens_per_chunk": 25,
                "n_cfm_timesteps": 2,
            }
            await ws.send(json.dumps(config))

            player.start()

            chunks_received = 0
            total_bytes = 0
            t_first_audio = None

            # Create tasks for sending and receiving
            async def send_stream():
                async for chunk in simulate_llm_stream(text, words_per_chunk, delay_ms):
                    await ws.send(json.dumps({"type": "text", "content": chunk}))
                    logger.debug(f"Sent: '{chunk.strip()}'")
                await ws.send(json.dumps({"type": "end"}))
                logger.info("Sent end signal")

            async def receive_audio():
                nonlocal chunks_received, total_bytes, t_first_audio

                async for message in ws:
                    if isinstance(message, bytes):
                        chunks_received += 1
                        total_bytes += len(message)
                        player.add_chunk(message)

                        if chunks_received == 1:
                            t_first_audio = time.perf_counter()
                            latency = (t_first_audio - t_start) * 1000
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
                            logger.info("LLM STREAM → TTS COMPLETE")
                            logger.info("=" * 55)
                            logger.info(f"  Text: '{text[:45]}...'")
                            logger.info(f"  Chunks received: {chunks_received}")
                            logger.info(f"  Total audio: {total_bytes / 1000:.1f} KB")
                            logger.info("")
                            logger.info("Server Metrics:")
                            logger.info(f"  First audio: {stats.get('t_first_audio_ms', 0):.0f}ms")
                            logger.info(f"  Total time: {stats.get('t_total_ms', 0):.0f}ms")
                            logger.info(f"  Audio duration: {stats.get('audio_duration_ms', 0):.0f}ms")
                            logger.info(f"  RTF: {stats.get('realtime_factor', 0):.2f}x")
                            logger.info(f"  Sentences: {stats.get('sentences_processed', 0)}")
                            logger.info("")
                            logger.info("Client Metrics:")
                            if t_first_audio:
                                logger.info(f"  Time to first audio: {(t_first_audio - t_start)*1000:.0f}ms")
                            if jitter_buffer_ms > 0 and player.playback_started_time:
                                playback_delay = (player.playback_started_time - t_start) * 1000
                                logger.info(f"  Time to playback: {playback_delay:.0f}ms")
                            logger.info("=" * 55)
                            break

            # Run both concurrently
            await asyncio.gather(send_stream(), receive_audio())

            logger.info("Waiting for playback to finish...")
            player.stop()

            t_end = time.perf_counter()
            logger.success(f"Total session: {(t_end - t_start)*1000:.0f}ms")

    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"Connection closed: {e}")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise
    finally:
        player.stop()


def main():
    parser = argparse.ArgumentParser(
        description="LLM Streaming → TTS Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulate LLM streaming
  python llm_client.py --simulate "Hello, this is a test of LLM streaming to TTS."

  # With jitter buffer for smooth playback
  python llm_client.py --simulate --jitter 800 "Hello world, this is smooth."

  # Adjust streaming speed
  python llm_client.py --simulate --words 5 --delay 30 "Faster streaming test."
        """,
    )
    parser.add_argument(
        "text",
        type=str,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated LLM streaming (required for now)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=DEFAULT_SERVER_URL,
        help=f"WebSocket server URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=0,
        help="Jitter buffer in ms (default: 0, try 800 for smooth playback)",
    )
    parser.add_argument(
        "--words",
        type=int,
        default=3,
        help="Words per chunk in simulated mode (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=50,
        help="Delay between chunks in ms (default: 50)",
    )

    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )

    if not args.simulate:
        logger.error("Currently only --simulate mode is supported")
        sys.exit(1)

    asyncio.run(run_llm_stream_client(
        args.text,
        args.server,
        jitter_buffer_ms=args.jitter,
        words_per_chunk=args.words,
        delay_ms=args.delay,
    ))


if __name__ == "__main__":
    main()
