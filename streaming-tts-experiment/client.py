"""
Chatterbox Turbo Streaming TTS Client (v0.2)

Test client that connects to the WebSocket server, sends text,
receives audio chunks in real-time, and plays them as they arrive.
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
    """Real-time audio player that plays chunks as they arrive."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.playing = False
        self.playback_thread = None
        self.stream = None
        self.first_chunk_time = None
        self.playback_started_time = None
        self.total_samples_played = 0

    def start(self):
        """Start the audio playback thread."""
        self.playing = True
        self.first_chunk_time = None
        self.playback_started_time = None
        self.total_samples_played = 0
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def stop(self):
        """Stop the audio playback."""
        self.playing = False
        # Clear any remaining audio
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)

    def add_chunk(self, audio_bytes: bytes):
        """Add an audio chunk to the playback queue."""
        if self.first_chunk_time is None:
            self.first_chunk_time = time.perf_counter()
        self.audio_queue.put(audio_bytes)

    def signal_end(self):
        """Signal that no more chunks are coming."""
        self.audio_queue.put(None)

    def _playback_loop(self):
        """Main playback loop running in separate thread."""
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
                blocksize=2048,
            )
            self.stream.start()

            while self.playing:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)

                    if chunk is None:
                        break

                    if self.playback_started_time is None:
                        self.playback_started_time = time.perf_counter()

                    audio = np.frombuffer(chunk, dtype=np.int16)
                    self.total_samples_played += len(audio)
                    self.stream.write(audio)

                except queue.Empty:
                    continue

            if self.stream:
                self.stream.stop()
                self.stream.close()

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
                        player.signal_end()
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

            logger.info("Waiting for playback...")
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
                        player.signal_end()
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
  # Streaming mode (chunks arrive during generation)
  python client.py "Hello, this is a test."

  # Simple mode (full generation then stream)
  python client.py "Hello world" --simple

  # Adjust chunk size (tokens per chunk)
  python client.py "Hello world" --tokens 15

  # Connect to remote server
  python client.py "Hello" --server ws://192.168.1.100:8765/ws/tts
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
        help=f"Tokens per audio chunk for streaming mode (default: {DEFAULT_TOKENS_PER_CHUNK})",
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
    else:
        asyncio.run(run_streaming_client(args.text, args.server, args.tokens))


if __name__ == "__main__":
    main()
