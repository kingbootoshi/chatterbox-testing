# Chatterbox Turbo Streaming TTS Server

Real-time streaming TTS over WebSocket with **1.3 second time-to-first-audio**.

## Quick Results

| Metric | Value |
|--------|-------|
| Time to first audio | **1.3s** |
| Voice loading | **0ms** (pre-cached) |
| Model loading | ~12s (one-time) |
| RTF (streaming) | 1.74x |
| RTF (simple) | 1.33x |

## Quick Start

```bash
cd streaming-tts-experiment

# Start server (loads model once)
uv run python server.py

# Test streaming (in another terminal)
uv run python client.py "Hello my big long beautiful world" --tokens 10

# Compare with simple mode
uv run python client.py "Hello world" --simple
```

## Current Status

**Working**:
- Fast time-to-first-audio via chunked streaming
- Pre-cached voice conditionals (instant load)
- MPS acceleration on Mac

**Known Issue**:
- Audio has audible breaks between chunks
- See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for solutions

## Documentation

| Doc | Description |
|-----|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full technical details, logs, and analysis |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | Three proposed experiments to fix chunking |

## Architecture

```
Text → T3 (generate tokens)
         ↓ yield every 10 tokens
      S3Gen (tokens → audio)
         ↓
      WebSocket → Client
```

## API

### Streaming Endpoint: `/ws/tts`
```json
{"text": "Hello world", "tokens_per_chunk": 10}
```

### Simple Endpoint: `/ws/tts/simple`
```json
{"text": "Hello world", "chunk_duration_ms": 200}
```

### Response
- Binary frames: PCM int16 audio @ 24kHz mono
- Final JSON: `{"type": "complete", "stats": {...}}`

## Next Steps

Three parallel experiments to eliminate chunk artifacts:

1. **Overlap-Add Crossfade** - Quick win, blend chunk boundaries
2. **Parallel Pipeline** - Run T3 and S3Gen concurrently
3. **S3Gen Context** - Pass state between chunks for seamless audio

See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for details.

## Files

```
streaming-tts-experiment/
├── server.py           # WebSocket server
├── client.py           # Test client
├── requirements.txt    # Dependencies
├── README.md           # This file
└── docs/
    ├── ARCHITECTURE.md # Technical deep-dive
    └── EXPERIMENTS.md  # Proposed experiments
```
