# Chatterbox Turbo Streaming TTS - Architecture & Experiment Log

## Overview

This document tracks the evolution of our streaming TTS implementation, from initial PRD through current working prototype, with proposed next steps.

---

## 1. Initial Proposed Architecture (PRD v0.1)

### Original Goal
WebSocket server that generates **complete audio first**, then streams it in chunks. This was meant to validate infrastructure before investing in true streaming.

### Original Flow
```
Client sends text
    ↓
Server generates FULL audio (wait for completion)
    ↓
Split into chunks (e.g., 200ms each)
    ↓
Stream chunks to client
    ↓
Client plays chunks
```

### Expected Metrics
- Time to first audio: **Equal to full generation time** (no improvement)
- Purpose: Establish baseline, validate WebSocket infrastructure

### Limitations
- No latency improvement - just chunked delivery of already-generated audio
- Full generation must complete before any audio plays

---

## 2. PRD Counter-Proposal: True Streaming Architecture

### The Pipeline
```
Text → T3 (autoregressive tokens) → S3Gen (tokens→mel via CFM) → HiFiGAN (mel→wav)
```

### Key Insight
Each stage has different streaming characteristics:

| Stage | Streamable? | Effort | Notes |
|-------|-------------|--------|-------|
| T3 (text→tokens) | Yes, trivially | Low | Already autoregressive, yields 1 token at a time |
| S3Gen (CFM decoder) | Partially | Medium-High | Has `finalize` param, but CFM processes full sequence |
| HiFiGAN (vocoder) | Yes | Low | Has `cache_source` param for continuation |

### Proposed Approach (Path A)
- Generate tokens in batches (10-25 at a time)
- Run S3Gen on each batch independently
- Accept quality tradeoffs at chunk boundaries
- Target: ~500ms time-to-first-audio

---

## 3. Current Implementation (v0.2)

### What We Built

```
┌─────────────────────────────────────────────────────────────┐
│                    WebSocket Server                          │
├─────────────────────────────────────────────────────────────┤
│  Startup:                                                    │
│    1. Load ChatterboxTurboTTS model (~12s)                  │
│    2. Load pre-cached voice conditionals (0ms!)             │
│                                                              │
│  Request Flow:                                               │
│    Text received                                             │
│         ↓                                                    │
│    T3 generates tokens (one at a time, with KV-cache)       │
│         ↓ every N tokens                                     │
│    S3Gen processes chunk → audio bytes                      │
│         ↓                                                    │
│    Send audio chunk via WebSocket                           │
│         ↓                                                    │
│    Continue until EOS token                                  │
│         ↓                                                    │
│    Send final stats JSON                                     │
└─────────────────────────────────────────────────────────────┘
```

### Two Endpoints

1. **`/ws/tts`** - Streaming mode
   - Chunks tokens during generation
   - Audio starts arriving before full generation completes

2. **`/ws/tts/simple`** - Baseline mode
   - Full generation, then chunked delivery
   - Better RTF, worse latency

### Key Optimizations

**1. Pre-cached Voice Conditionals**
```python
# OLD: Process .mp3 every time (~9 seconds)
model.prepare_conditionals("qb.mp3")

# NEW: Load pre-computed .pt file (0ms)
model.conds = Conditionals.load("voices/qb_voice.pt")
```

**2. Chunked Token Processing**
```python
# Generate tokens one at a time
for i in range(max_gen_len):
    token = generate_next_token()
    chunk_tokens.append(token)

    # Process chunk when we have enough
    if len(chunk_tokens) >= tokens_per_chunk:
        audio = s3gen.inference(chunk_tokens)
        yield audio  # Stream immediately!
        chunk_tokens = []
```

---

## 4. Performance Results

### Test Environment
- **Hardware**: MacBook Pro M-series (MPS acceleration)
- **Model**: Chatterbox Turbo 350M
- **Voice**: Pre-cached qb_voice.pt

### Benchmark Results

#### Short Text: "Hello world"
| Mode | First Audio | Total Time | Audio Duration | RTF |
|------|-------------|------------|----------------|-----|
| Streaming (10 tokens) | 3559ms | 5945ms | 1480ms | 4.02x |
| Simple | 1829ms | 1829ms | 1360ms | **1.33x** |

#### Longer Text: "Hello my big long beautiful world"
| Mode | First Audio | Total Time | Audio Duration | RTF |
|------|-------------|------------|----------------|-----|
| Streaming (10 tokens) | **1306ms** | 5221ms | 3000ms | 1.74x |
| Simple | ~2500ms | ~2500ms | ~3000ms | ~0.83x |

### Key Finding
**First audio in 1.3 seconds** for longer text! The streaming approach shines when:
- Text is longer (more tokens to generate)
- User perception matters (hearing audio start is better than waiting)

### Breakdown of Gains

| Optimization | Impact |
|--------------|--------|
| Pre-cached voice (.pt vs .mp3) | **9 seconds saved** at startup |
| Chunked streaming vs full generation | **~50% reduction** in time-to-first-audio |
| MPS acceleration | ~2x faster than CPU |

---

## 5. Test Logs

### Server Startup
```
2025-12-16 12:56:59 | INFO     | Starting Chatterbox Turbo Streaming TTS Server v0.2...
2025-12-16 12:56:59 | INFO     | Detected device: MPS
2025-12-16 12:56:59 | SUCCESS  | Using Metal Performance Shaders (GPU acceleration)
2025-12-16 12:56:59 | INFO     | Loading Chatterbox Turbo model...
2025-12-16 12:57:11 | SUCCESS  | Model loaded in 12.44s
2025-12-16 12:57:11 | INFO     | Loading voice: voices/qb_voice.pt
2025-12-16 12:57:11 | SUCCESS  | Voice loaded in 0.01s  ← KEY: Instant voice load!
2025-12-16 12:57:11 | INFO     | Server ready to accept connections
```

### Streaming Request: "Hello my big long beautiful world" (10 tokens/chunk)
```
2025-12-16 13:00:52 | INFO     | Client connected
2025-12-16 13:00:52 | INFO     | Request: 'Hello my big long beautiful world...' (tokens_per_chunk=10)

S3 Token -> Mel Inference...
100%|████████████████████| 2/2 [00:00<00:00, 12.43it/s]
2025-12-16 13:00:53 | DEBUG    | Sent chunk 1  ← 891ms from request

S3 Token -> Mel Inference...
100%|████████████████████| 2/2 [00:00<00:00, 10.58it/s]
2025-12-16 13:00:53 | DEBUG    | Sent chunk 2

S3 Token -> Mel Inference...
100%|████████████████████| 2/2 [00:00<00:00, 12.74it/s]
2025-12-16 13:00:54 | DEBUG    | Sent chunk 3

S3 Token -> Mel Inference...
100%|████████████████████| 2/2 [00:00<00:00, 13.35it/s]
2025-12-16 13:00:54 | DEBUG    | Sent chunk 4

S3 Token -> Mel Inference...
100%|████████████████████| 2/2 [00:00<00:00, 13.42it/s]
2025-12-16 13:00:55 | DEBUG    | Sent chunk 5

S3 Token -> Mel Inference...
100%|████████████████████| 2/2 [00:00<00:00, 12.57it/s]
2025-12-16 13:00:56 | DEBUG    | Sent chunk 6

S3 Token -> Mel Inference...
100%|████████████████████| 2/2 [00:00<00:00,  8.79it/s]
2025-12-16 13:00:57 | DEBUG    | Sent chunk 7

2025-12-16 13:00:57 | INFO     | Complete: total=5221ms, audio=3000ms, rtf=1.74x
```

### Client Output
```
13:00:52 | INFO     | Connecting to ws://localhost:8765/ws/tts...
13:00:52 | SUCCESS  | Connected!
13:00:52 | INFO     | Sent: 'Hello my big long beautiful world...'
13:00:53 | SUCCESS  | First audio chunk in 1306ms!  ← KEY METRIC

13:00:57 | INFO     | =======================================================
13:00:57 | INFO     | GENERATION COMPLETE
13:00:57 | INFO     | =======================================================
13:00:57 | INFO     |   Chunks received: 7
13:00:57 | INFO     |   Total audio: 144.0 KB
13:00:57 | INFO     |
13:00:57 | INFO     | Server Metrics:
13:00:57 | INFO     |   Total time: 5221ms
13:00:57 | INFO     |   Audio duration: 3000ms
13:00:57 | INFO     |   Realtime factor: 1.74x
13:00:57 | INFO     |   Tokens generated: 69
13:00:57 | INFO     |
13:00:57 | INFO     | Client Metrics:
13:00:57 | INFO     |   Time to first audio: 1306ms
13:00:57 | INFO     | =======================================================
```

---

## 6. Current Issue: Chunking Artifacts

### Problem
Audio chunks are processed independently through S3Gen, causing audible breaks/discontinuities at chunk boundaries.

### Why It Happens
```
Chunk 1: tokens [0-9]   → S3Gen → audio_1  ┐
Chunk 2: tokens [10-19] → S3Gen → audio_2  │ No continuity between chunks!
Chunk 3: tokens [20-29] → S3Gen → audio_3  ┘
```

Each S3Gen call:
1. Starts fresh with no context from previous chunk
2. Adds silence padding at the end
3. Has its own fade-in/fade-out characteristics

### Result
Audio sounds "chopped up" - noticeable breaks between chunks even though playback is continuous.

---

## 7. Proposed Experiments

Three parallel experiments to improve audio quality while maintaining low latency:

### Experiment A: Overlap-Add Crossfade

**Hypothesis**: Overlapping chunks and crossfading at boundaries will smooth transitions.

**Approach**:
```
Chunk 1: tokens [0-12]     → audio_1
Chunk 2: tokens [10-22]    → audio_2  (2 tokens overlap)
Chunk 3: tokens [20-32]    → audio_3

Crossfade:
audio_1[end-overlap:] × fade_out + audio_2[:overlap] × fade_in
```

**Implementation**:
1. Generate overlapping token ranges
2. Process each chunk through S3Gen
3. Apply crossfade at overlap region (50-100ms)
4. Concatenate smoothed audio

**Pros**: Simple, no model changes, predictable behavior
**Cons**: Requires re-processing overlap tokens, may have artifacts if prosody differs

**Effort**: ~2-4 hours

---

### Experiment B: Parallel T3 + S3Gen Pipeline

**Hypothesis**: Running T3 and S3Gen in parallel will reduce total latency and allow more continuous generation.

**Approach**:
```
Thread 1 (T3):     [generate tokens continuously] → token_queue
Thread 2 (S3Gen):  [consume from queue, process] → audio_queue
Thread 3 (Stream): [send audio as available]
```

**Implementation**:
1. T3 runs continuously, putting tokens into async queue
2. S3Gen worker consumes batches from queue
3. Audio sent to client as soon as available
4. Better pipelining = less waiting

**Pros**: Better utilization, smoother flow
**Cons**: Doesn't fix chunk boundary issue directly, more complex

**Effort**: ~4-6 hours

---

### Experiment C: Accumulating Context S3Gen

**Hypothesis**: Passing previous chunk's context to S3Gen will produce continuous audio.

**Approach**:
```python
# First chunk
audio_1, cache_1 = s3gen.inference(tokens[0:10], cache=None)

# Subsequent chunks - pass previous context
audio_2, cache_2 = s3gen.inference(tokens[10:20], cache=cache_1)
audio_3, cache_3 = s3gen.inference(tokens[20:30], cache=cache_2)
```

**Investigation Needed**:
- Does S3Gen support caching/continuation?
- HiFiGAN has `cache_source` parameter - can we use it?
- What state needs to be preserved between chunks?

**Implementation**:
1. Investigate S3Gen flow module for continuation support
2. Modify `flow_inference` to accept/return cache
3. Chain chunks with context preservation

**Pros**: Most natural solution, could eliminate artifacts entirely
**Cons**: Requires deeper model understanding, may need model modifications

**Effort**: ~1-2 days (exploration) + ~1 day (implementation)

---

## 8. Recommended Priority

| Priority | Experiment | Rationale |
|----------|------------|-----------|
| 1 | **A: Overlap-Add** | Quick win, no model changes, validates approach |
| 2 | **C: Accumulating Context** | Best long-term solution if S3Gen supports it |
| 3 | **B: Parallel Pipeline** | Optimization, do after quality is solved |

### Suggested Workflow
1. **Day 1**: Implement Experiment A (overlap-add)
2. **Day 1-2**: Test and measure audio quality
3. **Day 2-3**: Investigate Experiment C (check S3Gen internals)
4. **Day 3+**: Implement best solution based on findings

---

## 9. Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Time to first audio | 1.3s | <1s |
| Chunk artifacts | Audible breaks | Smooth/unnoticeable |
| RTF (streaming) | 1.74x | <1.5x |
| RTF (simple) | 1.33x | Maintain |

---

## 10. Files Reference

```
streaming-tts-experiment/
├── server.py              # WebSocket server with streaming
├── client.py              # Test client with audio playback
├── requirements.txt       # Dependencies
├── README.md              # Quick start guide
└── docs/
    └── ARCHITECTURE.md    # This document
```

---

## Appendix: Key Code Paths

### Voice Conditional Caching
```python
# Save voice conditionals (one-time)
model.prepare_conditionals("qb.mp3")
model.conds.save("voices/qb_voice.pt")

# Load pre-cached (instant)
model.conds = Conditionals.load("voices/qb_voice.pt")
```

### Streaming Generation Loop
```python
# server.py: generate_streaming()
async def generate_streaming(text, tokens_per_chunk=10):
    # ... T3 initialization ...

    for i in range(max_gen_len):
        token = generate_next_token()
        chunk_tokens.append(token)

        if len(chunk_tokens) >= tokens_per_chunk:
            audio = await process_token_chunk(chunk_tokens)
            yield audio  # Stream to client!
            chunk_tokens = []
```

### S3Gen Inference (Current)
```python
# Each chunk processed independently
def process_token_chunk(tokens):
    wav, _ = model.s3gen.inference(
        speech_tokens=tokens,
        ref_dict=model.conds.gen,
        n_cfm_timesteps=2,  # MeanFlow = 2 steps
    )
    return wav
```
