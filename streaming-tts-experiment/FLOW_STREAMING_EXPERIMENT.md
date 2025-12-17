# Flow-Level Streaming Experiment (v0.5)

## Date: 2025-12-16

## What We Built

Implemented TRUE flow-level streaming for Chatterbox TTS by caching the noise vector `z` between streaming calls. This ensures all chunks share a coherent latent trajectory through the CFM (Conditional Flow Matching) ODE solver.

### Files Modified

| File | Changes |
|------|---------|
| `src/chatterbox/models/s3gen/flow_matching.py` | Added `forward_streaming()` to `CausalConditionalCFM` |
| `src/chatterbox/models/s3gen/flow.py` | Added `inference_streaming()` to `CausalMaskedDiffWithXvec` |
| `src/chatterbox/models/s3gen/s3gen.py` | Added `flow_stream_step()` to `S3Token2Wav` |
| `streaming-tts-experiment/server.py` | Added `S3GenStreamer` class and `/ws/tts/flow` endpoint |
| `streaming-tts-experiment/client.py` | Added `--flow` flag |

### Key Insight

The original architecture already supported streaming via:
- `pre_lookahead_len=3` (3-token lookahead for causal attention)
- `finalize` parameter (controls whether lookahead frames are included)
- `cache_source` in HiFiGAN (waveform continuity)

What was missing: **noise continuity in CFM**. The ODE solver uses noise `z` to generate mel spectrograms. Without caching `z`, each chunk starts with fresh random noise, causing discontinuities.

### Solution: z_cache

```python
# In forward_streaming():
z = torch.randn_like(mu)

# Restore cached noise for previously generated frames
if z_cache is not None:
    T_cached = min(z_cache.shape[2], T_total)
    z[:, :, :T_cached] = z_cache[:, :, :T_cached]

# ... run ODE solver ...

# Cache ALL noise for next call (sequence grows cumulatively)
new_z_cache = z.clone()
```

---

## Test Results

### What Worked
- z_cache mechanism successfully maintains noise continuity
- Audio chunks DO sound connected (continuing context)
- HiFiGAN cache_source provides waveform continuity
- No tensor size mismatch errors after fixes

### What Didn't Work
- **48 chunks** for a ~4 second sentence (should be ~4 chunks)
- Each chunk is tiny (~0.25-0.5 seconds)
- Audio sounds "extruded" - gaps between chunks
- Client timeout before completion
- **TTFB: 2915ms** (worse than batch mode)

### Root Cause Analysis

The current implementation is **cumulative but inefficient**:

```
Token 1:   Run CFM on 1 token   → emit 0 audio (waiting for chunk)
Token 25:  Run CFM on 25 tokens → emit ~1s audio
Token 26:  Run CFM on 26 tokens → emit ~40ms audio (delta)
Token 50:  Run CFM on 50 tokens → emit ~1s audio
...
Token 200: Run CFM on 200 tokens → emit ~40ms audio (delta)
```

**Problems:**
1. **O(n²) complexity**: Re-running CFM on ALL tokens every step
2. **Diminishing returns**: Later chunks take longer but yield tiny deltas
3. **No parallelism**: T3 token generation is blocked during CFM inference

---

## Proposed Fixes

### Fix 1: Chunk-Based Emission (Quick Win)

Only run CFM when we have a full chunk of NEW tokens, not after every token:

```python
# Current (bad): Run CFM whenever tokens >= threshold
if len(streamer.speech_tokens) >= tokens_per_chunk:
    audio = streamer.step()

# Better: Run CFM only when NEW tokens since last emission >= threshold
if len(streamer.speech_tokens) - streamer.last_emitted_tokens >= tokens_per_chunk:
    audio = streamer.step()
    streamer.last_emitted_tokens = len(streamer.speech_tokens)
```

**Expected improvement**: 48 chunks → ~4-6 chunks

### Fix 2: Sliding Window CFM (Medium Effort)

Instead of re-running CFM on ALL tokens, only process recent tokens plus overlap:

```python
# Only process: overlap_context + new_tokens + lookahead
window_start = max(0, len(tokens) - window_size)
window_tokens = tokens[window_start:]
mels = cfm(window_tokens, z_cache=z_cache[window_start:])
```

**Expected improvement**: O(n²) → O(n), consistent latency per chunk

### Fix 3: Pipeline Parallelism (Best Quality)

Run T3 and CFM in parallel using a producer-consumer pattern:

```
Thread 1 (T3):     [tok1][tok2][tok3]...[tok25] → queue
Thread 2 (CFM):    [wait]...[process 25 tokens] → mel queue
Thread 3 (HiFiGAN): [wait]...[process mels] → audio out
```

**Expected improvement**: Much better TTFB, true real-time streaming

---

## Metrics Comparison

| Mode | TTFB | RTF | Chunks | Quality |
|------|------|-----|--------|---------|
| `/ws/tts/simple` (batch) | ~2000ms | 1.5x | 1 | Good |
| `/ws/tts/stateful` | ~1500ms | 1.7x | ~6 | Good |
| `/ws/tts/flow` (v1 - broken) | ~2900ms | N/A | 48 | Choppy |
| `/ws/tts/flow` (v2 - Fix 1) | ~1413ms | 1.49x | 4 | **Smooth!** |

### Fix 1 Results (2025-12-16)

```
Server Metrics:
  First chunk: 1414ms
  Total time: 5001ms
  Audio duration: 3360ms
  Realtime factor: 1.49x
  Tokens: 84
  Chunks: 4

Client Metrics:
  Time to first audio: 2412ms
```

**Key achievement**: Audio chunks continue smoothly from each other with no boundary artifacts!

### Final Working Version (2025-12-16)

After fixing client-side audio cutoff bug:

```
Server Metrics:
  First chunk: 1256ms
  Total time: 4422ms
  Audio duration: 3440ms
  Realtime factor: 1.28x
  Tokens: 87
  Chunks: 4

Client Metrics:
  Time to first audio: 2114ms
  Total session: 5839ms
```

**FULLY WORKING**: Complete audio plays smoothly with no cutoff!

#### Bugs Fixed:
1. **Double lookahead** - removed duplicate 6-frame crop in `step()`
2. **z_cache alignment** - moved crop to AFTER flow (preserves lookahead in cache)
3. **Client audio cutoff** - rewrote `StreamingAudioPlayer` to use sentinel-only termination

---

### Smaller First Chunk Optimization (2025-12-16)

**Change**: Use 12 tokens for first chunk instead of 25, then 25 for subsequent chunks.

**Rationale**: Faster TTFB by processing fewer tokens initially.

```
Server Metrics:
  First chunk: 723-881ms  (down from 1256ms!)
  Total time: 3519-6129ms
  Audio duration: 3080-4720ms
  Realtime factor: 1.14-1.30x
  Tokens: 78-119
  Chunks: 4-6

Client Metrics:
  Time to first audio: 1565-1836ms (down from 2114ms)
```

**Results**:
- Server TTFB improved by ~500ms ✅
- Client TTFB improved by ~500ms ✅
- RTF slightly improved ✅

**Remaining Issue**: ~900ms gap between server send and client receive
- Server says first chunk at 881ms
- Client says first audio at 1836ms
- Gap: ~955ms (consistent across tests)

**Observations**:
1. First chunk plays smoothly
2. **Hiccup between chunks** - if playback catches up to processing, there's a gap
3. The problem is the sequential nature: T3 → CFM → HiFiGAN → Send → Play

---

## Open Questions for Deep Analysis

### Question 1: Why is there a ~900ms client-side latency gap?

Server sends first chunk at 881ms, client receives at 1836ms. Where does the extra 955ms go?

Possible causes:
- WebSocket serialization/deserialization overhead
- asyncio event loop blocking during `asyncio.to_thread()`
- Client-side buffering before `add_chunk()` is called
- sounddevice/PortAudio initialization latency

### Question 2: How to eliminate chunk hiccups?

Current flow is BLOCKING:
```
T3: [gen 12 tok]────────[gen 25 tok]────────[gen 25 tok]
CFM:            [proc]              [proc]              [proc]
Play:                [play]              [play]
                      ^                   ^
                   hiccup if          hiccup if
                   play > proc        play > proc
```

The hiccup happens when:
- Chunk N finishes playing
- Chunk N+1 is still being processed by CFM
- Result: silence gap

### Question 3: Would pipeline parallelism fix this?

Proposed parallel flow:
```
T3:   [gen 12][gen 25][gen 25][gen 25]
CFM:       [proc 1][proc 2][proc 3]
Send:           [1][2][3][4]
Play:              [1][2][3][4]
```

Benefits:
- T3 keeps generating while CFM processes
- Audio buffer stays full
- No hiccups

Challenges:
- Thread synchronization
- z_cache needs to be updated atomically
- Memory management for queued tokens

---

## Remaining Paths for Improvement

### Path A: Reduce TTFB Further
Current TTFB is ~1.4s server-side. To get lower:
- **Smaller first chunk**: Emit after 15 tokens instead of 25 for first chunk only
- **Speculative decoding**: Start CFM while T3 is still generating
- **Model optimization**: Quantization, torch.compile, etc.

### Path B: Handle Long Utterances (O(n²) problem)
Current implementation re-runs CFM on ALL tokens each chunk. For long text:
- **Sliding window**: Only process `window_size` tokens + overlap
- **Incremental CFM**: Only compute new frames (requires architecture changes)

### Path C: Pipeline Parallelism
Run T3/CFM/HiFiGAN in parallel threads:
```
T3 ──tokens──> CFM ──mels──> HiFiGAN ──audio──> Client
```
This would achieve true real-time streaming with minimal latency.

### Path D: Quality Improvements
- **Silence padding**: Add S3GEN_SIL token at end for cleaner cutoff
- **Fade out**: Apply fade to final chunk to avoid abrupt end
- **Compare to batch**: Spectrogram analysis vs non-streaming output

---

## Code Artifacts

### S3GenStreamer (server.py:64-172)
Stateful wrapper maintaining z_cache and cache_source between steps.

### flow_stream_step (s3gen.py:330-380)
Single streaming step: tokens → mel with z_cache continuity.

### forward_streaming (flow_matching.py:236-294)
CFM forward pass with z_cache restore/save.

### inference_streaming (flow.py:200-295)
Flow inference wrapper passing z_cache through encoder/decoder.
