# Deep Analysis Request: Chatterbox Flow Streaming

## Context

We've implemented TRUE flow-level streaming for Chatterbox TTS by caching the noise vector `z` between streaming calls. This ensures chunks share a coherent latent trajectory through the CFM ODE solver.

**Current Status**: 95% working! Audio chunks flow smoothly into each other. But the VERY LAST frame gets cut off.

## What We Built

### Architecture
```
T3 (autoregressive) → S3Gen (CFM: tokens→mel) → HiFiGAN (mel→wav)
     ↓                      ↓                        ↓
  tokens            z_cache continuity         cache_source
```

### Key Innovation: z_cache
```python
# In forward_streaming():
z = torch.randn_like(mu)  # Fresh noise for full sequence

# Restore cached noise for previously generated frames
if z_cache is not None:
    T_cached = min(z_cache.shape[2], T_total)
    z[:, :, :T_cached] = z_cache[:, :, :T_cached]

# ... run ODE solver ...

# Cache ALL noise for next call
new_z_cache = z.clone()
```

### Current Metrics
- TTFB: ~1.4s (want < 500ms)
- RTF: 1.19-1.85x
- Chunks: 4 for ~3.4s audio
- Quality: Smooth continuity EXCEPT last frame cutoff

---

## Three Questions

### Question 1: Why is the VERY LAST audio frame getting cut off?

We fixed double-lookahead and z_cache alignment, but still losing the final ~50-100ms.

**Hypothesis A**: Token filtering removes EOS but z_cache was built including it
**Hypothesis B**: HiFiGAN cache_source truncation loses final samples
**Hypothesis C**: Client-side playback issue (doesn't flush final buffer)
**Hypothesis D**: finalize=True path has different behavior than expected

Key code paths to examine:
1. `_run_flow()` token filtering: `speech_tokens[:, speech_tokens[0] < 6561]`
2. `step()` final chunk: `streamer.step(True)` with finalize=True
3. HiFiGAN cache truncation in step()
4. Client audio buffer handling

### Question 2: How to implement Pipeline Parallelism for faster chunk delivery?

Current flow is SEQUENTIAL (blocking):
```
T3: [gen 25 tok]        [gen 25 tok]        [gen 25 tok]
CFM:            [proc]              [proc]              [proc]
Out:                 [send]              [send]
    |----1s----|----1s----|----1s----|
```

Want PARALLEL (pipelined):
```
T3:  [gen 25][gen 25][gen 25][gen 25]
CFM:      [proc 1][proc 2][proc 3]
Out:           [send 1][send 2][send 3]
     |--500ms--|--500ms--|--500ms--|
```

Questions:
1. Best async pattern? (asyncio.Queue, threading, multiprocessing?)
2. How to coordinate T3 and CFM without blocking?
3. How to handle EOS signal propagation?
4. Memory considerations for queued tokens/mels?

### Question 3: How to optimize TTFB to < 500ms?

Current TTFB breakdown (estimated):
- T3 generates 25 tokens: ~400ms
- CFM processes 25 tokens: ~500ms
- HiFiGAN converts mels: ~100ms
- Network/overhead: ~100ms
- **Total: ~1100-1400ms**

Potential optimizations:
1. Smaller first chunk (15 tokens instead of 25)?
2. Speculative CFM start while T3 is still generating?
3. torch.compile() for CFM?
4. Reduce CFM timesteps (currently 2)?
5. Warm-up/pre-allocation?

---

## Files for Context

### Core Model Files (provide these)

1. **s3gen.py** - S3Token2Wav class with `flow_stream_step()`
2. **flow.py** - CausalMaskedDiffWithXvec with `inference_streaming()`
3. **flow_matching.py** - CausalConditionalCFM with `forward_streaming()`

### Server/Client (provide these)

4. **server.py** - S3GenStreamer class and `/ws/tts/flow` endpoint
5. **client.py** - WebSocket client with `--flow` mode

---

## Specific Code to Analyze

### Token Filtering (potential cutoff cause)
```python
# server.py _run_flow()
speech_tokens = speech_tokens[:, speech_tokens[0] < 6561]
```

### Lookahead Handling (we moved crop to AFTER flow)
```python
# flow.py inference_streaming()
# 8. Apply lookahead crop AFTER flow (preserves z_cache for lookahead frames)
if finalize is False:
    crop_frames = self.pre_lookahead_len * self.token_mel_ratio  # 3 * 2 = 6
    feat = feat[:, :, :-crop_frames]
```

### HiFiGAN Cache Truncation
```python
# server.py step()
# Truncate cache_source if larger than current mel chunk needs
if cache.shape[2] > expected_source_size:
    cache = cache[:, :, -expected_source_size:]

# Limit cache after use
CACHE_TAIL_SAMPLES = 2400  # ~100ms at 24kHz
if new_cache.shape[2] > CACHE_TAIL_SAMPLES:
    new_cache = new_cache[:, :, -CACHE_TAIL_SAMPLES:]
```

### Final Chunk Emission
```python
# server.py endpoint
# Final chunk with finalize=True
remaining_tokens = len(streamer.speech_tokens) - last_emitted_count
audio = await asyncio.to_thread(streamer.step, True)
if len(audio) > 0:
    chunk_count += 1
    await websocket.send_bytes(audio.tobytes())
```

---

## Expected Output

Please provide:

1. **Root cause analysis** for last-frame cutoff with specific fix
2. **Pipeline parallelism design** with code structure
3. **TTFB optimization strategy** prioritized by impact/effort
4. **Any architectural concerns** with current approach

Focus on practical, implementable solutions. We're close to production-ready streaming!
