# Streaming TTS Experiments

Three parallel experiments to eliminate chunk boundary artifacts while maintaining ~1.3s time-to-first-audio.

---

## Current State

**What's Working**:
- 1.3s time-to-first-audio (excellent!)
- Pre-cached voice loading (0ms)
- Streaming chunks during generation

**The Problem**:
- Audio sounds "chopped up" between chunks
- Each chunk processed independently through S3Gen
- No continuity at boundaries

---

## Experiment A: Overlap-Add Crossfade

### Branch: `experiment/overlap-add`

### Concept
Generate overlapping token ranges, then crossfade audio at boundaries.

```
Tokens:    [0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 ...]
Chunk 1:   [0  1  2  3  4  5  6  7  8  9 10 11]
Chunk 2:                        [8  9 10 11 12 13 14 15 16 17]
                                ^^^^^^^^^^^^
                                  overlap

Audio:
chunk_1: [.......audio_1.......]
chunk_2:              [.......audio_2.......]
                      ^^^^^^^^
                      crossfade region
```

### Implementation Steps

1. **Modify token chunking** in `server.py`:
```python
OVERLAP_TOKENS = 3  # ~120ms overlap

def chunk_tokens_with_overlap(all_tokens, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(all_tokens):
        end = min(start + chunk_size, len(all_tokens))
        chunks.append(all_tokens[start:end])
        start = end - overlap  # Overlap with next chunk
    return chunks
```

2. **Add crossfade function**:
```python
def crossfade(audio1, audio2, overlap_samples):
    """Crossfade two audio arrays at overlap region."""
    fade_out = np.linspace(1, 0, overlap_samples)
    fade_in = np.linspace(0, 1, overlap_samples)

    # Blend overlap region
    audio1_end = audio1[-overlap_samples:] * fade_out
    audio2_start = audio2[:overlap_samples] * fade_in
    blended = audio1_end + audio2_start

    # Concatenate: audio1[:-overlap] + blended + audio2[overlap:]
    return np.concatenate([
        audio1[:-overlap_samples],
        blended,
        audio2[overlap_samples:]
    ])
```

3. **Buffer chunks client-side** for crossfade:
```python
# Client accumulates chunks, applies crossfade before playback
previous_chunk = None
for chunk in receive_chunks():
    if previous_chunk is not None:
        blended = crossfade(previous_chunk, chunk, overlap_samples)
        play(blended)
    previous_chunk = chunk
```

### Test Plan
- [ ] Generate "Hello my big long beautiful world" with overlap
- [ ] Compare audio quality vs non-overlap
- [ ] Measure latency impact (overlap tokens = extra processing)
- [ ] Try different overlap sizes (2, 3, 5 tokens)

### Success Criteria
- Smooth audio transitions (no audible breaks)
- Time-to-first-audio < 1.5s
- RTF < 2.5x

---

## Experiment B: Parallel T3 + S3Gen Pipeline

### Branch: `experiment/parallel-pipeline`

### Concept
Run T3 token generation and S3Gen audio synthesis in parallel threads/tasks.

```
Time →
T3:    [gen tok 1-10][gen tok 11-20][gen tok 21-30]...
S3Gen:        [process 1-10][process 11-20][process 21-30]...
Stream:              [send audio 1][send audio 2][send audio 3]...
```

### Implementation Steps

1. **Create async queues**:
```python
import asyncio

token_queue = asyncio.Queue()
audio_queue = asyncio.Queue()
```

2. **T3 producer task**:
```python
async def t3_producer(text, token_queue):
    """Generate tokens and push to queue."""
    # ... initialization ...
    chunk = []
    for i in range(max_gen_len):
        token = generate_next_token()
        chunk.append(token)

        if len(chunk) >= TOKENS_PER_CHUNK:
            await token_queue.put(chunk)
            chunk = []

        if is_eos(token):
            break

    if chunk:
        await token_queue.put(chunk)
    await token_queue.put(None)  # Signal done
```

3. **S3Gen consumer task**:
```python
async def s3gen_consumer(token_queue, audio_queue):
    """Process token chunks into audio."""
    while True:
        chunk = await token_queue.get()
        if chunk is None:
            await audio_queue.put(None)
            break

        audio = await asyncio.to_thread(process_s3gen, chunk)
        await audio_queue.put(audio)
```

4. **Streamer task**:
```python
async def audio_streamer(audio_queue, websocket):
    """Stream audio to client."""
    while True:
        audio = await audio_queue.get()
        if audio is None:
            break
        await websocket.send_bytes(audio)
```

5. **Run all tasks concurrently**:
```python
async def generate_parallel(text, websocket):
    token_queue = asyncio.Queue(maxsize=2)
    audio_queue = asyncio.Queue(maxsize=2)

    await asyncio.gather(
        t3_producer(text, token_queue),
        s3gen_consumer(token_queue, audio_queue),
        audio_streamer(audio_queue, websocket),
    )
```

### Test Plan
- [ ] Verify tasks run in parallel (check timing logs)
- [ ] Compare total time vs sequential
- [ ] Measure queue utilization
- [ ] Test with various text lengths

### Success Criteria
- Better pipeline utilization
- Reduced total generation time
- No deadlocks or race conditions

---

## Experiment C: Accumulating Context S3Gen

### Branch: `experiment/s3gen-context`

### Concept
Pass previous chunk's hidden state/cache to S3Gen for continuous generation.

```python
# Ideal API (if S3Gen supports it)
audio_1, cache_1 = s3gen.inference(tokens[0:10], cache=None)
audio_2, cache_2 = s3gen.inference(tokens[10:20], cache=cache_1)
audio_3, cache_3 = s3gen.inference(tokens[20:30], cache=cache_2)

# Audio is continuous across chunks!
```

### Investigation Steps

1. **Examine S3Gen flow module** (`src/chatterbox/models/s3gen/flow.py`):
   - Look for `flow_cache` parameters
   - Check `finalize` parameter behavior
   - Identify what state is preserved

2. **Check HiFiGAN vocoder** (`src/chatterbox/models/s3gen/hifigan.py`):
   - `cache_source` parameter exists
   - How is it used for continuation?

3. **Examine CausalMaskedDiffWithXvec**:
   - `pre_lookahead_len` suggests streaming consideration
   - What does `finalize=False` actually do?

### Key Code to Investigate

```python
# flow.py - CausalMaskedDiffWithXvec.inference()
def inference(self, token, token_len, prompt_token, prompt_token_len,
              prompt_feat, prompt_feat_len, embedding, finalize, n_timesteps=10, ...):
    # finalize parameter - what happens when False?
    # flow_cache - NotImplementedError currently raised

# hifigan.py - HiFTGenerator.inference()
def inference(self, speech_feat, cache_source=torch.zeros(1, 1, 0)):
    # cache_source is used to "avoid glitch"
    # How do we populate this from previous chunk?
```

### Implementation (if supported)

1. **Modify S3Gen inference** to return cache:
```python
def inference_streaming(self, tokens, ref_dict, cache=None):
    # ... process with cache ...
    return audio, new_cache
```

2. **Chain chunks in server**:
```python
cache = None
for chunk in token_chunks:
    audio, cache = s3gen.inference_streaming(chunk, ref_dict, cache)
    yield audio
```

### Test Plan
- [ ] Read and understand flow.py continuation logic
- [ ] Test `finalize=False` behavior
- [ ] Implement cache passing if possible
- [ ] Compare audio quality with/without context

### Success Criteria
- Seamless audio across chunk boundaries
- No artifacts or discontinuities
- Maintains current latency

---

## Comparison Matrix

| Aspect | A: Overlap-Add | B: Parallel | C: Context |
|--------|----------------|-------------|------------|
| Effort | Low (hours) | Medium (day) | High (days) |
| Risk | Low | Low | Medium |
| Solves artifacts? | Partially | No | Fully |
| Latency impact | Slight increase | Decrease | None |
| Requires model changes? | No | No | Maybe |

---

## Recommended Order

1. **Start with A** - Quick validation, low risk
2. **Investigate C** - Read the code, understand possibilities
3. **Implement C if viable**, else improve A
4. **B as optimization** - After quality solved

---

## Quick Start for Each Experiment

```bash
# Experiment A
git checkout -b experiment/overlap-add
# Edit server.py, add overlap logic
# Test with: uv run python client.py "Test text" --tokens 10

# Experiment B
git checkout -b experiment/parallel-pipeline
# Refactor to async tasks
# Test with: uv run python client.py "Long test text here"

# Experiment C
git checkout -b experiment/s3gen-context
# First: read src/chatterbox/models/s3gen/flow.py
# Then: implement cache passing
```

---

## Notes

- All experiments should maintain the 1.3s time-to-first-audio baseline
- Test with same phrases for comparison: "Hello my big long beautiful world"
- Log all timing metrics for comparison
- Audio quality is subjective - record samples for A/B testing
