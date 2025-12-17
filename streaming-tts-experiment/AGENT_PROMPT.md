# Deep Optimization Analysis: Chatterbox Flow Streaming v2

## Context

We've implemented TRUE flow-level streaming for Chatterbox TTS with z_cache continuity. It's working but we have two major performance issues:

1. **~900ms client-side latency gap** - Server sends at 881ms, client receives at 1836ms
2. **Chunk hiccups** - Gaps between chunks when playback catches up to processing

## Current Architecture

```
T3 (autoregressive) → S3Gen (CFM: tokens→mel) → HiFiGAN (mel→wav) → WebSocket → Client
     ↓                      ↓                        ↓
  tokens            z_cache continuity         cache_source
```

### Current Flow (BLOCKING/SEQUENTIAL)

```
T3:   [gen 12 tok]────────────[gen 25 tok]────────────[gen 25 tok]
CFM:              [proc 12 tok]            [proc 37 tok]           [proc 62 tok]
Send:                         [send]                   [send]                   [send]
Play:                              [play ~500ms]            [play ~1s]
                                    ^                        ^
                                 HICCUP                   HICCUP
                              (if play ends            (if play ends
                               before next              before next
                               send arrives)            send arrives)
```

### Current Metrics

| Metric | Value |
|--------|-------|
| Server TTFB | 723-881ms |
| Client TTFB | 1565-1836ms |
| Gap | ~900ms |
| RTF | 1.14-1.30x |
| First chunk tokens | 12 |
| Subsequent chunk tokens | 25 |

---

## Three Questions

### Question 1: Why is there a ~900ms client-side latency gap?

Server logs show first chunk sent at 881ms, but client doesn't receive until 1836ms.

**Hypothesis A**: asyncio event loop blocked during `asyncio.to_thread(streamer.step)`
- The thread runs CFM inference (~180ms for 2 ODE steps)
- Event loop waits for thread completion
- WebSocket send is queued but not executed until thread returns

**Hypothesis B**: Client-side websocket buffering
- websockets library may buffer incoming binary data
- `recv()` might not return immediately when data arrives

**Hypothesis C**: sounddevice/PortAudio initialization
- First call to `sd.OutputStream()` may have latency
- Audio device initialization happens on first chunk

**Hypothesis D**: Network/OS buffering (unlikely on localhost)

Key code paths to examine:
1. Server: `asyncio.to_thread(streamer.step, False)` then `websocket.send_bytes()`
2. Client: `await websocket.recv()` then `player.add_chunk()`

### Question 2: How to eliminate chunk hiccups?

The hiccup occurs when:
1. Chunk N finishes playing (e.g., ~500ms of audio)
2. Chunk N+1 is still being generated (T3) or processed (CFM)
3. Result: silence gap until chunk N+1 arrives

**Current timing analysis**:
- First chunk: 12 tokens → ~12 mel frames → ~240ms audio (too short!)
- CFM processing: ~180-200ms per step (but processes ALL tokens cumulatively)
- T3 generation: ~25 tokens/sec → 25 tokens = 1000ms
- Audio per 25 tokens: ~1000ms

**The math doesn't work**:
- We need 1000ms of audio buffered to cover 1000ms of T3 generation
- But first chunk is only ~240ms (12 tokens)
- By the time chunk 1 finishes playing, chunk 2 might not be ready

**Potential solutions**:
1. **Larger first chunk** (more tokens = longer audio buffer)
2. **Faster CFM** (torch.compile, quantization)
3. **Pipeline parallelism** (generate tokens while CFM processes)
4. **Predictive buffering** (pre-buffer 2 chunks before playing)

### Question 3: How to implement pipeline parallelism?

**Goal**: Run T3 token generation in parallel with CFM processing

**Proposed architecture**:
```python
# Thread 1: T3 token generation
async def token_generator():
    while not done:
        token = generate_next_token()
        token_queue.put(token)

# Thread 2: CFM processing (triggered when chunk ready)
async def cfm_processor():
    while not done:
        if len(accumulated_tokens) >= chunk_size:
            mels = run_cfm(accumulated_tokens)
            mel_queue.put(mels)

# Thread 3: HiFiGAN + Send (triggered when mels ready)
async def audio_sender():
    while not done:
        mels = mel_queue.get()
        audio = run_hifigan(mels)
        await websocket.send_bytes(audio)
```

**Challenges**:
1. **z_cache synchronization**: CFM needs z_cache from previous run
2. **Cumulative processing**: CFM re-processes ALL tokens each time
3. **Memory growth**: Token queue grows unbounded
4. **Thread safety**: PyTorch tensors, CUDA streams

**Alternative: Speculative CFM**
Start CFM on N tokens while T3 generates token N+1, N+2, etc.
When CFM finishes, if new tokens arrived, re-run CFM on extended sequence.

---

## Files for Context

### Server-side (provide these)

1. **`streaming-tts-experiment/server.py`**
   - `S3GenStreamer` class (lines 64-176)
   - `/ws/tts/flow` endpoint (lines 1043-1206)

2. **`src/chatterbox/models/s3gen/s3gen.py`**
   - `flow_stream_step()` method

3. **`src/chatterbox/models/s3gen/flow.py`**
   - `inference_streaming()` method

4. **`src/chatterbox/models/s3gen/flow_matching.py`**
   - `forward_streaming()` method

### Client-side (provide these)

5. **`streaming-tts-experiment/client.py`**
   - `StreamingAudioPlayer` class
   - `--flow` mode handler

---

## Specific Code to Analyze

### Server: asyncio.to_thread blocking pattern
```python
# server.py /ws/tts/flow endpoint
# This runs CFM in a thread - does it block the event loop?
audio = await asyncio.to_thread(streamer.step, False)
last_emitted_count = len(streamer.speech_tokens)

if len(audio) > 0:
    chunk_count += 1
    if t_first_chunk is None:
        t_first_chunk = time.perf_counter()  # Recorded AFTER thread returns
        logger.success(f"First chunk in {(t_first_chunk - t_start)*1000:.0f}ms")

    await websocket.send_bytes(audio.tobytes())  # Sent AFTER thread returns
```

### Client: WebSocket receive + audio queue
```python
# client.py flow mode
while True:
    message = await websocket.recv()

    if isinstance(message, bytes):
        player.add_chunk(message)  # When exactly is this called?
        chunks_received += 1
```

### CFM cumulative processing (O(n²) issue)
```python
# flow.py inference_streaming
# This runs on ALL accumulated tokens, not just new ones
h, h_masks = self.encoder(token, token_len)  # Full sequence!
# ...
feat, new_z_cache = self.decoder.forward_streaming(
    mu=h.transpose(1, 2).contiguous(),  # Full sequence!
    # ...
)
```

---

## Expected Output

Please provide:

1. **Root cause analysis** for the ~900ms client latency gap
   - Is it server-side (asyncio blocking)?
   - Is it client-side (websocket buffering, audio init)?
   - Provide instrumentation code to measure each segment

2. **Chunk hiccup elimination strategy**
   - Mathematical analysis: tokens needed for continuous playback
   - Recommendation: larger first chunk vs pipeline parallelism vs buffering

3. **Pipeline parallelism design** (if recommended)
   - Thread/async architecture
   - z_cache synchronization strategy
   - Memory management
   - Code structure

4. **Quick wins** that can be implemented in 10-15 minutes
   - Instrument latency
   - Adjust chunk sizes
   - Pre-buffer strategy

Focus on practical, implementable solutions. Current system is 90% there - we just need to smooth out the latency and hiccups!
