import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals

# Load environment variables from .env
load_dotenv()

# Set HF token for huggingface_hub
if os.getenv("HF_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
    print("🔑 HF token loaded from .env")

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

print(f"🔍 Detected device: {device.upper()}")
if device == "mps":
    print("✅ Using Metal Performance Shaders (GPU acceleration enabled)")
else:
    print("⚠️  Using CPU (Metal not available)")

# Patch torch.load for MPS compatibility
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

# Configuration
VOICE_REFERENCE = "qb.mp3"
VOICES_DIR = Path("voices")
OUTPUT_NAME = "qb_voice.pt"

# Create voices directory if it doesn't exist
VOICES_DIR.mkdir(exist_ok=True)
output_path = VOICES_DIR / OUTPUT_NAME

print(f"\n📂 Voice reference: {VOICE_REFERENCE}")
print(f"📁 Output directory: {VOICES_DIR.absolute()}")

# Load the Turbo model
print("\n🔄 Loading Chatterbox Turbo model...")
model = ChatterboxTurboTTS.from_pretrained(device=device)
print("✅ Model loaded!")

# Prepare conditionals from reference audio
print(f"\n🎤 Extracting voice from: {VOICE_REFERENCE}")
print("   This extracts:")
print("   - Speaker embedding (256-dim voice timbre)")
print("   - Conditioning tokens (up to 15s of acoustic context)")
print("   - Mel features for CFM decoder (up to 10s)")

model.prepare_conditionals(
    wav_fpath=VOICE_REFERENCE,
    exaggeration=0.0,      # Neutral - faithful to original voice
    norm_loudness=True     # Normalize loudness for consistency
)

# Save the voice clone
model.conds.save(output_path)

print(f"\n✅ Voice clone saved!")
print(f"\n{'='*60}")
print(f"📍 SAVED TO: {output_path.absolute()}")
print(f"{'='*60}")

print(f"""
To use this voice in the future:

```python
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals

model = ChatterboxTurboTTS.from_pretrained(device="mps")
model.conds = Conditionals.load("{output_path}").to("mps")

# Generate without needing the original MP3
wav = model.generate("Hello world")
```
""")

