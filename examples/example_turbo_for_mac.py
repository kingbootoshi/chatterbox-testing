import os
import torch
import torchaudio as ta
from dotenv import load_dotenv
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load environment variables from .env
load_dotenv()

# Set HF token for huggingface_hub
if os.getenv("HF_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
    print("🔑 HF token loaded from .env")

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

# Log detected device
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

# Load the Turbo model
print("🔄 Loading Chatterbox Turbo model...")
model = ChatterboxTurboTTS.from_pretrained(device=device)
print("✅ Model loaded!")

# Generate with Paralinguistic Tags (Turbo feature!)
# Supported tags: [clear throat], [sigh], [shush], [cough], [groan], [sniff], [gasp], [chuckle], [laugh]
text = "Hello"

print(f"📝 Generating: {text[:60]}...")

# Voice cloning with reference audio
VOICE_REFERENCE = "qb.mp3"
print(f"🎤 Cloning voice from: {VOICE_REFERENCE}")

wav = model.generate(text, audio_prompt_path=VOICE_REFERENCE)

ta.save("test-turbo.wav", wav, model.sr)
print("✅ Audio saved to: test-turbo.wav")

