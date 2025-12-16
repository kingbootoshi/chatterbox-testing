import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4, CUDA, or CPU)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

map_location = torch.device(device)

# Log detected device
print(f"🔍 Detected device: {device.upper()}")
if device == "mps":
    print("✅ Using Metal Performance Shaders (GPU acceleration enabled)")
elif device == "cuda":
    print("✅ Using CUDA (GPU acceleration enabled)")
else:
    print("⚠️  Using CPU (no GPU acceleration)")

# Patch torch.load for MPS compatibility
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

# Load the TTS model
print("🔄 Loading Chatterbox TTS model...")
model = ChatterboxTTS.from_pretrained(device=device)
print("✅ Model loaded!")

# Voice cloning configuration
VOICE_REFERENCE_PATH = "qb.mp3"  # Your reference audio file for voice cloning
TEXT_TO_SPEAK = "Hello! This is a voice clone. I'm speaking with the voice from the reference audio file. Pretty cool, right?"

print(f"🎤 Voice reference: {VOICE_REFERENCE_PATH}")
print(f"📝 Text to generate: {TEXT_TO_SPEAK}")

# Generate speech with voice cloning
print("🎙️ Generating voice-cloned speech...")
wav = model.generate(
    TEXT_TO_SPEAK,
    audio_prompt_path=VOICE_REFERENCE_PATH,
    exaggeration=0.5,  # Neutral expressiveness
    cfg_weight=0.5,    # Balanced pace
)

# Save the output
OUTPUT_PATH = "voice_clone_output.wav"
ta.save(OUTPUT_PATH, wav, model.sr)
print(f"✅ Audio saved to: {OUTPUT_PATH}")

