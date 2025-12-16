import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

# Log detected device
print(f"🔍 Detected device: {device.upper()}")
if device == "mps":
    print("✅ Using Metal Performance Shaders (GPU acceleration enabled)")
else:
    print("⚠️  Using CPU (Metal not available)")

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)
text = "CHICKEN LIVES MATTER! CHICKEN LIVES MATTER!"

wav = model.generate(
    text, 
    exaggeration=2.0,
    cfg_weight=0.5
    )
ta.save("test-2.wav", wav, model.sr)
