#!/usr/bin/env python3
"""
Generate speech using pre-saved voice clone.

Usage:
    python voice.py "Your text here"
    python voice.py "Hello world" -o custom_output.wav
"""
import sys
import os
import argparse
import time
import subprocess
from pathlib import Path
import torch
import torchaudio as ta
from dotenv import load_dotenv
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals

load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

# Patch torch.load for MPS compatibility
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

# Paths
VOICE_PATH = Path("voices/qb_voice.pt")
DEFAULT_OUTPUT = "output.wav"


def ts():
    """Current timestamp for logging."""
    return time.strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Generate speech with cloned voice")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output wav file")
    args = parser.parse_args()

    if not VOICE_PATH.exists():
        print(f"[{ts()}] ❌ Voice file not found: {VOICE_PATH}")
        print("   Run save_voice_clone.py first to create it")
        sys.exit(1)

    print(f"[{ts()}] 🔍 Device: {device.upper()}")
    
    t0 = time.perf_counter()
    print(f"[{ts()}] 🔄 Loading model...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    t1 = time.perf_counter()
    print(f"[{ts()}] ✅ Model loaded in {t1-t0:.2f}s")
    
    t0 = time.perf_counter()
    model.conds = Conditionals.load(VOICE_PATH, map_location=device).to(device)
    t1 = time.perf_counter()
    print(f"[{ts()}] 🎤 Voice loaded in {t1-t0:.2f}s")
    
    print(f"[{ts()}] 📝 Text: {args.text}")
    
    t0 = time.perf_counter()
    wav = model.generate(args.text)
    t1 = time.perf_counter()
    
    # Boost volume by 50%
    wav = wav * 1.5
    wav = torch.clamp(wav, -1.0, 1.0)  # Prevent clipping
    
    duration = wav.shape[1] / model.sr
    print(f"[{ts()}] 🔊 Generated {duration:.1f}s audio in {t1-t0:.2f}s")
    
    ta.save(args.output, wav, model.sr)
    print(f"[{ts()}] ✅ Saved: {args.output}")
    
    # Play audio
    print(f"[{ts()}] 🔊 Playing...")
    subprocess.run(["afplay", args.output])


if __name__ == "__main__":
    main()

