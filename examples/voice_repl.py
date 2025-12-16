#!/usr/bin/env python3
"""
Interactive voice generation REPL - model loads once, generate many.

Usage:
    python voice_repl.py
    
Then type text and press Enter to generate. Type 'quit' or Ctrl+C to exit.
"""
import os
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

VOICE_PATH = Path("voices/qb_voice.pt")
OUTPUT_DIR = Path("outputs")


def ts():
    """Current timestamp for logging."""
    return time.strftime("%H:%M:%S")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"[{ts()}] 🔍 Device: {device.upper()}")
    
    # === MODEL LOADING (one-time cost) ===
    t0 = time.perf_counter()
    print(f"[{ts()}] 🔄 Loading Chatterbox Turbo model...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    t1 = time.perf_counter()
    print(f"[{ts()}] ✅ Model loaded in {t1-t0:.2f}s")
    
    # === VOICE LOADING ===
    t0 = time.perf_counter()
    print(f"[{ts()}] 🎤 Loading voice: {VOICE_PATH}")
    model.conds = Conditionals.load(VOICE_PATH, map_location=device).to(device)
    t1 = time.perf_counter()
    print(f"[{ts()}] ✅ Voice loaded in {t1-t0:.2f}s")
    
    print(f"\n{'='*60}")
    print("🎙️  READY - Type text and press Enter to generate")
    print("    Type 'quit' or Ctrl+C to exit")
    print(f"{'='*60}\n")
    
    counter = 1
    
    while True:
        try:
            text = input("📝 > ").strip()
            
            if not text:
                continue
            if text.lower() in ('quit', 'exit', 'q'):
                print(f"[{ts()}] 👋 Bye!")
                break
            
            # === INFERENCE (the fast part) ===
            t0 = time.perf_counter()
            print(f"[{ts()}] 🔊 Generating...")
            
            wav = model.generate(text)
            
            # Boost volume by 50%
            wav = wav * 1.5
            wav = torch.clamp(wav, -1.0, 1.0)  # Prevent clipping
            
            t1 = time.perf_counter()
            
            # Save with incrementing filename
            output_file = OUTPUT_DIR / f"output_{counter:03d}.wav"
            ta.save(str(output_file), wav, model.sr)
            
            duration = wav.shape[1] / model.sr
            print(f"[{ts()}] ✅ Generated {duration:.1f}s audio in {t1-t0:.2f}s → {output_file}")
            
            # Play audio
            print(f"[{ts()}] 🔊 Playing...")
            subprocess.run(["afplay", str(output_file)])
            
            counter += 1
            
        except KeyboardInterrupt:
            print(f"\n[{ts()}] 👋 Bye!")
            break
        except Exception as e:
            print(f"[{ts()}] ❌ Error: {e}")


if __name__ == "__main__":
    main()

