import numpy as np
from scipy.io import wavfile
import os
import pyttsx3

print("Creating Extended Dataset for Training...")
print("="*60)

# Create 11 additional authentic samples
print("\n[1/2] Generating authentic speech samples...")
for i in range(9, 20):
    sr = 22050
    duration = 2 + (i % 3)
    t = np.linspace(0, duration, int(sr * duration))

    base_f = 500 + (i * 40)
    formants = [base_f, base_f + 500, base_f + 1800]

    signal = np.zeros_like(t)
    for f_idx, formant in enumerate(formants):
        amp = 1 / (f_idx + 1)
        mod_freq = formant * (1 + 0.015 * np.sin(2 * np.pi * (2 + f_idx * 0.5) * t))
        signal += amp * np.sin(2 * np.pi * mod_freq * t)

    for h in [2, 3, 4]:
        signal += 0.15 * np.sin(2 * np.pi * formants[0] * h * t)

    signal += 0.03 * np.random.randn(len(t))
    signal = signal / np.max(np.abs(signal)) * 0.8

    filename = f"audio_data/authentic/synthetic_authentic_{i}.wav"
    wavfile.write(filename, sr, (signal * 32767).astype(np.int16))
    print(f"  Generated {os.path.basename(filename)}")

print(f"  [OK] Generated 11 additional authentic samples")

# Generate 5 more TTS synthetic samples
print("\n[2/2] Generating synthetic TTS samples...")
engine = pyttsx3.init()
engine.setProperty('rate', 130)

texts = [
    "Welcome to the deepfake detection system powered by machine learning.",
    "Synthetic speech detection requires analyzing multiple audio features.",
    "Text to speech synthesis is becoming increasingly difficult to detect.",
    "Neural vocoders create artifacts that we can identify with signal processing.",
    "The future of audio authentication depends on robust detection methods.",
]

for i, text in enumerate(texts, start=3):
    output = f"audio_data/synthetic/synthetic_sample_{i}.wav"
    engine.save_to_file(text, output)
    engine.runAndWait()
    print(f"  Generated {os.path.basename(output)}")

print(f"  [OK] Generated 5 additional synthetic samples")

# Summary
authentic = len(os.listdir("audio_data/authentic"))
synthetic = len(os.listdir("audio_data/synthetic"))

print("\n" + "="*60)
print("Dataset Complete!")
print("="*60)
print(f"Authentic samples: {authentic}")
print(f"Synthetic samples: {synthetic}")
print(f"Total: {authentic + synthetic}")
print("\nNext step: python train_model.py --force")
print("="*60)
