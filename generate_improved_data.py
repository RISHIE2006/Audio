#!/usr/bin/env python3
"""
Generate realistic deepfake and TTS synthetic samples with characteristic artifacts
"""
import numpy as np
from scipy.io import wavfile
from scipy import signal
import os

def create_tts_like_deepfake(filename, text_idx=0, duration=3):
    """
    Create audio that mimics TTS/vocoder artifacts:
    - Robotic pitch contour
    - Harmonic stacking artifacts
    - Phase discontinuities
    - Unnatural formant transitions
    """
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))

    # TTS characteristic: constant pitch (no natural vibrato)
    fundamental = 150 + (text_idx * 15)  # Different speaker pitches

    # Create harmonic stack (like vocoders do)
    y = np.zeros_like(t)

    # Add harmonics with TTS-like uniformity
    for harmonic in range(1, 20):
        amp = 1.0 / (harmonic * 1.3)  # Vocoder amplitude decay
        # TTS: perfectly linear phase (unnatural)
        phase = 2 * np.pi * fundamental * harmonic * t
        y += amp * np.sin(phase)

    # TTS artifact: sudden pitch jumps (synthesis artifacts)
    jump_times = np.linspace(0.2, duration - 0.2, 8)
    for jump_time in jump_times:
        idx = int(jump_time * sr)
        if idx < len(t):
            y[idx:idx+100] *= 1.1  # Amplitude spike at transitions

    # TTS artifact: linear formant transitions (unnatural)
    f1 = signal.get_window('hamming', len(t))
    f1 = 700 + 300 * f1  # Smooth but unnatural formant movement

    formant_1 = 2 * np.pi * f1 * t / sr
    y += 0.3 * np.sin(formant_1)

    # TTS artifact: phase discontinuities from frame splicing
    frame_size = int(0.02 * sr)  # 20ms frames
    for i in range(0, len(y), frame_size):
        if i + frame_size < len(y):
            # Frame boundaries cause phase mismatches
            y[i:i+50] *= (1 + 0.05 * np.random.randn())

    # TTS: less natural noise floor
    y += 0.02 * np.random.randn(len(t))

    # Normalize
    y = y / np.max(np.abs(y)) * 0.8

    wavfile.write(filename, sr, (y * 32767).astype(np.int16))
    print(f"  Created deepfake-like sample: {os.path.basename(filename)}")

def create_realistic_authentic(filename, speaker_idx=0, duration=3):
    """
    Create more realistic authentic speech with natural characteristics:
    - Natural vibrato (slight frequency modulation)
    - Random pitch variations
    - Natural formant transitions
    - Realistic noise
    """
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))

    # Natural speech: variable fundamental frequency
    base_f0 = 100 + (speaker_idx * 20)

    # Intonation contour (natural speech goes down at phrase end)
    intonation = -30 * (t / duration) ** 2  # Falling tone

    # Vibrato (natural wobble)
    vibrato = 5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz vibrato

    f0 = base_f0 + intonation + vibrato + 3 * np.sin(2 * np.pi * 0.5 * t)  # Prosody

    # Create voice with harmonics
    y = np.zeros_like(t)

    # Harmonics with natural falloff
    for harmonic in range(1, 25):
        amp = 1.0 / (harmonic ** 1.5)  # Natural harmonic decay
        # Actual voice uses phase-tracking, not linear
        phase = 2 * np.pi * np.cumsum(f0 * harmonic) / sr
        y += amp * np.sin(phase)

    # Natural formants with gradual transitions
    for formant_f in [700, 1200, 2500]:
        # Formants change gradually in real speech
        formant = formant_f + 100 * np.sin(2 * np.pi * 0.3 * t)
        phase = 2 * np.pi * np.cumsum(formant) / sr
        y += 0.2 * np.sin(phase)

    # Jitter and shimmer (natural voice irregularities)
    jitter = np.random.normal(0, 0.005, len(t))
    shimmer = np.random.normal(1.0, 0.02, len(t))
    y = (y + jitter) * shimmer

    # Natural noise (breathing, etc)
    y += 0.015 * np.random.randn(len(t))

    # Slight formant ripple (natural resonances)
    y = signal.lfilter([1, 0.1], [1, -0.85], y)

    # Normalize
    y = y / np.max(np.abs(y)) * 0.8

    wavfile.write(filename, sr, (y * 32767).astype(np.int16))
    print(f"  Created realistic authentic sample: {os.path.basename(filename)}")

def main():
    print("="*60)
    print("Generating Improved Training Data")
    print("="*60)

    print("\nCreating realistic deepfake/TTS samples...")
    for i in range(10):
        filename = f"audio_data/synthetic/improved_deepfake_{i+1}.wav"
        create_tts_like_deepfake(filename, text_idx=i)

    print("\nCreating realistic authentic samples...")
    for i in range(10):
        filename = f"audio_data/authentic/improved_authentic_{i+1}.wav"
        create_realistic_authentic(filename, speaker_idx=i)

    # Count and report
    authentic = len([f for f in os.listdir("audio_data/authentic") if f.endswith('.wav')])
    synthetic = len([f for f in os.listdir("audio_data/synthetic") if f.endswith('.wav')])

    print("\n" + "="*60)
    print("Dataset Updated!")
    print("="*60)
    print(f"Authentic samples: {authentic}")
    print(f"Synthetic/Deepfake samples: {synthetic}")
    print(f"Total: {authentic + synthetic}")
    print("\nNext: python train_model.py --force")
    print("="*60)

if __name__ == "__main__":
    main()
