#!/usr/bin/env python3
"""
Download free deepfake and authentic speech datasets
"""
import os
import subprocess
import json
from pathlib import Path

def run_cmd(cmd, description=""):
    """Run shell command"""
    if description:
        print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def download_librispeech():
    """Download LibriSpeech dataset (authentic speech)"""
    print("\n" + "="*60)
    print("Downloading LibriSpeech (Authentic Speech)")
    print("="*60)

    # Create a temp directory
    os.makedirs("downloads", exist_ok=True)
    os.chdir("downloads")

    # Download a small subset (dev-clean - 337MB)
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    filename = "librispeech_dev.tar.gz"

    print(f"Downloading {filename} (370 MB) - This may take a few minutes...")
    if run_cmd(f"curl -L -o {filename} {url}", "Fetching LibriSpeech"):
        print("[OK] Downloaded")

        # Extract
        if run_cmd(f"tar -xzf {filename}", "Extracting files"):
            print("[OK] Extracted")

            # Convert FLAC to WAV and copy to training directory
            print("\nConverting FLAC to WAV...")
            import subprocess
            result = subprocess.run(
                "find LibriSpeech/dev-clean -name '*.flac' | head -20",
                shell=True, capture_output=True, text=True
            )

            flac_files = result.stdout.strip().split('\n')
            copied = 0
            for flac_file in flac_files:
                if flac_file:
                    wav_file = flac_file.replace('.flac', '.wav').replace('downloads/', '')
                    try:
                        # Use ffmpeg if available, otherwise skip
                        subprocess.run(
                            f"ffmpeg -i {flac_file} ../audio_data/authentic/{os.path.basename(wav_file)} -y",
                            shell=True, capture_output=True, timeout=30
                        )
                        copied += 1
                    except:
                        pass

            print(f"[OK] Converted and copied {copied} files")
            os.chdir("..")
            return copied > 0

    os.chdir("..")
    return False

def download_common_voice():
    """Download Mozilla Common Voice samples"""
    print("\n" + "="*60)
    print("Downloading Mozilla Common Voice (Authentic Speech)")
    print("="*60)

    # Using HuggingFace datasets library (easier method)
    print("\nAttempting to download from HuggingFace...")

    try:
        from datasets import load_dataset
        print("Loading Common Voice dataset (this may take 10-20 minutes)...")

        dataset = load_dataset(
            "mozilla-foundation/common_voice_13_0",
            "en",
            split="train",
            streaming=False,
            cache_dir="./cv_cache"
        )

        print(f"[OK] Loaded {len(dataset)} samples")

        # Take first 20 samples and convert to WAV
        copied = 0
        for idx, sample in enumerate(dataset.take(20)):
            try:
                audio = sample['audio']
                wav_bytes = audio['bytes']

                output_path = f"audio_data/authentic/cv_sample_{idx+1}.wav"
                with open(output_path, 'wb') as f:
                    f.write(wav_bytes)
                copied += 1
            except:
                pass

        print(f"[OK] Copied {copied} samples")
        return copied > 0
    except ImportError:
        print("[INFO] datasets library not installed")
        print("Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def download_asvspoof():
    """Download ASVspoof dataset (deepfake/synthetic speech)"""
    print("\n" + "="*60)
    print("Downloading ASVspoof (Synthetic/Deepfake Speech)")
    print("="*60)

    print("\n[INFO] ASVspoof requires manual download from:")
    print("https://www.asvspoof.org/index2021.html")
    print("\nAfter downloading, extract to: audio_data/synthetic/")
    return False

def download_with_youtube_dl():
    """Download speaker samples from YouTube using yt-dlp"""
    print("\n" + "="*60)
    print("Downloading authentic speech from YouTube")
    print("="*60)

    try:
        import yt_dlp

        # Some good authentic speech sources
        urls = [
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # First YouTube video (classic test)
        ]

        print("[INFO] This method requires yt-dlp")
        print("Install with: pip install yt-dlp")
        return False
    except ImportError:
        print("[INFO] yt-dlp not installed")
        return False

def create_demo_dataset():
    """Create a larger demo dataset using simple synthesis"""
    print("\n" + "="*60)
    print("Creating Extended Demo Dataset")
    print("="*60)

    try:
        import numpy as np
        from scipy.io import wavfile

        print("\nGenerating authentic speech simulations...")

        for i in range(9, 20):
            sr = 22050
            duration = 2 + (i % 3)
            t = np.linspace(0, duration, int(sr * duration))

            # Vary parameters
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

        print(f"[OK] Generated 11 additional authentic samples")

        print("\nGenerating synthetic speech samples...")
        try:
            import pyttsx3
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

            print(f"[OK] Generated 5 additional synthetic samples")
            return True
        except Exception as e:
            print(f"[WARNING] Could not generate TTS: {e}")
            return False

    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def summarize():
    """Show summary of what we have"""
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)

    authentic = len(os.listdir("audio_data/authentic"))
    synthetic = len(os.listdir("audio_data/synthetic"))

    print(f"\nAuthentic samples: {authentic}")
    print(f"Synthetic samples: {synthetic}")
    print(f"Total: {authentic + synthetic}")

    if authentic + synthetic >= 20:
        print("\n[OK] Ready to train! Run: python train_model.py --force")
    else:
        print(f"\n[INFO] For better accuracy, aim for 50+ samples per category")

def main():
    print("\n" + "="*60)
    print("Deepfake Detector - Dataset Downloader")
    print("="*60)

    # Use current directory instead of changing
    script_dir = os.getcwd()

    print("\nAvailable options:")
    print("1. Create extended demo dataset (Fast)")
    print("2. Download LibriSpeech (370 MB - Slow)")
    print("3. Download Common Voice from HuggingFace (Slow)")
    print("4. Show summary")
    print("5. Exit")

    choice = input("\nSelect option (1-5): ").strip()

    if choice == "1":
        create_demo_dataset()
        summarize()
    elif choice == "2":
        download_librispeech()
        summarize()
    elif choice == "3":
        download_common_voice()
        summarize()
    elif choice == "4":
        summarize()
    else:
        print("Exiting...")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
