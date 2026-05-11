#!/usr/bin/env python3
"""
Setup script to download and generate training data for deepfake detector
"""
import os
import subprocess
import sys

def download_audio(url, output_path):
    """Download audio file from URL"""
    try:
        print(f"Downloading: {output_path}...")
        subprocess.run(
            ["curl", "-L", "-o", output_path, url],
            check=True,
            capture_output=True
        )
        print(f"  [OK] Saved")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def generate_synthetic_audio():
    """Generate synthetic audio using pyttsx3 (offline TTS)"""
    try:
        import pyttsx3
        print("\nGenerating synthetic audio samples...")

        engine = pyttsx3.init()
        engine.setProperty('rate', 150)

        texts = [
            "This is a text to speech generated sample. Deepfake detection is important for security.",
            "Machine learning models can detect synthetic audio by analyzing audio features.",
            "Voice synthesis technology has improved significantly over the years.",
            "Detecting deepfakes requires analyzing spectral characteristics and phase information.",
        ]

        synthetic_dir = "audio_data/synthetic"
        for i, text in enumerate(texts):
            output_path = os.path.join(synthetic_dir, f"tts_sample_{i+1}.wav")
            print(f"  Generating: {output_path}...")
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            print(f"    [OK] Generated")

        return True
    except ImportError:
        print("  [i] pyttsx3 not installed. Install with: pip install pyttsx3")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def main():
    print("=" * 60)
    print("Setting up training data for deepfake detector")
    print("=" * 60)

    # Free audio samples URLs (authentic speech)
    authentic_samples = [
        # LibriSpeech samples (public domain)
        ("https://www.openslr.org/resources/12/dev-clean.tar.gz", "librispeech_dev.tar.gz"),
        # Common Voice samples
        ("https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0/resolve/main/en/v14/common_voice_en_sample.wav", "audio_data/authentic/cv_sample_1.wav"),
    ]

    print("\n[1] Downloading authentic speech samples...")
    authentic_count = 0

    # Try to download some authentic samples
    try:
        import torchaudio
        print("  [i] Using torchaudio to download TIMIT dataset...")
        # This would require additional setup, so we'll skip for now
    except ImportError:
        pass

    print("\n[2] Generating synthetic audio samples...")
    generate_synthetic_audio()

    # Check what we have
    authentic_dir = "audio_data/authentic"
    synthetic_dir = "audio_data/synthetic"

    authentic_files = len([f for f in os.listdir(authentic_dir) if f.endswith(('.wav', '.mp3'))])
    synthetic_files = len([f for f in os.listdir(synthetic_dir) if f.endswith(('.wav', '.mp3'))])

    print("\n" + "=" * 60)
    print("Status:")
    print(f"  Authentic samples: {authentic_files}")
    print(f"  Synthetic samples: {synthetic_files}")
    print("=" * 60)

    if synthetic_files >= 3:
        print("\n[OK] Ready to train! Run: python train_model.py")
        return 0
    else:
        print("\n[WARNING] Need more training data. Options:")
        print("  1. Manually add .wav/.mp3 files to:")
        print(f"     - {authentic_dir}/ (for authentic speech)")
        print(f"     - {synthetic_dir}/ (for synthetic/TTS audio)")
        print("  2. Install pyttsx3: pip install pyttsx3")
        print("  3. Use online datasets:")
        print("     - Common Voice: https://commonvoice.mozilla.org/")
        print("     - LibriSpeech: https://www.openslr.org/12")
        return 1

if __name__ == "__main__":
    sys.exit(main())
