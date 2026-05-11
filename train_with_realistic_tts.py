import argparse
import json
import logging
import os
import pyttsx3
import soundfile as sf
import numpy as np
from datetime import datetime
from utils.dsp_logic import extract_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DEFAULT_AUTHENTIC_DIR = "training_audio/authentic_realistic"
DEFAULT_TTS_DIR = "training_audio/tts_deepfake"
DEFAULT_METADATA = "training_audio/generation_metadata.json"

SAMPLE_TEXTS = [
    "Hello, this is an AI-generated message",
    "Welcome to the synthetic voice demonstration",
    "This audio was created by a text-to-speech system",
    "Artificial intelligence can now produce realistic voices",
    "Machine learning enables speech synthesis",
    "This is computer-generated audio content",
    "Digital voice synthesis technology is advancing rapidly",
    "Text-to-speech systems use neural networks",
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clear_wav_files(path: str):
    if not os.path.isdir(path):
        return
    for filename in os.listdir(path):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(path, filename)
            try:
                os.remove(file_path)
            except Exception as exc:
                logger.warning("Failed to remove %s: %s", file_path, exc)


def generate_tts_audio(text: str, output_path: str, voice_idx: int = 0, rate: int = 150, volume: float = 0.9) -> bool:
    """Generate realistic TTS audio using pyttsx3."""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")

        if voices:
            selected = voice_idx % len(voices)
            engine.setProperty("voice", voices[selected].id)
        else:
            logger.warning("No pyttsx3 voices available; using default voice settings")

        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)

        engine.save_to_file(text, output_path)
        engine.runAndWait()
        engine.stop()

        return True
    except Exception as e:
        logger.error("TTS generation failed for %s: %s", output_path, e)
        return False


def generate_realistic_authentic_audio(output_path: str, sr: int = 22050, duration: float = 3.0) -> bool:
    """Generate human-like synthetic audio with natural prosody and texture."""
    try:
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        f0 = 150 + 60 * np.sin(2 * np.pi * 0.4 * t) + 15 * np.cos(2 * np.pi * 1.2 * t)
        f0 = np.clip(f0, 80, 250)

        formants = [
            (700, 120),
            (1220, 130),
            (2600, 200),
        ]

        signal = np.zeros_like(t, dtype=np.float32)
        for harmonic in range(1, 30):
            freq = f0 * harmonic
            amplitude = 1.0 / (harmonic ** 1.3)
            for formant_freq, bw in formants:
                distance = np.abs(freq - formant_freq)
                amplitude *= 1.0 + 2.0 * np.exp(-((distance / bw) ** 2))
            phase_mod = 0.05 * np.sin(2 * np.pi * 2.5 * t)
            signal += amplitude * np.sin(2 * np.pi * freq * t + phase_mod)

        noise = 0.03 * np.random.randn(len(signal))
        noise_mask = np.random.random(len(signal)) > 0.7
        signal[noise_mask] += noise[noise_mask]

        attack = int(0.08 * sr)
        release = int(0.15 * sr)
        envelope = np.ones_like(signal)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        signal *= envelope

        for delay_ms in [3, 7, 15]:
            delay_samples = int(delay_ms * sr / 1000)
            if delay_samples < len(signal):
                signal += 0.08 * np.roll(signal, delay_samples)

        signal = signal / (np.max(np.abs(signal)) + 1e-8)
        signal = (signal * 0.85).astype(np.float32)

        sf.write(output_path, signal, sr)
        return True
    except Exception as e:
        logger.error("Failed to generate synthetic authentic audio %s: %s", output_path, e)
        return False


def generate_tts_deepfakes(num_samples: int, output_dir: str, regenerate: bool, rate: int = 150, volume: float = 0.9) -> dict:
    ensure_dir(output_dir)
    if regenerate:
        clear_wav_files(output_dir)

    voices_available = len(pyttsx3.init().getProperty("voices"))
    logger.info("Generating %d TTS deepfake samples in %s", num_samples, output_dir)

    success = 0
    failed = 0
    for i in range(num_samples):
        output_path = os.path.join(output_dir, f"tts_deepfake_{i:04d}.wav")
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        if generate_tts_audio(text, output_path, voice_idx=i, rate=rate, volume=volume):
            success += 1
        else:
            failed += 1

    logger.info("TTS generation complete: %d succeeded, %d failed", success, failed)
    return {"success": success, "failed": failed, "target": num_samples, "directory": output_dir, "voices": voices_available}


def generate_authentic_samples(num_samples: int, output_dir: str, duration: float, regenerate: bool, sr: int = 22050) -> dict:
    ensure_dir(output_dir)
    if regenerate:
        clear_wav_files(output_dir)

    logger.info("Generating %d synthetic authentic-like samples in %s", num_samples, output_dir)
    success = 0
    failed = 0
    for i in range(num_samples):
        output_path = os.path.join(output_dir, f"authentic_{i:04d}.wav")
        if generate_realistic_authentic_audio(output_path, sr=sr, duration=duration):
            success += 1
        else:
            failed += 1

    logger.info("Authentic-like generation complete: %d succeeded, %d failed", success, failed)
    return {"success": success, "failed": failed, "target": num_samples, "directory": output_dir}


def extract_features_from_folder(folder_path: str, label: int, description: str = ""):
    if not os.path.isdir(folder_path):
        logger.warning("Folder does not exist: %s", folder_path)
        return None, None

    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.wav')])
    logger.info("Extracting features from %s (%d files)", description, len(files))

    features = []
    failed = 0
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            feature_vector = extract_features(filepath)
            features.append(feature_vector[0])
        except Exception as e:
            failed += 1
            logger.warning("Feature extraction failed for %s: %s", filepath, e)

    if not features:
        logger.error("No valid feature vectors extracted from %s", folder_path)
        return None, None

    X = np.array(features)
    y = np.full(len(features), label, dtype=int)
    logger.info("Extracted %d valid samples (%d failed) from %s", len(X), failed, folder_path)
    return X, y


def write_metadata(metadata: dict, metadata_path: str):
    try:
        ensure_dir(os.path.dirname(metadata_path) or ".")
        with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        logger.info("Saved generation metadata to %s", metadata_path)
    except Exception as e:
        logger.error("Failed to save metadata %s: %s", metadata_path, e)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic audio datasets for deepfake detection experiments.")
    parser.add_argument("--authentic-samples", type=int, default=120, help="Number of realistic authentic-like samples to generate.")
    parser.add_argument("--tts-samples", type=int, default=120, help="Number of TTS deepfake samples to generate.")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration of each generated authentic-like sample in seconds.")
    parser.add_argument("--authentic-dir", type=str, default=DEFAULT_AUTHENTIC_DIR, help="Output directory for generated authentic-like audio.")
    parser.add_argument("--tts-dir", type=str, default=DEFAULT_TTS_DIR, help="Output directory for generated TTS audio.")
    parser.add_argument("--metadata-path", type=str, default=DEFAULT_METADATA, help="Path to write dataset generation metadata.")
    parser.add_argument("--regenerate", action="store_true", help="Remove existing generated samples and regenerate from scratch.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Sample rate for generated audio.")
    parser.add_argument("--tts-rate", type=int, default=150, help="Speech rate for TTS generation.")
    parser.add_argument("--tts-volume", type=float, default=0.9, help="Volume for TTS generation.")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Starting dataset generation")
    logger.info("Authentic output directory: %s", args.authentic_dir)
    logger.info("TTS output directory: %s", args.tts_dir)
    logger.info("Regenerate mode: %s", args.regenerate)

    authentic_results = generate_authentic_samples(
        num_samples=args.authentic_samples,
        output_dir=args.authentic_dir,
        duration=args.duration,
        regenerate=args.regenerate,
        sr=args.sample_rate,
    )

    tts_results = generate_tts_deepfakes(
        num_samples=args.tts_samples,
        output_dir=args.tts_dir,
        regenerate=args.regenerate,
        rate=args.tts_rate,
        volume=args.tts_volume,
    )

    X_authentic, y_authentic = extract_features_from_folder(
        args.authentic_dir, 0, "generated authentic-like audio"
    )
    X_tts, y_tts = extract_features_from_folder(
        args.tts_dir, 1, "generated TTS deepfakes"
    )

    generation_metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "authentic_dir": os.path.abspath(args.authentic_dir),
        "tts_dir": os.path.abspath(args.tts_dir),
        "authentic_requested": args.authentic_samples,
        "tts_requested": args.tts_samples,
        "duration_seconds": args.duration,
        "sample_rate": args.sample_rate,
        "regenerate": args.regenerate,
        "tts_settings": {"rate": args.tts_rate, "volume": args.tts_volume},
        "generated_authentic": authentic_results,
        "generated_tts": tts_results,
        "authentic_feature_count": len(X_authentic) if X_authentic is not None else 0,
        "tts_feature_count": len(X_tts) if X_tts is not None else 0,
        "notes": "Generated authentic-like audio is synthetic and should be compared with recorded authentic speech for generalization."
    }

    write_metadata(generation_metadata, args.metadata_path)

    if X_authentic is None or X_tts is None:
        logger.error("Feature extraction failed for one or more generated datasets. Check warnings above.")
        return

    logger.info("Generation complete. Ready for model training with train_model.py.")
    logger.info("Generated authentic-like features: %d", len(X_authentic))
    logger.info("Generated TTS features: %d", len(X_tts))


if __name__ == "__main__":
    main()
