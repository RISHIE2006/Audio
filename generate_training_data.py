import numpy as np
import librosa
import soundfile as sf
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import joblib

# Import feature extraction from existing module
from utils.dsp_logic import extract_features

def ensure_directories():
    """Create necessary directories for training data."""
    os.makedirs("authentic_audio", exist_ok=True)
    os.makedirs("synthetic_audio", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)

def generate_synthetic_deepfake(text, output_path, sr=22050, duration=3):
    """
    Generate a synthetic audio sample using simple DSP techniques.
    This simulates what a vocoder/TTS system might produce.
    """
    # Create a simple synthetic signal with characteristics of vocoder output:
    # - Less natural phase variation
    # - Over-smooth spectral envelope
    # - Unnatural formant transitions
    t = np.linspace(0, duration, int(sr * duration))

    # Combine multiple sine waves with less natural variation
    fundamental = 100 + 50 * np.sin(2 * np.pi * 0.5 * t)  # Smooth pitch variation
    signal = 0.3 * np.sin(2 * np.pi * fundamental * t)

    # Add harmonics with unnatural smoothness
    for harmonic in [2, 3, 4]:
        signal += (0.15 / harmonic) * np.sin(2 * np.pi * fundamental * harmonic * t)

    # Add slight vocoder-like artifacts: phase discontinuities
    noise = 0.05 * np.random.randn(len(signal))
    signal = signal + noise

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    signal = (signal * 0.9).astype(np.float32)

    sf.write(output_path, signal, sr)
    return output_path

def load_real_audio_samples():
    """Load real audio samples from authentic_audio folder."""
    real_samples = []
    if os.path.exists("authentic_audio"):
        for filename in os.listdir("authentic_audio"):
            if filename.endswith((".wav", ".mp3", ".flac", ".m4a")):
                filepath = os.path.join("authentic_audio", filename)
                try:
                    y, sr = librosa.load(filepath, sr=22050, mono=True)
                    real_samples.append(filepath)
                    print(f"Loaded authentic sample: {filename}")
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

    return real_samples

def generate_synthetic_samples(num_samples=100):
    """Generate synthetic deepfake samples."""
    texts = [
        "Hello this is a test",
        "Machine learning is fascinating",
        "Audio synthesis creates realistic sound",
        "Deepfake detection is important",
        "This is a synthesized voice",
    ]

    synthetic_samples = []
    for i in range(num_samples):
        text = texts[i % len(texts)]
        output_path = f"synthetic_audio/synthetic_{i:04d}.wav"
        try:
            generate_synthetic_deepfake(text, output_path)
            synthetic_samples.append(output_path)
            if (i + 1) % 20 == 0:
                print(f"Generated {i + 1}/{num_samples} synthetic samples")
        except Exception as e:
            print(f"Failed to generate synthetic sample {i}: {e}")

    return synthetic_samples

def extract_features_from_samples(sample_paths, label):
    """Extract features from audio samples."""
    features_list = []
    for filepath in sample_paths:
        try:
            features = extract_features(filepath)
            features_list.append(features[0])
        except Exception as e:
            print(f"Failed to extract features from {filepath}: {e}")
            continue

    if features_list:
        X = np.array(features_list)
        y = np.array([label] * len(features_list))
        return X, y
    return None, None

def train_model_with_real_and_synthetic_data():
    """Train RandomForest on real and synthetic audio data."""
    ensure_directories()

    print("\n=== Deepfake Detector Model Training ===\n")

    # Check for existing real audio samples
    real_samples = load_real_audio_samples()
    if not real_samples:
        print("\nNo authentic audio samples found in 'authentic_audio/' folder.")
        print("Proceeding with synthetic-only training (will be less accurate).")
        num_real = 0
    else:
        num_real = len(real_samples)
        print(f"Found {num_real} authentic samples")

    # Generate synthetic samples
    print(f"\nGenerating 200 synthetic deepfake samples...")
    synthetic_samples = generate_synthetic_samples(200)
    print(f"Generated {len(synthetic_samples)} synthetic samples")

    # Extract features
    print("\nExtracting features from authentic audio...")
    X_real, y_real = extract_features_from_samples(real_samples, label=0)

    print("Extracting features from synthetic audio...")
    X_synthetic, y_synthetic = extract_features_from_samples(synthetic_samples, label=1)

    # Combine datasets
    if X_real is not None and X_synthetic is not None:
        X_train = np.vstack([X_real, X_synthetic])
        y_train = np.hstack([y_real, y_synthetic])
        print(f"\nTraining set: {len(X_train)} samples ({len(X_real)} authentic, {len(X_synthetic)} synthetic)")
    elif X_synthetic is not None:
        X_train = X_synthetic
        y_train = y_synthetic
        print(f"\nTraining set: {len(X_train)} synthetic samples only")
    else:
        print("ERROR: No features extracted!")
        return

    # Train RandomForest
    print("\nTraining RandomForest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Save model
    model_path = 'deepfake_detector.pkl'
    joblib.dump(rf_model, model_path)
    print(f"\n[OK] Model trained and saved to '{model_path}'")
    print(f"  Feature count: {X_train.shape[1]}")
    print(f"  Training accuracy: {rf_model.score(X_train, y_train):.2%}")

if __name__ == "__main__":
    train_model_with_real_and_synthetic_data()
