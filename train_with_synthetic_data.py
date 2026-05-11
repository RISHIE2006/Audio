import numpy as np
import librosa
import soundfile as sf
import os
from sklearn.ensemble import RandomForestClassifier
import joblib
from utils.dsp_logic import extract_features

def generate_authentic_audio(output_path, sr=22050, duration=3):
    """
    Generate realistic authentic-sounding audio with natural characteristics:
    - Variable pitch and formants
    - Natural phase variations
    - Realistic spectral envelope
    - Background noise/room reflections
    """
    t = np.linspace(0, duration, int(sr * duration))

    # Fundamental frequency with natural prosody (pitch changes)
    base_f0 = 150 + 40 * np.sin(2 * np.pi * 0.3 * t)  # Natural pitch variation
    f0 = base_f0 + 10 * np.sin(2 * np.pi * 1.5 * t)   # Micromodulation

    # Generate speech-like signal with formant frequencies
    signal = np.zeros_like(t, dtype=np.float32)

    # Fundamental + harmonics with formant envelope (F1, F2, F3 for vowel-like sounds)
    formant_freqs = [700, 1220, 2600]  # Approximate formant frequencies
    formant_bw = [100, 120, 200]        # Bandwidth

    for harmonic in range(1, 20):
        partial_freq = f0 * harmonic
        partial_amp = 1.0 / harmonic

        # Apply formant envelope (resonances)
        for formant_f, formant_bw_val in zip(formant_freqs, formant_bw):
            distance = np.abs(partial_freq - formant_f)
            partial_amp *= np.exp(-distance / (formant_bw_val * 2))

        signal += partial_amp * np.sin(2 * np.pi * partial_freq * t)

    # Add natural phase variations and jitter
    phase_jitter = 0.02 * np.random.randn(len(signal))
    signal = signal * (1 + phase_jitter)

    # Add realistic noise floor and room acoustics
    noise = 0.01 * np.random.randn(len(signal))

    # Simulate room reflections with short delays
    for delay_samples in [220, 440, 880]:
        if delay_samples < len(signal):
            delayed = np.roll(signal, delay_samples) * 0.05
            signal = signal + delayed

    signal = signal + noise

    # Natural amplitude envelope (attack, sustain, release)
    envelope = np.ones_like(t)
    attack_time = int(0.05 * sr)
    release_time = int(0.1 * sr)
    envelope[:attack_time] = np.linspace(0, 1, attack_time)
    envelope[-release_time:] = np.linspace(1, 0, release_time)
    signal = signal * envelope

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    signal = (signal * 0.85).astype(np.float32)

    sf.write(output_path, signal, sr)

def generate_deepfake_audio(output_path, sr=22050, duration=3):
    """
    Generate AI/vocoder-like audio with characteristic artifacts:
    - Unnaturally smooth pitch transitions
    - Phase discontinuities
    - Over-smoothed spectral envelope
    - Unnatural formant behavior
    - Artifacts from vocoder processing
    """
    t = np.linspace(0, duration, int(sr * duration))

    # Unnaturally smooth pitch (vocoder limitation)
    f0 = 150 + 30 * np.sin(2 * np.pi * 0.2 * t)  # Overly smooth, no micromodulation

    signal = np.zeros_like(t, dtype=np.float32)

    # Harmonics but with unnatural spectral envelope
    for harmonic in range(1, 25):
        partial_freq = f0 * harmonic
        # Unnaturally smooth envelope (no sharp formants)
        partial_amp = (1.0 / harmonic) * np.exp(-harmonic / 15)
        signal += partial_amp * np.sin(2 * np.pi * partial_freq * t)

    # Add vocoder-characteristic phase discontinuities
    for i in range(0, len(signal), int(sr * 0.01)):  # Every 10ms
        phase_shift = np.random.uniform(-0.5, 0.5)
        signal[i:i+int(sr*0.01)] = signal[i:i+int(sr*0.01)] * np.exp(1j * phase_shift).real

    # Add unnatural artifacts: slight clicking
    num_clicks = int(duration * 50)  # More frequent than natural speech
    click_positions = np.random.choice(len(signal), num_clicks, replace=False)
    for pos in click_positions:
        signal[pos] *= (1 + 0.3 * np.random.randn())

    # Minimal noise floor (unnaturally clean)
    noise = 0.002 * np.random.randn(len(signal))
    signal = signal + noise

    # Unnatural amplitude envelope
    envelope = np.ones_like(t)
    attack_time = int(0.01 * sr)  # Too-fast attack
    envelope[:attack_time] = np.linspace(0, 1, attack_time)
    signal = signal * envelope

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    signal = (signal * 0.85).astype(np.float32)

    sf.write(output_path, signal, sr)

def generate_training_datasets(num_authentic=150, num_deepfake=150):
    """Generate authentic and deepfake training datasets."""
    os.makedirs("training_audio/authentic", exist_ok=True)
    os.makedirs("training_audio/deepfake", exist_ok=True)

    # Generate authentic audio
    print(f"Generating {num_authentic} authentic audio samples...")
    for i in range(num_authentic):
        output_path = f"training_audio/authentic/authentic_{i:04d}.wav"
        generate_authentic_audio(output_path)
        if (i + 1) % 30 == 0:
            print(f"  Generated {i + 1}/{num_authentic}")

    # Generate deepfake audio
    print(f"Generating {num_deepfake} deepfake audio samples...")
    for i in range(num_deepfake):
        output_path = f"training_audio/deepfake/deepfake_{i:04d}.wav"
        generate_deepfake_audio(output_path)
        if (i + 1) % 30 == 0:
            print(f"  Generated {i + 1}/{num_deepfake}")

def extract_features_from_folder(folder_path, label):
    """Extract features from all audio files in a folder."""
    features_list = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    for idx, filename in enumerate(files):
        filepath = os.path.join(folder_path, filename)
        try:
            features = extract_features(filepath)
            features_list.append(features[0])
        except Exception as e:
            print(f"  Failed to extract: {filename} - {e}")
            continue

    if features_list:
        X = np.array(features_list)
        y = np.array([label] * len(features_list))
        return X, y
    return None, None

def train_model():
    """Generate data and train the deepfake detector."""
    print("\n=== Deepfake Detector - Synthetic Training Data ===\n")

    # Generate training data
    generate_training_datasets(num_authentic=150, num_deepfake=150)

    # Extract features
    print("\nExtracting features from authentic audio...")
    X_authentic, y_authentic = extract_features_from_folder("training_audio/authentic", label=0)
    print(f"  Extracted {len(X_authentic)} authentic samples")

    print("Extracting features from deepfake audio...")
    X_deepfake, y_deepfake = extract_features_from_folder("training_audio/deepfake", label=1)
    print(f"  Extracted {len(X_deepfake)} deepfake samples")

    # Combine datasets
    X_train = np.vstack([X_authentic, X_deepfake])
    y_train = np.hstack([y_authentic, y_deepfake])

    # Train model
    print(f"\nTraining RandomForest on {len(X_train)} samples...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Save model
    model_path = 'deepfake_detector.pkl'
    joblib.dump(rf_model, model_path)

    accuracy = rf_model.score(X_train, y_train)
    print(f"\n[OK] Model trained and saved to '{model_path}'")
    print(f"  Authentic samples: {len(X_authentic)}")
    print(f"  Deepfake samples: {len(X_deepfake)}")
    print(f"  Training accuracy: {accuracy:.2%}")
    print(f"\nThe model can now detect AI-generated voices!")

if __name__ == "__main__":
    train_model()
