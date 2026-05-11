import numpy as np
import librosa
import os
from sklearn.ensemble import RandomForestClassifier
import joblib
from utils.dsp_logic import extract_features

def extract_segments_from_audio(audio_path, segment_duration=3, sr=22050):
    """
    Extract multiple segments from a long audio file for training.
    This creates more training samples from a single file.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        segment_samples = int(segment_duration * sr)

        segments = []
        # Extract overlapping segments
        for start in range(0, len(y) - segment_samples, segment_samples // 2):
            segment = y[start:start + segment_samples]
            if len(segment) == segment_samples:
                segments.append(segment)

        return segments, sr
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return [], sr

def save_audio_segment(audio_data, output_path, sr=22050):
    """Save audio segment to file."""
    import soundfile as sf
    sf.write(output_path, audio_data.astype(np.float32), sr)

def extract_features_from_audio(audio_path):
    """Extract features from audio file."""
    try:
        features = extract_features(audio_path)
        return features[0]
    except Exception as e:
        print(f"Failed to extract features from {audio_path}: {e}")
        return None

def train_on_real_data():
    """Train model on real music and FineVoice AI voice."""
    print("\n=== Training on Real Audio Files ===\n")

    # Paths to training files
    music_file = "real_music.mp3"
    finevoice_file = "finevoice_ai.mp3"

    if not os.path.exists(music_file):
        print(f"ERROR: Music file not found at {music_file}")
        return

    if not os.path.exists(finevoice_file):
        print(f"ERROR: FineVoice file not found at {finevoice_file}")
        return

    print(f"Using music file: {music_file}")
    print(f"Using FineVoice file: {finevoice_file}\n")

    # Create training directories
    os.makedirs("training_audio/real_music", exist_ok=True)
    os.makedirs("training_audio/real_finevoice", exist_ok=True)

    # Extract segments from music (authentic)
    print("Processing real music into segments...")
    music_segments, sr = extract_segments_from_audio(music_file)
    print(f"  Extracted {len(music_segments)} segments from music")

    # Save music segments
    for i, segment in enumerate(music_segments):
        path = f"training_audio/real_music/music_{i:04d}.wav"
        save_audio_segment(segment, path, sr)

    # Extract segments from FineVoice (deepfake)
    print("Processing FineVoice AI voice into segments...")
    finevoice_segments, sr = extract_segments_from_audio(finevoice_file)
    print(f"  Extracted {len(finevoice_segments)} segments from FineVoice")

    # Save FineVoice segments
    for i, segment in enumerate(finevoice_segments):
        path = f"training_audio/real_finevoice/finevoice_{i:04d}.wav"
        save_audio_segment(segment, path, sr)

    # Extract features
    print("\nExtracting features from authentic music...")
    X_authentic = []
    for filename in os.listdir("training_audio/real_music"):
        if filename.endswith(".wav"):
            features = extract_features_from_audio(os.path.join("training_audio/real_music", filename))
            if features is not None:
                X_authentic.append(features)
    X_authentic = np.array(X_authentic)
    y_authentic = np.zeros(len(X_authentic))
    print(f"  Extracted {len(X_authentic)} feature vectors from music")

    print("Extracting features from FineVoice AI...")
    X_finevoice = []
    for filename in os.listdir("training_audio/real_finevoice"):
        if filename.endswith(".wav"):
            features = extract_features_from_audio(os.path.join("training_audio/real_finevoice", filename))
            if features is not None:
                X_finevoice.append(features)
    X_finevoice = np.array(X_finevoice)
    y_finevoice = np.ones(len(X_finevoice))
    print(f"  Extracted {len(X_finevoice)} feature vectors from FineVoice")

    # Combine datasets
    X_train = np.vstack([X_authentic, X_finevoice])
    y_train = np.hstack([y_authentic, y_finevoice])

    print(f"\nTotal training samples: {len(X_train)}")
    print(f"  Authentic (music): {len(X_authentic)}")
    print(f"  Deepfake (FineVoice): {len(X_finevoice)}")

    # Train model
    print("\nTraining RandomForest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Save model
    model_path = 'deepfake_detector.pkl'
    joblib.dump(rf_model, model_path)

    accuracy = rf_model.score(X_train, y_train)
    print(f"\n[OK] Model trained and saved to '{model_path}'")
    print(f"  Training accuracy: {accuracy:.2%}")
    print(f"\nModel is now trained to detect FineVoice AI voices!")

if __name__ == "__main__":
    train_on_real_data()
