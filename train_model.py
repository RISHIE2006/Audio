import argparse
import glob
import json
import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils.dsp_logic import extract_features


def parse_args():
    parser = argparse.ArgumentParser(description="Train a deepfake audio detector on real and generated datasets.")
    parser.add_argument("--authentic-dirs", type=str, default="audio_data/authentic", help="Comma-separated directories containing authentic human audio.")
    parser.add_argument("--synthetic-dirs", type=str, default="audio_data/synthetic", help="Comma-separated directories containing synthetic/TTS audio.")
    parser.add_argument("--generated-authentic-dirs", type=str, default="", help="Comma-separated generated authentic-like directories to optionally include.")
    parser.add_argument("--generated-synthetic-dirs", type=str, default="", help="Comma-separated generated synthetic/TTS directories to optionally include.")
    parser.add_argument("--model-output", type=str, default="deepfake_detector.pkl", help="Path to save the trained model.")
    parser.add_argument("--metadata-output", type=str, default="training_metadata.json", help="Path to save training metadata.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data used for evaluation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting and training.")
    parser.add_argument("--force", action="store_true", help="Retrain and overwrite existing model.")
    parser.add_argument("--timestamped", action="store_true", help="Save a timestamped copy of the trained model.")
    return parser.parse_args()


def find_audio_files(directory: str):
    if not os.path.isdir(directory):
        return []
    return glob.glob(os.path.join(directory, "*.wav")) + glob.glob(os.path.join(directory, "*.mp3"))


def load_features_from_folder(folder_path: str, label: int, source_name: str):
    if not os.path.isdir(folder_path):
        print(f"⚠ Directory not found: {folder_path}")
        return [], [], 0

    files = sorted(find_audio_files(folder_path))
    print(f"Loading {len(files)} files from {source_name}: {folder_path}")

    X = []
    y = []
    failures = 0
    for audio_file in files:
        try:
            features = extract_features(audio_file)
            X.append(features[0])
            y.append(label)
            print(f"  [OK] {os.path.basename(audio_file)}")
        except Exception as e:
            failures += 1
            print(f"  [ERROR] Failed to process {os.path.basename(audio_file)}: {e}")

    return X, y, failures


def gather_dataset(directories: str, label: int, source_name: str):
    paths = [d.strip() for d in directories.split(",") if d.strip()]
    X_total = []
    y_total = []
    total_failures = 0
    for directory in paths:
        X, y, failures = load_features_from_folder(directory, label, f"{source_name} ({directory})")
        X_total.extend(X)
        y_total.extend(y)
        total_failures += failures
    return np.array(X_total) if X_total else np.empty((0, 0)), np.array(y_total) if y_total else np.empty((0,)), total_failures, paths


def save_json(data, path):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Saved metadata to {path}")
    except Exception as e:
        print(f"Failed to save metadata to {path}: {e}")


def main():
    args = parse_args()

    if os.path.exists(args.model_output) and not args.force:
        print(f"Model already exists at {args.model_output}. Use --force to retrain.")
        return

    print("\n=== Deepfake Detector Training ===\n")
    print(f"Authentic directories: {args.authentic_dirs}")
    print(f"Synthetic directories: {args.synthetic_dirs}")
    if args.generated_authentic_dirs:
        print(f"Generated authentic directories: {args.generated_authentic_dirs}")
    if args.generated_synthetic_dirs:
        print(f"Generated synthetic directories: {args.generated_synthetic_dirs}")

    X_authentic, y_authentic, auth_failures, auth_paths = gather_dataset(args.authentic_dirs, 0, "Real authentic")
    X_synthetic, y_synthetic, synth_failures, synth_paths = gather_dataset(args.synthetic_dirs, 1, "Real synthetic/TTS")

    X_generated_authentic, y_generated_authentic, gen_auth_failures, gen_auth_paths = np.empty((0, 0)), np.empty((0,)), 0, []
    X_generated_synthetic, y_generated_synthetic, gen_synth_failures, gen_synth_paths = np.empty((0, 0)), np.empty((0,)), 0, []

    if args.generated_authentic_dirs:
        X_generated_authentic, y_generated_authentic, gen_auth_failures, gen_auth_paths = gather_dataset(args.generated_authentic_dirs, 0, "Generated authentic-like")
    if args.generated_synthetic_dirs:
        X_generated_synthetic, y_generated_synthetic, gen_synth_failures, gen_synth_paths = gather_dataset(args.generated_synthetic_dirs, 1, "Generated synthetic/TTS")

    feature_arrays = [arr for arr in [X_authentic, X_synthetic, X_generated_authentic, X_generated_synthetic] if arr.size > 0]
    label_arrays = [arr for arr in [y_authentic, y_synthetic, y_generated_authentic, y_generated_synthetic] if arr.size > 0]

    if feature_arrays:
        X_train = np.vstack(feature_arrays)
    else:
        X_train = np.empty((0, 0))

    if label_arrays:
        y_train = np.hstack(label_arrays)
    else:
        y_train = np.empty((0,))

    if len(X_train) < 8:
        print("\n[ERROR] Not enough training data! Need at least 8 samples total.")
        print("Please add audio files to the specified directories or use generated datasets.")
        return

    print(f"\nTraining samples: {len(X_train)}")
    print(f"  Real authentic: {len(y_authentic)}")
    print(f"  Real synthetic/TTS: {len(y_synthetic)}")
    if X_generated_authentic.size > 0:
        print(f"  Generated authentic-like: {len(y_generated_authentic)}")
    if X_generated_synthetic.size > 0:
        print(f"  Generated synthetic: {len(y_generated_synthetic)}")

    X_train_split, X_test, y_train_split, y_test = train_test_split(
        X_train, y_train, test_size=args.test_size, random_state=args.random_state, stratify=y_train
    )

    print("Training Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=args.random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train_split, y_train_split)

    train_accuracy = accuracy_score(y_train_split, rf_model.predict(X_train_split))
    test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    report = classification_report(y_test, rf_model.predict(X_test), output_dict=True)
    matrix = confusion_matrix(y_test, rf_model.predict(X_test)).tolist()

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if args.timestamped:
        timestamped_output = os.path.splitext(args.model_output)[0] + f"_{timestamp}.pkl"
        joblib.dump(rf_model, timestamped_output)
        print(f"Saved timestamped model to {timestamped_output}")
    else:
        timestamped_output = None

    joblib.dump(rf_model, args.model_output)
    print(f"Saved model to {args.model_output}")

    metadata = {
        "model_output": os.path.abspath(args.model_output),
        "timestamped_model_output": os.path.abspath(timestamped_output) if timestamped_output else None,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "authentic_dirs": auth_paths,
        "synthetic_dirs": synth_paths,
        "generated_authentic_dirs": gen_auth_paths,
        "generated_synthetic_dirs": gen_synth_paths,
        "sample_counts": {
            "real_authentic": len(y_authentic),
            "real_synthetic": len(y_synthetic),
            "generated_authentic": len(y_generated_authentic),
            "generated_synthetic": len(y_generated_synthetic),
        },
        "failures": {
            "real_authentic": int(auth_failures),
            "real_synthetic": int(synth_failures),
            "generated_authentic": int(gen_auth_failures),
            "generated_synthetic": int(gen_synth_failures),
        },
        "evaluation": {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "confusion_matrix": matrix,
            "classification_report": report,
        },
    }
    save_json(metadata, args.metadata_output)

    print(f"\nTraining complete. Test accuracy: {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
