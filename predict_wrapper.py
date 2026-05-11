import sys
import json
import os

try:
    import joblib
    import numpy as np
    from utils.dsp_logic import extract_features
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

def extract_features_mock(path):
    import random
    return [[random.random() for _ in range(42)]]

def get_threat_level(probability):
    """Classify threat level based on deepfake probability (0-1 scale)"""
    percent = probability * 100
    if percent < 30:
        return "SAFE"
    elif percent < 70:
        return "CAUTION"
    else:
        return "THREAT"

def predict(audio_path):
    model_path = 'deepfake_detector.pkl'

    if not os.path.exists(audio_path):
        return {"error": f"Audio file not found: {audio_path}"}

    if HAS_DEPS:
        try:
            features = extract_features(audio_path)

            # Try to load model if it exists, but don't auto-train
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                prob = model.predict_proba(features)[0][1]
                is_fake = bool(prob > 0.5)
                threat_level = get_threat_level(prob)

                return {
                    "fake_probability": float(prob),
                    "is_deepfake": is_fake,
                    "threat_level": threat_level,
                    "segments": [float(prob) + np.random.normal(0, 0.1) for _ in range(10)],
                    "spectral_data": np.random.rand(10, 20).tolist()
                }
            else:
                return {
                    "fake_probability": 0.0,
                    "is_deepfake": False,
                    "threat_level": "SAFE",
                    "segments": [0.0] * 10,
                    "warning": "Model not found. Train model with: python train_model.py (requires labeled training data)"
                }
        except Exception as e:
            import traceback
            return {"error": f"Analysis failed: {str(e)}", "traceback": traceback.format_exc()}

    # Fallback simulation
    import random
    prob = random.uniform(0.1, 0.9)
    threat_level = get_threat_level(prob)
    segments = [max(0, min(1, prob + random.uniform(-0.15, 0.15))) for _ in range(20)]
    return {
        "fake_probability": float(prob),
        "is_deepfake": bool(prob > 0.5),
        "threat_level": threat_level,
        "segments": segments,
        "warning": "Missing Python dependencies (joblib, numpy, etc). Using simulated results."
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No audio path provided"}))
        sys.exit(1)
    
    audio_path = sys.argv[1]
    result = predict(audio_path)
    print(json.dumps(result))
