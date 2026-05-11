"""
evaluate_model.py
─────────────────
Loads the trained deepfake_detector.pkl, rebuilds the same train/test split
used during training (random_state=42), then prints:
  • accuracy_score
  • confusion_matrix
  • classification_report
"""

import os, glob
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -- 1. Re-load the features exactly as train_model.py does ------------------
import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.dsp_logic import extract_features

print("=" * 60)
print("  DEEPFAKE DETECTOR - Model Evaluation")
print("=" * 60)

X_all, y_all = [], []

for label, folder in [(0, "audio_data/authentic"), (1, "audio_data/synthetic")]:
    label_name = "Authentic" if label == 0 else "Synthetic"
    files = (glob.glob(f"{folder}/*.wav") + glob.glob(f"{folder}/*.mp3"))
    print(f"\n[{label_name}] Found {len(files)} file(s) in '{folder}'")
    for f in files:
        try:
            feat = extract_features(f)
            X_all.append(feat[0])
            y_all.append(label)
            print(f"   *  {os.path.basename(f)}")
        except Exception as e:
            print(f"   !  {os.path.basename(f)} - {e}")

if len(X_all) < 6:
    print("\n[ERROR] Not enough samples to evaluate (need >= 6).")
    sys.exit(1)

X_all = np.array(X_all)
y_all = np.array(y_all)

# -- 2. Recreate the exact same split ----------------------------------------
_, X_test, _, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# -- 3. Load model & predict -------------------------------------------------
model_path = "deepfake_detector.pkl"
if not os.path.exists(model_path):
    print(f"\n[ERROR] Model not found at '{model_path}'. Run train_model.py first.")
    sys.exit(1)

model = joblib.load(model_path)
y_pred = model.predict(X_test)

# -- 4. Print metrics ---------------------------------------------------------
print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)

acc = accuracy_score(y_test, y_pred)
print(f"\n[+] Accuracy Score : {acc:.4f}  ({acc * 100:.2f}%)")
print("""
  ->  Fraction of total predictions that are correct.
      Range: 0 (worst) -> 1 (best).
""")

cm = confusion_matrix(y_test, y_pred)
print("[+] Confusion Matrix:")
print(f"""
       Predicted
       Authentic  Synthetic
  A  [ {cm[0][0]:>5}      {cm[0][1]:>5} ]   <- True Authentic
  c
  t  [ {cm[1][0]:>5}      {cm[1][1]:>5} ]   <- True Synthetic
  u
  a
  l
""")
print("""  Cells explained:
    [0,0] True Negatives  - real audio correctly called authentic
    [0,1] False Positives - real audio wrongly flagged as deepfake
    [1,0] False Negatives - deepfake missed (called authentic)
    [1,1] True Positives  - deepfake correctly detected
""")

print("[+] Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["Authentic", "Synthetic"]))
print("""  Column meanings:
    precision  - of all samples predicted as X, how many truly are X?
    recall     - of all true X samples, how many did we catch?
    f1-score   - harmonic mean of precision & recall (balanced measure)
    support    - actual number of samples in each class in y_test
""")
print("=" * 60)
