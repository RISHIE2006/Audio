import librosa
import numpy as np
from scipy import signal

def extract_features(audio_path: str) -> np.ndarray:
    """
    Extracts acoustic features designed to detect deepfake/TTS artifacts.
    Focus on characteristics that distinguish synthetic from authentic speech.
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {audio_path}: {str(e)}")

    # 1. Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # 2. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # 3. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # 4. Spectral Centroid & Roll-off
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # 5. Phase-based features (TTS artifacts show as phase discontinuities)
    stft_matrix = librosa.stft(y)
    phase = np.angle(stft_matrix)
    phase_derivative = np.diff(phase, axis=1)
    phase_variance = np.var(phase_derivative)

    # 6. DEEPFAKE-SPECIFIC: Harmonic-to-Noise Ratio
    # Real speech has more harmonics, deepfakes tend to have more noise
    S = np.abs(stft_matrix)
    harmonic_sum = np.sum(S, axis=0)
    noise_sum = np.std(S, axis=0)
    hnr = np.mean(harmonic_sum) / (np.mean(noise_sum) + 1e-8)

    # 7. DEEPFAKE-SPECIFIC: Zero Crossing Rate (ZCR) variability
    # Deepfakes have more uniform ZCR, real speech is more varied
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_std = np.std(zcr)
    zcr_mean = np.mean(zcr)

    # 8. DEEPFAKE-SPECIFIC: Fundamental Frequency stability
    # TTS has very stable F0, real speech varies naturally
    try:
        f0 = librosa.yin(y, fmin=80, fmax=400)
        f0_valid = f0[f0 > 0]
        if len(f0_valid) > 0:
            f0_std = np.std(f0_valid)  # Lower std = more synthetic
            f0_mean = np.mean(f0_valid)
        else:
            f0_std = 0
            f0_mean = 0
    except:
        f0_std = 0
        f0_mean = 0

    # 9. DEEPFAKE-SPECIFIC: Spectral flatness
    # Synthetic audio often has flatter spectrum
    spectral_energy = np.mean(np.abs(stft_matrix) ** 2)
    spectral_flatness = np.mean(np.abs(stft_matrix)) / (spectral_energy + 1e-8)

    # 10. DEEPFAKE-SPECIFIC: Temporal modulation
    # Real speech has more amplitude modulation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_std = np.std(onset_env)

    # Combine all features into a single vector
    feature_vector = np.hstack([
        mfccs_mean,           # 20 features
        mfccs_std,            # 20 features (new: captures variability)
        contrast_mean,        # 7 features
        chroma_mean,          # 12 features
        centroid,             # 1 feature
        rolloff,              # 1 feature
        phase_variance,       # 1 feature
        hnr,                  # 1 feature (deepfake indicator)
        zcr_std,              # 1 feature (deepfake indicator)
        zcr_mean,             # 1 feature
        f0_std,               # 1 feature (deepfake indicator)
        f0_mean,              # 1 feature
        onset_std             # 1 feature (deepfake indicator)
    ])

    return feature_vector.reshape(1, -1)
