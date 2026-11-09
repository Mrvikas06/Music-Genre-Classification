import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import librosa


def extract_features_from_audio(file_path):
    """
    Extracts time-frequency and spectral features from an audio file (.wav/.mp3).
    Matches the schema of features_30_sec.csv (chroma, rms, spectral centroid,
    spectral bandwidth, rolloff, zero crossing rate, harmony, perceptr, tempo, and 20 MFCCs).
    """
    signal, sr = librosa.load(file_path, duration=30)

    chroma_stft = librosa.feature.chroma_stft(y=signal, sr=sr)
    rms = librosa.feature.rms(y=signal)
    spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(signal)
    harmony, perceptr = librosa.effects.hpss(signal)
    tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)

    features = [
        np.mean(chroma_stft), np.var(chroma_stft),
        np.mean(rms), np.var(rms),
        np.mean(spec_cent), np.var(spec_cent),
        np.mean(spec_bw), np.var(spec_bw),
        np.mean(rolloff), np.var(rolloff),
        np.mean(zcr), np.var(zcr),
        np.mean(harmony), np.var(harmony),
        np.mean(perceptr), np.var(perceptr),
        float(tempo) if np.isscalar(tempo) else float(tempo[0]),
    ]

    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
    for i in range(20):
        features.append(np.mean(mfccs[i]))
        features.append(np.var(mfccs[i]))

    return np.array(features).reshape(1, -1)
