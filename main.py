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


def predict_genre(audio_path, model_dir="models"):
    model_path = os.path.join(model_dir, "champion_genre_classifier.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "genre_classifier.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, le_path]):
        print("Model artifacts not found! Run 'python train.py' first to train and generate models.")
        sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(le_path)

    print(f"Extracting features from: {audio_path}")
    raw_features = extract_features_from_audio(audio_path)
    scaled_features = scaler.transform(raw_features)

    prediction = model.predict(scaled_features)
    predicted_genre = label_encoder.inverse_transform(prediction)[0]
    probabilities = model.predict_proba(scaled_features)[0]

    print(f"\n==========================================")
    print(f" Predicted Genre: {predicted_genre.upper()}")
    print(f" Confidence:      {np.max(probabilities) * 100:.2f}%")
    print(f"==========================================")

    # Top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    print("\nTop Candidates:")
    for idx in top_indices:
        print(f" - {label_encoder.classes_[idx].ljust(12)}: {probabilities[idx] * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Predict music genre from an audio file.")
    parser.add_argument("--audio", type=str, help="Path to audio file (.wav or .mp3)")
    args = parser.parse_args()

    if not args.audio:
        print("Usage: python main.py --audio <path_to_audio_file>")
        return

    if not os.path.exists(args.audio):
        print(f"Audio file '{args.audio}' does not exist.")
        return

    predict_genre(args.audio)


if __name__ == "__main__":
    main()
