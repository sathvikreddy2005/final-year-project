import argparse
import json

import joblib
from audio_module import extract_features


MODEL_PATH = "models/audio_model.pkl"
ID_TO_LABEL = {0: "stress", 1: "depression", 2: "anxiety"}


def normalize_label(raw_cls):
    # Supports both string labels (train_model.py) and numeric labels (train_model2.py).
    if isinstance(raw_cls, str):
        return raw_cls.strip().lower()
    return ID_TO_LABEL.get(int(raw_cls), str(raw_cls))


def predict_audio(file_path):
    model = joblib.load(MODEL_PATH)

    features = extract_features(file_path).reshape(1, -1)

    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    results = {"stress": 0.0, "depression": 0.0, "anxiety": 0.0}

    for cls, prob in zip(classes, probabilities):
        label = normalize_label(cls)
        if label in results:
            results[label] = float(prob * 100.0)

    print("\nMental Health Audio Analysis\n")
    print(f"Input file: {file_path}")
    print(f"Stress: {results['stress']:.2f}%")
    print(f"Depression: {results['depression']:.2f}%")
    print(f"Anxiety: {results['anxiety']:.2f}%")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict stress, depression, and anxiety from an audio file."
    )
    parser.add_argument(
        "--file",
        default="test.wav",
        help="Path to input WAV file (default: test.wav)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON output (for API integration).",
    )
    args = parser.parse_args()

    out = predict_audio(args.file)
    if args.json_only:
        print(json.dumps(out))
