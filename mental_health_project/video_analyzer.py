import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "best_mobilenetv2_emotion.h5"
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
RISK_EMOTIONS = {
    "stress": "angry",
    "depression": "sad",
    "anxiety": "fear",
}


def _load_dependencies():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import tensorflow as tf  # type: ignore
        from tensorflow.keras.layers import Dense  # type: ignore
        from tensorflow.keras.models import load_model  # type: ignore
        from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Video dependencies missing. Install opencv-python and tensorflow in "
            "mental_health_project/venv before using video analysis."
        ) from exc

    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass

    return cv2, np, tf, Dense, load_model, img_to_array


def _load_model_compat(model_path, Dense, load_model):
    original_from_config = Dense.from_config

    @classmethod
    def patched_from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return original_from_config.__func__(cls, config)

    Dense.from_config = patched_from_config
    try:
        return load_model(model_path, compile=False)
    finally:
        Dense.from_config = original_from_config


def _load_face_detector(cv2):
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    return detector


def _prepare_face(frame, face_box, cv2, np, img_to_array):
    x, y, w, h = face_box
    face_roi = frame[y : y + h, x : x + w]
    if face_roi.size == 0:
        raise ValueError("Empty face region")
    face_roi = cv2.resize(face_roi, (224, 224))
    face_roi = img_to_array(face_roi).astype("float32") / 255.0
    return np.expand_dims(face_roi, axis=0)


def _predict_frame(frame, model, detector, cv2, np, img_to_array):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    frame_predictions = []
    for face_box in faces:
        try:
            face_input = _prepare_face(frame, face_box, cv2, np, img_to_array)
            prediction = model.predict(face_input, verbose=0)[0]
            frame_predictions.append(prediction)
        except Exception:
            continue

    if not frame_predictions:
        return None, 0

    return np.mean(np.stack(frame_predictions), axis=0), len(frame_predictions)


def _scores_from_prediction(prediction, face_count):
    emotion_scores = {
        label: round(float(score) * 100.0, 2)
        for label, score in zip(EMOTION_LABELS, prediction)
    }
    return {
        "available": True,
        "face_detected": True,
        "face_count": face_count,
        "stress": round(emotion_scores[RISK_EMOTIONS["stress"]], 2),
        "depression": round(emotion_scores[RISK_EMOTIONS["depression"]], 2),
        "anxiety": round(emotion_scores[RISK_EMOTIONS["anxiety"]], 2),
        "emotions": emotion_scores,
        "dominant_emotion": max(emotion_scores, key=emotion_scores.get),
    }


def analyze_image(image_path: Path):
    cv2, np, tf, Dense, load_model, img_to_array = _load_dependencies()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    model = _load_model_compat(MODEL_PATH, Dense, load_model)
    detector = _load_face_detector(cv2)
    prediction, face_count = _predict_frame(frame, model, detector, cv2, np, img_to_array)
    if prediction is None:
        return {
            "available": False,
            "face_detected": False,
            "face_count": 0,
            "reason": "No face was detected clearly enough in the captured frame.",
            "stress": 0.0,
            "depression": 0.0,
            "anxiety": 0.0,
        }
    return _scores_from_prediction(prediction, face_count)


def main():
    parser = argparse.ArgumentParser(
        description="Predict stress, depression, and anxiety percentages from a webcam image."
    )
    parser.add_argument("--image", required=True, help="Path to a captured image frame.")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON output (for API integration).",
    )
    args = parser.parse_args()

    result = analyze_image(Path(args.image))
    if args.json_only:
        print(json.dumps(result))
        
    else:
        print("\nVideo Mental Health Output")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
