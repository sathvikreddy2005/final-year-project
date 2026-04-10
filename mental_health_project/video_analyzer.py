import argparse
import json
import os
from pathlib import Path
from typing import Iterable

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
    return cv2.CascadeClassifier(str(cascade_path))


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


def _unavailable_result(reason: str):
    return {
        "available": False,
        "face_detected": False,
        "face_count": 0,
        "reason": reason,
        "stress": 0.0,
        "depression": 0.0,
        "anxiety": 0.0,
    }


def _frame_stride(total_frames: int) -> int:
    if total_frames <= 0:
        return 3
    return max(1, total_frames // 24)


def _iter_sampled_frames(video_capture, total_frames: int) -> Iterable[object]:
    stride = _frame_stride(total_frames)
    frame_index = 0

    while True:
        ok, frame = video_capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            yield frame
        frame_index += 1


def analyze_video(video_path: Path):
    cv2, np, tf, Dense, load_model, img_to_array = _load_dependencies()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    model = _load_model_compat(MODEL_PATH, Dense, load_model)
    detector = _load_face_detector(cv2)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sampled_frame_count = 0
    face_frame_count = 0
    face_count_total = 0
    predictions = []

    try:
        for frame in _iter_sampled_frames(video, total_frames):
            sampled_frame_count += 1
            prediction, face_count = _predict_frame(frame, model, detector, cv2, np, img_to_array)
            if prediction is None:
                continue
            predictions.append(prediction)
            face_frame_count += 1
            face_count_total += face_count
    finally:
        video.release()

    if not predictions:
        return _unavailable_result("No face was detected clearly enough in the recorded video.")

    averaged_prediction = np.mean(np.stack(predictions), axis=0)
    avg_face_count = max(1, round(face_count_total / face_frame_count)) if face_frame_count else 0
    result = _scores_from_prediction(averaged_prediction, avg_face_count)
    result.update(
        {
            "source": "video",
            "total_frames": total_frames,
            "sampled_frames": sampled_frame_count,
            "frames_with_faces": face_frame_count,
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict stress, depression, and anxiety percentages from a recorded video clip."
    )
    parser.add_argument("--video", required=True, help="Path to a recorded video clip.")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON output (for API integration).",
    )
    args = parser.parse_args()

    result = analyze_video(Path(args.video))
    if args.json_only:
        print(json.dumps(result))
    else:
        print("\nVideo Mental Health Output")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
