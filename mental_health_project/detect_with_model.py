import os
# silence TensorFlow and absl logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR only
os.environ['ABSL_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# temporarily suppress stderr during TensorFlow import
import sys
import cv2
import numpy as np
orig_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
# set TF logger to error to suppress warnings
try:
    tf.get_logger().setLevel('ERROR')
except Exception:
    pass
# also silence absl Python logger
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass
sys.stderr.close()
sys.stderr = orig_stderr

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import img_to_array


def load_model_compat(model_path):
    # Older H5 models may carry fields that newer Keras versions reject.
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

# Load the trained MobileNetV2 model
# try primary filename, fall back to checkpoint
model_path = None
for candidate in ['mobilenetv2_emotion_model.h5', 'best_mobilenetv2_emotion.h5']:
    if os.path.isfile(candidate):
        model_path = candidate
        break
if model_path is None:
    raise FileNotFoundError("No trained model file found. Run training script first.")
print(f"Loading model from {model_path}")
model = load_model_compat(model_path)

# Emotion labels (7 emotions from FER2013)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get video path from command line or use default
video_path = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
video = cv2.VideoCapture(video_path)

print(f"Analyzing video: {video_path}")

# check that a trained model file exists
if model_path:
    print("Model file found; training appears to have completed.")
    # try to show stored metrics
    try:
        import json
        mdata = json.load(open('model_metrics.json'))
        print(f"Model test accuracy: {mdata['test_accuracy']*100:.2f}%")
        print(f"Model F1 score: {mdata['f1_score']:.2f}")
    except Exception:
        pass
else:
    print("No trained model available – please run training script.")

frame_count = 0
counts = {e:0 for e in emotion_labels}

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = img_to_array(face_roi) / 255.0
        prediction = model.predict(np.expand_dims(face_roi, axis=0), verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        counts[emotion] += 1

video.release()

print("\nVideo summary")
print("-" * 40)
print(f"Total frames processed: {frame_count}")
for e, c in counts.items():
    if c > 0:
        print(f"{e.capitalize()}: {c} ({c/frame_count*100:.1f}%)")
print("Analysis Complete!")
