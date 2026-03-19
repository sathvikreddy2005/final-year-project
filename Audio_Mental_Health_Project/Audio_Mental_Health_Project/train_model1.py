import os
import numpy as np
from audio_module import extract_features

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

import joblib

DATASET_PATH = "dataset/ravdess"

X = []
y = []

# Read dataset
for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".wav"):

            emotion_code = file.split("-")[2]

            # Select only required emotions
            if emotion_code in ["04", "05", "06"]:

                path = os.path.join(root, file)

                features = extract_features(path)

                X.append(features)

                if emotion_code == "05":
                    y.append("stress")

                elif emotion_code == "04":
                    y.append("depression")

                elif emotion_code == "06":
                    y.append("anxiety")

X = np.array(X)

print("Total samples:", len(X))
print("Total labels:", len(y))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC(kernel='rbf', probability=True)

print("Training SVM model...")

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/audio_model.pkl")

print("\nModel saved in models/audio_model.pkl")