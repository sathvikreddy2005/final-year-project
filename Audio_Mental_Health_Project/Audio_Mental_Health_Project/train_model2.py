import os
import numpy as np
from audio_module import extract_features

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier
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

# Convert labels to numbers
label_map = {
    "stress": 0,
    "depression": 1,
    "anxiety": 2
}

y = np.array([label_map[label] for label in y])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=3
)

print("Training XGBoost model...")

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/audio_model.pkl")

print("\nModel saved to models/audio_model.pkl")