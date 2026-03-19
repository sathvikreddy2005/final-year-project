import os
import numpy as np
from audio_module import extract_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

DATASET_PATH = "dataset/ravdess"

X = []
y = []

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".wav"):

            emotion_code = file.split("-")[2]

            # only select 3 emotions
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
print("Random Forest")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

joblib.dump(model, "models/audio_model.pkl")