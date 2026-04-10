import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# configuration for memory-limited environments
BATCH_SIZE = 16  # lower if you still hit OOM errors
IMAGE_SIZE = (128, 128)  # smaller images reduce memory usage

# Step 1: Load FER2013 dataset
# Assuming FER2013 is downloaded and extracted to 'dataset' folder in the workspace
# Download from: https://www.kaggle.com/datasets/msambare/fer2013
data_dir = 'dataset'  # Path to FER2013 dataset

# FER2013 has train and test folders
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Print dataset info
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
total_train_samples = sum([len(os.listdir(os.path.join(train_dir, emotion))) for emotion in os.listdir(train_dir)])
total_test_samples = sum([len(os.listdir(os.path.join(test_dir, emotion))) for emotion in os.listdir(test_dir)])
print(f"Total Training Samples: {total_train_samples}")
print(f"Total Test Samples: {total_test_samples}")
print(f"Total Labels (Emotions): 7")
for emotion in sorted(os.listdir(train_dir)):
    count = len(os.listdir(os.path.join(train_dir, emotion)))
    print(f"  - {emotion.capitalize()}: {count} images")
print("=" * 60)

# Data generators for augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

# Step 2: Build MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)  # 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the model
print("\n" + "=" * 60)
print("TRAINING MODEL - MobileNetV2")
print("=" * 60 + "\n")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_mobilenetv2_emotion.h5', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=callbacks
)

# Step 4: Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# F1 Score and other metrics
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print(f"F1 Score: {f1:.2f}")

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# Ensure accuracy > 80%
if test_acc > 0.80:
    print("Model meets the minimum accuracy requirement of 80%.")
else:
    print("Model accuracy is below 80%. Consider further tuning.")

# Save metrics to JSON for later reference
metrics = {
    'test_accuracy': float(test_acc),
    'f1_score': float(f1)
}
with open('model_metrics.json', 'w') as mf:
    import json
    json.dump(metrics, mf)
print("Evaluation metrics written to model_metrics.json")

# Save the model
model.save('mobilenetv2_emotion_model.h5')
print("Model saved as 'mobilenetv2_emotion_model.h5'")
