# Mental Health Detection System - Complete Guide

## Project Structure
```
mental_health_project/
├── train_emotion_model.py           # Train MobileNetV2 model with FER2013 dataset
├── detect_with_model.py             # Analyze video with trained model
├── fer2013/                         # FER2013 dataset folder
│   ├── train/                       # Training images organized by emotion
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/                        # Test images organized by emotion
├── video.mp4                        # Video file to analyze
└── mobilenetv2_emotion_model.h5     # Trained model (generated after training)
```

## Step 1: Download FER2013 Dataset
- **FER2013 ✅ (Recommended)**: Best for emotion detection
- Ravdness: Designed for speech-based emotion (not suitable for facial expressions)

**Download FER2013:**
1. Go to https://www.kaggle.com/datasets/msambare/fer2013
2. Download and extract to `fer2013/` folder in your project
3. Should have 7 emotion folders (angry, disgust, fear, happy, neutral, sad, surprise)

## Step 2: Prepare the Dataset
Check what you have:
```powershell
.\.venv\Scripts\python.exe
>>> import os
>>> train_dir = 'fer2013/train'
>>> for emotion in os.listdir(train_dir):
>>>     count = len(os.listdir(os.path.join(train_dir, emotion)))
>>>     print(f"{emotion}: {count} images")
```

## Step 3: Train the Model
Run the training script:
```powershell
.\.venv\Scripts\python.exe train_emotion_model.py
```

**Expected Output:**
```
Total samples: XXX
Total labels: 7 (angry, disgust, fear, happy, neutral, sad, surprise)
Test Accuracy: XX.XX%
F1 Score: X.XX
Model meets the minimum accuracy requirement of 80%.
Model saved as 'mobilenetv2_emotion_model.h5'
```

## Step 4: Analyze Video with Trained Model
Run video analysis (no loop - processes once):
```powershell
.\.venv\Scripts\python.exe detect_with_model.py video.mp4
```

**Expected Output:**
```
Analyzing video: video.mp4
------------------------------------------------------------
Frame 1 | Emotion: HAPPY | Confidence: 95.23%
Frame 2 | Emotion: SAD | Confidence: 87.56%
Frame 3 | Emotion: NEUTRAL | Confidence: 92.10%
...
------------------------------------------------------------
Total Frames Analyzed: 150
Analysis Complete!
```

## Model Architecture: MobileNetV2
- **Base**: MobileNetV2 (pretrained on ImageNet)
- **Processing**: Global Average Pooling
- **Dense Layers**: 1024 neurons + 50% dropout
- **Output**: 7 emotions
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy, F1 Score

## Key Features
✅ Fast analysis (no looping)
✅ MobileNetV2 for efficiency
✅ F1 Score & Accuracy tracking
✅ Minimum 80% accuracy requirement
✅ Confidence scores per frame
✅ Total frame count at end

## Troubleshooting
1. **"fer2013 folder not found"**: Download dataset first
2. **"Model file not found"**: Train the model first with train_emotion_model.py
3. **"Video file not found"**: Ensure video.mp4 exists or specify correct path
