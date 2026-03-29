# Mental Health Detection System Documentation

## Overview

This project performs multimodal mental health screening using:

- Text
- Audio
- Video

The system works in four layers:

1. The frontend collects user input.
2. The backend API receives that input.
3. The backend calls the correct model inference script.
4. The frontend fuses the returned scores and displays the final result.

## Main Files

### Frontend

- [UI/mental-health-detection.html](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.html)
- [UI/mental-health-detection.js](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.js)

### Backend

- [backend_api.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\backend_api.py)

### Text

- [TEXT/text_input_inference.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\TEXT\text_input_inference.py)
- [TEXT/text_pipeline.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\TEXT\text_pipeline.py)
- [TEXT/lightgbm_text_model.pkl](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\TEXT\lightgbm_text_model.pkl)
- [TEXT/lightgbm_text_metadata.json](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\TEXT\lightgbm_text_metadata.json)

### Audio

- [Audio_Mental_Health_Project/Audio_Mental_Health_Project/predict_audio.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\predict_audio.py)
- [Audio_Mental_Health_Project/Audio_Mental_Health_Project/audio_module.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\audio_module.py)
- [Audio_Mental_Health_Project/Audio_Mental_Health_Project/train_model.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\train_model.py)
- [Audio_Mental_Health_Project/Audio_Mental_Health_Project/models/audio_model.pkl](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\models\audio_model.pkl)

### Video

- [mental_health_project/video_analyzer.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\mental_health_project\video_analyzer.py)
- [mental_health_project/best_mobilenetv2_emotion.h5](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\mental_health_project\best_mobilenetv2_emotion.h5)

## 1. Text Analysis Flow

### User input

The user enters text in the frontend UI.

- HTML page structure: [UI/mental-health-detection.html](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.html)
- Frontend logic: [UI/mental-health-detection.js](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.js)

### Frontend to backend

When the user clicks Analyze Text:

1. `analyzeText()` in [UI/mental-health-detection.js](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.js) is called.
2. It sends a request to `POST /predict/text`.

### Backend processing

The backend route is in [backend_api.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\backend_api.py).

Flow:

1. `predict_text()` receives the text.
2. The backend runs [TEXT/text_input_inference.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\TEXT\text_input_inference.py) using `subprocess`.
3. The script prints JSON.
4. Backend reads that JSON and returns it to the frontend.

### Text inference script

[TEXT/text_input_inference.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\TEXT\text_input_inference.py) does the following:

1. Loads the trained LightGBM model from [TEXT/lightgbm_text_model.pkl](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\TEXT\lightgbm_text_model.pkl).
2. Loads SBERT sentence embedding model `all-MiniLM-L6-v2`.
3. Converts user text into embeddings.
4. Runs the LightGBM model.
5. Produces a depression score.

Important:

- Text currently contributes only `depression`.
- Stress and anxiety are not predicted from text anymore.

### Where the text model was trained

The LightGBM text model was trained and saved by [TEXT/text_pipeline.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\TEXT\text_pipeline.py).

That file:

1. Loads the text dataset.
2. Combines posts per user.
3. Generates SBERT embeddings.
4. Trains multiple LightGBM candidates.
5. Selects the best one.
6. Saves:
   - `lightgbm_text_model.pkl`
   - `lightgbm_text_metadata.json`

## 2. Audio Analysis Flow

### User input

The user records audio in the frontend UI.

- UI logic: [UI/mental-health-detection.js](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.js)

### Frontend to backend

When the user clicks Analyze Audio:

1. `analyzeAudio()` in [UI/mental-health-detection.js](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.js) is called.
2. It uploads the recorded audio file to `POST /predict/audio`.

### Backend processing

The backend audio route is in [backend_api.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\backend_api.py).

Flow:

1. `predict_audio()` receives the uploaded audio file.
2. It stores the file temporarily.
3. It runs [Audio_Mental_Health_Project/Audio_Mental_Health_Project/predict_audio.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\predict_audio.py).
4. That script prints JSON.
5. The backend returns the parsed JSON to the frontend.

### Audio inference script

[Audio_Mental_Health_Project/Audio_Mental_Health_Project/predict_audio.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\predict_audio.py) does the following:

1. Loads the trained audio model from [Audio_Mental_Health_Project/Audio_Mental_Health_Project/models/audio_model.pkl](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\models\audio_model.pkl).
2. Calls `extract_features()` from [Audio_Mental_Health_Project/Audio_Mental_Health_Project/audio_module.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\audio_module.py).
3. Extracts audio features such as:
   - MFCC
   - Chroma
   - ZCR
   - RMS
4. Runs `predict_proba()` on the trained model.
5. Produces:
   - stress
   - depression
   - anxiety

### Which training file produced the current audio model

The current saved audio model is a Random Forest model.

So the active `audio_model.pkl` most likely came from:

- [Audio_Mental_Health_Project/Audio_Mental_Health_Project/train_model.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\train_model.py)

This is because:

- `train_model.py` uses `RandomForestClassifier`
- `train_model1.py` uses `SVC`
- `train_model2.py` uses `XGBClassifier`

The current saved `.pkl` is a `RandomForestClassifier`.

## 3. Video Analysis Flow

### User input

The user starts the webcam and records a short video clip in the frontend.

- UI structure: [UI/mental-health-detection.html](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.html)
- UI logic: [UI/mental-health-detection.js](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.js)

### Frontend to backend

When the user clicks Record & Analyze Video:

1. `analyzeVideo()` in [UI/mental-health-detection.js](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.js) records a short clip.
2. It uploads that clip to `POST /predict/video`.

### Backend processing

The backend video route is in [backend_api.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\backend_api.py).

Flow:

1. `predict_video()` receives the uploaded video.
2. It stores the clip temporarily.
3. It runs [mental_health_project/video_analyzer.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\mental_health_project\video_analyzer.py) with `--video`.
4. The script prints JSON.
5. The backend returns that JSON to the frontend.

### Video inference script

[mental_health_project/video_analyzer.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\mental_health_project\video_analyzer.py) does the following:

1. Loads the trained model from [mental_health_project/best_mobilenetv2_emotion.h5](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\mental_health_project\best_mobilenetv2_emotion.h5).
2. Opens the uploaded video clip using OpenCV.
3. Samples frames from the clip.
4. Detects faces in the sampled frames.
5. Predicts emotion probabilities from the detected face.
6. Aggregates those frame-level predictions.
7. Maps emotions to mental health scores:
   - `angry` -> stress
   - `fear` -> anxiety
   - `sad` -> depression

So the video branch contributes:

- stress
- anxiety
- depression

## 4. Role of backend_api.py

[backend_api.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\backend_api.py) is the bridge between the frontend and all model scripts.

It contains three main routes:

- `/predict/text`
- `/predict/audio`
- `/predict/video`

Its job is:

1. receive data from the frontend
2. run the correct inference script
3. collect the script output
4. return the output in API format

It also returns:

- `raw`: the full original model output
- `scores`: simplified numeric values used for fusion

## 5. Meaning of Raw and Scores

### Raw

`raw` contains the complete output returned by the inference script.

This may include:

- score
- level
- source
- metadata
- additional explanation-related fields

### Scores

`scores` contains simplified numeric values extracted from `raw`.

These values are used in the frontend for fusion and display.

## 6. How the Final Scores Are Fused

Score fusion is done in:

- [UI/mental-health-detection.js](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.js)

Main logic:

- `computeFusedScores()`
- `averageMetric()`

### Current fusion rule

Each metric is averaged only across the modalities that actually provide that metric.

That means:

- `stress` = average of audio stress and video stress
- `anxiety` = average of audio anxiety and video anxiety
- `depression` = average of text depression, audio depression, and video depression

### Current modality contribution

- Text contributes only to depression
- Audio contributes to stress, anxiety, depression
- Video contributes to stress, anxiety, depression

## 7. How Final Results Are Displayed

After all three branches finish:

1. Frontend stores per-modality results in `modalityScores`
2. Frontend stores detailed raw outputs in `modalityRaw`
3. `computeFusedScores()` calculates final fused values
4. `animResults()` animates the result cards
5. `renderExplainability()` fills the explanation section
6. The results page displays:
   - Stress score
   - Anxiety score
   - Depression score
   - Explanation bullets
   - Assistant/chat suggestions

The final result screen is defined in:

- [UI/mental-health-detection.html](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.html)

The logic for filling and animating it is in:

- [UI/mental-health-detection.js](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\UI\mental-health-detection.js)

## 8. Simple End-to-End Summary

### Text

User text  
-> frontend sends to backend  
-> backend runs [TEXT/text_input_inference.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\TEXT\text_input_inference.py)  
-> script loads [TEXT/lightgbm_text_model.pkl](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\TEXT\lightgbm_text_model.pkl)  
-> returns depression score  
-> frontend stores result

### Audio

User audio  
-> frontend sends to backend  
-> backend runs [Audio_Mental_Health_Project/Audio_Mental_Health_Project/predict_audio.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\predict_audio.py)  
-> script loads [Audio_Mental_Health_Project/Audio_Mental_Health_Project/models/audio_model.pkl](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\models\audio_model.pkl)  
-> extracts features using [Audio_Mental_Health_Project/Audio_Mental_Health_Project/audio_module.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\Audio_Mental_Health_Project\Audio_Mental_Health_Project\audio_module.py)  
-> returns stress, depression, anxiety  
-> frontend stores result

### Video

User video clip  
-> frontend sends to backend  
-> backend runs [mental_health_project/video_analyzer.py](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\mental_health_project\video_analyzer.py)  
-> script loads [mental_health_project/best_mobilenetv2_emotion.h5](c:\Users\SATHVIK\OneDrive\Desktop\textanalysis\mental_health_project\best_mobilenetv2_emotion.h5)  
-> samples frames and predicts emotions  
-> maps emotions to stress, depression, anxiety  
-> frontend stores result

### Final fusion

Frontend fuses all available scores  
-> shows final stress, anxiety, depression  
-> displays explanation and assistant guidance
