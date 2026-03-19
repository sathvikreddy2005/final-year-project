import librosa
import numpy as np

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=22050)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)

    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)

    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)

    features = np.hstack([mfcc, chroma, zcr, rms])

    return features