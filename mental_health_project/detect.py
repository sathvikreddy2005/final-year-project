import cv2
import sys
from deepface import DeepFace

# allow video path as a command-line argument; default to video.mp4
video_path = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
video = cv2.VideoCapture(video_path)

# process video once and exit
while True:

    ret, frame = video.read()

    if not ret:
        break

    try:

        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        emotions = result[0]['emotion']
        dominant = result[0].get('dominant_emotion', 'unknown')

        depression = emotions['sad']
        anxiety = emotions['fear']
        stress = emotions['angry']

        # display emotion percentages
        info = (f"Depression: {depression:.1f}% | Anxiety: {anxiety:.1f}% | "
                f"Stress: {stress:.1f}%")
        print(f"Dominant: {dominant} | {info}")

    except Exception as e:
        # log error if needed
        print(f"Analysis failed: {e}")
        continue

# release resources when done
video.release()
print("Done analyzing video.")