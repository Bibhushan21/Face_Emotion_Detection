import cv2
import numpy as np
from fer import FER

def detect_faces_and_emotions(image):
    # Initialize the FER detector
    emotion_detector = FER(mtcnn=True)
    
    # Detect emotions
    emotions = emotion_detector.detect_emotions(image)
    
    # Draw rectangles and emotions on the image
    for face in emotions:
        x, y, w, h = face['box']
        emotions = face['emotions']
        dominant_emotion = max(emotions, key=emotions.get)
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display emotion
        emotion_text = f"{dominant_emotion}: {emotions[dominant_emotion]:.2f}"
        cv2.putText(image, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def main():
    # Initialize the video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect faces and emotions in the frame
        frame_with_emotions = detect_faces_and_emotions(frame)
        
        # Display the result
        cv2.imshow('Face and Emotion Detection', frame_with_emotions)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
