import cv2 as cv
import mediapipe as mp
import numpy as np
import mediapipe.tasks as mp_tasks
from utils import CvFpsCalc

# Initialize MediaPipe Hands
BaseOptions = mp_tasks.BaseOptions
HandLandmarker = mp_tasks.vision.HandLandmarker
HandLandmarkerOptions = mp_tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp_tasks.vision.HandLandmarkerResult
VisionRunningMode = mp_tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='C:\Users\Arnav Waghdhare\Desktop\Arnav20\Coding\Python\temp_hand_gest\hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.7
# )
# mp_draw = mp.solutions.drawing_utils

# Start webcam capture
cap = cv.VideoCapture(0)

# FPS calculator instance
fps_calc = CvFpsCalc(buffer_len=10)

print("Starting webcam... Press 'q' to quit.")

while cap.isOpened():        
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image horizontally for natural mirror effect
    frame = cv.flip(frame, 1)
    
    # Get FPS
    fps = fps_calc.get()

    # Convert BGR to RGB for MediaPipe processing
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    with HandLandmarker.create_from_options(options) as landmarker:
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Optional: Extract landmark coordinates (63 values)
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            print(f"Extracted Keypoints: {keypoints}")  # Should be 63

    # Display FPS on screen
    cv.putText(frame, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display video feed
    cv.imshow('MediaPipe Hand Detection + FPS', frame)

    # Exit loop on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()