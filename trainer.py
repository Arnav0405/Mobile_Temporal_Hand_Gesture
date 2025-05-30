import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import time
import csv
from datetime import datetime
from collections import deque
import mediapipe.tasks as mp_tasks
 
BaseOptions = mp_tasks.BaseOptions
HandLandmarker = mp_tasks.vision.HandLandmarker
HandLandmarkerOptions = mp_tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp_tasks.vision.HandLandmarkerResult
VisionRunningMode = mp_tasks.vision.RunningMode

# FPS Calculator Class
class CvFpsCalc(object):
    def __init__(self, buffer_len=10):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        diff_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick
        self._difftimes.append(diff_time)
        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes)) if len(self._difftimes) > 0 else 0
        return round(fps, 2)

# Global variable to store latest hand landmarks
latest_result = None

# Result callback function
def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Create dataset directories
gesture_classes = ['swipe_up', 'swipe_down', 'thumbs_up', 'idle_gestures']
sequence_length = 30
dataset_dir = 'dataset'
metadata_file = os.path.join(dataset_dir, 'metadata.csv')

for cls in gesture_classes:
    os.makedirs(os.path.join(dataset_dir, cls), exist_ok=True)

# Create metadata CSV file if it doesn't exist
if not os.path.exists(metadata_file):
    with open(metadata_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'label', 'class_id', 'file_path'])

# FPS calculator
fps_calc = CvFpsCalc()

# Video capture
cap = cv.VideoCapture(0)
print("Starting webcam... Press 'q' to quit.")
print("To record a gesture:")
print(" - Press 'r' to start countdown")
print(" - Prepare your gesture during the 3-second countdown")
print(" - Label the gesture by pressing the corresponding key:")
print("   1: Swipe Up")
print("   2: Swipe Down")
print("   3: Thumbs Up")
print("   4: Idle Gesture")

record_key_pressed = False
recording = False
counting_down = False
buffer = []
countdown_start_time = None
countdown_duration = 3  # seconds

# Set up MediaPipe Hand Landmarker
model_path = 'hand_landmarker.task'

base_options = BaseOptions(model_asset_path=model_path)
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=result_callback
)

with HandLandmarker.create_from_options(options) as detector:

    frame_timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        fps = fps_calc.get()

        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        frame_timestamp_ms = int(time.time() * 1000)

        # Send frame to MediaPipe for async processing
        detector.detect_async(mp_image, frame_timestamp_ms)

        annotated_image = frame.copy()

        if counting_down:
            elapsed = time.time() - countdown_start_time
            remaining = countdown_duration - int(elapsed)
            if remaining > 0:
                cv.putText(annotated_image, f"Get Ready: {remaining}", (150, 100),
                           cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            else:
                print("Recording started...")
                counting_down = False
                recording = True
                buffer = []

        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Draw landmarks on image
                for lm in hand_landmarks:
                    x_px = int(lm.x * frame.shape[1])
                    y_px = int(lm.y * frame.shape[0])
                    cv.circle(annotated_image, (x_px, y_px), 4, (0, 255, 0), -1)

                # Extract 63D keypoints
                keypoints = []
                for lm in hand_landmarks:
                    keypoints.extend([lm.x, lm.y, lm.z])

                if recording:
                    buffer.append(keypoints)
                    cv.putText(annotated_image, f"Recording: {len(buffer)}", (10, 60),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if len(buffer) == sequence_length:
                        print("Recording complete. Waiting for label...")
                        record_key_pressed = True
                        recording = False

        # Show FPS
        cv.putText(annotated_image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Instructions
        cv.putText(annotated_image, "Press '5' to record", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv.imshow('Gesture Recorder', annotated_image)

        key = cv.waitKey(1)

        # Start countdown
        if key == ord('5') and not recording and not counting_down and not record_key_pressed:
            print("Starting countdown...")
            counting_down = True
            countdown_start_time = time.time()

        # Labeling after recording
        if record_key_pressed:
            if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                class_idx = int(chr(key)) - 1
                class_name = gesture_classes[class_idx]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                save_path = os.path.join(dataset_dir, class_name, f"seq_{len(os.listdir(os.path.join(dataset_dir, class_name)))}.npy")
                np.save(save_path, np.array(buffer))

                # Save metadata
                with open(metadata_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, class_name, class_idx, save_path])

                print(f"Saved {save_path}")
                record_key_pressed = False

        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()