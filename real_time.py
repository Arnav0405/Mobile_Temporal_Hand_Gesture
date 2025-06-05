import cv2 as cv
import mediapipe as mp
import numpy as np
import torch
from collections import deque
import time
import os
from utils.cvfpscalc import CvFpsCalc 


# Import MediaPipe Tasks API
mp_tasks = mp.tasks
BaseOptions = mp_tasks.BaseOptions
HandLandmarker = mp_tasks.vision.HandLandmarker
HandLandmarkerOptions = mp_tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp_tasks.vision.RunningMode
HandLandmarkerResult = mp_tasks.vision.HandLandmarkerResult

# Load your trained model
from model.basic_tcn import TCNModel  # Use correct import

# Load model
device = torch.device("cpu")
our_model = TCNModel().to(device)

# Check if model file exists
our_model_path = 'best_tcn_optuna_model.pth'
if not os.path.exists(our_model_path):
    print(f"Error: Model file {our_model_path} not found!")
    exit(1)

our_model.load_state_dict(torch.load(our_model_path))
our_model.eval()

# Gesture classes
classes = ['Swipe Up', 'Swipe Down', 'Thumbs Up', 'Idle Gestures']


# Circular buffer to store sequence of keypoints
sequence_length = 30
keypoint_buffer = deque(maxlen=sequence_length)

# Smoothing predictions
prediction_history = deque(maxlen=5)

# FPS calculator
fps_calc = CvFpsCalc()

# Variable to hold the latest processed frame
latest_frame = None

# Create a flag to track when we have a new result
new_result = False
result_lock = False

def process_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_frame, new_result, result_lock
    
    if result_lock:
        return
    
    result_lock = True
    
    # Convert to OpenCV format
    annotated_image = output_image.numpy_view().copy()
    annotated_image = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)
    
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Extract 63D keypoints
            keypoints = []
            for lm in hand_landmarks:
                keypoints.extend([lm.x, lm.y, lm.z])
                
            # Add to buffer
            keypoint_buffer.append(keypoints)
            
            # Draw landmarks
            for lm in hand_landmarks:
                x_px = int(lm.x * annotated_image.shape[1])
                y_px = int(lm.y * annotated_image.shape[0])
                cv.circle(annotated_image, (x_px, y_px), 4, (0, 255, 0), -1)

            # If we have enough frames, run inference
            if len(keypoint_buffer) == sequence_length:
                input_seq = np.array(keypoint_buffer, dtype=np.float32)
                print(f"Input sequence shape: {input_seq.shape}")
                print(f"Input sequence unsequeeezed shape: {torch.from_numpy(input_seq).unsqueeze(0).shape}")
                input_tensor = torch.from_numpy(input_seq).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = our_model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    label = classes[predicted.item()]
                    
                    prediction_history.append(label)
                    
                    # Majority voting
                    if prediction_history:
                        most_common = max(set(prediction_history), key=prediction_history.count)
                        # Display prediction
                        cv.putText(annotated_image, f"Predicted: {most_common}", (10, 60),
                                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show FPS
    fps = fps_calc.get()
    cv.putText(annotated_image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    latest_frame = annotated_image
    new_result = True
    result_lock = False

# Check if the hand landmarker file exists
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print(f"Error: Hand landmarker model file {model_path} not found!")
    exit(1)

# Initialize MediaPipe Hand Landmarker
base_options = BaseOptions(model_asset_path=model_path)
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=process_result
)

# Start webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam!")
    exit(1)

print("Starting webcam... Press 'q' to quit.")

# Main loop
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting...")
            break

        frame = cv.flip(frame, 1)  # Mirror
        
        # Convert to MediaPipe format and send for processing
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Process with MediaPipe
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        
        # Display the frame we already have while waiting for new results
        if latest_frame is not None:
            cv.imshow('Real-Time Gesture Recognition', latest_frame)
            new_result = False
        else:
            # Show original frame until we get the first result
            cv.imshow('Real-Time Gesture Recognition', frame)
        
        # Small delay to allow for callback processing
        if cv.waitKey(5) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()