import numpy as np
import cv2 as cv
import os

# MediaPipe HAND_CONNECTIONS for drawing lines between keypoints
import mediapipe as mp
mp_hands = mp.solutions.hands

# Configuration
dataset_dir = 'dataset'  # Folder containing gesture class folders
sequence_length = 30     # Must match the length used during recording
frame_size = (640, 480)  # Frame size for visualization window
landmark_color = (0, 255, 0)
connection_color = (255, 0, 0)
circle_radius = 5
line_thickness = 2

def draw_landmarks(image, landmarks):
    """
    Draw landmarks and connections on the image.
    :param image: Image to draw on
    :param landmarks: List of (x, y, z) normalized coordinates
    """
    h, w, _ = image.shape
    coords = []

    # Convert normalized coordinates to pixel positions
    for i in range(0, len(landmarks), 3):
        x = int(landmarks[i] * w)
        y = int(landmarks[i + 1] * h)
        coords.append((x, y))

    # Draw connections
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        cv.line(image, coords[start_idx], coords[end_idx], connection_color, line_thickness)

    # Draw circles on each landmark
    for (x, y) in coords:
        cv.circle(image, (x, y), circle_radius, landmark_color, -1)

    return image


def visualize_sequences():
    print("Available gesture classes:")
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    for i, cls in enumerate(classes):
        print(f"{i}: {cls}")

    cls_index = int(input("Enter class index to visualize: "))
    selected_class = classes[cls_index]

    class_dir = os.path.join(dataset_dir, selected_class)
    files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
    files.sort()

    print(f"Found {len(files)} sequences for '{selected_class}'. Press arrow keys to navigate.")
    idx = 0


    
    while True:
        print(f"Playing: {files[idx]}")
        file_path = os.path.join(class_dir, files[idx])
        sequence = np.load(file_path)
        for frame_num, frame in enumerate(sequence):
            # Create blank frame
            frame_vis = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

            # Draw landmarks
            draw_landmarks(frame_vis, frame)

            # Display text
            cv.putText(frame_vis, f"Class: {selected_class}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(frame_vis, f"Frame: {frame_num + 1}/{sequence_length}", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Show frame
            cv.imshow("Gesture Sequence Viewer", frame_vis)
            cv.waitKey(100)  # Adjust speed here (ms per frame)

        # Wait for user input
        key = cv.waitKey(0)

        if key == ord('q'):
            break
        elif key == ord('a') or key == ord('d'):  # A & D Keys
            if key == ord('a'):  # Left
                print("Left arrow pressed")
                idx = max(0, idx - 1)
            elif key == ord('d'):  # Right
                print("Right arrow pressed")
                idx = min(len(files) - 1, idx + 1)

    cv.destroyAllWindows()


if __name__ == "__main__":
    visualize_sequences()