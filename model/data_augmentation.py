import numpy as np
from scipy.interpolate import interp1d
import os
import csv
from datetime import datetime

def time_warp(seq, sigma=0.2):
    seq_len = seq.shape[0]
    warp_sigma = sigma * seq_len
    t = np.arange(seq_len)
    t_warped = t + np.random.normal(loc=0.0, scale=warp_sigma, size=seq_len) + 0.000001 * np.random.rand(seq_len)
    t_warped = np.clip(t_warped, -1, seq_len - 1).astype('float32')
    f = interp1d(t_warped, seq, axis=0, fill_value="extrapolate")
    return f(t).astype('float32')

def scale(seq, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(1, seq.shape[1]))
    return seq * factor.astype('float32')

def translate(seq, sigma=0.05):
    offset = np.random.normal(loc=0.0, scale=sigma, size=(1, seq.shape[1]))
    return seq + offset.astype('float32')

def add_noise(seq, sigma=0.01):
    noise = np.random.normal(loc=0.0, scale=sigma, size=seq.shape)
    return seq + noise.astype('float32')

def augment_sequence(seq):
    if np.random.rand() < 0.5:
        seq = time_warp(seq)
    if np.random.rand() < 0.5:
        seq = scale(seq)
    if np.random.rand() < 0.5:
        seq = translate(seq)
    if np.random.rand() < 0.5:
        seq = add_noise(seq)
    return seq

def save_augmented_sequence(original_path, augmented_data, suffix):
    base, ext = os.path.splitext(original_path)
    new_path = f"{base}_aug{suffix}{ext}"
    # Check if file already exists before saving
    if os.path.exists(new_path):
        # np.save(new_path, augmented_data)
        print(f"Saved augmented data to {new_path}")
        return True
    return False

def cleanup_double_aug_files():
    """Delete files with double augmentation patterns like '_augX_augY.npy'"""
    print("Cleaning up files with double augmentation patterns...")
    deleted_count = 0
    
    for cls in classes:
        class_dir = os.path.join(path, cls)
        for file in os.listdir(class_dir):
            # Check for double augmentation pattern using regex
            if file.endswith('.npy') and '_aug1' in file:
                file_path = os.path.join(class_dir, file)
                print(f"Deleting: {file_path}")
                os.remove(file_path)
                deleted_count += 1
    print(f"Cleanup completed. {deleted_count} files deleted.")

path = "dataset"
classes = ['swipe_up', 'swipe_down', 'thumbs_up', 'idle_gestures']
new_files_created = 0

# Run cleanup first to remove any doubled augmentation files
length_dict = {
    'swipe_up': 0,
    'swipe_down': 0,
    'thumbs_up': 0,
    'idle_gestures': 0
}

for cls in classes:
    class_dir = os.path.join(path, cls)
    for file in os.listdir(class_dir):
        if file.endswith('.npy'):
            length_dict[cls] += 1

            
print("Initial file counts:", length_dict)
cleanup_double_aug_files()
print(f"Data augmentation completed. {new_files_created} new files created.")
print(len(os.listdir(path + '/swipe_up')))  # Check the augmented files in the swipe_up class
print(len(os.listdir(path + '/swipe_down')))  # Check the augmented files in the swipe_down class
print(len(os.listdir(path + '/thumbs_up')))  # Check the augmented files in the thumbs_up class
print(len(os.listdir(path + '/idle_gestures')))  # Check the augmented files in the idle_gestures class


# augmented_seq = time_warp(original_seq)  # shape (30, 63)