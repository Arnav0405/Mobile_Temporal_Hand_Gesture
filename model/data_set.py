import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

class GestureDataset(Dataset):
    def __init__(self, root_dir, classes, ignore_idle=True):
        self.root_dir = root_dir
        self.ignore_idle = ignore_idle
        self.classes = classes
        self.file_paths = []
        self.labels = []

        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            print(os.listdir(cls_dir))  # Debugging line to check directory contents
            for file in os.listdir(cls_dir):
                self.file_paths.append(os.path.join(cls_dir, file))
                self.labels.append(label)

        if not ignore_idle:
            idle_dir = os.path.join(root_dir, 'idle_gestures')
            for file in os.listdir(idle_dir):
                self.file_paths.append(os.path.join(idle_dir, file))
                self.labels.append(-1)  # Ignore label

        self.label_encoder = LabelEncoder()
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        if label == -1:
            return self.__getitem__((idx + 1) % len(self))  # Skip idle

        data = np.load(path).astype(np.float32)
        data = torch.from_numpy(data)
        label = torch.tensor(label, dtype=torch.long)
        return data, label

if __name__ == "__main__":
    # Example usage
    dataset = GestureDataset('dataset', ['swipe_up', 'swipe_down', 'thumbs_up', 'idle_gestures'])
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    for inputs, labels in train_loader:
        print(inputs.shape, labels.shape)
        print(inputs[0], labels[0])
        break  # Just to show one batch