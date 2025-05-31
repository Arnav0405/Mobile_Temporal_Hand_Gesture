from skorch import NeuralNetClassifier
from basic_tcn import TCNModel
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_gesture_data(dataset_path='dataset', classes=None):
    """
    Load all gesture data into memory for use with scikit-learn.
    
    Args:
        dataset_path: Path to the dataset directory
        classes: List of class folder names to include
        
    Returns:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of labels
    """
    if classes is None:
        classes = ['swipe_up', 'swipe_down', 'thumbs_up', 'idle_gestures']
    
    X_data = []
    y_data = []
    
    # Loop through each class folder
    print("Loading dataset into memory...")
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f"Warning: {class_path} is not a directory")
            continue
            
        files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        
        for file in tqdm(files, desc=f"Loading {class_name}"):
            file_path = os.path.join(class_path, file)
            # Load the sequence data
            sequence = np.load(file_path).astype(np.float32)
            X_data.append(sequence)
            y_data.append(np.int64(class_idx))  
    
    # Convert to numpy arrays
    X = np.array(X_data)
    y = np.array(y_data)
    
    print(f"Loaded {len(X)} samples with shape {X[0].shape}")
    return X, y

if __name__ == "__main__":
    # Define hyperparameter grid
    param_grid = {
        'module__dropout': [0.05, 0.1, 0.15],
        'module__kernel_size': [5],
        'module__num_channels': [[32, 64]],
        'lr': [0.0003, 0.00035, 0.0004],
        'max_epochs': [50],
        'batch_size': [64]
    }
    
    # Load data
    classes = ['swipe_up', 'swipe_down', 'thumbs_up', 'idle_gestures']
    X, y = load_gesture_data(dataset_path='dataset', classes=classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Create the skorch model wrapper
    input_size = 63  # Based on your training.py
    output_size = len(classes)
    
    # Create NeuralNetClassifier with your TCN model
    net = NeuralNetClassifier(
        module=TCNModel,
        module__input_size=input_size,
        module__output_size=output_size,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=1
    )
    net.set_params(train_split=False, verbose=0)

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=net,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        refit=False,
        verbose=3,
        n_jobs=1  # Use 1 for CUDA compatibility
    )
    
    # Fit the grid search
    print("Starting grid search...")
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    
    # Evaluate on test set
    # test_score = grid_search.score(X_test, y_test)
    # print(f"Test accuracy with best model: {test_score:.4f}")
    
    # Save results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv('grid_search_results.csv', index=False)
    
    # Save best model
    torch.save(grid_search.best_estimator_.module_.state_dict(), 'best_tcn_grid_search.pth')