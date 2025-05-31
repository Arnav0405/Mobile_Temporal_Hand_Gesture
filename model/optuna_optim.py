import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import optuna
import numpy as np
import json
import os
from tqdm import tqdm

from data_set import GestureDataset
from basic_tcn import TCNModel

def load_data(batch_size):
    # Dataset preparation
    classes = ['swipe_up', 'swipe_down', 'thumbs_up', 'idle_gestures']
    dataset = GestureDataset('dataset', classes)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    return train_loader, val_loader

def create_model(trial, model_type="TCN"):
    """Create a model with parameters suggested by Optuna"""
    if model_type == "TCN":
        # Suggest hyperparameters for TCN model
        dropout = trial.suggest_float("tcn_dropout", 0.1, 0.5)
        leaky_slope = trial.suggest_float("tcn_leaky_slope", 0.01, 0.2)
        kernel_size = trial.suggest_categorical("tcn_kernel_size", [3, 5])
        
        # For num_channels, suggest different architectures
        num_channels_option = trial.suggest_categorical("tcn_channels_option", [0, 1])
        if num_channels_option == 0:
            num_channels = [32, 32]
        else:
            num_channels = [32, 64]
        
        model = TCNModel(
            dropout=dropout,
            leaky_slope=leaky_slope,
            kernel_size=kernel_size,
            num_channels=num_channels
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def create_optimizer(trial, model):
    """Create an optimizer with parameters suggested by Optuna"""
    # Suggest optimizer type
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop"])
    
    # Suggest learning rate & weight decay
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    
    # Create optimizer
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        momentum = trial.suggest_float("momentum", 0.0, 0.9)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    return optimizer

def objective(trial, model_type="TCN", epochs=10):
    """Optuna objective function for hyperparameter optimization"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Suggest batch size
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Load data
    train_loader, val_loader = load_data(batch_size)
    
    # Create model & optimizer
    model = create_model(trial, model_type)
    model.to(device)
    optimizer = create_optimizer(trial, model)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_acc = correct_val / total_val
        
        # Report value to Optuna
        trial.report(val_acc, epoch)
        
        # Update best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_acc

def tune_hyperparameters(model_type="TCN", n_trials=50, epochs=10):
    """Run Optuna hyperparameter tuning"""
    print(f"Starting hyperparameter tuning for {model_type} model...")
    
    # Create study object
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=f"{model_type}_optimization"
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, model_type, epochs),
        n_trials=n_trials
    )
    
    # Print results
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial
    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")
    
    return trial.params

def train_best_model(params, model_type="TCN", epochs=50):
    """Train a model with the best parameters found by Optuna"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create data loaders
    batch_size = params["batch_size"]
    train_loader, val_loader = load_data(batch_size)
    
    # Create model with best parameters
    if model_type == "TCN":
        # Extract parameters
        dropout = params["tcn_dropout"]
        leaky_slope = params["tcn_leaky_slope"]
        kernel_size = params["tcn_kernel_size"]
        
        # Set channels based on option
        num_channels_option = params["tcn_channels_option"]
        if num_channels_option == 0:
            num_channels = [32, 32]
        elif num_channels_option == 1:
            num_channels = [32, 64]
        else:
            num_channels = [64, 64]
        
        model = TCNModel(
            dropout=dropout,
            leaky_slope=leaky_slope,
            kernel_size=kernel_size,
            num_channels=num_channels
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.to(device)
    
    # Create optimizer with best parameters
    optimizer_name = params["optimizer"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        momentum = params.get("momentum", 0.0)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # Train model
    criterion = nn.CrossEntropyLoss()
    best_model_state = None
    best_val_acc = 0
    
    progress_bar = tqdm(range(epochs), desc=f"Training {model_type} with best params")
    for epoch in progress_bar:
        # Training
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        # Calculate metrics
        train_loss /= len(train_loader.dataset)
        train_acc = correct_train / total_train
        val_loss /= len(val_loader.dataset)
        val_acc = correct_val / total_val
        
        # Update progress bar
        progress_bar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.4f}'
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Save best model
    save_path = f'best_{model_type.lower()}_optuna_model.pth'
    torch.save(best_model_state, save_path)
    print(f"Best model saved to {save_path} with validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import optuna
    except ImportError:
        print("Installing optuna...")
        import subprocess
        subprocess.check_call(["pip", "install", "optuna"])
        import optuna
    
    # Run hyperparameter tuning for TCN
    best_params = tune_hyperparameters(model_type="TCN", n_trials=20, epochs=10)
    
    # Save best hyperparameters
    with open("best_hyperparameters_optuna.json", "w") as f:
        json.dump({"TCN": best_params}, f, indent=4)
    
    # Train model with best parameters for more epochs
    train_best_model(best_params, model_type="TCN", epochs=50)
    
    print("Hyperparameter tuning and final training complete!")