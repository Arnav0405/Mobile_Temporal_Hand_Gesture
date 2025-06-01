import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_set import GestureDataset

# from tcn_model import TCNModel
from lstm_gru_models import LSTMModel, GRUModel
from basic_tcn import TCNModel

batch_size = 128
epochs = 50
lr = 0.0014584183844302903
weight_decay=0.000200795620999658
early_stop_patience = 20

# Dataset & Dataloader
classes = ['swipe_up', 'swipe_down', 'thumbs_up', 'idle_gestures']
dataset = GestureDataset('dataset', classes)
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
def train_model(model, model_name='model', num_epochs=50, save_path='best_tcn_gesture_model.pth', config=None):
    model.to(device)
    
# Create dataloaders with potentially tuned batch size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    no_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_error': [], 'val_error': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_grad_norms = []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        epoch_error = 1.0 - epoch_acc

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
        val_error = 1.0 - val_acc

        print(f"Epoch {epoch+1}: "
              f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Error: {epoch_error:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Error: {val_error:.4f}")

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(val_acc)
        history['train_error'].append(epoch_error)
        history['val_error'].append(val_error)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= early_stop_patience:
            print("Early stopping triggered.")
            break

    # Calculate mean errors
    mean_train_error = sum(history['train_error']) / len(history['train_error'])
    mean_val_error = sum(history['val_error']) / len(history['val_error'])
    print(f"Mean Train Error: {mean_train_error:.4f}")
    print(f"Mean Val Error: {mean_val_error:.4f}")
    
    # Plot training history
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title(f'{model_name} Loss')

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.legend()
    plt.title(f'{model_name} Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_error'], label=f'Train Error (Mean: {mean_train_error:.4f})')
    plt.plot(history['val_error'], label=f'Val Error (Mean: {mean_val_error:.4f})')
    plt.axhline(y=mean_train_error, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=mean_val_error, color='g', linestyle='--', alpha=0.3)
    plt.legend()
    plt.title(f'{model_name} Error')

    plt.tight_layout()
    plt.show()
    return history

if __name__ == "__main__":
    tcn_model = TCNModel()
    tcn_history = train_model(tcn_model, model_name="TCN", 
                             save_path='best_tcn_gesture_model.pth')
    
    # lstm_model = LSTMModel()
    # lstm_history = train_model(lstm_model, model_name='LSTM', 
    #                           save_path='best_lstm_gesture_model.pth',
    #                           config=tuned_params.get("LSTM", {}))
    
    # gru_model = GRUModel()
    # gru_history = train_model(gru_model, model_name='GRU', 
    #                          save_path='best_gru_gesture_model.pth',
    #                          config=tuned_params.get("GRU", {}))

    print("Training complete with tuned hyperparameters!")