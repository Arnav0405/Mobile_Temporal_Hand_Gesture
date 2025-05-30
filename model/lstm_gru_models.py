import torch
import torch.nn as nn
from tcn_model import SelfAttention
class LSTMModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=4, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)  # out: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]    # Use last time step's output
        out = self.fc(out)     # (batch_size, num_classes)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=64, num_layers=2, num_classes=4, dropout=0.2, bidirectional= True):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional= bidirectional)
        # self.attention = SelfAttention(hidden_size * 2 if bidirectional else hidden_size)
        self.norm = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)
        self.global_pool = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        out, _ = self.gru(x)
        out = self.global_pool(out.transpose(1, 2)).squeeze(-1)  # Global average pooling
        out = self.fc(out)
        return out