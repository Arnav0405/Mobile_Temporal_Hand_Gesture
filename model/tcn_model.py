import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torchinfo import summary

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class SelfAttention(nn.Module):
    """Self attention layer for TCN output"""
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(x)    # (batch_size, seq_len, hidden_dim)
        V = self.value(x)  # (batch_size, seq_len, hidden_dim)
        
        # Calculate attention scores
        attention = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # (batch_size, seq_len, seq_len)
        attention = torch.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V)  # (batch_size, seq_len, hidden_dim)
        return out

    def pool(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        Q = self.query(x)
        K = self.key(x)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # (batch, seq, seq)
        attention_weights = torch.softmax(attention_scores.mean(dim=1), dim=-1)  # (batch, seq)
        # Weighted sum of values
        pooled = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
        return pooled
    
class TCNModel(nn.Module):
    def __init__(self, num_inputs=63, output_size= 4, num_channels= [32, 64], kernel_size=7, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)

        self.feature_attention = SelfAttention(num_channels[-1])
            
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        
        # Pass through TCN
        y = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        
        # Feature attention - focus on important timesteps
        y = self.feature_attention(y)
        # y_pooled = self.feature_attention.pool(y)  # Pooling to get a fixed-size output
        y_pooled = y.mean(dim=1)

        return self.fc(y_pooled)


if __name__ == "__main__":
    num_inputs = 63
    output_size = 4
    num_channels = [32,64]  
    kernel_size = 7
    dropout = 0.2

    model = TCNModel(num_inputs, output_size, num_channels, kernel_size, dropout)
    # Input shape: (batch_size, sequence_length, num_inputs)
    summary(model, input_size=(16, 30, 63))