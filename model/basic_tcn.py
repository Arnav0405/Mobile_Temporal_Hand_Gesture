import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, 
                 dropout=0.2, leaky_slope=0.01):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.batchnorm1 = nn.BatchNorm1d(n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.leakyRelu1 = nn.LeakyReLU(leaky_slope)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.batchnorm2 = nn.BatchNorm1d(n_outputs)
        self.chomp2 = Chomp1d(padding)
        self.leakyRelu2 = nn.LeakyReLU(leaky_slope)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
                    self.conv1, self.chomp1, self.batchnorm1, self.leakyRelu1, self.dropout1,
                    self.conv2, self.chomp2, self.batchnorm2, self.leakyRelu2, self.dropout2
                    )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')
        

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, leaky_slope):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation, padding=padding, 
                                        dropout=dropout, leaky_slope=leaky_slope))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, input_size=63, output_size=4, num_channels=[32, 64], 
                 kernel_size=5, dropout=0.4780003395861791, leaky_slope=0.1284525066146185):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, 
                                  dropout=dropout, leaky_slope=leaky_slope)
        self.linear = nn.Linear(num_channels[-1], output_size)
        
        # Store configuration for reference
        self.config = {
            'num_channels': num_channels,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'leaky_slope': leaky_slope
        }

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # (B, C, T)
        y = self.tcn(x)
        o = self.linear(y[:, :, -1])  # Use last timestep
        return o
    
    def get_config(self):
        """Return the model's configuration parameters"""
        return self.config