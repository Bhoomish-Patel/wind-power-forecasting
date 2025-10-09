import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_mixture_params(out, n_components):
    pi = F.softmax(out[:, :n_components], dim=1)
    mu = out[:, n_components:2*n_components]
    sigma = torch.exp(out[:, 2*n_components:]) 
    return pi, mu, sigma

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class LinearQuantileReg(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.fc = nn.Linear(seq_len, 1)
    def forward(self, x):
        return self.fc(x)

class LinearTubeReg(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.fc = nn.Linear(seq_len, 2)
    def forward(self, x):
        return self.fc(x)

class LSTMQuantileReg(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class LSTMTubeReg(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class ANNQuantileReg(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  
        )
    def forward(self, x):
        return self.net(x)


class ANNTubeReg(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        return self.net(x)

class MDNQuantileReg(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64, n_components=5):
        super().__init__()
        self.n_components = n_components
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_components * 3)
        )

    def forward(self, x):
        out = self.net(x)
        pi, mu, sigma = get_mixture_params(out, self.n_components)
        return pi, mu, sigma  
class MDNTubeReg(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64, n_components=5):
        super().__init__()
        self.n_components = n_components
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_components * 3)
        )

    def forward(self, x):
        out = self.net(x)
        pi, mu, sigma = get_mixture_params(out, self.n_components)
        return pi, mu, sigma 
class DeepARQuantileReg(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        x = x.unsqueeze(-1)  
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
class DeepARTubeReg(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2) 

    def forward(self, x):
        x = x.unsqueeze(-1) 
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TCN(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TCNQuantileReg(nn.Module):
    def __init__(self, input_dim=1, num_channels=[32, 32, 32, 32], kernel_size=2, dropout=0.2):
        super(TCNQuantileReg, self).__init__()
        self.tcn = TCN(input_dim, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        out = self.tcn(x)
        out = out[:, :, -1]  # Take last time step
        return self.fc(out)

class TCNTubeReg(nn.Module):
    def __init__(self, input_dim=1, num_channels=[32, 32, 32, 32], kernel_size=2, dropout=0.2):
        super(TCNTubeReg, self).__init__()
        self.tcn = TCN(input_dim, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], 2)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        out = self.tcn(x)
        out = out[:, :, -1]  # Take last time step
        return self.fc(out)
