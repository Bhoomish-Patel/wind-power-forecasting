import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class SeqDataset(Dataset):
    def __init__(self, series, seq_len):
        self.X, self.y = [], []
        for i in range(len(series) - seq_len):
            self.X.append(series[i:i+seq_len])
            self.y.append(series[i+seq_len])
        self.X, self.y = np.array(self.X), np.array(self.y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

def load_data(filepath, seq_len=24, train_split=0.8):
    df = pd.read_csv(filepath)
    y = df.iloc[:, 1:].sum(axis=1).values.astype(float)

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    split_idx = int(len(y_scaled) * train_split)
    train_series = y_scaled[:split_idx]
    test_series = y_scaled[split_idx - seq_len:]

    train_dataset = SeqDataset(train_series, seq_len)
    test_dataset = SeqDataset(test_series, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader, test_dataset, scaler, y, y_scaled, split_idx
def load_data_1(filepath, seq_len=24):
    df = pd.read_csv(filepath)
    y = df.iloc[:, 1:].sum(axis=1).values.astype(float)
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    full_dataset = SeqDataset(y_scaled, seq_len)
    return full_dataset, scaler, y, y_scaled

