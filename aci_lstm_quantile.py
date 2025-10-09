import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from preprocessing import load_data_1
from models import LSTMQuantileReg
from loss_functions import quantile_loss
from torch.utils.data import TensorDataset, DataLoader

SEQ_LEN = 5
EPOCHS = 50
LR = 0.001
SEED = 42
BATCH_SIZE = 32
WINDOW_SIZE = 20 

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

full_dataset, scaler, y, y_scaled = load_data_1("TransnetBW.csv", SEQ_LEN)

def run_lstm_quantile(alpha=0.1, train_split=0.5):
    T = len(full_dataset)
    train_size = int(T * train_split)
    train_idx = list(range(train_size))
    test_idx = list(range(train_size, T))
    
    X_train = full_dataset.X[train_idx].astype(np.float32)
    y_train = full_dataset.y[train_idx].astype(np.float32)
    X_test = full_dataset.X[test_idx].astype(np.float32)
    y_test = full_dataset.y[test_idx].astype(np.float32)
    
    quantiles = [alpha/2.0, 1.0 - alpha/2.0]
    y_u_preds = np.zeros(len(test_idx), dtype=np.float32)
    y_l_preds = np.zeros(len(test_idx), dtype=np.float32)
    for q in quantiles:
        model = LSTMQuantileReg().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        X_train_tensor = torch.tensor(X_train.reshape(-1, SEQ_LEN), dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        model.train()
        for epoch in range(EPOCHS):
            for X_batch, y_batch in train_loader:
                preds = model(X_batch)
                loss = quantile_loss(y_batch, preds, q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        model.eval()
        X_test_tensor = torch.tensor(X_test.reshape(-1, SEQ_LEN), dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = model(X_test_tensor).cpu().numpy().reshape(-1)
            if q == quantiles[0]:
                y_l_preds = preds
            else:
                y_u_preds = preds
    
    return y_u_preds, y_l_preds, y_test


y_u_preds, y_l_preds, y_true = run_lstm_quantile()


y_u_preds_inv = scaler.inverse_transform(y_u_preds.reshape(-1,1)).flatten()
y_l_preds_inv = scaler.inverse_transform(y_l_preds.reshape(-1,1)).flatten()
y_true_inv = scaler.inverse_transform(y_true.reshape(-1,1)).flatten()


inside = (y_true_inv >= y_l_preds_inv) & (y_true_inv <= y_u_preds_inv)
PICP = inside.mean()
MPIW = np.mean(y_u_preds_inv - y_l_preds_inv)
