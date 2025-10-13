dataset = "TransnetBW.csv"
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from preprocessing import load_data_1
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import pandas as pd
def pad(arr, length):
    arr = np.array(arr, dtype=np.float32)
    if len(arr) < length:
        arr = np.concatenate([arr, np.full(length - len(arr), np.nan)])
    return arr
from models import GRUQuantileReg
from loss_functions import quantile_loss

SEQ_LEN = 5
EPOCHS = 200
LR = 0.001
SEED = 42
BATCH_SIZE = 32
ALPHA = 0.1

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

full_dataset, scaler, y, y_scaled = load_data_1(dataset, SEQ_LEN)

split_idx = int(0.8 * len(full_dataset.X))
X_train = full_dataset.X[:split_idx]
y_train = full_dataset.y[:split_idx]
X_test = full_dataset.X[split_idx:]
y_test = full_dataset.y[split_idx:]

start_time = time.time()
quantiles = [ALPHA/2.0, 1.0 - ALPHA/2.0]
models = []

for q in quantiles:
    model = GRUQuantileReg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"\nTraining quantile {q}")
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        
        for X_batch, y_batch in train_loader:
            preds = model(X_batch)
            loss = quantile_loss(y_batch, preds, q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {np.mean(epoch_losses):.4f}")
    
    models.append(model)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
models[0].eval()
models[1].eval()

with torch.no_grad():
    y_l_preds = models[0](X_test_tensor).cpu().numpy().flatten()
    y_u_preds = models[1](X_test_tensor).cpu().numpy().flatten()

end_time = time.time()
time_taken = end_time - start_time

y_u_preds_inv = scaler.inverse_transform(y_u_preds.reshape(-1,1)).flatten()
y_l_preds_inv = scaler.inverse_transform(y_l_preds.reshape(-1,1)).flatten()
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

inside = (y_test_inv >= y_l_preds_inv) & (y_test_inv <= y_u_preds_inv)
PICP = inside.mean()
MPIW = np.mean(y_u_preds_inv - y_l_preds_inv)

plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label="True", color='black')
plt.plot(y_u_preds_inv, label="Upper Bound", linestyle="--", color='blue')
plt.plot(y_l_preds_inv, label="Lower Bound", linestyle="--", color='red')
plt.fill_between(range(len(y_test_inv)), y_l_preds_inv, y_u_preds_inv, alpha=0.2, color='lightblue')
plt.legend()
title = f"GRU Quantile Regression\nPICP={PICP:.3f}, MPIW={MPIW:.3f}, Time={time_taken:.2f}s"
plt.title(title)
plt.xlabel("Time")
plt.ylabel("Value")
filename = "results/withoutaci/gru_quantile.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
print(f"\nMetrics:")
print(f"PICP: {PICP:.3f}")
print(f"MPIW: {MPIW:.3f}")
print(f"Training time: {time_taken:.2f} seconds")
print(f"Plot saved as {filename}")
plt.show()


max_len = max(
    len(y_u_preds_inv),
    len(y_l_preds_inv),
)
results_df = pd.DataFrame({
    "y_u_preds_inv":pad(y_u_preds_inv,max_len),
    "y_l_preds_inv":pad(y_l_preds_inv,max_len),
    "y_true_vals_inv":pad(y_test_inv,max_len),
    "time": [time_taken] + [np.nan] * (max_len - 1),
})
csv_filename = "results/withoutaci/gru_quantile_results.csv"
results_df.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")