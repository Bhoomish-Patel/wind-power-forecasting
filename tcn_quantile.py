import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from preprocessing import load_data_1
from models import TCNQuantileReg
from loss_functions import quantile_loss
from torch.utils.data import TensorDataset, DataLoader
import time

SEQ_LEN = 5
EPOCHS = 20
LR = 0.001
SEED = 42
BATCH_SIZE = 32
ALPHA = 0.1

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

full_dataset, scaler, y, y_scaled = load_data_1("TransnetBW.csv", SEQ_LEN)

split_idx = int(0.8 * len(full_dataset.X))
X_train = full_dataset.X[:split_idx]
y_train = full_dataset.y[:split_idx]
X_test = full_dataset.X[split_idx:]
y_test = full_dataset.y[split_idx:]

start_time = time.time()
quantiles = [ALPHA/2.0, 1.0 - ALPHA/2.0]
models = []

for q in quantiles:
    model = TCNQuantileReg(input_dim=1).to(device)
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
title = f"TCN Quantile Regression\nPICP={PICP:.3f}, MPIW={MPIW:.3f}, Time={time_taken:.2f}s"
plt.title(title)
plt.xlabel("Time")
plt.ylabel("Value")
filename = "tcn_quantile.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
print(f"\nMetrics:")
print(f"PICP: {PICP:.3f}")
print(f"MPIW: {MPIW:.3f}")
print(f"Training time: {time_taken:.2f} seconds")
print(f"Plot saved as {filename}")
plt.show()

window_size = 20
local_picp = []
local_mpiw = []

for i in range(len(y_test_inv) - window_size + 1):
    window_inside = inside[i:i+window_size]
    window_width = y_u_preds_inv[i:i+window_size] - y_l_preds_inv[i:i+window_size]
    local_picp.append(window_inside.mean())
    local_mpiw.append(window_width.mean())

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(range(window_size-1, len(y_test_inv)), local_picp, label='Local PICP', color='blue')
plt.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
plt.legend()
plt.title('Local Coverage (PICP) with Window Size 20')
plt.xlabel('Time')
plt.ylabel('PICP')

plt.subplot(2, 1, 2)
plt.plot(range(window_size-1, len(y_test_inv)), local_mpiw, label='Local MPIW', color='green')
plt.axhline(y=MPIW, color='r', linestyle='--', label=f'Global MPIW: {MPIW:.3f}')
plt.legend()
plt.title('Local Prediction Interval Width (MPIW) with Window Size 20')
plt.xlabel('Time')
plt.ylabel('MPIW')

plt.tight_layout()
filename = "local_metrics_tcn_quantile.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
print(f"Local metrics plot saved as {filename}")
plt.show()