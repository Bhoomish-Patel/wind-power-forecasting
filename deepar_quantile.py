import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from preprocessing import load_data
from models import DeepARQuantileReg
from loss_functions import quantile_loss

SEQ_LEN = 30
EPOCHS = 100
LR = 0.001
quantiles = [0.05, 0.95]

train_loader, test_dataset, scaler, y, y_scaled, split_idx = load_data("TenneTTSO.csv", SEQ_LEN)

start_time = time.time()
preds_all = []

for q in quantiles:
    model = DeepARQuantileReg()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            preds = model(X_batch)
            loss = quantile_loss(y_batch, preds, q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"q={q}, Epoch {epoch + 1}, Loss = {loss.item():.4f}")

    test_series = y_scaled[split_idx - SEQ_LEN:]
    X_test = []
    for i in range(len(test_series) - SEQ_LEN):
        X_test.append(test_series[i:i + SEQ_LEN])
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)

    preds_scaled = model(X_test).detach().numpy()
    preds_all.append(preds_scaled)

training_time = time.time() - start_time
preds_all = np.concatenate(preds_all, axis=1)
preds = scaler.inverse_transform(preds_all)
y_true = scaler.inverse_transform(test_series[SEQ_LEN:].reshape(-1, 1)).flatten()

lower = preds[:, 0]
upper = preds[:, 1]
inside = (y_true >= lower) & (y_true <= upper)
PICP = inside.mean()
MPIW = np.mean(upper - lower)

print(f"PICP: {PICP:.3f}")
print(f"MPIW: {MPIW:.3f}")
print(f"Training Time: {training_time:.2f} seconds")

plt.figure(figsize=(12, 6))
plt.plot(range(len(y)), y, label="Actual", color="black")
plt.plot(range(split_idx, len(y)), lower, label="Lower Bound", color="red", linestyle="--")
plt.plot(range(split_idx, len(y)), upper, label="Upper Bound", color="blue", linestyle="--")
plt.fill_between(range(split_idx, len(y)), lower, upper, color="lightblue", alpha=0.4, label="Prediction Interval")
plt.axvline(split_idx, color="gray", linestyle="--", label="Train/Test split")
plt.legend()

title = f"DeepAR Quantile Regression\nPICP={PICP:.3f}, MPIW={MPIW:.3f}, Time={training_time:.2f}s"
plt.title(title)
plt.xlabel("Time steps")
plt.ylabel("Load")

filename = "deep_ar_quantile.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.show()
print(f"Plot saved as {filename}")
