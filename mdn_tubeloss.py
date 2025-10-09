import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_data
from models import MDNTubeReg
from loss_functions import tube_loss
from scipy.stats import norm

def gmm_quantiles(pi, mu, sigma, quantiles):
    results = []
    for q in quantiles:
        q_value = torch.sum(pi * (mu + sigma * torch.tensor(norm.ppf(q))), dim=1)
        results.append(q_value.unsqueeze(1))
    return torch.cat(results, dim=1) 

SEQ_LEN = 24
EPOCHS = 200
LR = 0.001

train_loader, test_dataset, scaler, y, y_scaled, split_idx = load_data("TenneTTSO.csv", SEQ_LEN)

model = MDNTubeReg(input_dim=SEQ_LEN, n_components=10)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    for X_batch, y_batch in train_loader:
        pi, mu, sigma = model(X_batch)
        quantile_preds = gmm_quantiles(pi, mu, sigma, quantiles=[0.025, 0.975])
        loss = tube_loss(y_batch, quantile_preds, q=0.95, r=0.5, delta=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss = {loss.item():.4f}")

training_time = time.time() - start_time

model.eval()
test_series = y_scaled[split_idx - SEQ_LEN:]
X_test = [test_series[i:i + SEQ_LEN] for i in range(len(test_series) - SEQ_LEN)]
X_test = torch.tensor(np.array(X_test), dtype=torch.float32)

with torch.no_grad():
    pi, mu, sigma = model(X_test)
    preds_scaled = gmm_quantiles(pi, mu, sigma, quantiles=[0.025, 0.975]).numpy()

preds = scaler.inverse_transform(preds_scaled)
upper, lower = preds[:, 1], preds[:, 0]
y_true = scaler.inverse_transform(test_series[SEQ_LEN:].reshape(-1, 1)).flatten()

inside = (y_true >= lower) & (y_true <= upper)
PICP = inside.mean()
MPIW = np.mean(upper - lower)

print(f"PICP: {PICP:.3f}, MPIW: {MPIW:.3f}, Training Time: {training_time:.2f} sec")

plt.figure(figsize=(12, 6))
plt.plot(range(len(y)), y, label="Actual", color="black")
plt.plot(range(split_idx, len(y)), lower, label="Lower Bound", color="red", linestyle="--")
plt.plot(range(split_idx, len(y)), upper, label="Upper Bound", color="blue", linestyle="--")
plt.fill_between(range(split_idx, len(y)), lower, upper, color="lightblue", alpha=0.4, label="Prediction Interval")
plt.axvline(split_idx, color="gray", linestyle="--", label="Train/Test split")
plt.legend()

title = f"MDN Tube Loss Regression\nPICP={PICP:.3f}, MPIW={MPIW:.3f}, Time={training_time:.2f}s"
plt.title(title)
plt.xlabel("Time steps")
plt.ylabel("Load")
plt.savefig("mdn_tube.png", dpi=300, bbox_inches="tight")
plt.show()

print("Plot saved as mdn_tube.png")
