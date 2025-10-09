import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from preprocessing import load_data_1
from models import LSTMTubeReg
from loss_functions import tube_loss
from torch.utils.data import TensorDataset, DataLoader

SEQ_LEN = 5
EPOCHS = 50
LR = 0.001
SEED = 42
BATCH_SIZE = 32
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

full_dataset, scaler, y, y_scaled = load_data_1("TransnetBW.csv", SEQ_LEN)

def run_adaptive_conformal_lstm_tube(alpha=0.05, gamma=0.01, tinit=50, splitSize=0.5):
    T = len(full_dataset)
    alphaTrajectory = np.zeros(T - tinit, dtype=np.float32)
    adaptErrSeq = np.zeros(T - tinit, dtype=np.float32)
    noAdaptErrorSeq = np.zeros(T - tinit, dtype=np.float32)
    y_u_preds = np.zeros(T - tinit, dtype=np.float32)
    y_l_preds = np.zeros(T - tinit, dtype=np.float32)
    y_true_vals = np.zeros(T - tinit, dtype=np.float32)
    alphat = alpha

    for t in range(tinit, T):
        idx = list(range(t))
        train_size = int(max(1, t * splitSize))
        if train_size >= len(idx):
            trainPoints = idx.copy()
        else:
            trainPoints = random.sample(idx, train_size)
        calPoints = [i for i in idx if i not in trainPoints]

        X_train = full_dataset.X[trainPoints].astype(np.float32)
        y_train = full_dataset.y[trainPoints].astype(np.float32)
        X_cal = full_dataset.X[calPoints].astype(np.float32) if len(calPoints) > 0 else np.empty((0, SEQ_LEN), dtype=np.float32)
        y_cal = full_dataset.y[calPoints].astype(np.float32) if len(calPoints) > 0 else np.empty((0,), dtype=np.float32)
        y_t = float(full_dataset.y[t])

        model = LSTMTubeReg().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        X_train_tensor = torch.tensor(X_train.reshape(-1, SEQ_LEN), dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE, len(train_ds)), shuffle=True)

        model.train()
        for epoch in range(EPOCHS):
            for X_batch, y_batch in train_loader:
                preds = model(X_batch)
                loss = tube_loss(y_batch, preds, q=1-alphat, r=0.5, delta=0.0006)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()

        if len(calPoints) > 0:
            X_cal_tensor = torch.tensor(X_cal.reshape(-1, SEQ_LEN), dtype=torch.float32, device=device)
            with torch.no_grad():
                preds_cal = model(X_cal_tensor).cpu().numpy()
            y_u_cal, y_l_cal = preds_cal[:, 0], preds_cal[:, 1]
            scores = np.maximum(y_l_cal - y_cal, y_cal - y_u_cal)
        else:
            scores = np.array([0.0], dtype=np.float32)

        X_t_tensor = torch.tensor(full_dataset.X[t].reshape(1, SEQ_LEN), dtype=torch.float32, device=device)
        with torch.no_grad():
            preds_t = model(X_t_tensor).cpu().numpy()[0]
            y_u_t, y_l_t = preds_t[0], preds_t[1]

        newScore = max(y_l_t - y_t, y_t - y_u_t)
        confQuantNaive = float(np.quantile(scores, 1.0 - alphat)) if scores.size > 0 else 0.0
        noAdaptError = float(newScore > confQuantNaive)
        if alphat >= 1.0:
            adaptErr = 1.0
        elif alphat <= 0.0:
            adaptErr = 0.0
        else:
            adaptErr = float(newScore > confQuantNaive)
        alphat = float(alphat + gamma * (alpha - adaptErr))
        alphat = max(0.0, min(1.0, alphat))

        alphaTrajectory[t - tinit] = alphat
        adaptErrSeq[t - tinit] = adaptErr
        noAdaptErrorSeq[t - tinit] = noAdaptError
        y_u_preds[t - tinit] = y_u_t
        y_l_preds[t - tinit] = y_l_t
        y_true_vals[t - tinit] = y_t

        if t % 10 == 0:
            print(f"t={t}, alphat={alphat:.4f}, adaptErr={adaptErr}, noAdaptErr={noAdaptError}")

    return alphaTrajectory, adaptErrSeq, noAdaptErrorSeq, y_u_preds, y_l_preds, y_true_vals

alphaTrajectory, adaptErrSeq, noAdaptErrorSeq, y_u_preds, y_l_preds, y_true_vals = run_adaptive_conformal_lstm_tube()

y_u_preds_inv = scaler.inverse_transform(y_u_preds.reshape(-1,1)).flatten()
y_l_preds_inv = scaler.inverse_transform(y_l_preds.reshape(-1,1)).flatten()
y_true_vals_inv = scaler.inverse_transform(y_true_vals.reshape(-1,1)).flatten()

inside = (y_true_vals_inv >= y_l_preds_inv) & (y_true_vals_inv <= y_u_preds_inv)
PICP = inside.mean()
MPIW = np.mean(y_u_preds_inv - y_l_preds_inv)

plt.figure(figsize=(12,6))
plt.plot(y_true_vals_inv, label="True")
plt.plot(y_u_preds_inv, label="Upper Bound", linestyle="--")
plt.plot(y_l_preds_inv, label="Lower Bound", linestyle="--")
plt.fill_between(range(len(y_true_vals_inv)), y_l_preds_inv, y_u_preds_inv, alpha=0.2)
plt.legend()
title = f"Adaptive Conformal LSTM Tube\nPICP={PICP:.3f}, MPIW={MPIW:.3f}"
plt.title(title)
plt.xlabel("Time")
plt.ylabel("Value")
filename = "aci_lstm_tube.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
print(f"Plot saved as {filename}")
plt.show()