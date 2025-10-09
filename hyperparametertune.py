import torch
import torch.nn as nn
import numpy as np
import random
import optuna

from preprocessing import load_data
from models import LSTMTubeReg
from loss_functions import tube_loss

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100
LR = 0.001

def objective(trial):
    seq_len = trial.suggest_int("seq_len", 5, 60, step=1)
    delta = trial.suggest_float("delta", 0.0, 1.0)

    train_loader, test_dataset, scaler, y, y_scaled, split_idx = load_data("TransnetBW.csv", seq_len)

    model = LSTMTubeReg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = tube_loss(y_batch, preds, q=0.95, r=0.5, delta=delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_series = y_scaled[split_idx - seq_len:]
    X_test = []
    for i in range(len(test_series) - seq_len):
        X_test.append(test_series[i:i+seq_len])
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)

    preds_scaled = model(X_test).detach().cpu().numpy()
    preds = scaler.inverse_transform(preds_scaled)
    y_true = scaler.inverse_transform(test_series[seq_len:].reshape(-1, 1)).flatten()

    upper = preds[:, 0]
    lower = preds[:, 1]
    inside = (y_true >= lower) & (y_true <= upper)

    PICP = inside.mean()
    MPIW = np.mean(upper - lower)

    trial.set_user_attr("PICP", PICP)
    trial.set_user_attr("MPIW", MPIW)

    if PICP < 0.95:
        return MPIW + (0.95 - PICP) * 1e9
    else:
        return MPIW

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("\nBest trial:")
best_trial = study.best_trial
print(f"  Seq_len: {best_trial.params['seq_len']}")
print(f"  Delta: {best_trial.params['delta']:.4f}")
print(f"  MPIW: {best_trial.value:.4f}")
print(f"  PICP: {best_trial.user_attrs['PICP']:.4f}")
