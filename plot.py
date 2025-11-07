import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

setting = ['withoutaci', 'aci']
models = ["gru", "lstm", "tcn"]
losses = ["quantile", "tube"]

def find_picp(lower, upper, predictions):
    inside = (predictions >= lower) & (predictions <= upper)
    coverage = np.mean(inside)
    return coverage

def find_mpiw(lower, upper, predictions):
    mpiw = np.mean(upper - lower)
    return mpiw

def find_local_picp(lower, upper, predictions, window=10):
    half = window // 2
    n = len(lower)
    local_picp = []
    for i in range(n):
        start = i - half + 1
        end = i + half
        if start < 0 or end >= n:
            continue
        cur = np.mean((predictions[start:end] >= lower[start:end]) & (predictions[start:end] <= upper[start:end]))
        local_picp.append(cur)
    return local_picp

def find_local_mpiw(lower, upper, predictions, window=10):
    half = window // 2
    n = len(lower)
    local_mpiw = []
    for i in range(n):
        start = i - half + 1
        end = i + half
        if start < 0 or end >= n:
            continue
        cur = np.mean(upper[start:end] - lower[start:end])
        local_mpiw.append(cur)
    return local_mpiw


for s in setting:
    for model in models:
        for loss in losses:
            filename = f"results/{s}/{model}_{loss}_results.csv"
            df = pd.read_csv(filename)
            lower = np.asarray(df['y_l_preds_inv'])
            upper = np.asarray(df['y_u_preds_inv'])
            predictions = np.asarray(df['y_true_vals_inv'])
            time = df['time'][0]

            picp = find_picp(lower, upper, predictions)
            mpiw = find_mpiw(lower, upper, predictions)
            local_picp = find_local_picp(lower, upper, predictions)
            local_mpiw = find_local_mpiw(lower, upper, predictions)

            # Create plots directory if not exist
            save_dir = f"results/{s}"
            os.makedirs(save_dir, exist_ok=True)

            # Plot and save local PICP
            plt.figure(figsize=(8, 4))
            plt.plot(local_picp, label=f'Local PICP ({s}-{model}-{loss})')
            plt.title(f'Local PICP - {s.upper()} | {model.upper()} | {loss.upper()}')
            plt.xlabel('Window Index')
            plt.ylabel('Local PICP')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{model}_{loss}_local_picp.png")
            plt.close()

            # Plot and save local MPIW
            plt.figure(figsize=(8, 4))
            plt.plot(local_mpiw, label=f'Local MPIW ({s}-{model}-{loss})', color='orange')
            plt.title(f'Local MPIW - {s.upper()} | {model.upper()} | {loss.upper()}')
            plt.xlabel('Window Index')
            plt.ylabel('Local MPIW')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{model}_{loss}_local_mpiw.png")
            plt.close()

