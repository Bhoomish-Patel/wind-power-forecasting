import pandas as pd
import numpy as np
setting = ['aci', 'withoutaci']
models = ["ann", "lstm", "tcn"]
losses = ["quantile", "tube"]
def find_picp(lower,upper,predictions):
    inside = (predictions >= lower) & (predictions <= upper)
    coverage = np.mean(inside)
    return coverage
def find_mpiw(lower, upper, predictions):
    mpiw = np.mean(upper - lower)
    return mpiw
def find_local_picp(lower, upper,predictions):
    window = 20
    half = window // 2
    n = len(lower)
    local_picp = []
    for i in range(n):
        start = i-half+1
        end = i+half
        if(start<0):
            continue
        if(end>=n):
            continue
        cur = 0
        for i in range(start,end):
            cur += (lower[i]<=predictions[i] and predictions[i]<=upper[i])
        cur/=window
        local_picp.append(cur)
    return local_picp
def find_local_mpiw(lower, upper, predictions):
    window = 20
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
def find_lpice(local_picp, req_cov):
    local_picp = np.asarray(local_picp) 
    diff = req_cov - local_picp
    return np.mean(np.maximum(diff, 0))
def find_mpice(local_mpiw, req_cov):
    local_mpiw = np.asarray(local_mpiw)
    diff = req_cov - local_mpiw
    return np.mean(np.maximum(diff, 0))
for s in setting:
    targetfile = f"results/{s}/{s}_summary_results.csv"
    results_df = pd.DataFrame(columns=[
        "setting",
        "model",
        "loss",
        "time",
        "picp",
        "mpiw",
        "mlpicp",
        "mlmpiw",
        "lpice",
        "mpice"
    ])
    for model in models:
        for loss in losses:
            filename = f"results/{s}/{model}_{loss}_results.csv"
            df = pd.read_csv(filename)
            lower = np.asarray(df['y_l_preds_inv'])
            upper = np.asarray(df['y_u_preds_inv'])
            predictions = np.asarray(df['y_true_vals_inv'])
            time = df['time'][0]
            picp = find_picp(lower, upper,predictions)
            mpiw = find_mpiw(lower,upper,predictions)
            local_mpiw = find_local_mpiw(lower,upper,predictions)
            local_picp = find_local_picp(lower,upper,predictions)
            mean_local_picp = np.mean(local_picp)
            mean_local_mpiw = np.mean(local_mpiw)
            lpice = find_lpice(local_picp,0.9)
            mpice = find_mpice(local_mpiw,0.9)
            new_row = {
                "setting":s,
                "model":model,
                "loss":loss,
                "time":time,
                "picp":picp,
                "mpiw":mpiw,
                "mlpicp":mean_local_picp,
                "mlmpiw":mean_local_mpiw,
                "lpice":lpice,
                "mpice":mpice,
            }
            results_df.loc[len(results_df)] = new_row
    results_df.to_csv(targetfile,index=False)

            
            
