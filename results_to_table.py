import pandas as pd
import os
filetype = ['aci','withoutaci']
models = ["ann", "lstm", "tcn"]
losses = ["quantile", "tube"]
for f in filetype:
    summary_path = f"results/{f}/{f}_summary_results.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    summary_df = pd.DataFrame(columns=[
        "model", "loss", "picp", "mpiw","standard_dev",
        "time",
    ])
    for model in models:
        for loss_function in losses:
            cur_filename = f"results/{f}/{model}_{loss_function}_results.csv"
            if not os.path.exists(cur_filename):
                print(f"File not found: {cur_filename}")
                continue
            df = pd.read_csv(cur_filename)
            first_row = df.iloc[0]
            PICP = first_row.get("picp", 0.0)
            MPIW = first_row.get("mpiw", 0.0)
            std_dev = first_row.get("standard_dev", 0.0) 
            time_taken = first_row.get("time", 0.0)
            entry = pd.DataFrame([{
                "model": model,
                "loss": loss_function,
                "picp": PICP,
                "mpiw": MPIW,
                "standard_dev": std_dev,
                "time": time_taken,
            }])
            summary_df = pd.concat([summary_df, entry], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"{f} summary table saved to {summary_path}")
