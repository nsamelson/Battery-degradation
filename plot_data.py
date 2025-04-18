import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_signals(x, y_true, y_pred, params):
    colors = ['b', 'g', 'r', 'm']
    linestyles = ['-', '--']

    print(params)

    # duration = params["durations"][0]

    # for i, (fb, fs, duration, resp, exc) in enumerate(zip(fbs, fss, durations, response, excitation)):
    fig, ax = plt.subplots( figsize=(20, 10))
    time = np.linspace(0, len(x)/params['fbs'], len(x))
    
    # Plot response and excitation on the same subplot
    ax.plot(time, y_pred, color=colors[0], linestyle=linestyles[1], alpha=0.4, label=f'y_pred')
    ax.plot(time, y_true, color=colors[1], linestyle=linestyles[1], alpha=0.4, label=f'y_true')
    ax.plot(time, x, color=colors[2], linestyle=linestyles[0],  label=f'Excitation')
    ax.set_title(f'Simulation vs Ground truth')
    ax.set_xlim([0, time[-1]])
    ax.legend()
    ax.grid(True)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")

    fig.tight_layout()
    plt.savefig("plots/sim_v_truth.png", dpi=300)


def extract_experiments(exp_name):
    folder = os.path.join("ray_results", exp_name)
    trial_paths = [
        os.path.join(folder, trial)
        for trial in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, trial))
    ]

    records = []
    for trial_path in trial_paths:
        result_path = os.path.join(trial_path, "result.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)

            flat_record = {
                "bic": result["bic"],
                "mse": result["mse"]
            }
            # Flatten config
            for k, v in result.get("config", {}).items():
                flat_record[k] = v

            records.append(flat_record)

    df = pd.DataFrame(records)

    out_folder = os.path.join("output", exp_name)
    os.makedirs(out_folder, exist_ok=True)
    df.to_csv(os.path.join(out_folder, "results.csv"), index=False)

        
def plot_boxplots(exp_name):
    # Load data
    df_path = os.path.join("output", exp_name, "results.csv")
    df = pd.read_csv(df_path)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("BIC and MSE over N number of parameters", fontsize=16)

    # BIC plot
    sns.boxplot(data=df, x="N", y="bic", ax=axs[0])
    axs[0].set_title("BIC over N")
    axs[0].set_xlabel("N")
    axs[0].set_ylabel("BIC")
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')
    min_bic_row = df.loc[df["bic"].idxmin()]
    axs[0].plot(min_bic_row["N"], min_bic_row["bic"], marker='x', color='black', markersize=8, label='Min BIC')
    axs[0].legend()

    # MSE plot
    sns.boxplot(data=df, x="N", y="mse", ax=axs[1])
    axs[1].set_yscale("log")
    axs[1].set_title("MSE over N")
    axs[1].set_xlabel("N")
    axs[1].set_ylabel("MSE")
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')
    min_mse_row = df.loc[df["mse"].idxmin()]
    axs[1].plot(min_mse_row["N"], min_mse_row["mse"], marker='x', color='black', markersize=8, label='Min MSE')
    axs[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join("output", exp_name, "bic_and_mse.png")
    plt.savefig(out_path)
    plt.close()





if __name__ == "__main__":
    # data = np.load("output/signals.npz",allow_pickle=True)

    # samples = 100

    # x = data["x"][0:samples]
    # y_true = data["y_true"][0][0:samples]
    # y_pred = data["y_pred"][0][0:samples]
    # params = data["params"].item()

    # # print(x.shape,y_true.shape,y_pred.shape)

    # plot_signals(x, y_true, y_pred, params)

    # extract_experiments("bic_30000_hz")
    plot_boxplots("bic_30000_hz")


