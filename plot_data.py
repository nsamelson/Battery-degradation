import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator


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


def plot_param_to_perf(exp_name, parameter, freq):
    df_path = os.path.join("output", exp_name, "results.csv")
    df = pd.read_csv(df_path)

    if parameter not in df.columns:
        raise ValueError(f"Parameter '{parameter}' not found in results.")

    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"BIC and MSE over parameter {parameter} at {freq} Hz", fontsize=16)

    # Primary axis - BIC
    sns.scatterplot(data=df, x=parameter, y="bic", ax=ax1, color="royalblue", s=30, alpha=0.6)
    ax1.set_ylabel("BIC", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(True, which="major", linestyle='--', linewidth=0.5, color='lightgray')

    # Secondary axis - MSE (log scale)
    ax2 = ax1.twinx()
    sns.scatterplot(data=df, x=parameter, y="mse", ax=ax2, color="royalblue", marker="o", s=30, alpha=0.6)
    ax2.set_ylabel("MSE (log scale)", color="black")
    ax2.set_yscale("log")
    ax2.tick_params(axis='y', labelcolor="black")

    # Enable grid for log scale (right axis)
    ax2.grid(True, which="both", axis="y", linestyle=":", linewidth=0.5, color="lightgray")
    ax2.grid(False, axis='x')  # Prevent double x-axis grid

    # Improve x-axis readability
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    plt.xticks(rotation=45)

    plt.tight_layout()
    out_path = os.path.join("output", exp_name, f"{parameter}_to_perf.png")
    plt.savefig(out_path)
    plt.close()
        
def plot_boxplots(exp_name, freq):
    # Load data
    df_path = os.path.join("output", exp_name, "results.csv")
    df = pd.read_csv(df_path)

    # Compute count per N
    counts_per_n = df["N"].value_counts().sort_index()
    count_dict = counts_per_n.to_dict()
    total_samples = len(df)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"BIC and MSE over N number of parameters at {freq} Hz (n={total_samples} samples)", fontsize=16)

    # BIC plot
    sns.boxplot(data=df, x="N", y="bic", color="dodgerblue", ax=axs[0])
    axs[0].set_title("BIC over N")
    axs[0].set_xlabel("N")
    axs[0].set_ylabel("BIC")

    min_bic_row = df.loc[df["bic"].idxmin()]
    min_bic_val = round(min_bic_row["bic"], 3)
    axs[0].plot(min_bic_row["N"], min_bic_row["bic"], marker='x', color='black', markersize=8,
                label=f'Best BIC: {min_bic_val}')
    axs[0].legend()

    # Annotate counts under each tick
    for xtick in axs[0].get_xticks():
        n_value = int(axs[0].get_xticklabels()[xtick].get_text())
        count = count_dict.get(n_value, 0)
        axs[0].text(xtick, axs[0].get_ylim()[0] - 0.9, f'n={count}', 
                    ha='center', va='top', fontsize=9, color='gray')

    # MSE plot
    sns.boxplot(data=df, x="N", y="mse", color="dodgerblue", ax=axs[1])
    axs[1].set_yscale("log")
    axs[1].set_title("MSE over N")
    axs[1].set_xlabel("N")
    axs[1].set_ylabel("MSE (log scale)")

    min_mse_row = df.loc[df["mse"].idxmin()]
    min_mse_val = round(min_mse_row["mse"], 3)
    axs[1].plot(min_mse_row["N"], min_mse_row["mse"], marker='x', color='black', markersize=8,
                label=f'Best MSE: {min_mse_val}')
    axs[1].legend()

    # Annotate counts under each tick
    for xtick in axs[1].get_xticks():
        n_value = int(axs[1].get_xticklabels()[xtick].get_text())
        count = count_dict.get(n_value, 0)
        axs[1].text(xtick, axs[1].get_ylim()[0] * 0.4, f'n={count}', 
                    ha='center', va='top', fontsize=9, color='gray')

    plt.tight_layout(rect=[0, 0.01, 1, 1])
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
    freq = 30000
    exp_name = f"newtest_{freq}_hz"
    parameters = ["Rs","C_0","R_0","alpha_0"]

    extract_experiments(exp_name)
    plot_boxplots(exp_name,freq)

    for param in parameters:
        plot_param_to_perf(exp_name,param,freq)


