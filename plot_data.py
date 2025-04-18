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
    folder = os.path.join("ray_results",exp_name)
    trial_paths = [os.path.join(folder,trial) for trial in os.listdir(folder) if os.path.isdir(os.path.join(folder,trial))]

    all_trials = []

    for trial_path in trial_paths:
        files_list = os.listdir(trial_path)
        if "result.json" in files_list:
            with open(os.path.join(trial_path,"result.json"),'r') as f:
                result = json.load(f)

                all_trials.append({
                    "bic": result["bic"],
                    "mse": result["mse"],
                    "config": result["config"]
                })
    
    out_folder = os.path.join("output",exp_name)
    os.makedirs(out_folder,exist_ok=True)

    result_file = os.path.join(out_folder,"results.json")
    with open(result_file,"w+") as f:
        json.dump(all_trials, f)

        
def load_and_plot_results(folder_name):
    # 1. Load data
    filepath = os.path.join("output",folder_name, "results.json")
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # 2. Extract bic, mse, config["N"]
    records = []
    for entry in data:
        bic = entry.get("bic")
        mse = entry.get("mse")
        config = entry.get("config", {})
        N = config.get("N")
        records.append({"bic": bic, "mse": mse, "N": N})
    
    df = pd.DataFrame(records)
    
    # 3. Plot boxplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("BIC and MSE over N number of parameters", fontsize=16)

    sns.boxplot(data=df, x="N", y="bic", ax=axs[0])
    axs[0].set_title("BIC over N")
    axs[0].set_xlabel("N")
    axs[0].set_ylabel("BIC")
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')

    # Highlight min BIC
    min_bic_row = df.loc[df["bic"].idxmin()]
    axs[0].plot(min_bic_row["N"], min_bic_row["bic"], marker='x', color='black', markersize=8, label='Min BIC')

    sns.boxplot(data=df, x="N", y="mse", ax=axs[1])
    axs[1].set_yscale("log")
    axs[1].set_title("MSE over N (log scale)")
    axs[1].set_xlabel("N")
    axs[1].set_ylabel("MSE (log scale)")
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')

    # Highlight min MSE
    min_mse_row = df.loc[df["mse"].idxmin()]
    axs[1].plot(min_mse_row["N"], min_mse_row["mse"], marker='x', color='black', markersize=8, label='Min MSE')

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join("output",folder_name,"bic_and_mse.png"))





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
    load_and_plot_results("bic_30000_hz")


