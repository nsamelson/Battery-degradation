import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from main import correct_signal, decimate_signal, load_data, sim_z
from run_model import simulation, load_matlab_data, reduce_sampling, cumulative_absolute_error, root_mean_squared_error

COLORS = {
        'U1': 'orange', 
        'U2': 'darkgreen',    
        'U3': 'orangered',    
        'U4': 'darkblue',     
        'U5': 'brown',    
        'U6': 'violet',  
    }

def plot_signals(x, y_true, y_pred, params):
    colors = ['b', 'g', 'r', 'm']
    linestyles = ['-', '--']

    print(params)

    # duration = params["durations"][0]

    # for i, (fb, fs, duration, resp, exc) in enumerate(zip(fbs, fss, durations, response, excitation)):
    fig, ax = plt.subplots( figsize=(20, 10))
    time = np.linspace(0, len(x)/params['fbs'], len(x))
    
    # Plot response and excitation on the same subplot
    ax.plot(time, y_pred, color=colors[0], linestyle=linestyles[0], alpha=0.4, label=f'y_pred')
    ax.plot(time, y_true, color=colors[1], linestyle=linestyles[0], alpha=0.4, label=f'y_true')
    # ax.plot(time, x, color=colors[2], linestyle=linestyles[0],  label=f'Excitation')
    ax.set_title(f'Simulation vs Ground truth')
    ax.set_xlim([0, time[-1]])
    ax.legend()
    ax.grid(True)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")

    fig.tight_layout()
    plt.savefig("plots/sim_v_truth.png", dpi=300)


def extract_experiments(full_name,exp_name):
    folder = os.path.join("ray_results", full_name)
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
                "cae": result["cae"],
                "rmse": result["rmse"],
                "nrmse": result["nrmse"],
                "timestamp":result["timestamp"]
            }
            # Flatten config
            for k, v in result.get("config", {}).items():
                flat_record[k] = v

            records.append(flat_record)

    df = pd.DataFrame(records)

    out_folder = os.path.join("output", exp_name,full_name)
    os.makedirs(out_folder, exist_ok=True)
    df.to_csv(os.path.join(out_folder, f"trials.csv"), index=False)


def extract_metrics(exp_name, freq=10):
    dirs = os.listdir(os.path.join("output", exp_name))
    
    combined_metrics = []

    for directory in dirs:
        if "." in directory:
            continue
        test_path = os.path.join("output",exp_name,directory,"test_metrics.json")
        train_path = os.path.join("output",exp_name,directory,"train_metrics.json")

        with open(test_path,"r+") as f:
            test_metrics = json.load(f)
        
        with open(train_path,"r+") as f:
            train_metrics = json.load(f)

        # get stuff from the train json
        flat_record = {
            "cae": train_metrics["cae"],
            "rmse": train_metrics["rmse"],
            "nrmse": train_metrics["nrmse"],
        }

        # get stuff from the test json
        for cell, metrics in test_metrics.items():
            for k,v in metrics.items():
                flat_record[f"{k}_{cell}"] = v

        # get config
        for k, v in train_metrics.get("config", {}).items():
            flat_record[k] = v


        combined_metrics.append(flat_record)

    df = pd.DataFrame(combined_metrics)

    df.to_csv(os.path.join("output", exp_name, f"metrics.csv"), index=False)
  
def get_cell_metric(metric, cell):
    return metric if cell == "U4" else f"{metric}_{cell}"



def plot_param_to_perf(exp_name, parameter, freq):
    df_path = os.path.join("output", exp_name, "results.csv")
    df = pd.read_csv(df_path)

    if parameter not in df.columns:
        raise ValueError(f"Parameter '{parameter}' not found in results.")
    
    # palette = sns.color_palette("viridis", as_cmap=False, n_colors=6)


    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"BIC and MSE over parameter {parameter} at {freq} Hz", fontsize=16)

    # Primary axis - BIC
    sns.scatterplot(data=df, x=parameter, y="bic", ax=ax1, color="royalblue", s=30, alpha=0.6,)
    # sns.scatterplot(data=df, x=parameter, y="bic", ax=ax1, hue="alpha_0", palette="viridis", color="royalblue", s=30, alpha=0.6, legend=False)
    ax1.set_ylabel("BIC", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(True, which="major", linestyle='--', linewidth=0.5, color='lightgray')

    # Secondary axis - MSE (log scale)
    ax2 = ax1.twinx()
    sns.scatterplot(data=df, x=parameter, y="mse", ax=ax2, color="royalblue", marker="o", s=30, alpha=0.6,)
    # sns.scatterplot(data=df, x=parameter, y="mse", ax=ax2, hue="alpha_0", palette="viridis", color="royalblue", marker="o", s=30, alpha=0.6, legend=False)
    ax2.set_ylabel("MSE (log scale)", color="black")
    ax2.set_yscale("log")
    ax2.tick_params(axis='y', labelcolor="black")

    # Enable grid for log scale (right axis)
    ax2.grid(True, which="both", axis="y", linestyle=":", linewidth=0.5, color="lightgray")
    ax2.grid(False, axis='x')  # Prevent double x-axis grid

    # set x axis to log scale
    if "R_" in parameter:
        ax1.set_xscale("log")
        ax1.set_xscale("log")
    else:

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
    axs[0].plot(min_bic_row["N"] - 1, min_bic_row["bic"], marker='x', color='black', markersize=8,
                label=f'Best BIC: {min_bic_val}')
    axs[0].legend()

    # Annotate counts under each tick
    # for xtick in axs[0].get_xticks():
    #     n_value = int(axs[0].get_xticklabels()[xtick].get_text())
    #     count = count_dict.get(n_value, 0)
    #     axs[0].text(xtick, axs[0].get_ylim()[0] - 0.1, f'n={count}', 
    #                 ha='center', va='top', fontsize=9, color='gray')

    # MSE plot
    sns.boxplot(data=df, x="N", y="mse", color="dodgerblue", ax=axs[1])
    axs[1].set_yscale("log")
    axs[1].set_title("MSE over N")
    axs[1].set_xlabel("N")
    axs[1].set_ylabel("MSE (log scale)")

    min_mse_row = df.loc[df["mse"].idxmin()]
    min_mse_val = round(min_mse_row["mse"], 3)
    axs[1].plot(min_mse_row["N"] - 1, min_mse_row["mse"], marker='x', color='black', markersize=8,
                label=f'Best MSE: {min_mse_val}')
    axs[1].legend()

    # Annotate counts under each tick
    # for xtick in axs[1].get_xticks():
    #     n_value = int(axs[1].get_xticklabels()[xtick].get_text())
    #     count = count_dict.get(n_value, 0)
    #     axs[1].text(xtick, axs[1].get_ylim()[0] * 0.9, f'n={count}', 
    #                 ha='center', va='top', fontsize=9, color='gray')

    plt.tight_layout(rect=[0, 0, 1, 1])
    out_path = os.path.join("output", exp_name, "bic_and_mse.png")
    plt.savefig(out_path)
    plt.close()

      
def plot_errors(exp_name):
    df_path = os.path.join("output", exp_name, "metrics.csv")
    df = pd.read_csv(df_path)

    metrics = ['cae',"rmse","nrmse"]
    cells = [f'U{i}' for i in range(1, 7)]

    

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.1
        x = df['N']
        offsets = [i * width - width*3 for i in range(len(cells))]

        for i, cell in enumerate(cells):

            y = df[get_cell_metric(metric, cell)] # U4 was used to "train"
            cell_label = cell +" (train)" if cell == "U4" else cell
            ax.bar(x + offsets[i], y, width=width, label=cell_label, color=COLORS[cell], alpha=0.8)
        
        ax.set_xlabel("Model Complexity (N)")
        ax.set_ylabel(f"{metric.upper()} Value")
        ax.set_title(f"{metric.upper()} Across Model Complexity (Train and All Cells)")
        ax.legend()
        ax.grid(True)
        plt.xticks(df['N'])
        plt.tight_layout()
        plt.savefig(os.path.join("output",exp_name,f"{metric}_over_N.png"))
            


def plot_error_heatmaps(exp_name):
    df_path = os.path.join("output", exp_name, "metrics.csv")
    df = pd.read_csv(df_path)
    metrics = ['cae', 'rmse', 'nrmse']
    cells = [f'U{i}' for i in range(1, 7)]

    for metric in metrics:
        heatmap_data = pd.DataFrame({cell: df[get_cell_metric(metric,cell)] for cell in cells})
        heatmap_data.index = df['N']

        plt.figure(figsize=(10,6))
        sns.heatmap(heatmap_data.T, annot=True, fmt=".4g", cmap="YlGnBu", cbar_kws={'label': metric.upper()})
        plt.title(f"{metric.upper()} per Cell vs Model Complexity (N)")
        plt.xlabel("Model Complexity (N)")
        plt.ylabel("Cell")
        plt.tight_layout()
        plt.savefig(os.path.join("output", exp_name, f"{metric}_heatmap.png"))

def plot_train_vs_test_error(exp_name):
    df_path = os.path.join("output", exp_name, "metrics.csv")
    df = pd.read_csv(df_path)

    metrics = ['cae', 'rmse', 'nrmse']
    test_cells = [f'U{i}' for i in range(1, 7) if f'U{i}' != 'U4']
    x = df['N']

    for metric in metrics:
        train_vals = df[metric]  # U4 used as training, stored under the raw name
        test_matrix = np.array([df[f"{metric}_{cell}"] for cell in test_cells])
        test_mean = test_matrix.mean(axis=0)
        test_std = test_matrix.std(axis=0)

        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(x, train_vals, label='Train (U4)', color='darkblue', linestyle='--', marker='o')

        for i, cell in enumerate(test_cells):
            ax.plot(x, df[get_cell_metric(metric,cell)], label=cell, color=COLORS[cell], linestyle='--', marker='o')

        ax.plot(x, test_mean, label='Test Avg (U1,2,3,5,6)', color='black', linestyle='-', marker='s')
        # ax.fill_between(x, test_mean - test_std, test_mean + test_std, color='gray', alpha=0.3, label='Test ±1 std dev')

        ax.set_xlabel("Model Complexity (N)")
        ax.set_ylabel(f"{metric.upper()} Value")
        ax.set_title(f"{metric.upper()} – Train vs. Test Error Across Complexity")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("output", exp_name, f"{metric}_train_vs_test.png"))

def plot_bic_per_cell_with_mean_lines(exp_name):
    df_path = os.path.join("output", exp_name, "metrics.csv")
    df = pd.read_csv(df_path)

    metrics = ['bic_cae', 'bic_rmse', 'bic_nrmse']
    cells = [f'U{i}' for i in range(1, 7) if i != 4]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10,6))
        x = df['N']

        for cell in cells:
            y = df[f"{metric}_{cell}"]
            ax.plot(x, y, marker='o', label=cell, color=COLORS[cell])

        # Mean BIC line
        y_avg = df[[f"{metric}_{cell}" for cell in cells]].mean(axis=1)
        ax.plot(x, y_avg, marker='s', linestyle='--', color='black', label='Mean BIC', linewidth=2)

        ax.set_xlabel("Model Complexity (N)")
        ax.set_ylabel("BIC Score")
        ax.set_title(f"{metric.upper()} per Cell (Excl. U4) + Mean Across Complexity")
        ax.grid(True)
        ax.legend(title="Cells")
        plt.tight_layout()
        plt.savefig(os.path.join("output", exp_name, f"{metric}_bic_lines_mean.png"))

def plot_trials_errors(exp_name):
    base_path = os.path.join("output", exp_name)
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')]

    colors = plt.cm.viridis(np.linspace(0, 1, len(dirs)))
    metrics = ["cae","rmse","nrmse"]
    metric_caps = {
        "cae":1e6,
        "rmse":100,
        "nrmse":1000
    }

    for metric in metrics:

        plt.figure(figsize=(12, 6))
        for i, directory in enumerate(sorted(dirs)):
            if i not in [2]:
                continue
            trials_path = os.path.join(base_path, directory, "trials.csv")

            df = pd.read_csv(trials_path)
            df = df.sort_values(by='timestamp').reset_index(drop=True)

            # Limit
            metric_values = df[metric].clip(upper=metric_caps[metric])

            x = np.arange(len(metric_values))
            plt.plot(x, metric_values, label=directory, color=colors[i], alpha=0.6)
            

        plt.xlabel("Trial Number")
        plt.ylabel(metric.upper())
        plt.yscale("log")
        plt.title(f"{metric.upper()} over the trials")
        plt.legend(title="Model Complexity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f"trials_{metric}_over_index.png"))
        plt.close()



def plot_model_output(config: dict, cell="U4", save_path=None):
    import jax.numpy as jnp
    freq = config["freq"]
    N = config["N"]

    
    # Load real data
    data = load_matlab_data("data", freq)
    I = jnp.array(data.get("I"))[0]
    y_true = data.get(cell)

    params = {
        "fbs": np.array([freq]),
        "durations": data.get("duration")[0],
        "fss": data.get("fs")[0],
        "Rs": jnp.array(config["Rs"]),
        "R": jnp.array([config[f"R_{i}"] for i in range(N)]),
        "C": jnp.array([config[f"C_{i}"] for i in range(N)]),
        "alpha": jnp.array([config[f"alpha_{i}"] for i in range(N)]),
    }
    print(params)

    # downsample
    down_sample_factor = 20
    target_fss = params["fss"] // down_sample_factor
    I, y_true = reduce_sampling(I, y_true, data["fs"][0], target_fss, params["durations"])
    params["fss"] = np.array([target_fss])

    # Correct signals
    i_corr = I * (-1) * 50 / 0.625
    y_true[0] -= np.mean(y_true[0])



    el = 5000
    y_pred = simulation.main(i_corr[:el] - np.mean(i_corr[:el]), params, apply_noise=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[0][:el], label="y_true",)
    plt.plot(y_pred[0][:el], label="y_pred", )
    plt.title(f"Simulated Output vs True Signal (Cell {cell})")
    plt.xlabel("Timestep")
    plt.xlim(0,el)
    plt.ylabel("Voltage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    # Return error metrics too if needed
    # rmse = root_mean_squared_error(y_true, y_pred)
    # cae = cumulative_absolute_error(y_true, y_pred)
    # print( {"rmse": rmse, "cae": cae})

def plot_loss_curve(losses: list, avg_losses:list, config:dict):

    N = config["N"]
    save_path = os.path.join("output",config["model_name"],"loss.png")

    # --- plot loss curve ---
    plt.figure(figsize=(6,4))
    plt.plot(losses, label="best training loss")
    plt.plot(avg_losses, label="avg loss (all cells)")
    # plt.yscale('log')
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss (N={N} blocks)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def plot_signal(U, y_pred, config:dict, el):

    N = config["N"]
    save_path = os.path.join("output",config["model_name"],"signal.png")

    plt.figure(figsize=(8,6))
    plt.plot(U[:el], label="true")
    plt.plot(y_pred[:el], label="pred")
    plt.xlim(0,min(el, len(U)))
    plt.xlabel("Timestep")
    plt.ylabel("Voltage")
    plt.title(f"Simulated vs True (N={N} blocks)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def plot_all_losses(full_history, model_name):
    save_path = os.path.join("output",f"{model_name}_losses.png")

    plt.figure(figsize=(6,4))
    for key, val in full_history.items():
        losses = val["losses"]
        plt.plot(losses, label=f"{key} blocks")
        # plt.plot(avg_losses, label="avg loss (all cells)")
    # plt.yscale('log')
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    # data = np.load("output/signals.npz",allow_pickle=True)

    # samples = 100

    # x = data["x"][0:samples]
    # y_true = data["y_true"][0][0:samples]
    # y_pred = data["y_pred"][0][0:samples]
    # params = data["params"].item()

    # # print(x.shape,y_true.shape,y_pred.shape)

    # plot_signals(x, y_true, y_pred, params)
    # parameters = ["Rs","C_0","R_0","alpha_0"]
    # parameters = ["C_1","R_1","alpha_1", "C_2","R_2","alpha_2", "C_3","R_3","alpha_3", "C_4","R_4","alpha_4", ]

    # ------ Parameters ------
    freq = 10
    exp_name = f"bic_corr"

    # ------ extract trials from experiments ------
    # for N in range (1,7):
    #     full_name = f"{exp_name}_{N}_blocks_{freq}_hz"
    #     try:
    #         extract_experiments(full_name,exp_name)
    #     except:
    #         print(f"experiment {N} not found")

    # ------ combine metrics ------
    # extract_metrics(exp_name, freq)

    # ------ plot errors ------
    # plot_errors(exp_name)
    # plot_error_heatmaps(exp_name)
    # plot_train_vs_test_error(exp_name)
    # plot_bic_per_cell_with_mean_lines(exp_name)
    # plot_trials_errors(exp_name)

    # ------ plot signals -------
    # path = "output/bic_corr/bic_corr_3_blocks_10_hz"
    # with open(os.path.join(path,"best_config.json"), "r") as f:
    #     config = json.load(f)

    # plot_model_output(config, cell="U1", save_path=os.path.join(path,"simulation_vs_true_U4.png"))

    # plot_boxplots(exp_name,freq)

    # for param in parameters:
    #     plot_param_to_perf(exp_name,param,freq)


    # ------- plot optax -------
    import jax.numpy as jnp

    model_name = "optaxrun"
    full_hist = {}

    for N in range(1,7):
        path = f"output/{model_name}_{N}_blocks_10_hz/"
        with open(os.path.join(path,"history.json"), "r") as f:
            history = json.load(f)


        losses = history["U1"]["best_seed"]["losses"]
        avg_losses = history["avg_best_seed"]["losses"]

        config = history["config"]
        config["N"] = N

        plot_loss_curve(losses,avg_losses,config)

        # get stuff from hist
        best_loss = history["U1"]["best_seed"]["loss"]
        best_bic = history["U1"]["best_seed"]["bic"]
        params = history["U1"]["best_seed"]["params"]
        params["Rs"] = jnp.array(params["Rs"])
        params["R"] = jnp.array(params["R"])
        params["C"] = jnp.array(params["C"])
        params["alpha"] = jnp.array(params["alpha"])

        # get data
        data = load_data("data", freq)
        fs = float(data["fs"])

        # decimate and correct offset
        I = correct_signal(decimate_signal(data["I"],fs,2000))
        U = decimate_signal(data["U1"],fs,2000)

        # normalise
        I -= jnp.mean(I)
        U -= jnp.mean(U)

        # scale up
        I *= 200
        U *= 200

        # plot signal
        y_pred = sim_z(I=I[:2000],fs=config["fs"], **params)
        plot_signal(U, y_pred, config, 2000)


        print(f"{N} blocks: Loss={best_loss}, BIC={best_bic}")
        full_hist[N] = {
            "losses": losses,
            "avg_losses":avg_losses,
            "params":params
        }


    plot_all_losses(full_hist,model_name)