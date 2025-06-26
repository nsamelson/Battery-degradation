import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jax.numpy as jnp

from load_data import load_data
from models import sim_z
from preprocess_data import *


def plot_loss_curve(history):

    N = history["config"]["N"]
    save_path = os.path.join("output",history["config"]["model_name"],"loss.png")

    best_seed = history["best_seed"]
    folds = history["seeds"][best_seed]["folds"]
    loss_by_fold = [fold["val_loss"] for fold in folds]

    # get best fold loss
    best_fold_index = np.argmin(loss_by_fold)
 
    # get progresses
    losses = folds[best_fold_index]["train_losses"]
    val_losses = folds[best_fold_index]["val_losses"]

    train_cells = folds[0]["train_cells"]
    val_cell = folds[0]["val_cell"]

    # --- plot loss curve ---
    plt.figure(figsize=(6,4))
    # avoid weird step labels that are not integers
    step = max(1, len(losses) // 10)  # Adjust step size based on the number of items
    plt.xticks(np.arange(0, len(losses), step))

    plt.plot(losses, label=f"Training Loss (cells [{','.join(train_cells)}])")
    plt.plot(val_losses, label=f"Validation Loss (cell {val_cell})")
    # plt.yscale('log')
    plt.xlabel("Step")
    plt.ylabel(f"Loss ({history['config']['loss_type']})")
    plt.title(f"Training Loss (N={N} blocks)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

def plot_params_progress(params_progress: dict, losses: list, config: dict, params_to_plot=["R", "Rs", "C", "alpha"]):
    N = config["N"]
    save_path = os.path.join("output", config["model_name"], "params_over_loss.png")

    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axs = axs.flatten()
    
    param_groups = {
        "R": 0,
        "Rs": 0,
        "C": 1,
        "alpha": 2
    }

    titles = ["R and Rs (log-scale)", "C (log-scale)", "Alpha", "Loss only"]

    for pname in params_to_plot:
        if pname not in params_progress:
            continue
        ax_idx = param_groups.get(pname, 3)
        ax = axs[ax_idx]

        param_array = np.array(params_progress[pname])  # shape: (steps, dim) or (steps,)
        if param_array.ndim == 1:
            param_array = param_array[:, np.newaxis]

        for i in range(param_array.shape[1]):
            values = param_array[:, i]
            if pname in ["R", "Rs", "C"]:
                values = 10**values  # Stabilize / preserve scale
                ax.set_yscale("log")
            ax.plot(values, label=f"{pname}[{i}]", linestyle="--")

    # Plot loss in all subplots
    for i in range(4):
        if i == 3:
            axs[i].plot(losses, label="Loss", color="black")
        axs[i].set_title(titles[i])
        axs[i].legend()
        axs[i].set_xlabel("Step")
        if i in [0, 2]:
            axs[i].set_ylabel("Value")

    fig.suptitle(f"Training Loss and Parameters (N={N} blocks)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path)
    plt.close()

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
        plt.plot(losses, label=f"{key} blocks",linestyle="--")
        # plt.plot(avg_losses, label="avg loss (all cells)")
    # plt.yscale('log')
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

def plot_param_evolution(history, param_names=["R", "C", "alpha"], cell="U1"):
    save_path = os.path.join("output", history["config"]["model_name"], "params_evolution.png")
    data = history[cell]["trainings"]
    config = history["config"]
    N = config["N"]

    fig, axes = plt.subplots(len(param_names), N, figsize=(4*N, 3 * len(param_names)), squeeze=False)
    axes = np.atleast_2d(axes)

    all_handles = []
    all_labels = []
    
    for i, pname in enumerate(param_names):
        for j in range(N):
            ax = axes[i][j]
            for entry in data:
                init = entry["init_params"][pname]
                final = entry["params"][pname]

                init = init if isinstance(init, list) else [init]
                final = final if isinstance(final, list) else [final]

                if j >= len(init) or j >= len(final):
                    continue

                if pname in ["R","C"]:
                    ax.set_yscale("log")

                line,= ax.plot([0, 1], [init[j], final[j]], marker='o',linestyle="--", label=f"Seed {entry['seed']}")
                if i == 0 and j == 0:  # Collect legend items only from first subplot
                    all_handles.append(line)
                    all_labels.append(f"Seed {entry['seed']}")


            if i==0:
                ax.set_title(f"N={j}")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Init", "Trained"])
            ax.set_ylabel(f"{pname}_{j}")
            ax.grid(True)


    fig.legend(
        all_handles,
        all_labels,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        title="Seeds"
    )

    plt.suptitle(f"Init vs trained params (N={N} block(s))")
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path,bbox_inches='tight')


def plot_loss_vs_parameter(history, parameter_name="Parameter"):
    """
    Plots the loss (y-axis) over the parameter range (x-axis).
    """
    N = history["config"]["N"]
    parameter_values = [seed["folds"][0]["params"][parameter_name] for seed in history["seeds"] if len(seed["folds"]) > 0]
    losses = [seed["avg_val_loss"] for seed in history["seeds"] if len(seed["folds"]) > 0]
    
    initial_values = [10**seed["folds"][0]["params_progress"][parameter_name][0] for seed in history["seeds"] if len(seed["folds"]) > 0]
    initial_losses = [seed["folds"][0]["val_losses"][0] for seed in history["seeds"] if len(seed["folds"]) > 0]

    # get the best param value for the minimum loss
    best_index = np.argmin(losses)
    best_param_value = parameter_values[best_index]
    best_loss = losses[best_index]

    plt.figure(figsize=(6, 4))
    plt.scatter(initial_values, initial_losses, color='green', label=f'Initial {parameter_name}', alpha=0.8)
    plt.scatter(parameter_values, losses, color='royalblue', label=f'Trained {parameter_name}', alpha=0.8)
    plt.scatter(best_param_value, best_loss, color='black', label=f'Best: {best_param_value:.2e}',marker="x")

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel(f"{parameter_name} Range", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Loss vs Initial and Trained {parameter_name} (N={N} block(s))", fontsize=14)
    plt.legend(fontsize=10, loc='upper right')

    save_path = os.path.join("output", history["config"]["model_name"], f"{parameter_name}_over_loss.png")
    plt.savefig(save_path,bbox_inches='tight',dpi=300)
    plt.tight_layout()
    plt.close()


if __name__ == "__main__":

    # ------ Parameters ------
    freq = 10
    model_name = "caparange"
    full_hist = {}


    for N in range(1,7):
        path = f"output/{model_name}_{N}_blocks_10_hz/"
        with open(os.path.join(path,"history.json"), "r") as f:
            history = json.load(f)


        losses = history["U1"]["best_seed"]["losses"]
        params_progress = history["U1"]["best_seed"]["params_progress"]
        avg_losses = history["avg_best_seed"]["losses"]

        config = history["config"]
        config["N"] = N

        plot_loss_curve(losses,avg_losses,config)
        plot_params_progress(params_progress,losses,config)
        plot_param_evolution(history)

        # get stuff from hist
        best_loss = history["avg_best_seed"]["loss"]
        best_bic = history["avg_best_seed"]["bic"]
        best_aic = history["avg_best_seed"]["aic"]
        params = history["U1"]["best_seed"]["params"]
        params["Rs"] = jnp.log10(jnp.array(params["Rs"]))
        params["R"] = jnp.log10(jnp.array(params["R"]))
        params["C"] = jnp.log10(jnp.array(params["C"]))
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
        # print(I, U)

        # plot signal
        y_pred = sim_z(I=I[:2000],fs=config["fs"], **params)

        # y_pred *= 200

        plot_signal(U, y_pred, config, 2000)


        print(f"{N} blocks: Loss={best_loss:.5f}, BIC={best_bic:.2f}, AIC={best_aic:.2f}")
        full_hist[N] = {
            "losses": losses,
            "avg_losses":avg_losses,
            "params":params
        }


    plot_all_losses(full_hist,model_name)