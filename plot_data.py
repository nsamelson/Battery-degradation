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
    save_path = os.path.join("output", history["config"]["model_name"], "loss.png")

    best_seed = history["best_seed"]
    folds = history["seeds"][best_seed]["folds"]
    loss_by_fold = [fold["val_loss"] for fold in folds]

    # Get best fold loss
    best_fold_index = np.argmin(loss_by_fold)

    # Get progresses
    losses = folds[best_fold_index]["train_losses"]
    val_losses = folds[best_fold_index]["val_losses"]

    train_cells = folds[0]["train_cells"]
    val_cell = folds[0]["val_cell"]

    # --- Plot loss curve ---
    plt.figure(figsize=(8, 5))  # Slightly larger for better readability
    step = max(2, len(losses) // 10)  # Adjust step size based on the number of items
    plt.xticks(np.arange(0, len(losses), step))

    plt.plot(losses, label=f"Training Loss (cells [{', '.join(train_cells)}])", linestyle="--", color="blue")
    plt.plot(val_losses, label=f"Validation Loss (cell {val_cell})", linestyle="-", color="orange")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel(f"Loss ({history['config']['loss_type']})", fontsize=12)
    plt.title(f"Training Loss (N={N} blocks)", fontsize=14)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_params_progress(history, params_to_plot=["R", "Rs", "C", "alpha"]):
    """
    Plots the progression of parameters and loss during training.

    Args:
        history (dict): Training history containing parameter progress and losses.
        params_to_plot (list): List of parameter names to plot.
    """
    config = history["config"]
    N = config["N"]

    best_seed = history["best_seed"]
    folds = history["seeds"][best_seed]["folds"]
    loss_by_fold = [fold["val_loss"] for fold in folds]

    # Get the best fold index
    best_fold_index = np.argmin(loss_by_fold)
    losses = folds[best_fold_index]["train_losses"]
    val_losses = folds[best_fold_index]["val_losses"]
    params_progress = folds[best_fold_index]["params_progress"]

    save_path = os.path.join("output", config["model_name"], "params_over_loss.png")

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    axs = axs.flatten()

    param_groups = {
        "R": 0,
        "Rs": 0,
        "C": 1,
        "alpha": 2
    }

    titles = ["R and Rs (log-scale)", "C (log-scale)", "Alpha", "Loss"]

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
                values = 10**values  # Convert to original scale
                ax.set_yscale("log")
            ax.plot(values, label=f"{pname}[{i}]", linestyle="--")

    # Plot losses in the last subplot
    axs[3].plot(losses, label="Train Loss", color="blue", linestyle="--")
    axs[3].plot(val_losses, label="Validation Loss", color="orange", linestyle="-")

    # Configure subplots
    for i, ax in enumerate(axs):
        ax.set_title(titles[i])
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.6)
        if i >= 2:  # Bottom row
            ax.set_xlabel("Step")
        if i % 2 == 0:  # Left column
            ax.set_ylabel("Value")

        step = max(2, len(losses) // 10)  # Adjust step size for x-axis ticks
        ax.set_xticks(np.arange(0, len(losses), step))

    # Add a global title and save the figure
    fig.suptitle(f"Training Loss and Parameter Progression (N={N} blocks)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300)
    plt.close()

def plot_signal(history, el=2000):
    config = history["config"]
    N = config["N"]

    best_seed = history["best_seed"]
    folds = history["seeds"][best_seed]["folds"]

    # Get the best fold index
    loss_by_fold = [fold["val_loss"] for fold in folds]
    best_fold_index = np.argmin(loss_by_fold)
    params = folds[best_fold_index]["params"]

    # weird thing is happening, sometimes I need it sometimes not
    params["Rs"] = jnp.log10(jnp.array(params["Rs"]))
    params["R"] = jnp.log10(jnp.array(params["R"]))
    params["C"] = jnp.log10(jnp.array(params["C"]))
    params["alpha"] = jnp.array(params["alpha"])    

    # Load and preprocess data
    data = load_data("data", config["freq"])
    fs = float(data["fs"])

    I = correct_signal(decimate_signal(data["I"], fs, 2000))
    U = decimate_signal(data["U1"], fs, 2000)

    # Normalize and scale signals
    I -= np.mean(I)
    U -= np.mean(U)
    I *= 200
    U *= 200

    # Simulate the predicted signal
    y_pred = sim_z(I=I[:el], fs=config["fs"], **params)

    # Save path for the plot
    save_path = os.path.join("output", config["model_name"], "signal.png")

    # Plot the true and predicted signals
    plt.figure(figsize=(8, 5))
    plt.plot(U[:el], label="True Signal", color="blue", linestyle="-")
    plt.plot(y_pred[:el], label="Simulated Signal", color="orange", linestyle="-")
    plt.xlim(0, min(el, len(U)))
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Voltage", fontsize=12)
    plt.title(f"Simulated vs True Signal Response (N={N} blocks)", fontsize=14)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


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
    plt.close()

def plot_param_evolution(history, param_names=["R", "C", "alpha"], cell="U1"):
    save_path = os.path.join("output", history["config"]["model_name"], "params_evolution.png")
    config = history["config"]
    N = config["N"]
    # data = history[cell]["trainings"]
    seeds = history["seeds"]
    best_seed = history["best_seed"]
    trainings = {seed["seed"]:fold["params_progress"] for seed in seeds for fold in seed["folds"] if fold["val_cell"] == cell}

    fig, axes = plt.subplots(len(param_names), N, figsize=(4*N, 3 * len(param_names)), squeeze=False)
    axes = np.atleast_2d(axes)

    all_handles = []
    all_labels = []
    
    for i, pname in enumerate(param_names):
        for j in range(N):
            ax = axes[i][j]
            for k, entry in trainings.items():
                # init = entry["init_params"][pname]
                # final = entry["params"][pname]
                init = entry[pname][0]
                final = entry[pname][-1]

                init = init[j] if isinstance(init, list) else [init]
                final = final[j] if isinstance(final, list) else [final]

                # if j >= len(init) or j >= len(final):
                #     continue

                if pname in ["R","C"]:
                    ax.set_yscale("log")
                    init = 10**init
                    final = 10** final

                best_comm = " (best)" if k== best_seed else ""

                line,= ax.plot([0, 1], [init,final], marker='o',linestyle="--", label=f"Seed {k}")
                if i == 0 and j == 0:  # Collect legend items only from first subplot
                    all_handles.append(line)
                    all_labels.append(f"Seed {k}{best_comm}")


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
    model_name = "params_search"
    full_hist = {}


    for N in range(1,7):
        path = f"output/{model_name}_{N}_blocks_10_hz/"

        try:
            with open(os.path.join(path,"history.json"), "r") as f:
                history = json.load(f)
        except:
            continue

        best_seed = history["best_seed"]
        folds = history["seeds"][best_seed]["folds"]
        loss_by_fold = [fold["val_loss"] for fold in folds]

        # get best fold loss and BIC
        avg_val_loss = history["seeds"][best_seed]["avg_val_loss"]
        best_loss = min(loss_by_fold)
        best_fold_index = loss_by_fold.index(best_loss)
        best_bic = folds[best_fold_index]["bic"]
        print(f"N={N}, Seed: {best_seed}, Avg loss: {avg_val_loss:.4f}, Min loss: {best_loss:.4f}, BIC: {best_bic:.2f}")

        # get progresses
        losses = folds[best_fold_index]["train_losses"]
        val_losses = folds[best_fold_index]["val_losses"]
        params_progress = folds[best_fold_index]["params_progress"]



        config = history["config"]
        config["N"] = N

        plot_loss_curve(history)
        plot_params_progress(history)
        plot_signal(history)
        
        plot_param_evolution(history)        
        plot_loss_vs_parameter(history, parameter_name="Rs")

        # # get stuff from hist
        params = folds[best_fold_index]["params"]
        params["Rs"] = jnp.log10(jnp.array(params["Rs"]))
        params["R"] = jnp.log10(jnp.array(params["R"]))
        params["C"] = jnp.log10(jnp.array(params["C"]))
        params["alpha"] = jnp.array(params["alpha"])
  



        # print(f"{N} blocks: Loss={best_loss:.5f}, BIC={best_bic:.2f}, AIC={best_aic:.2f}")
        full_hist[N] = {
            "losses": losses,
            "avg_losses":val_losses,
            "params":params
        }


    plot_all_losses(full_hist,model_name)