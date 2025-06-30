import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jax.numpy as jnp

from load_data import load_data
from models import sim_z
from preprocess_data import *
import seaborn as sns


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
    axs = np.atleast_1d(axs).flatten()

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
    save_path = os.path.join("output", f"{model_name}_losses.png")

    plt.figure(figsize=(6, 4))
    for key, val in full_history.items():
        fig, ax1 = plt.subplots()

        # Plot training loss on the right y-axis
        ax2 = ax1.twinx()
        ax2.plot(full_history.keys(), [val["best_train_loss"] for val in full_history.values()], color="royalblue", linestyle="--", marker="o", label="Train Loss")
        ax2.set_ylabel("Train Loss", color="royalblue")
        ax2.tick_params(axis="y", labelcolor="royalblue")

        # Plot validation loss on the left y-axis
        ax1.plot(full_history.keys(), [val["best_loss"] for val in full_history.values()], color="orangered", linestyle="--", marker="x", label="Val Loss")  # Red color for validation loss
        ax1.set_ylabel("Val Loss", color="orangered")
        ax1.tick_params(axis="y", labelcolor="orangered")

        # Combine legends from both axes and place them in the upper right corner within the plot
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=10)

        # Configure x-axis and title
        ax1.set_xlabel("Number of Blocks (N)")
        ax1.set_title("Best Training and Validation Losses for Each N")
        ax1.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_val_losses_vs_BIC(full_history, model_name):
    """
    Plots validation losses and BIC for each N in the full history.

    Args:
        full_history (dict): Dictionary containing training history for different N values.
        model_name (str): Name of the model for saving the plot.
    """
    save_path = os.path.join("output", f"{model_name}_val_losses_vs_BIC.png")

    # Extract N values, validation losses, and BICs
    N_values = sorted(full_history.keys())
    val_losses = [full_history[N]["best_loss"] for N in N_values]
    bics = [full_history[N]["best_bic"] for N in N_values]

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Plot validation loss on the left y-axis
    ax1.plot(N_values, val_losses, color="orangered", marker="o", linestyle="-", label="Validation Loss")
    ax1.set_xlabel("Number of Blocks (N)", fontsize=12)
    ax1.set_ylabel("Val Loss (CAE)", color="orangered", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="orangered")
    ax1.grid(True, linestyle="-", alpha=0.6)

    # Plot BIC on the right y-axis
    ax2 = ax1.twinx()
    ax2.plot(N_values, bics, color="royalblue", marker="x", linestyle="--", label="BIC", alpha=1)
    ax2.set_ylabel("BIC", color="royalblue", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="royalblue")

    # Add a title and save the figure
    plt.title("Validation Loss and BIC vs Number of Blocks (N)", fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_param_evolution(history, param_names=["R", "C", "alpha"], cell="U1"):
    """
    Plots the evolution of parameters from initialization to trained values for each block.

    Args:
        history (dict): Training history containing parameter progress.
        param_names (list): List of parameter names to plot.
        cell (str): Validation cell to filter the folds.
    """
    save_path = os.path.join("output", history["config"]["model_name"], "params_evolution.png")
    config = history["config"]
    N = config["N"]
    seeds = history["seeds"]
    best_seed = history["best_seed"]

    # Extract parameter progress for the specified cell
    trainings = {
        seed["seed"]: fold["params_progress"]
        for seed in seeds
        for fold in seed["folds"]
        if fold["val_cell"] == cell
    }

    # Get the best seed's parameters
    best_params = history["seeds"][best_seed]["folds"][0]["params"]

    # Create subplots
    fig, axes = plt.subplots(len(param_names), N, figsize=(3 * N, 2 * len(param_names)), squeeze=False)
    axes = np.atleast_2d(axes)

    handles, labels = [], []  # Collect handles and labels for the legend

    for i, pname in enumerate(param_names):
        for j in range(N):
            ax = axes[i][j]
            for k, entry in trainings.items():
                init = entry[pname][0]
                final = entry[pname][-1]

                # Handle single or multi-dimensional parameters
                init = init[j] if isinstance(init, list) else [init]
                final = final[j] if isinstance(final, list) else [final]

                # Convert to original scale for log-scale parameters
                if pname in ["R", "C"]:
                    ax.set_yscale("log")
                    init = 10**init
                    final = 10**final

                # Plot parameter evolution for all seeds
                linestyle = "--" if k == best_seed else ":"
                color = "red" if k == best_seed else "gray"
                alpha = 1.0 if k == best_seed else 0.5
                ax.plot([0, 1], [init, final], marker="o", linestyle=linestyle, color=color, alpha=alpha)

            # Configure subplot titles and labels
            if i == 0:
                ax.set_title(f"N={j}")
            if i == len(param_names) - 1:  # Add x-axis labels only on the last row
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Init", "Trained"])
            else:
                ax.set_xticks([])  # Remove x-axis ticks for other rows
            ax.set_ylabel(f"{pname}_{j}" if pname != "Rs" else pname)
            ax.grid(True)

    # Add the best seed's parameters to the legend
    best_params_str = ", ".join([
        f"{key}={', '.join([f'{10**float(v):.2e}' for v in np.atleast_1d(val)])}" if key in ["R", "C"] and isinstance(val, (np.ndarray, list, jnp.ndarray)) else
        f"{key}={', '.join([f'{float(v):.2e}' for v in np.atleast_1d(val)])}" if isinstance(val, (np.ndarray, list, jnp.ndarray)) else
        f"{key}={10**float(np.atleast_1d(val)[0]):.2e}" if key in ["R", "C"] else
        f"{key}={float(np.atleast_1d(val)[0]):.2e}"
        for key, val in best_params.items()
    ])
    handles.append(plt.Line2D([0], [0], color="red", linestyle="--"))  # Ensure 'linestyle' is correctly spelled
    labels.append(f"Best Seed {best_seed}: {best_params_str}")

    # Add a global legend at the bottom center
    fig.legend(handles, labels, loc="lower center", fontsize=10, ncol=1, bbox_to_anchor=(0.5, -0.1))

    # Add a global title and save the figure
    plt.suptitle(f"Init vs Trained Parameters (N={N} block(s))", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_loss_vs_parameter(history, parameter_name="Rs"):
    """
    Plots the loss (y-axis) over the parameter range (x-axis) in a grid layout.
    Handles multi-dimensional parameters by creating subplots in a grid.
    """
    N = history["config"]["N"]
    seeds = [seed for seed in history["seeds"] if len(seed["folds"]) > 0]
    losses = [seed["avg_val_loss"] for seed in seeds]

    # Check if the parameter is multi-dimensional
    is_multi_dim = isinstance(seeds[0]["folds"][0]["params"][parameter_name], list)

    if is_multi_dim:
        # Determine the number of rows and columns dynamically
        rows = (N + 1) // 2  # Number of rows (2 columns per row, except for odd N)
        cols = 2 if N > 1 else 1  # Use 2 columns if N > 1, otherwise 1 column
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
        axs = np.atleast_1d(axs).flatten()  # Flatten to handle indexing uniformly

        for i in range(N):
            parameter_values = [seed["folds"][0]["params"][parameter_name][i] if parameter_name in ["Rs", "R", "C"] else seed["folds"][0]["params"][parameter_name][i] for seed in seeds]
            initial_values = [10**seed["folds"][0]["params_progress"][parameter_name][0][i] if parameter_name in ["Rs", "R", "C"] else seed["folds"][0]["params_progress"][parameter_name][0][i] for seed in seeds]
            
            # Get the best param value for the minimum loss
            best_index = np.argmin(losses)
            best_param_value = parameter_values[best_index]
            best_loss = losses[best_index]

            ax = axs[i]
            ax.scatter(initial_values, losses, color='green', label=f'Initial {parameter_name}[{i}]', alpha=0.8)
            ax.scatter(parameter_values, losses, color='royalblue', label=f'Trained {parameter_name}[{i}]', alpha=0.8)
            ax.scatter(best_param_value, best_loss, color='black', label=f'Best: {best_param_value:.2e}', marker="x")

            # Set x-axis and y-axis to log scale for log-scale parameters
            if parameter_name in ["Rs", "R", "C"]:
                ax.set_xscale("log")

            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel(f"{parameter_name}[{i}] Range", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title(f"Loss vs {parameter_name}[{i}]", fontsize=14)
            ax.legend(fontsize=10, loc='upper right')

        # Hide unused subplots
        for j in range(N, len(axs)):
            axs[j].axis('off')

        # Add a main title for the grid
        fig.suptitle(f"Seeds of init and trained {parameter_name} over Loss (N={N} Blocks)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join("output", history["config"]["model_name"], f"{parameter_name}_over_loss.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    else:
        parameter_values = [seed["folds"][0]["params"][parameter_name] if parameter_name in ["Rs", "R", "C"] else seed["folds"][0]["params"][parameter_name] for seed in seeds]
        initial_values = [10**seed["folds"][0]["params_progress"][parameter_name][0] if parameter_name in ["Rs", "R", "C"] else seed["folds"][0]["params_progress"][parameter_name][0] for seed in seeds]
        # Get the best param value for the minimum loss
        best_index = np.argmin(losses)
        best_param_value = parameter_values[best_index]
        best_loss = losses[best_index]

        plt.figure(figsize=(6, 4))
        plt.scatter(initial_values, losses, color='green', label=f'Initial {parameter_name}', alpha=0.8)
        plt.scatter(parameter_values, losses, color='royalblue', label=f'Trained {parameter_name}', alpha=0.8)
        plt.scatter(best_param_value, best_loss, color='black', label=f'Best: {best_param_value:.2e}', marker="x")

        # Set x-axis and y-axis to log scale for log-scale parameters
        if parameter_name in ["Rs", "R", "C"]:
            plt.xscale("log")
        plt.yscale("log")

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel(f"{parameter_name} Range", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(f"Seeds of init and trained {parameter_name} over Loss (N={N} Blocks)", fontsize=14)
        plt.legend(fontsize=10, loc='upper right')

        save_path = os.path.join("output", history["config"]["model_name"], f"{parameter_name}_over_loss.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.tight_layout()
        plt.close()


def plot_best_loss_per_N(full_history, model_name):
    """
    Plots the best validation loss for each N from the full history, along with a regression plot
    for all trained seeds' validation losses.

    Args:
        full_history (dict): Dictionary containing training history for different N values.
        model_name (str): Name of the model for saving the plot.
    """

    save_path = os.path.join("output", f"{model_name}_losses_trend.png")

    # Extract N values, best losses, and all seeds' validation losses
    N_values = sorted(full_history.keys())
    best_losses = [full_history[N]["best_loss"] for N in N_values]
    all_seeds_losses = [full_history[N]["seeds_val_losses"] for N in N_values]

    # Flatten all seeds' losses for regression plot
    all_N = []
    all_losses = []
    for i, losses in enumerate(all_seeds_losses):
        all_N.extend([N_values[i]] * len(losses))
        all_losses.extend(losses)

    # Create a DataFrame for seaborn
    data = pd.DataFrame({"N": all_N, "Validation Loss": all_losses})

    # Plot the best losses and all seeds' losses with regression
    plt.figure(figsize=(6, 4))
    plt.scatter([], [], color="cornflowerblue", s=12, alpha=0.8, label="Random Seeds")  # Add an invisible scatter for legend
    sns.regplot(
        x="N", 
        y="Validation Loss", 
        data=data, 
        color="cornflowerblue", 
        scatter_kws={"s": 12, "alpha": 0.8}, 
        line_kws={"color": "royalblue", "label": "Trend"}, 
        order=2
    )
    plt.plot(N_values, best_losses, marker="o", linestyle="--", color="orangered", label="Best Seeds")

    # Configure plot
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel("Number of Blocks (N)", fontsize=12)
    plt.ylabel("Loss (CAE)", fontsize=12)
    plt.title("Validation Loss over Model Complexity (N)", fontsize=14)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
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
        # folds = history["seeds"][best_seed]["folds"]
        # loss_by_fold = [fold["val_loss"] for fold in folds]

        # Find the best seed by matching the "seed" field
        best_seed_data = next(seed for seed in history["seeds"] if seed["seed"] == best_seed)

        # Get best fold loss and BIC
        avg_val_loss = best_seed_data["avg_val_loss"]
        folds = best_seed_data["folds"]
        loss_by_fold = [fold["val_loss"] for fold in folds]
        best_loss = min(loss_by_fold)
        best_fold_index = loss_by_fold.index(best_loss)
        best_train_loss = folds[best_fold_index]["train_losses"][-1]  # Last training loss
        best_bic = folds[best_fold_index]["bic"]
        print(f"N={N}, Seed: {best_seed}, Train loss: {best_train_loss:.4f}, Val loss: {best_loss:.4f}, BIC: {best_bic:.2f}")
        # get progresses
        losses = folds[best_fold_index]["train_losses"]
        val_losses = folds[best_fold_index]["val_losses"]
        params_progress = folds[best_fold_index]["params_progress"]



        config = history["config"]
        config["N"] = N

        # plot_loss_curve(history.copy())
        # plot_params_progress(history.copy())
        # plot_signal(history.copy())
        
        # plot_param_evolution(history.copy())        
        # plot_loss_vs_parameter(history.copy(), parameter_name="R")
        # plot_loss_vs_parameter(history.copy(), parameter_name="Rs")
        # plot_loss_vs_parameter(history.copy(), parameter_name="C")
        # plot_loss_vs_parameter(history.copy(), parameter_name="alpha")

        # # get stuff from hist
        params = folds[best_fold_index]["params"]
        print(f"Best params for N={N}: {params}")

        params["Rs"] = jnp.log10(jnp.array(params["Rs"]))
        params["R"] = jnp.log10(jnp.array(params["R"]))
        params["C"] = jnp.log10(jnp.array(params["C"]))
        params["alpha"] = jnp.array(params["alpha"])
  

        # grab all avg_val_losses from all the seeds, that are not NaN
        seeds_val_losses = []
        for seed in history["seeds"]:
            if "avg_val_loss" in seed and not jnp.isnan(seed["avg_val_loss"]) and not jnp.isinf(seed["avg_val_loss"]):
                seeds_val_losses.append(seed["avg_val_loss"])

        # print(f"{N} blocks: Loss={best_loss:.5f}, BIC={best_bic:.2f}, AIC={best_aic:.2f}")
        full_hist[N] = {
            "losses": losses,
            "avg_losses":val_losses,
            "params":params,
            "best_loss": best_loss,
            "best_train_loss": best_train_loss,
            "best_bic": best_bic,
            "best_seed": best_seed,
            "seeds_val_losses": seeds_val_losses,
        }

    
    # Call the function to plot best loss per N
    plot_best_loss_per_N(full_hist, model_name)
    plot_all_losses(full_hist,model_name)
    plot_val_losses_vs_BIC(full_hist, model_name)