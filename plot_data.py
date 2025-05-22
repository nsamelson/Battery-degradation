import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jax.numpy as jnp

from load_data import load_data
from models import sim_z
from preprocess_data import *


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
        plt.plot(losses, label=f"{key} blocks",linestyle="--")
        # plt.plot(avg_losses, label="avg loss (all cells)")
    # plt.yscale('log')
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)




if __name__ == "__main__":

    # ------ Parameters ------
    freq = 10
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


        print(f"{N} blocks: Loss={best_loss:.5f}, BIC={best_bic:.2f}")
        full_hist[N] = {
            "losses": losses,
            "avg_losses":avg_losses,
            "params":params
        }


    plot_all_losses(full_hist,model_name)