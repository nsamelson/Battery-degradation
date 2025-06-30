import argparse
import os
import json
import jax
import jax.numpy as jnp

import load_data
from models import *
import preprocess_data as preprocess
import train


def main(model_name, N, iters, freq, debug, sampling_frequency, n_seeds, start_seed, minibatch, optimizer, loss_type="MSE", loco_cv=True):
    work_dir = os.getcwd()
    el = 4000
    cells = ["U1","U2","U4","U5","U6"] # exclude U3

    loss_codes = {"MSE": 0, "RMSE": 1, "CSE": 2, "CAE": 3, "MAPE": 4}
    loss_code = loss_codes[loss_type]

    # get data
    data = load_data.load_data(os.path.join(work_dir, "data"), freq)
    fs = float(data["fs"])

    # decimate and correct offset
    I = preprocess.correct_signal(preprocess.decimate_signal(data["I"],fs,sampling_frequency))
    U_cells = {cell: preprocess.decimate_signal(data[cell], fs, sampling_frequency) for cell in cells}
    
    if debug:
        U_cells = {cell: U[:el] for cell, U in U_cells.items()}
        I= I[:el]

    # normalise
    I -= jnp.mean(I)
    U_cells = {cell: U - jnp.mean(U) for cell, U in U_cells.items()}

    # init parameters
    base_params = {
        'Rs':    jnp.array(jnp.log10(3e-3)),                   # initial supply resistance
        'R':     jnp.ones((N,)) * jnp.log10(5e-2) ,             # block resistances
        'C':     jnp.log10(jnp.array([20.,100.,1000,500.,500.,500.])[:N]) , # block capacitances
        'alpha': jnp.ones((N,)) * 0.75,             # fractional factors
    }

    best_seed_overall = None
    best_avg_val_loss = float("inf")
    


    history = {
        "config": {
            "model_name": model_name,
            "freq": freq,
            "debug": debug,
            "sampling_frequency": sampling_frequency,
            "fs": fs,
            "N": N,
            "loss_type": loss_type,
        },
        "seeds": []
    }

    for s in range(start_seed, start_seed + n_seeds + 1):

        rng_key = jax.random.PRNGKey(s)
        init_params = base_params if s == 0 else preprocess.sample_params(key=rng_key, N=N)

        print(f"\n🔁 Seed {s}, init params: {preprocess.log_to_exp(init_params)}")
        seed_result = {"seed": s, "folds": [], "avg_val_loss": None}
        val_losses = []

        # Leave-One-Cell-Out Cross-Validation
        loco_cells = cells if loco_cv else ["U1"]
        for val_cell in loco_cells:
            train_cells = [c for c in cells if c != val_cell] # leave one cell out for val
            U_train = [U_cells[c] for c in train_cells]
            U_val = U_cells[val_cell]

            # pilot loss
            try:
                pilot_loss = compute_loss(init_params.copy(), I, U_val, fs, loss_code =loss_code )
                if not jnp.isfinite(pilot_loss):
                    print(f"🚫 Pilot loss is NaN or inf at seed {s}, skipping.")
                    break
            except:
                print("Seed unstable, skip")
                break

            if s > 0 and pilot_loss > 100 * best_avg_val_loss:
                print(f"🚫 Skipping bad seed {s} (pilot val loss {pilot_loss:.2e} vs {best_avg_val_loss:.2e})")
                break
            
            try:
                trained_params, train_losses, val_losses_progress, params_progress = train.train_loop(
                    init_params.copy(), I, U_train, fs, U_val,
                    num_steps=iters, minibatch=minibatch, opt_type=optimizer, loss_code =loss_code 
                )
            except Exception as e:
                print(f"❌ Exception in training seed {s}, val cell {val_cell}: {e}")
                break


            if not train_losses:
                print("Seed became unstable while training, skip")
                break
            
            try:
                val_loss = compute_loss(trained_params.copy(), I, U_val, fs, loss_code =loss_code )
                val_losses.append(val_loss)
            except:
                print(f"🚫 Validation loss is NaN or inf at seed {s}, skipping.")
                break

            seed_result["folds"].append({
                "val_cell": val_cell,
                "train_cells": train_cells,
                "val_loss": float(val_loss),
                "bic": compute_bic(len(U_val), val_loss, 3*N+1),
                "params": preprocess.log_to_exp(trained_params),
                "train_losses": train_losses,
                "val_losses": val_losses_progress,
                "params_progress": params_progress,
            })

        # Aggregate performance of this seed
        avg_val_loss = float(jnp.mean(jnp.array(val_losses)))
        seed_result["avg_val_loss"] = avg_val_loss
        history["seeds"].append(seed_result)

        # Track best
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            best_seed_overall = s
        
        history["best_seed"] = best_seed_overall

        # Save incrementally
        dir_path = os.path.join(work_dir, "output", model_name)
        os.makedirs(dir_path, exist_ok=True)
        with open(f"{dir_path}/history_{start_seed}.json", "w") as outfile:
            json.dump(load_data.clean_for_json(history), outfile)

    # save history
    # dir_path = os.path.join(work_dir,"output",model_name)
    # os.makedirs(dir_path, exist_ok=True)
    # with open(f"{dir_path}/history.json", "w") as outfile: 
    #     json.dump(load_data.clean_for_json(history), outfile)
    
    print(f"Saved history of training into {dir_path}/history.json")


    
def join_experiments(model_name, start_seed_indices=[]):
    work_dir = os.getcwd()
    dir_path = os.path.join(work_dir, "output", model_name)
    history = {"config": {}, "seeds": []}

    for s in start_seed_indices:
        with open(f"{dir_path}/history_{s}.json", "r") as infile:
            seed_history = json.load(infile)
            if not history["config"]:
                history["config"] = seed_history["config"]
            history["seeds"].extend(seed_history["seeds"])

    # Find the best seed
    best_seed = min(history["seeds"], key=lambda x: x["avg_val_loss"])
    history["best_seed"] = best_seed["seed"]
    print(f"Best seed overall: {history['best_seed']} with avg val loss {best_seed['avg_val_loss']:.4f}")

    with open(f"{dir_path}/history.json", "w") as outfile:
        json.dump(load_data.clean_for_json(history), outfile)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",help="Name of the experiment",default="default")
    parser.add_argument("-N","--number_blocks",help="Number of blocks" ,default=1)
    parser.add_argument("-i","--iters",help="Number of iterations",default=100)
    parser.add_argument("-f","--frequency",choices=["10","100","1000","10000","30000"],help="Chose the frequency",default="30000")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-s", "--sampling_frequency", help="Reduce sampling frequency for faster simulation",default=20)
    parser.add_argument("-rs","--random_seeds",help="Number of random seeds to explore",default=100)
    parser.add_argument("-ss","--start_seed",help="Seed index to start",default=0)
    parser.add_argument("-m", "--minibatch", action="store_true", help="minibatch the process")
    parser.add_argument("-o", "--optimizer", choices=["adam","adamw",], help="optimizer type",default="adam")
    parser.add_argument("-l", "--loss", choices=["MSE","RMSE","MAPE","CAE","CSE"], help="loss type",default="MSE")
    parser.add_argument("-cv", "--loco_cv", action="store_true", help="apply leave one cell out cross validation")

    args = parser.parse_args()


    model_name = f"{args.name}_{args.number_blocks}_blocks_{args.frequency}_hz"

    # Launch your hyperparameter search
    main(model_name, 
        N=int(args.number_blocks),
        iters=int(args.iters),
        freq=int(args.frequency), 
        debug=args.debug, 
        sampling_frequency=int(args.sampling_frequency),
        n_seeds = int(args.random_seeds),
        start_seed= int(args.start_seed),
        minibatch=args.minibatch,
        optimizer=args.optimizer,
        loss_type=args.loss,
        loco_cv=args.loco_cv
    )

    # If you want to join multiple experiments, uncomment the line below and specify the start seed indices
    # join_experiments(model_name, start_seed_indices=[0,25,50,75])