import argparse
import os
import json
import jax
import jax.numpy as jnp

import load_data
from models import *
import preprocess_data as preprocess
import train


def main(model_name, N, iters, freq, debug, sampling_frequency, n_seeds):
    work_dir = os.getcwd()
    el = 4000
    cells = ["U1","U2","U3","U4","U5","U6"]
    minibatch = True

    # get data
    data = load_data.load_data(os.path.join(work_dir, "data"), freq)
    fs = float(data["fs"])

    # decimate and correct offset
    I = preprocess.correct_signal(preprocess.decimate_signal(data["I"],fs,sampling_frequency))
    U_cells = [preprocess.decimate_signal(data[cell],fs,sampling_frequency) for cell in cells]

    if debug:
        U_cells = [cell[:el] for cell in U_cells]
        I= I[:el]

    # normalise
    I -= jnp.mean(I)
    U_cells = [cell - jnp.mean(cell) for cell in U_cells]

    # scale up
    I *= 200
    U_cells = [cell* 200 for cell in U_cells]

    # init parameters
    params = {
        'Rs':    jnp.array(jnp.log10(3e-3)),                   # initial supply resistance
        'R':     jnp.ones((N,)) * jnp.log10(5e-2) ,             # block resistances
        'C':     jnp.log(jnp.array([10.,100.,1000,500.,500.,500.])[:N]) ,            # block capacitances
        'alpha': jnp.ones((N,)) * 0.75,             # fractional factors
    }

    bad_seeds = set()
    history = {}

    # run training for each cell
    for i in range(len(U_cells)):
        cell = cells[i]

        # skip computing other cells for now
        if i>=1:
            continue

        print(f"Training on cell {cell}...")

        # setup U_val as all U cells except U1, when U_train is U_1 only
        U_train = U_cells[i]
        U_val = U_cells[:i] + U_cells[i+1:] if cell == "U1" else None

        # setup best seed tracking
        best_seed_loss = float('inf')
        best_seed_params = None
        best_seed_losses = []
        best_seed_val_losses = []
        best_seed = 0

        history[cell] = {}

        for s in range(n_seeds + 1):
            if s in bad_seeds:
                continue

            rng_key = jax.random.PRNGKey(s)
            p = params if s==0 else preprocess.sample_params(key=rng_key, N=N)

            # Pilot run
            pilot_loss = compute_loss(p, I, U_train, fs)
            if pilot_loss > best_seed_loss * 500:
                print(f"Seed {s} is worse than best seed loss by {pilot_loss // best_seed_loss} times: {pilot_loss:.4f} vs {best_seed_loss:.4f}, skipping.")
                bad_seeds.add(s)
                continue

            # full run
            # TODO: maybe send the best seed loss within the train loop and break if after x epochs we see still a big difference
            trained_params, losses, avg_val_losses = train.train_loop(p, I, U_train, fs, U_val, num_steps=iters,minibatch=minibatch)

            # compute metrics
            # y_pred = sim_z(I=I,fs=fs, **trained_params)
            # loss = jnp.mean(optax.squared_error(y_pred, U_train))
            loss = compute_loss(trained_params,I, U_train, fs)
            print(f"Seed {s}: Last training loss: {losses[-1]}, Loss: {loss}, best seed loss: {best_seed_loss}")

            # skip seeds if loss is 10* larger than the best loss
            if loss > best_seed_loss * 50:
                print(f"Seed {s} is worse than best seed loss by {loss // best_seed_loss} times, {loss:.4f} vs {best_seed_loss:.4f}, skipping.")
                bad_seeds.add(s)
                continue

            history[cell]["trainings"]={
                "seed": s,
                "params": trained_params,
                "loss":loss,
            }

            if loss < best_seed_loss:
                best_seed_loss = loss
                best_seed_params = trained_params
                best_seed_losses = losses
                best_seed = s
                if cell == "U1":
                    best_seed_val_losses = avg_val_losses

        bic = compute_bic(len(U_train),best_seed_loss,3*N+1)

        history[cell]["best_seed"] = {
            "losses": best_seed_losses,
            "params": best_seed_params,
            "loss": best_seed_loss,
            "seed": best_seed,
            "bic": bic
        }

        if cell == "U1":
            avg_loss = jnp.mean(jnp.array([compute_loss(trained_params,I, U_cell, fs)for U_cell in U_val]))
            history["avg_best_seed"] = {
                "losses": best_seed_val_losses,
                "loss":avg_loss,
                "bic": compute_bic(len(U_train),avg_loss,3*N+1),
                "aic": compute_aic(len(U_train),avg_loss,3*N+1)
            }


    history["config"]= {
        "model_name": model_name,
        "path": os.path.join(work_dir, "output"),
        "best_seed": best_seed,
        "freq":freq,
        "debug":debug,
        "sampling_frequency": sampling_frequency,
        "fs": fs,
        "N":N
    }

    # save history
    dir_path = os.path.join(work_dir,"output",model_name)
    os.makedirs(dir_path, exist_ok=True)

    with open(f"{dir_path}/history.json", "w") as outfile: 
        json.dump(load_data.clean_for_json(history), outfile)
    
    print(f"Saved history of training into {dir_path}/history.json")

    # compute metrics
    
    # print(f"\nTrained params: {trained_params}")
    # print(f"Final RMSE: {rmse:.4e},   CAE: {cae:.4f}")

    # best_y_pred = sim_z(I=I,fs=fs, **best_seed_params)


    # # --- plot loss curve ---
    # plt.figure(figsize=(6,4))
    # plt.plot(best_seed_losses, label="training loss")
    # plt.plot(history["avg_losses"], label="avg loss (all cells)")
    # # plt.yscale('log')
    # plt.xlabel("Step")
    # plt.ylabel("MSE Loss")
    # plt.title(f"Training Loss (N={N} blocks)")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("plots/loss.png")

    # # --- plot signals ---
    # plt.figure(figsize=(8,6))
    # plt.plot(U_cells[0], label="true")
    # plt.plot(best_y_pred, label="pred")
    # plt.xlim(0,min(el, len(U_train)))
    # plt.xlabel("Timestep")
    # plt.ylabel("Voltage")
    # plt.title(f"Simulated vs True (N={N} blocks)")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("plots/signals.png")
    # plt.close()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",help="Name of the experiment",default="default")
    parser.add_argument("-N","--number_blocks",help="Number of blocks" ,default=1)
    parser.add_argument("-i","--iters",help="Number of iterations",default=100)
    parser.add_argument("-f","--frequency",choices=["10","100","1000","10000","30000"],help="Chose the frequency",default="30000")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-s", "--sampling_frequency", help="Reduce sampling frequency for faster simulation",default=20)
    parser.add_argument("-rs","--random_seeds",help="Number of random seeds to explore",default=100)

    args = parser.parse_args()


    model_name = f"{args.name}_{args.number_blocks}_blocks_{args.frequency}_hz"

    # Launch your hyperparameter search
    main(model_name, 
        N=int(args.number_blocks),
        iters=int(args.iters),
        freq=int(args.frequency), 
        debug=args.debug, 
        sampling_frequency=int(args.sampling_frequency),
        n_seeds = int(args.random_seeds)
    )