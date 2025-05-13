import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
import run_model
import optax
import jax
import jax.numpy as jnp
from jax import jit
import vb_eis.state_space_sim as state_space_sim
import h5py
import scipy.signal as signal
from tqdm import tqdm


@jax.jit
def sim_z(Rs, R, C, alpha,fs, I, init=0.):
    # print(Rs, R, C, alpha, fs, I, init)
    A, bl, m, d, T_end = state_space_sim.jgen(Rs,R,C,alpha,fs,len(I))
    mask = state_space_sim.generate_mask(A.shape)
    x_init = np.zeros(A.shape)
    x_init[0,:] = init
    return state_space_sim.forward_sim(A, bl, m, d, jnp.array(x_init), I, mask)

@jax.jit
def compute_loss(params, y, U):
    y_pred = sim_z(I=y, **params)
    loss = jnp.sum(optax.squared_error(y_pred, U))
    return loss

def step(params, opt_state, I, U, optimizer):
    loss, grads = jax.value_and_grad(compute_loss)(params, I, U)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Clip to pre-defined ranges
    params['R']   = jnp.clip(params['R'],   a_min=1e-5, a_max=100.0)
    params['Rs']  = jnp.clip(params['Rs'],  a_min=1e-5, a_max=100.0)
    params['alpha'] = jnp.clip(params['alpha'], a_min=0.55, a_max=1.0)
    params['C'] = jnp.clip(params['C'], a_min=0.1, a_max=1000.0)
    return params, opt_state, loss

def train_loop(params, I, y_true, num_steps=1000, lr=1e-3):
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    losses = []
    pbar = tqdm(range(num_steps), desc="Training")
    for _ in pbar:
        params, opt_state, loss = step(params, opt_state, I, y_true, optimizer)
        losses.append(loss.item())
        pbar.set_description(f"Loss={loss:.3e}")
    return params, losses


def load_data(path, freq):
    
    freq_str = f"{freq}hz"
    pattern = os.path.join(path, f"*{freq_str}*.mat")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No dataset file found for frequency: {freq_str}")
    
    with h5py.File(matches[0], 'r') as f:
        data = {
            "fs": f['fs'][()][0].squeeze().item(),
            "I": f['I'][()].squeeze(),
            **{f"U{i}": f[f'U{i}'][()].squeeze() for i in range(1,7)}
        }
    return data

def decimate_signal(orig_signal, fs, new_sampling_freq):
    factor = fs // new_sampling_freq
    return signal.decimate(orig_signal,int(factor),ftype='fir',n=20)[10:-10]

def correct_signal(orig_signal):
    corr_signal = orig_signal*(-1) * 50/.625
    return jnp.array(corr_signal)-np.mean(corr_signal)

def main(model_name, N, iters, freq, debug, sampling_frequency):
    work_dir = os.getcwd()
    key = 42
    el = 2000
    rng = jax.random.PRNGKey(key)

    # get data
    data = load_data(os.path.join(work_dir, "data"), freq)
    fs = data["fs"]

    # decimate and correct offset
    I = correct_signal(decimate_signal(data["I"],fs,sampling_frequency))[:el]
    U = decimate_signal(data["U1"],fs,sampling_frequency)[:el]

    # init parameters
    params = {
        'Rs':    jnp.array(3e-3),                   # initial supply resistance
        'R':     jnp.ones((N,)) * 1e-2,             # block resistances
        'C':     jnp.ones((N,)) * 10.0,            # block capacitances
        'alpha': jnp.ones((N,)) * 0.75,             # fractional factors
        'fs':    float(sampling_frequency),
    }
    params = {'R': jnp.array([3e-3, 5e-3, 1e-2]), 'Rs':jnp.array(3e-3), 'alpha':jnp.array([.75, .75, .75]), 'C':jnp.array([10., 100., 1000.]),'fs':float(sampling_frequency)}


    # run training
    trained_params, losses = train_loop(params, I, U, num_steps=iters, lr=1e-2)

    # final simulation
    y_pred = sim_z(I=I, **trained_params)

    config = {
        "model_name": model_name,
        "path": os.path.join(work_dir, "data"),
        "seed_value": key,
        "freq":freq,
        "debug":debug,
        "sampling_frequency": sampling_frequency,
    }

    # compute metrics
    rmse  = run_model.root_mean_squared_error(U, y_pred)
    cae   = run_model.cumulative_absolute_error(U, y_pred)
    print(f"\nTrained params: {trained_params}")
    print(f"Final RMSE: {rmse:.4d},   CAE: {cae:.4d}")


    # --- plot loss curve ---
    plt.figure(figsize=(6,4))
    plt.plot(losses, label="training loss")
    plt.yscale('log')
    plt.xlabel("Step")
    plt.ylabel("Squared–Error Loss")
    plt.title(f"Training Loss (N={N})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/loss.png")

    # --- plot signals ---
    plt.figure(figsize=(8,4))
    plt.plot(U - np.mean(U), label="true", linewidth=2)
    plt.plot(y_pred, label="pred", linestyle='--')
    plt.xlabel("Timestep")
    plt.xlim(0,min(el, len(U)))
    plt.ylabel("Voltage")
    plt.title(f"Simulated vs True (N={N})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/signals.png")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",help="Name of the experiment",default="default")
    parser.add_argument("-N","--number_blocks",help="Number of blocks" ,default=1)
    parser.add_argument("-i","--iters",help="Number of iterations",default=100)
    parser.add_argument("-f","--frequency",choices=["10","100","1000","10000","30000"],help="Chose the frequency",default="30000")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-s", "--sampling_frequency", help="Reduce sampling frequency for faster simulation",default=20)

    args = parser.parse_args()


    model_name = f"{args.name}_{args.number_blocks}_blocks_{args.frequency}_hz"

    # Launch your hyperparameter search
    main(model_name, 
        N=int(args.number_blocks),
        iters=int(args.iters),
        freq=int(args.frequency), 
        debug=args.debug, 
        sampling_frequency=int(args.sampling_frequency),
    )