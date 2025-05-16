import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
from EarlyStopping import EarlyStopping
import run_model
import optax
import jax
import jax.numpy as jnp
from jax import jit
import vb_eis.state_space_sim as state_space_sim
import h5py
import scipy.signal as signal
from tqdm import tqdm

# jax.config.update("jax_debug_nans", True)

def clean_for_json(obj):

    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, jnp.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # catches all np.float32, np.int64, etc.
        return obj.item()
    else:
        return obj

@jax.jit
def sim_z(Rs, R, C, alpha,fs, I):
    A, bl, m, d, T_end = state_space_sim.jgen(10**Rs,10**R,10**C,alpha,fs,len(I))
    mask = state_space_sim.generate_mask(A.shape)
    x_init = np.zeros(A.shape)
    return state_space_sim.forward_sim(A, bl, m, d, jnp.array(x_init), I, mask)

@jax.jit
def compute_loss(params, y, U, fs):
    y_pred = sim_z(I=y,fs=fs, **params)
    # loss = jnp.sum(optax.squared_error(y_pred, U))
    # loss = jnp.sum(jnp.abs(y_pred - U))
    # loss = jnp.mean(jnp.abs(y_pred - U))
    loss = jnp.mean(optax.squared_error(y_pred, U))

    return loss


def make_optimizer(params, lr_res=1e-3, lr_alpha=1e-3, lr_cap=1e-2):
    warmup = 40
    decay = 200
    
    res_optim   = optax.adamw(learning_rate=optax.warmup_cosine_decay_schedule(0.,lr_res,warmup,decay,lr_res*0.1),
                                weight_decay=1e-4)
    alpha_optim = optax.adamw(learning_rate=optax.warmup_cosine_decay_schedule(0.,lr_alpha,warmup,decay,lr_alpha*0.1),
                                weight_decay=1e-4)
    cap_optim   = optax.adamw(learning_rate=optax.warmup_cosine_decay_schedule(0.,lr_cap,warmup,decay,lr_cap*0.1),
                                weight_decay=1e-4)

    res_mask   = {k: (k == 'Rs' or k == 'R')   for k in params}
    alpha_mask = {k: (k == 'alpha')            for k in params}
    cap_mask   = {k: (k == 'C' or k == 'Q')    for k in params}

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),   
        optax.masked(res_optim,   res_mask),
        optax.masked(alpha_optim, alpha_mask),
        optax.masked(cap_optim,   cap_mask),
    )
    # optimizer = optax.apply_if_finite(optimizer, max_consecutive_errors=1)

    return optimizer

def data_stream(signals, batch_size):
    for i in range(0, len(signals[0]), batch_size):
        yield (signals[j][i:i+batch_size] for j in range(len(signals)))

def step(params, opt_state, I, U_train, optimizer, fs, minibatch=True, U_val = None):

    # setup batches
    batches = data_stream([I, U_train], 2000) if minibatch else [(I, U_train)]

    tot_loss = 0.
    avg_val_loss = 0.
    for I_batch, U_batch in batches:
        # still differentiate w.r.t. params
        loss, grads = jax.value_and_grad(compute_loss)(params, I_batch, U_batch, fs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        tot_loss += loss

        # Clip parameters
        params['R']     = jnp.clip(params['R'],     a_min=-5., a_max=2.0)
        params['Rs']    = jnp.clip(params['Rs'],    a_min=-5., a_max=2.0)
        params['alpha'] = jnp.clip(params['alpha'], a_min=0.6,  a_max=1.0)
        params['C']     = jnp.clip(params['C'],     a_min=1.0,  a_max=4.0)

    
    # Simulate once for full val loss
    if U_val:
        y_pred_val = sim_z(I=I, fs=fs, **params)

        val_losses = [jnp.mean(optax.squared_error(y_pred_val, U_cell_val)) for U_cell_val in U_val]
        avg_val_loss = jnp.mean(jnp.array(val_losses))

    return params, opt_state, tot_loss, avg_val_loss


def train_loop(params, I, U_train, fs, U_val= None, num_steps=1000,):
    optimizer = make_optimizer(params)
    opt_state = optimizer.init(params)

    losses = []
    avg_val_losses = []
    early_stopper = EarlyStopping(patience=30, min_delta=1e-4)
    pbar = tqdm(range(num_steps), desc="Training")

    for _ in pbar:
        params, opt_state, loss, avg_val_loss = step(params, opt_state, I, U_train, optimizer, fs, U_val=U_val)   

        if not jnp.isfinite(loss):
            print("Loss is NaN or Inf — stopping...")
            break

        losses.append(loss.item())

        if U_val and avg_val_loss:
            avg_val_losses.append(avg_val_loss.item())

        # Early stop
        early_stopper(loss.item(), params)
        if early_stopper.should_stop:
            # print(f"Early stopping triggered after {early_stopper.patience} epochs without improvement.")
            break

        pbar.set_description(
            f"Train loss={loss:.4e}, Rs={10**params['Rs']:.5f},"
            f"R={[f'{10**r:.5f}' for r in params['R'].tolist()]}, "
            f"C={[f'{10**c:.2f}' for c in params['C'].tolist()]}, "
            f"a={[f'{a:.4f}' for a in params['alpha'].tolist()]}"
        )
            
    return early_stopper.best_params, losses, avg_val_losses


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
    return corr_signal

def sample_params(key, N):
    keys = jax.random.split(key, 4)

    Rs = jax.random.uniform(keys[0], (), minval=-5, maxval=2)
    R = jax.random.uniform(keys[1], (N,), minval=-5, maxval=2)
    alpha = jax.random.uniform(keys[2], (N,), minval=0.65, maxval=1.0)
    C = jax.random.uniform(keys[3], (N,), minval=1.0, maxval=4.0)

    return {
        'Rs': Rs,
        'R': R,
        'C': C,  # because your model expects log(C)
        'alpha': alpha
    }

def main(model_name, N, iters, freq, debug, sampling_frequency):
    work_dir = os.getcwd()
    el = 2000
    cells = ["U1","U2","U3","U4","U5","U6"]
    n_seeds = 5

    # get data
    data = load_data(os.path.join(work_dir, "data"), freq)
    fs = float(data["fs"])

    # decimate and correct offset
    I = correct_signal(decimate_signal(data["I"],fs,sampling_frequency))
    U_cells = [decimate_signal(data[cell],fs,sampling_frequency) for cell in cells]

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

        for s in range(n_seeds + 1):
            if s in bad_seeds:
                continue

            rng_key = jax.random.PRNGKey(s)
            p = params if s==0 else sample_params(key=rng_key, N=N)

            # Pilot run
            pilot_loss = compute_loss(p, I, U_train, fs)
            if pilot_loss > best_seed_loss * 100:
                print(f"Seed {s} is worse than best seed loss by 100 times {pilot_loss} vs {best_seed_loss}, skipping.")
                bad_seeds.add(s)
                continue

            # full run
            trained_params, losses, avg_val_losses = train_loop(p, I, U_train, fs, U_val, num_steps=iters,)

            # compute metrics
            y_pred = sim_z(I=I,fs=fs, **trained_params)
            loss = jnp.mean(optax.squared_error(y_pred, U_train))

            # skip seeds if loss is 10* larger than the best loss
            if loss > best_seed_loss * 10:
                print(f"Seed {s} is worse than best seed loss by 10 times, skipping.")
                bad_seeds.add(s)
                continue

            if loss < best_seed_loss:
                best_seed_loss = loss
                best_seed_params = trained_params
                best_seed_losses = losses
                best_seed_val_losses = avg_val_losses
                best_seed = s

        bic = run_model.compute_bic(len(U_train),best_seed_loss,3*N+1)

        history[cell] = {
            "losses": best_seed_losses,
            "best_params": best_seed_params,
            "best_loss": best_seed_loss,
            "bic": bic
        }

        if cell == "U1":
            history["avg_losses"] = best_seed_val_losses
            history["avg_best_loss"] = jnp.mean(jnp.array([jnp.sum(optax.squared_error(y_pred, U_cell)) for U_cell in U_val]))
            history["avg_bic"] = run_model.compute_bic(len(U_train),history["avg_best_loss"],3*N+1)
            history["avg_aic"] = run_model.compute_aic(len(U_train),history["avg_best_loss"],3*N+1)


    history["config"]= {
        "model_name": model_name,
        "path": os.path.join(work_dir, "data"),
        "best_seed": best_seed,
        "freq":freq,
        "debug":debug,
        "sampling_frequency": sampling_frequency,
        "fs": fs
    }

    # save history
    dir_path = os.path.join(work_dir,"output",model_name)
    os.makedirs(dir_path, exist_ok=True)

    with open(f"{dir_path}/history.json", "w") as outfile: 
        json.dump(clean_for_json(history), outfile)
    
    print(f"Saved history of training into {dir_path}/history.json")

    # compute metrics
    
    # print(f"\nTrained params: {trained_params}")
    # print(f"Final RMSE: {rmse:.4e},   CAE: {cae:.4f}")

    best_y_pred = sim_z(I=I,fs=fs, **best_seed_params)


    # --- plot loss curve ---
    plt.figure(figsize=(6,4))
    plt.plot(best_seed_losses, label="training loss")
    plt.plot(avg_val_losses, label="validation loss")
    # plt.yscale('log')
    plt.xlabel("Step")
    plt.ylabel("Squared–Error Loss")
    plt.title(f"Training Loss (N={N})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/loss.png")

    # --- plot signals ---
    plt.figure(figsize=(8,6))
    plt.plot(U_cells[0], label="true", linewidth=2)
    plt.plot(best_y_pred, label="pred", linestyle='--')
    plt.xlim(0,min(el, len(U_train)))
    plt.xlabel("Timestep")
    plt.ylabel("Voltage")
    plt.title(f"Simulated vs True (N={N})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/signals.png")
    plt.close()






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