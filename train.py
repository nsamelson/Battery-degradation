

import jax
import optax
import jax.numpy as jnp
from tqdm import tqdm

from EarlyStopping import EarlyStopping
from models import *

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

        if not jnp.isfinite(loss):
            print(f"Loss is {loss}, params are {params}")
            return None, opt_state, float('inf'), float('inf')

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        tot_loss += loss

        # Clip parameters
        params['R']     = jnp.clip(params['R'],     a_min=-4., a_max=2.0)
        params['Rs']    = jnp.clip(params['Rs'],    a_min=-4, a_max=2.0)
        params['alpha'] = jnp.clip(params['alpha'], a_min=0.6,  a_max=1.0)
        params['C']     = jnp.clip(params['C'],     a_min=0.0,  a_max=5.0)

    
    # Simulate once for full val loss
    if U_val:
        y_pred_val = sim_z(I=I, fs=fs, **params)

        val_losses = [jnp.mean(optax.squared_error(y_pred_val, U_cell_val)) for U_cell_val in U_val]
        avg_val_loss = jnp.mean(jnp.array(val_losses))

    return params, opt_state, tot_loss, avg_val_loss


def train_loop(params, I, U_train, fs, U_val= None, num_steps=1000,minibatch=True, opt_type="adam"):
    optimizer = make_optimizer(params,opt_type=opt_type)
    opt_state = optimizer.init(params)

    losses = []
    params_progress = {k: [] for k in params}
    avg_val_losses = []
    early_stopper = EarlyStopping(patience=30, min_delta=5e-3)
    pbar = tqdm(range(num_steps), desc="Training")

    for _ in pbar:
        params, opt_state, loss, avg_val_loss = step(params, opt_state, I, U_train, optimizer, fs,minibatch, U_val=U_val)   

        if not jnp.isfinite(loss):
            return early_stopper.best_params, losses, avg_val_losses, params_progress

        losses.append(loss.item())
        for k in params_progress:
            params_progress[k].append(params[k])

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
            
    return early_stopper.best_params, losses, avg_val_losses, params_progress

