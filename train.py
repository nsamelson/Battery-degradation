

import jax
import optax
import jax.numpy as jnp
from tqdm import tqdm

from EarlyStopping import EarlyStopping
from models import *

def data_stream(signals, batch_size):
    for i in range(0, len(signals[0]), batch_size):
        yield tuple(signals[j][i:i+batch_size] for j in range(len(signals)))

def step(params, opt_state, I, U_train, optimizer, fs, minibatch=True, U_val = None, loss_code =0):
    cell_losses = []

    for U_cell_train in U_train:
        train_loss  = 0.
        batches = data_stream([I, U_cell_train], 2000) if minibatch else [(I, U_cell_train)]

        for I_batch, U_batch in batches:
            # still differentiate w.r.t. params
            loss, grads = jax.value_and_grad(compute_loss)(params, I_batch, U_batch, fs, loss_code )

            if not jnp.isfinite(loss):
                print(f"Loss is {loss}, params are {params}")
                return None, opt_state, float('inf'), float('inf')

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            train_loss  += loss

            # Clip parameters
            params['R']     = jnp.clip(params['R'],     a_min=-4., a_max=2.0)
            params['Rs']    = jnp.clip(params['Rs'],    a_min=-4, a_max=2.0)
            params['alpha'] = jnp.clip(params['alpha'], a_min=0.6,  a_max=1.0)
            params['C']     = jnp.clip(params['C'],     a_min=0.0,  a_max=5.0)

        cell_losses.append(train_loss)

    # average across cells
    avg_train_loss = jnp.mean(jnp.array(cell_losses))

    # Simulate once for val loss
    val_loss = compute_loss(params, I, U_val, fs, loss_code =loss_code )

    return params, opt_state, avg_train_loss , val_loss


def train_loop(params, I, U_train, fs, U_val, num_steps=1000,minibatch=True, opt_type="adam", loss_code =0):
    optimizer = make_optimizer(params,opt_type=opt_type)
    early_stopper = EarlyStopping(patience=15, min_delta=0.001, relative=True)
    opt_state = optimizer.init(params)

    losses = []
    val_losses = []
    params_progress = {k: [] for k in params}

    pbar = tqdm(range(num_steps), desc="Training")
    for _ in pbar:
        params, opt_state, loss, val_loss = step(params, opt_state, I, U_train, optimizer, fs,minibatch, U_val, loss_code =loss_code )   

        if not jnp.isfinite(loss):
            return early_stopper.best_params, losses, val_losses, params_progress

        # add losses
        losses.append(loss.item())
        val_losses.append(val_loss.item())

        for k in params_progress:
            params_progress[k].append(params[k])

        # Early stop
        early_stopper(val_loss.item(), params)
        if early_stopper.should_stop:
            break

        pbar.set_description(
            f"Train loss={loss:.4e}, Rs={10**params['Rs']:.5f},"
            f"R={[f'{10**r:.5f}' for r in params['R'].tolist()]}, "
            f"C={[f'{10**c:.2f}' for c in params['C'].tolist()]}, "
            f"a={[f'{a:.4f}' for a in params['alpha'].tolist()]}"
        )
            
    return early_stopper.best_params, losses, val_losses, params_progress

