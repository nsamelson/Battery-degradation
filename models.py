import jax
import numpy as np
import jax.numpy as jnp
import optax

from vb_eis import state_space_sim

def compute_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic

def compute_aic(n, mse, num_params):
    aic = n * np.log(mse) + num_params * 2
    return aic

def compute_exact_bic(n, mle, num_params):
    bic_exact = -2 * mle + num_params * np.log(n)
    return bic_exact

def compute_log_likelihood(mse, n):
    return -0.5 * n * (np.log(2 * np.pi * mse) + 1)

def cumulative_absolute_error(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    # return np.cumsum(errors)[-1]# this is really stupid
    return np.sum(errors)

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


def make_optimizer(params, lr_res=1e-2, lr_alpha=5e-4, lr_cap=1e-2):
    warmup = 40
    decay = 200

    res_optim = optax.adam(lr_res)
    alpha_optim = optax.adam(lr_alpha)
    cap_optim = optax.adam(lr_cap)
    
    # res_optim   = optax.adamw(learning_rate=optax.warmup_cosine_decay_schedule(0.,lr_res,warmup,decay,lr_res*0.1),
    #                             weight_decay=1e-4)
    # alpha_optim = optax.adamw(learning_rate=optax.warmup_cosine_decay_schedule(0.,lr_alpha,warmup,decay,lr_alpha*0.1),
    #                             weight_decay=1e-4)
    # cap_optim   = optax.adamw(learning_rate=optax.warmup_cosine_decay_schedule(0.,lr_cap,warmup,decay,lr_cap*0.1),
    #                             weight_decay=1e-4)

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