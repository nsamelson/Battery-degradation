
import vb_eis.state_space_sim as state_space_sim
import numpy as np


def sim_z(Rs, R, C, alfa,fs, I):
    import jax.numpy as jnp
    A, bl, m, d, T_end = state_space_sim.jgen(Rs,R,C,alfa,fs,len(I))
    mask = state_space_sim.generate_mask(A.shape)
    x_init = jnp.zeros(A.shape)
    return state_space_sim.forward_sim(A, bl, m, d, x_init, I, mask)

def add_white_noise(data, noise_level, key):
    import jax
    return data + noise_level * jax.random.normal(key, data.shape)



def main(I, parameters:dict, apply_noise=False):
    import jax
    import jax.numpy as jnp
    # Set global seed

    fbs = parameters["fbs"]
    fss = parameters["fss"]

    Rs = parameters["Rs"]
    R = parameters["R"]
    C = parameters["C"]
    alpha = parameters["alpha"]

    # wraps functions with jit
    sim = jax.jit(sim_z)
    white_noise = jax.jit(add_white_noise)

    def simulate_single(fb, i):
        y = sim(Rs, R, C, alpha, fss, I)

        if apply_noise:
            key = jax.random.fold_in(jax.random.PRNGKey(42), i)
            y = white_noise(y, 0.01, key)
        return jnp.asarray(y, copy=True)

    responses = jax.vmap(simulate_single, in_axes=(0, 0))(fbs, jnp.arange(len(fbs)))

    return responses


if __name__ == "__main__":
    main()