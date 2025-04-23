
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
    np.random.seed(42)
    jax_key = jax.random.PRNGKey(42)

    fbs = parameters["fbs"]
    fss = parameters["fss"]

    Rs = parameters["Rs"]
    R = parameters["R"]
    C = parameters["C"]
    alpha = parameters["alpha"]

    # Output parameters
    responses = []

    for i in enumerate(fbs):

        # generate output response
        y = sim_z(Rs, R, C, alpha, fss, I)

        # add noise
        if apply_noise:
            y = add_white_noise(y, 0.05, jax.random.fold_in(jax_key, 100))

        responses.append(jnp.asarray(y, copy=True))

    return responses


if __name__ == "__main__":
    main()