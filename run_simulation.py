import jax
import vb_eis.state_space_sim as state_space_sim
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tqdm import trange, tqdm

# Set global seed
np.random.seed(42)
jax_key = jax.random.PRNGKey(42)



def estimate_pdf(u_cof, i_cof):
    """
    Implementation of equation (7) in the paper https://doi.org/10.1016/j.jpowsour.2014.06.110
    """
    s11 = np.nanmean(np.multiply(u_cof,np.conj(u_cof)),axis=1)
    s12 = np.nanmean(np.multiply(i_cof,np.conj(u_cof)),axis=1)
    s21 = np.nanmean(np.multiply(u_cof,np.conj(i_cof)),axis=1)
    s22 = np.nanmean(np.multiply(i_cof,np.conj(i_cof)),axis=1)
    
    rho = np.divide(np.divide(s12,np.sqrt(s11)),np.sqrt(s22))
    s1 = np.sqrt(s11)
    s2 = np.sqrt(s22)
    z = np.divide(np.multiply(np.conj(rho),s1),s2)
    
    return rho, s1, s2,z


def drbsgen(fs, fb, prbs_length):
    """
    DRBSGEN Random binary signal generator.
    
    DRBSGEN generates signal with:
    fs - sampling frequency
    fb - bandwith of the DRBS signal
    prbs_length - length of the signal in seconds
    seed - set seed for random number generator (in order to exactly recreate results) 
    """

    f_prbs = 3*fb
    N = int(np.around(fs/f_prbs, decimals=0))
    Ns = int(np.ceil((prbs_length*fs)/N))
    lb = int(np.ceil(prbs_length*fs))
    prbs = np.ones(int(lb))#*np.nan;
    
    for idx in range(1,Ns):
        x = np.around(np.random.uniform(0,1),decimals=0)
        if(x==0):
            x = 0
        prbs[((idx-1)*N+1):idx*N+1] = x
   
    t = np.arange(0,len(prbs))/fs
    t = t[0:lb]
    prbs = np.append(prbs[1:lb],prbs[-1])
    return prbs,t


def sim_z(Rs, R, C, alfa,fs, I):
    A, bl, m, d, T_end = state_space_sim.jgen(Rs,R,C,alfa,fs,len(I))
    mask = state_space_sim.generate_mask(A.shape)
    x_init = jnp.zeros(A.shape)
    return state_space_sim.forward_sim(A, bl, m, d, x_init, I, mask)

def add_white_noise(data, noise_level, key):
    return data + noise_level * jax.random.normal(key, data.shape)



def main(I, parameters:dict, apply_noise=False):

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
            y = add_white_noise(y, 0.01, jax.random.fold_in(jax_key, 100))

        responses.append(jnp.asarray(y, copy=True))

    return responses


if __name__ == "__main__":
    main()