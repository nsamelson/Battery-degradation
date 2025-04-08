import numpy as np
from sklearn.linear_model import LinearRegression
import jax.numpy as jnp
import h5py
from sklearn.metrics import mean_squared_error

import run_simulation

    # Simulation frequency parameters
    # fbs = np.array([1]) # bandwidths of the DRBS signal
    # durations = [10]    # duration of the sampling (s)
    # fss = [2000]         # sampling frequency




# MATLAB loader function
def load_matlab_data(filename):
    with h5py.File(filename, 'r') as f:
        data = {key: f[key][()] for key in f.keys()}
    return data

def compute_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic


def run_model(config:dict):
    debug = config["debug"]
    freq = config["freq"]
    N = config["N"]

    # Load real data
    data = load_matlab_data(f"data/data_prbs_{freq}hz_1000000_20190207_013502.mat")
    I = jnp.array(data.get("I"))[0]
    y_true = data.get("U4")


    if debug:
        sample_size = 1000
        I = I[0:sample_size]
        y_true = y_true[:,0:sample_size]

    params = {
        "fbs": np.array([freq]),                                        # bandwidth freq
        "durations": data.get("duration")[0],           
        "fss": data.get("fs")[0],                                       # sampling freq
        "Rs": jnp.array(config["Rs"]),                                  # Resistance of supply?
        "R": jnp.array([config[f"R_{i}"] for i in range(N)]),           # Resistance
        "C": jnp.array([config[f"C_{i}"] for i in range(N)]),           # Capacitance
        "alpha": jnp.array([config[f"alpha_{i}"] for i in range(N)])    # Fractional factor
    }

    # run simulation
    y_pred = run_simulation.main(I,params,apply_noise=True)

    # compute metrics
    n = len(y_true)   
    mse = mean_squared_error(y_true, y_pred)
    bic = compute_bic(n,mse, N) # maybe N*3+1 to count the real number of parameters?

    return bic


def main(freq=30000, debug=False):

    # Load real data
    data = load_matlab_data(f"data/data_prbs_{freq}hz_1000000_20190207_013502.mat")
    I = jnp.array(data.get("I"))[0]
    y_true = data.get("U4")
    durations = data.get("duration")[0]
    fss = data.get("fs")[0]

    if debug:
        sample_size = 1000
        I = I[0:sample_size]
        y_true = y_true[:,0:sample_size]

    # bandwidth freq
    fbs = np.array([freq])

    # Electrical circuit blocks
    Rs = jnp.array(1.5)                  # Resistance of supply?
    R = jnp.array([1., 2.])             # Resistance
    C = jnp.array([.1, 3.])            # Capacitance
    alpha = jnp.array([0.88, 0.92])    # Fractional factor linked with the capacitance

    params = {
        "fbs": fbs,
        "durations": durations,
        "fss": fss,
        "Rs": Rs,
        "R": R,
        "C": C,
        "alpha": alpha
    }

    # run simulation
    y_pred = run_simulation.main(I,params,apply_noise=True)

    # compute metrics
    n = len(y_true)   
    mse = mean_squared_error(y_true, y_pred)
    num_params = 1 + len(R) + len(C) + len(alpha)
    bic = compute_bic(n,mse,num_params)

    # print(bic, mse, num_params)

    # # Save file
    # np.savez("output/signals.npz",x=I, y_true=y_true, y_pred=y_pred, params= params, allow_pickle=True)
    
    return bic


if __name__ == "__main__":
    bic = main(debug=True)