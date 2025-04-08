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

def main():

    # Load real data
    data = load_matlab_data("data/data_prbs_30000hz_1000000_20190207_013502.mat")
    I = jnp.array(data.get("I"))[0]
    y_true = data.get("U4")
    durations = data.get("duration")[0]
    fss = data.get("fs")[0]

    # bandwidth freq
    fbs = np.array([30000])

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

    print(bic, mse, num_params)

    # Save file
    np.savez("output/signals.npz",x=I, y_true=y_true, y_pred=y_pred, params= params, allow_pickle=True)




if __name__ == "__main__":
    main()