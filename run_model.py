import argparse
import glob
import math
import os
import random
import numpy as np
from sklearn.linear_model import LinearRegression
import jax.numpy as jnp
import h5py
from sklearn.metrics import mean_squared_error
from ray.air import session


import run_simulation

    # Simulation frequency parameters
    # fbs = np.array([1]) # bandwidths of the DRBS signal
    # durations = [10]    # duration of the sampling (s)
    # fss = [2000]         # sampling frequency




# MATLAB loader function
def load_matlab_data(path, freq):
    # data_path = os.path.join(, f"data_prbs_{freq}hz_1000000_20190207_013502.mat")
    freq_str = f"{freq}hz"
    pattern = os.path.join(path, f"*{freq_str}*.mat")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No dataset file found for frequency: {freq_str}")
    elif len(matches) > 1:
        print(f"Warning: Multiple files match {freq_str}, using the first one:\n{matches}")


    with h5py.File(matches[0], 'r') as f:
        data = {key: f[key][()] for key in f.keys()}
    return data

def compute_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic


def reduce_sampling(I, y, original_fss, target_fss, duration_s):
    """
    Downsamples and trims the signal to a shorter duration.

    Args:
        I (np.ndarray or jax array): input signal (1D or 2D shape [1, N])
        y (np.ndarray or jax array): output signal (same shape as I)
        original_fss (float): original sampling frequency
        target_fss (float): desired sampling frequency
        duration_s (float): desired duration in seconds

    Returns:
        (I_down, y_down): downsampled and truncated input/output
    """
    import jax.numpy as jnp

    downsample_ratio = int(original_fss[0] // target_fss)
    total_samples = int(duration_s * original_fss[0])
    new_length = int(duration_s * target_fss)

    I_trimmed = I[:total_samples]
    y_trimmed = y[0, :total_samples]

    I_down = I_trimmed[::downsample_ratio]
    y_down = y_trimmed[::downsample_ratio]

    # Reshape to keep it consistent with original shape
    return I_down, jnp.reshape(y_down, (1, -1))



def run_model(config:dict, is_searching=True, verbose=False):
    debug = config["debug"]
    freq = config["freq"]
    N = config["N"]
    
    # Load real data
    data = load_matlab_data(config["path"], freq)
    I = jnp.array(data.get("I"))[0]
    y_true = data.get("U4")



    params = {
        "fbs": np.array([freq]),                                        # bandwidth freq
        "durations": data.get("duration")[0],           
        "fss": data.get("fs")[0],                                       # sampling freq
        "Rs": jnp.array(config["Rs"]),                                  # Resistance of supply?
        "R": jnp.array([config[f"R_{i}"] for i in range(N)]),           # Resistance
        "C": jnp.array([config[f"C_{i}"] for i in range(N)]),           # Capacitance
        "alpha": jnp.array([config[f"alpha_{i}"] for i in range(N)])    # Fractional factor
    }


    if config.get("reduce_sampling",False):
        original_fss = data["fs"][0]  # 500000
        target_fss = config.get("target_fss", 25000)
        duration = config.get("sim_duration", 20.0)  # seconds
        I, y_true = reduce_sampling(I, y_true, original_fss, target_fss, duration)
        # Update fss and durations in params to reflect downsampling
        params["fss"] = np.array([target_fss])
        params["durations"] = np.array([duration])


    if verbose:
        print(params)
        print(I.shape, y_true.shape)

    if debug:
        sample_size = 10000
        I = I[0:sample_size]
        y_true = y_true[:,0:sample_size]


    # run simulation
    y_pred = run_simulation.main(I,params,apply_noise=True)

    if np.isnan(y_pred).any():
        bic = float("inf")
        session.report(metrics={"bic":bic,"mse":float("inf")}) # Penalize the trial heavily
        return bic

    # compute metrics
    n = len(y_true)   
    mse = mean_squared_error(y_true, y_pred)
    bic = compute_bic(n,mse, N*3+1) # maybe N*3+1 to count the real number of parameters?

    if is_searching:
        session.report(metrics={"bic":bic,"mse":mse})

    return bic


def main(model_name, freq=30000, debug=False):
    work_dir = os.getcwd()

    config = {
        "model_name": model_name,
        "path": os.path.join(work_dir, "data"),
        "seed_value": 42,
        "freq":freq,
        "debug":debug,
        "reduce_sampling": True,
        "target_fss":25000,
        "sim_duration": 20.0,


        # search space       
        "Rs": 0.2,
        "N": 3,
        **{f"R_{i}": random.uniform(0.5, 5.0) for i in range(3)},
        **{f"C_{i}": random.uniform(0.05, 1.0) for i in range(3)},
        **{f"alpha_{i}": random.uniform(0.5, 1.0) for i in range(3)},
    }

    bic = run_model(config, False, False)
    print(bic)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",help="Name of the experiment",default="default")
    parser.add_argument("-f","--frequency",choices=["10","100","1000","10000","30000"],help="Chose the frequency",default="10")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")

    args = parser.parse_args()

    random.seed(42)
    

    model_name = f"{args.name}_{args.frequency}_hz"

    main(model_name, freq=int(args.frequency), debug=args.debug)
    
