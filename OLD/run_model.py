import argparse
import glob
import math
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import random
import numpy as np
from sklearn.linear_model import LinearRegression
# import jax.numpy as jnp
import h5py
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from ray.air import session

from ray import tune


import simulation



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

    I_trimmed = I[:total_samples]
    y_trimmed = y[0, :total_samples]

    I_down = I_trimmed[::downsample_ratio]
    y_down = y_trimmed[::downsample_ratio]

    # Reshape to keep it consistent with original shape
    return I_down, jnp.reshape(y_down, (1, -1))


from collections import defaultdict

def test_model(config: dict, cells=["U1", "U2", "U3", "U5", "U6"]):
    performances = {}
    totals = defaultdict(float)
    N = config["N"]

    for cell in cells:
        metrics, n = run_model(config, is_searching=False, cell=cell)

        metrics = {
            # "mse": metrics['mse'],
            # "mle": metrics["mle"],
            "rmse": metrics["rmse"],
            "nrmse": metrics["nrmse"],
            "cae": metrics["cae"],
            "bic_rmse": compute_bic(n, metrics['rmse'], 3 * N + 1),
            "bic_cae": compute_bic(n, metrics['cae'], 3 * N + 1),
            "bic_nrmse": compute_bic(n, metrics['nrmse'], 3 * N + 1),
            # "bic": compute_bic(n, metrics['mse'], 3 * N + 1),
            # "aic": compute_aic(n, metrics['mse'], 3 * N + 1),
            # "bic_exact": compute_exact_bic(n, metrics['mle'], 3 * N + 1),
        }

        performances[cell] = metrics

        # sum the BICs and AICs 
        for key in ["bic_rmse", "bic_cae", "bic_nrmse"]:
            totals[key] += metrics[key]

    performances["total"] = dict(totals)
    return performances



def run_model(config:dict, is_searching=True, verbose=False,cell="U4"):
    import jax
    import jax.numpy as jnp

    debug = config["debug"]
    freq = config["freq"]
    N = config["N"]
    
    # Load real data
    data = load_matlab_data(config["path"], freq)
    I = jnp.array(data.get("I"))[0]
    y_true = data.get(cell)

    params = {
        "fbs": np.array([freq]),                                        # bandwidth freq
        "durations": data.get("duration")[0],           
        "fss": data.get("fs")[0],                                       # sampling freq
        "Rs": jnp.array(config["Rs"]),                                  # Resistance of supply?
        "R": jnp.array([config[f"R_{i}"] for i in range(N)]),           # Resistance
        "C": jnp.array([config[f"C_{i}"] for i in range(N)]),           # Capacitance
        "alpha": jnp.array([config[f"alpha_{i}"] for i in range(N)])    # Fractional factor
    }



    if config.get("reduce_sampling_factor",1) != 1:
        # duration = config.get("sim_duration", 20.0)  # seconds
        target_fss = params["fss"] // config["reduce_sampling_factor"]

        # for the time being, keep same
        I, y_true = reduce_sampling(I, y_true, data["fs"][0], target_fss, params["durations"])

        # Update fss and durations in params to reflect downsampling
        params["fss"] = np.array([target_fss])
        # params["durations"] = np.array([duration])


    if verbose:
        print(params)
        print(I.shape, y_true.shape)

    if debug:
        sample_size = 50000
        I = I[0:sample_size]
        y_true = y_true[:,0:sample_size]

    # Correct signals
    i_corr = I* (-1) * 50/.625 
    y_true[0] -= np.mean(y_true[0])

    # run simulation
    y_pred = simulation.main(i_corr - np.mean(i_corr) ,params,apply_noise=False)
    n = y_true.shape[1] 

    # Compute error
    if np.isnan(y_pred).any():
        rmse = float("inf")
        nrmse = float("inf")
        cae = float("inf")
    else:       
        rmse = root_mean_squared_error(y_true,y_pred)
        nrmse = rmse / (y_true[0].max() - y_true[0].min())
        cae = cumulative_absolute_error(y_true,y_pred)

    # metrics={"mse":mse, "mle":mle, "rmse":rmse, "nrmse":nrmse}
    metrics={"cae":cae, "rmse":rmse, "nrmse":nrmse}

    if is_searching:
        try:
            tune.report(metrics=metrics)
        finally:
            import jax
            try:
                jax.clear_backends()
            except:
                print("parameters lead to system unstable")

    return metrics, n


def main(model_name, freq=30000, debug=False):
    work_dir = os.getcwd()

    config = {
        "model_name": model_name,
        "path": os.path.join(work_dir, "data"),
        "seed_value": 42,
        "freq":freq,
        "debug":debug,
        "reduce_sampling_factor": True,
        # "target_fss":25000,
        # "sim_duration": 20.0,


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
    
