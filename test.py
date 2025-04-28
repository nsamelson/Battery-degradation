import glob
import os
import h5py
import numpy as np
import pandas as pd

def model_diagnostics(y_true, y_pred, n_params):
    """
    Computes MSE, log-likelihood (assuming Gaussian errors),
    and two versions of BIC:
    - 'bic_exact': using full log-likelihood
    - 'bic_mse': simplified MSE-only version (up to constant)
    
    Parameters:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
        n_params (int): Number of fitted model parameters (k)
    
    Returns:
        dict: {
            'mse': float,
            'log_likelihood': float,
            'bic_exact': float,
            'bic_mse': float
        }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    
    # Mean Squared Error
    residuals = y_true - y_pred
    mse = np.mean(residuals**2)
    
    # Log-likelihood for Gaussian errors
    log_likelihood = -n / 2 * (np.log(2 * np.pi * mse) + 1)
    
    # Full BIC using log-likelihood
    bic_exact = -2 * log_likelihood + n_params * np.log(n)
    
    # Simplified BIC using only MSE (drops constant)
    bic_mse = n * np.log(mse) + n_params * np.log(n)
    
    return {
        'mse': mse,
        'log_likelihood': log_likelihood,
        'bic_exact': bic_exact,
        'bic_mse': bic_mse
    }


def compute_grouped_bic(csv_path):
    df = pd.read_csv(csv_path)

    results = []
    for N, group in df.groupby("N"):
        # n_trials = len(group)
        n_trials = 40
        mse_mean = group["mse"].mean()
        bic_mean = group["bic"].mean()

        # Option 1: Mean MSE
        bic_mean_mse = n_trials * np.log(group["mse"].min()) + N * np.log(n_trials)

        # Option 2: Avg BIC from individual trials
        bics = np.log(group["mse"]) + ((N*3+1) * np.log(n_trials)) / n_trials
        bic_avg = bics.mean()

        log_likelihood = -n_trials / 2 * (np.log(2 * np.pi * group["mse"].min()) + 1)
        bic_exact = -2 * log_likelihood + N * np.log(n_trials)

        results.append({
            "N": N,
            "n_trials": n_trials,
            "mean_mse": mse_mean,
            "bic_mean_mse": bic_mean_mse,
            "bic_avg_individual": bic_avg,
            "bic_exact":bic_exact,
            "log likelihood": log_likelihood
        })

    return pd.DataFrame(results)


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


def test_dataset():
    import jax.numpy as jnp


    data = load_matlab_data("data", 10)
    I = jnp.array(data.get("I"))[0]
    y_true = data.get("U4")

    return I, y_true


# Example usage:
if __name__ == "__main__":
    # y_true = [3.1, 2.9, 3.0, 3.2]
    # y_pred = [3.0, 3.0, 2.8, 3.3]
    # k = 3  # e.g., 2 parameters (intercept + slope)

    # results = model_diagnostics(y_true, y_pred, n_params=k)
    # for key, val in results.items():
    #     print(f"{key}: {val:.4f}")

    # result = compute_grouped_bic("output/bic_full_10_hz/results.csv")
    # print(result)

    x, y_true = test_dataset()

    print(x.shape, x.mean(), x.std(), x.min(), x.max())
    print(y_true.shape, y_true.mean(), y_true.std(), y_true.min(), y_true.max())