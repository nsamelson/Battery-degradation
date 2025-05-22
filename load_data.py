import glob
import os
import h5py
import numpy as np
import jax.numpy as jnp
import json

def clean_for_json(obj):

    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, jnp.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # catches all np.float32, np.int64, etc.
        return obj.item()
    else:
        return obj
    

def load_data(path, freq):
    
    freq_str = f"{freq}hz"
    pattern = os.path.join(path, f"*{freq_str}*.mat")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No dataset file found for frequency: {freq_str}")
    
    with h5py.File(matches[0], 'r') as f:
        data = {
            "fs": f['fs'][()][0].squeeze().item(),
            "I": f['I'][()].squeeze(),
            **{f"U{i}": f[f'U{i}'][()].squeeze() for i in range(1,7)}
        }
    return data
