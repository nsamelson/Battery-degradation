import jax
from scipy import signal


def decimate_signal(orig_signal, fs, new_sampling_freq):
    factor = fs // new_sampling_freq
    return signal.decimate(orig_signal,int(factor),ftype='fir',n=20)[10:-10]

def correct_signal(orig_signal):
    corr_signal = orig_signal*(-1) * 50/.625
    return corr_signal

def sample_params(key, N):
    keys = jax.random.split(key, 4)

    Rs = jax.random.uniform(keys[0], (), minval=-5, maxval=2)
    R = jax.random.uniform(keys[1], (N,), minval=-5, maxval=2)
    alpha = jax.random.uniform(keys[2], (N,), minval=0.65, maxval=1.0)
    C = jax.random.uniform(keys[3], (N,), minval=1.0, maxval=4.0)

    return {
        'Rs': Rs,
        'R': R,
        'C': C,  # because your model expects log(C)
        'alpha': alpha
    }

def log_to_exp(params, selected=["R","Rs","C"]):
    out_params = {}

    for key, val in params.items():
        if key in selected:
            try:
                out_params[key] = [10**param for param in val.tolist()]
            except:
                out_params[key] = 10**val
        else:
            out_params[key] = val

    return out_params

