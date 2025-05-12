import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
import run_model
import optax
import jax
import jax.numpy as jnp
from jax import jit



def main(model_name, N, trials, freq, debug, sampling_frequency):

    work_dir = os.getcwd()

    search_space = {
        "model_name": model_name,
        "path": os.path.join(work_dir, "data"),
        "seed_value": 42,
        "freq":freq,
        "debug":debug,
        "sampling_frequency": sampling_frequency,
        # "target_fss":25000,
        # "sim_duration": 20.0,


        # search space       
        # "Rs": tune.loguniform(0.001, 10),
        # # "N": tune.randint(1,7),
        # "N": N,
        # **{f"R_{i}": tune.loguniform(1e-4, 10.0) for i in range(N)},
        # **{f"C_{i}": tune.loguniform(0.1, 1000.0) for i in range(N)},
        # **{f"alpha_{i}": tune.uniform(0.55, 1.0) for i in range(N)},

    }









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",help="Name of the experiment",default="default")
    parser.add_argument("-N","--number_blocks",help="Number of blocks" ,default=1)
    parser.add_argument("-t","--trials",help="Number of trials",default=100)
    parser.add_argument("-f","--frequency",choices=["10","100","1000","10000","30000"],help="Chose the frequency",default="30000")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-s", "--sampling_frequency", help="Reduce sampling frequency for faster simulation",default=20)

    args = parser.parse_args()


    model_name = f"{args.name}_{args.number_blocks}_blocks_{args.frequency}_hz"

    # Launch your hyperparameter search
    main(model_name, 
        N=int(args.number_blocks),
        trials=int(args.trials),
        freq=int(args.frequency), 
        debug=args.debug, 
        sampling_frequency=int(args.sampling_frequency),
    )