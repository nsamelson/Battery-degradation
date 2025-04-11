import argparse
import os
import json
# import pandas as pd
from ray import tune
import ray
from ray.tune import RunConfig, FailureConfig
# from ray.train.torch import TorchTrainer
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import multiprocessing

import run_model

def custom_trial_dirname_creator(trial):
    # Create a shorter name for the trial directory
    return f"trial_{trial.trial_id}"



def main(model_name, num_samples=1, gpus_per_trial=float(1/4),freq=30000,debug=False ):

    # create dirs and stuff
    work_dir = os.getcwd()
    storage_path = os.path.join(work_dir,"ray_results")
    # tmp_dir = os.path.join(work_dir,"tmp")
    tmp_dir = "/tmp/ray_tmp"  # much shorter path

    # trials_dir = os.path.join(storage_path, model_name)
    
    os.makedirs(tmp_dir, exist_ok=True)
    os.chmod(tmp_dir, 0o777)  # Adjust permissions as needed
    os.environ["RAY_TMPDIR"] = tmp_dir

    is_searching = True if num_samples > 1 else False

    search_space = {
        "model_name": model_name,
        "path": os.path.join(work_dir, "data"),
        "seed_value": 42,
        "freq":freq,
        "debug":debug,


        # search space       
        "Rs": tune.uniform(0.1,1000),
        "N": tune.randint(1,7),
        **{f"R_{i}": tune.uniform(0.5, 5.0) for i in range(6)},
        **{f"C_{i}": tune.loguniform(0.05, 10.0) for i in range(6)},
        **{f"alpha_{i}": tune.uniform(0.5, 1.0) for i in range(6)},

    }

    hyperopt_search = HyperOptSearch(metric="bic", mode="min")

    # ASHA scheduler
    asha_scheduler = ASHAScheduler(
        metric="bic",
        mode="min",
        max_t=1,   # Maximum number of training iterations
        # grace_period=1,         # Number of iterations before considering early stopping
        reduction_factor=3,      # keeps the top 1/reduction_factor running, the rest is pruned
        brackets=2              # more brackets = more exploration in the parameters
    )
    
    
    # running config
    run_config = RunConfig(
        name=model_name,
        storage_path=storage_path,
        failure_config=FailureConfig(max_failures=0),
    )

    # Restore or run a new tuning
    # if tune.Tuner.can_restore(trials_dir):
    #     tuner = tune.Tuner.restore(
    #         trials_dir, 
    #         trainable=tune.with_resources(
    #             tune.with_parameters(train_model),
    #             resources={"cpu": 8}
    #         ), 
    #         resume_errored=True,
    #         param_space= search_space
    #     )
    # else:

    if is_searching:
        tuner = tune.Tuner(
            trainable=tune.with_resources(
                tune.with_parameters(run_model.run_model),
                resources={"cpu": 4,"gpu": gpus_per_trial}
            ),
            tune_config=tune.TuneConfig(
                search_alg=hyperopt_search,  
                scheduler=asha_scheduler,  
                num_samples=num_samples,  # Adjust based on budget
                trial_dirname_creator= custom_trial_dirname_creator
            ),
            param_space=search_space,
            run_config=run_config
        )
        
        results = tuner.fit()
        best_result = results.get_best_result("bic", "min","last")



    print(f"Best config: {best_result.config}")
    print(f"Best BIC : {best_result.metrics['bic']}")
    # print(f"path is {best_model_path}")
    print(best_result)

    # save best model
    dir_path = os.path.join(work_dir,"output",model_name)
    os.makedirs(dir_path, exist_ok=True)
    try:
        # get config
        with open(f"{dir_path}/best_config.json", "w") as outfile: 
            json.dump(best_result.config, outfile)
        with open(f"{dir_path}/best_metrics.json", "w") as outfile: 
            json.dump(best_result.metrics, outfile)
        

    except Exception as e:
        print(f"Couldn't save the model because of {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",help="Name of the experiment",default="default")
    parser.add_argument("-s","--samples",help="Number of samples",default=100)
    parser.add_argument("-f","--frequency",choices=[10,100,1000,10000,30000],help="Chose the frequency",default=30000)
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-g", "--gpus", help="Number of GPUs, default is 1",default=0.25)

    args = parser.parse_args()

    # grab cpu and gpu from available info
    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", "8"))
    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) if "CUDA_VISIBLE_DEVICES" in os.environ else 0


    # Set multiprocessing start method first (critical)
    multiprocessing.set_start_method("spawn", force=True)
    # model_name = "param_search"

    # # Then initialize Ray 
    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        runtime_env={"env_vars": {
            "JAX_PLATFORM_NAME": "gpu",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",  # Optional: disables JAX CPU threading
            "OMP_NUM_THREADS": "1",  # For numpy/BLAS/OpenMP conflicts
            "MKL_NUM_THREADS": "1"
        }},
        # _plasma_directory="/tmp",  # Optional, good for tmpdir management
        ignore_reinit_error=True,
    )

    model_name = f"{args.name}_{args.frequency}_hz"

    # Launch your hyperparameter search
    main(model_name, num_samples=args.samples, gpus_per_trial=args.gpus, freq=args.frequency, debug=args.debug)