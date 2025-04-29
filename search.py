import argparse
import os
import json
import ray
from ray import tune
from ray.tune import RunConfig, FailureConfig

from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

# Init ray before jax
ray.init(ignore_reinit_error=True, num_cpus=32, num_gpus=1)

import run_model

def custom_trial_dirname_creator(trial):
    # Create a shorter name for the trial directory
    return f"trial_{trial.trial_id}"



def main(model_name, num_samples=1, N=1, gpus_per_trial=float(1/4),freq=30000,debug=False, cpus=32, reduce_sampling=False ):

    # create dirs and stuff
    work_dir = os.getcwd()
    storage_path = os.path.join(work_dir,"ray_results")
    tmp_dir = "/tmp/ray_tmp"  # much shorter path

    trials_dir = os.path.join(storage_path, model_name)
    
    os.makedirs(tmp_dir, exist_ok=True)
    os.chmod(tmp_dir, 0o777)  # Adjust permissions as needed
    os.environ["RAY_TMPDIR"] = tmp_dir

    search_space = {
        "model_name": model_name,
        "path": os.path.join(work_dir, "data"),
        "seed_value": 42,
        "freq":freq,
        "debug":debug,
        "reduce_sampling": reduce_sampling,
        "target_fss":25000,
        "sim_duration": 20.0,


        # search space       
        "Rs": tune.uniform(0.5, 3.5),
        # "N": tune.randint(1,7),
        "N": N,
        **{f"R_{i}": tune.loguniform(0.01, 100.0) for i in range(N)},
        **{f"C_{i}": tune.loguniform(0.01, 100.0) for i in range(N)},
        **{f"alpha_{i}": tune.loguniform(0.1, 1.0) for i in range(N)},

    }

    hyperopt_search = HyperOptSearch(metric="nrmse", mode="min")

    asha_scheduler = ASHAScheduler(
        metric="nrmse",
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

    trainable = tune.with_resources(
        tune.with_parameters(run_model.run_model),
        resources={"cpu": cpus, "gpu": gpus_per_trial}
    )

    # Restore or run a new tuning
    if tune.Tuner.can_restore(trials_dir):
        tuner = tune.Tuner.restore(
            trials_dir, 
            trainable=trainable,
            resume_errored=True,
            param_space= search_space
        )
    else:

        tuner = tune.Tuner(
            trainable=trainable,
            tune_config=tune.TuneConfig(
                search_alg=hyperopt_search,  
                scheduler=asha_scheduler, 
                num_samples=num_samples,  # Adjust based on budget
                max_concurrent_trials= 8,
                trial_dirname_creator= custom_trial_dirname_creator
            ),
            param_space=search_space,
            run_config=run_config
        )
        
    results = tuner.fit()
    best_result = results.get_best_result("nrmse", "min","last")
    print(f"Best config: {best_result.config}")


    test_metrics = run_model.test_model(best_result.config)
    print(test_metrics)

    # save best model
    dir_path = os.path.join(work_dir,"output",model_name)
    os.makedirs(dir_path, exist_ok=True)
    try:
        # get config
        with open(f"{dir_path}/best_config.json", "w") as outfile: 
            json.dump(best_result.config, outfile)
        with open(f"{dir_path}/train_metrics.json", "w") as outfile: 
            json.dump(best_result.metrics, outfile)
        with open(f"{dir_path}/test_metrics.json", "w") as outfile: 
            json.dump(test_metrics, outfile)
        

    except Exception as e:
        print(f"Couldn't save the model because of {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",help="Name of the experiment",default="default")
    parser.add_argument("-N","--number_blocks",help="Number of blocks" ,default=1)
    parser.add_argument("-s","--samples",help="Number of samples",default=100)
    parser.add_argument("-f","--frequency",choices=["10","100","1000","10000","30000"],help="Chose the frequency",default="30000")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-g", "--gpus", help="Number of GPUs, default is 1",default=0.25)
    parser.add_argument("-c", "--cpus", help="Number of CPUs, default is 16",default=4)
    parser.add_argument("-r", "--reduce_sampling", help="Reduce sampling frequency for faster simulation",default=False)

    args = parser.parse_args()


    model_name = f"{args.name}_{args.number_blocks}_blocks_{args.frequency}_hz"

    # Launch your hyperparameter search
    main(model_name, 
        num_samples=int(args.samples), 
        N=int(args.number_blocks),
        gpus_per_trial=float(args.gpus), 
        freq=int(args.frequency), 
        debug=args.debug, 
        cpus=int(args.cpus), 
        reduce_sampling=bool(args.reduce_sampling)
    )