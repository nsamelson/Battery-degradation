import os
import json
import pandas as pd
from ray import tune
import ray
from ray.train import Checkpoint, RunConfig
from ray.train.torch import TorchTrainer
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session, config, ScalingConfig



def custom_trial_dirname_creator(trial):
    # Create a shorter name for the trial directory
    return f"trial_{trial.trial_id}"



def main(model_name, dataset_path, num_samples=1 ):

    # create dirs and stuff
    work_dir = os.getcwd()
    storage_path = os.path.join(work_dir,"ray_results")
    tmp_dir = os.path.join(work_dir,"tmp")
    # trials_dir = os.path.join(storage_path, model_name)
    
    os.makedirs(tmp_dir, exist_ok=True)
    os.chmod(tmp_dir, 0o777)  # Adjust permissions as needed
    os.environ["RAY_TMPDIR"] = tmp_dir

    # parameters
    # max_num_epochs = 50
    grace_period = 2
    is_searching = True if num_samples > 1 else False

    search_space = {
        "model_name": model_name,
        "dataset_path": os.path.join(work_dir, dataset_path),
        # "num_epochs": max_num_epochs,
        "seed_value": 42,


        # search space       

    }

    hyperopt_search = HyperOptSearch(metric="val_acc", mode="max")

    # ASHA scheduler
    asha_scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_num_epochs,   # Maximum number of training iterations
        grace_period=grace_period,         # Number of iterations before considering early stopping
        reduction_factor=3,      # keeps the top 1/reduction_factor running, the rest is pruned
        brackets=2              # more brackets = more exploration in the parameters
    )
    

    # Early stopper
    stopper = TrialPlateauStopper(
        metric="val_loss",
        num_results=5,
        grace_period=grace_period,
        mode="min"
    )
    
    # running config
    run_config = RunConfig(
        name=model_name,
        storage_path=storage_path,
        failure_config=config.FailureConfig(max_failures=0),
        stop=stopper,
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
                tune.with_parameters(train_model),
                resources={"cpu": 4}
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
        best_result = results.get_best_result("val_acc", "max","all")
        # best_checkpoint = best_result.checkpoint

    else:
        ray.init(runtime_env={"env_vars": {"USE_LIBUV": "0"}}, configure_logging=False)
        scaling_config = ScalingConfig(num_workers=1, use_gpu=False, resources_per_worker={"CPU": 8})

        trainer = TorchTrainer(
            train_loop_per_worker=train_model,
            train_loop_config=search_space,
            scaling_config=scaling_config,
            run_config=run_config
        )

        best_result = trainer.fit()
        # best_checkpoint = best_result.get_best_checkpoint("val_loss", "min")

    best_checkpoint = best_result.checkpoint
    best_model_path = best_checkpoint.path


    print(f"Best trial config: {best_result.config}")
    print(f"Best trial validation accuracy: {best_result.metrics['val_acc']}")
    print(f"Best trial final validation loss: {best_result.metrics['val_loss']}")
    print(f"path is {best_model_path}")

    # save best model
    dir_path = os.path.join(work_dir,"saved_models",model_name)
    os.makedirs(dir_path, exist_ok=True)
    try:
        best_checkpoint.to_directory(dir_path)

        # get config
        with open(f"{dir_path}/config.json", "w") as outfile: 
            json.dump(best_result.config, outfile)
        
        # get metrics
        progress_path = os.path.join("/".join(best_model_path.split("/")[:-1]),"progress.csv")
        df = pd.read_csv(progress_path)

        df.dropna(how='all', inplace=True)
        useful_columns = ['loss', 'acc', 'val_loss', 'val_acc', 'training_iteration']

        df_cleaned = df[useful_columns]
        df_cleaned.to_csv(f"{dir_path}/progress.csv", index=False)
        

    except Exception as e:
        print(f"Couldn't save the model because of {e}")





if __name__=="__main__":
    model_name = "mnist_modular_search"
    dataset_path = "datasets/mnist/train.csv"

    main(model_name,dataset_path, 100)