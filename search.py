import os
import json
import pandas as pd
from ray import tune
import ray
from ray.train import Checkpoint, RunConfig
# from ray.train.torch import TorchTrainer
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session, config, ScalingConfig

import run_model

def custom_trial_dirname_creator(trial):
    # Create a shorter name for the trial directory
    return f"trial_{trial.trial_id}"



def main(model_name, num_samples=1, gpus_per_trial=float(1/4) ):

    # create dirs and stuff
    work_dir = os.getcwd()
    storage_path = os.path.join(work_dir,"ray_results")
    tmp_dir = os.path.join(work_dir,"tmp")
    # trials_dir = os.path.join(storage_path, model_name)
    
    os.makedirs(tmp_dir, exist_ok=True)
    os.chmod(tmp_dir, 0o777)  # Adjust permissions as needed
    os.environ["RAY_TMPDIR"] = tmp_dir

    is_searching = True if num_samples > 1 else False

    search_space = {
        "model_name": model_name,
        "path": os.path.join(work_dir, "data"),
        "seed_value": 42,
        "freq":30000,
        "debug":True,


        # search space       
        "Rs": tune.uniform(0.1,1000),
        "N": tune.randint(1,7),
        **{f"R_{i}": tune.uniform(0.5, 5.0) for i in range(6)},
        **{f"C_{i}": tune.loguniform(0.01, 10.0) for i in range(6)},
        **{f"alpha_{i}": tune.uniform(0.1, 1.0) for i in range(6)},

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
        failure_config=config.FailureConfig(max_failures=0),
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
        # best_checkpoint = best_result.checkpoint

    else:
        ray.init(runtime_env={"env_vars": {"USE_LIBUV": "0"}}, configure_logging=False)
        scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"CPU": 4,"gpu": gpus_per_trial})

        # trainer = TorchTrainer(
        #     train_loop_per_worker=run_model.run_model,
        #     train_loop_config=search_space,
        #     scaling_config=scaling_config,
        #     run_config=run_config
        # )

        # best_result = trainer.fit()
        # best_checkpoint = best_result.get_best_checkpoint("val_loss", "min")

    # best_checkpoint = best_result.checkpoint
    # best_model_path = best_checkpoint.path


    print(f"Best config: {best_result.config}")
    print(f"Best BIC : {best_result.metrics['bic']}")
    # print(f"path is {best_model_path}")
    print(best_result)

    # save best model
    # dir_path = os.path.join(work_dir,"saved_models",model_name)
    # os.makedirs(dir_path, exist_ok=True)
    # try:
    #     best_checkpoint.to_directory(dir_path)

    #     # get config
    #     with open(f"{dir_path}/config.json", "w") as outfile: 
    #         json.dump(best_result.config, outfile)
        
    #     # get metrics
    #     progress_path = os.path.join("/".join(best_model_path.split("/")[:-1]),"progress.csv")
    #     df = pd.read_csv(progress_path)

    #     df.dropna(how='all', inplace=True)
    #     useful_columns = ['loss', 'acc', 'val_loss', 'val_acc', 'training_iteration']

    #     df_cleaned = df[useful_columns]
    #     df_cleaned.to_csv(f"{dir_path}/progress.csv", index=False)
        

    # except Exception as e:
    #     print(f"Couldn't save the model because of {e}")





if __name__=="__main__":
    model_name = "bic_experiment"

    main(model_name, 20,0)  