#!/bin/bash

#SBATCH --job-name=optax
#SBATCH --error=logs/err_%A_%a.err           # %A = master job ID, %a = array index
#SBATCH --output=logs/log_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=24:10:00
#SBATCH --gpus=1
#SBATCH --partition=e8
#SBATCH --cpus-per-task=16
#SBATCH --array=1-6


# Debug output
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# nvidia-smi
# echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"

# Set N from the array index
N=${SLURM_ARRAY_TASK_ID}


apptainer exec --nv jaxtainer python3 main.py -n smallsearch -i 1000 -f 10 -s 2000 -rs 25 -m -o adam -N ${N}

