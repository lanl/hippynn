#!/bin/bash
#SBATCH --job-name=hippynn_train
#SBATCH --output=./logs/%x_%A_%a_hippynn_train.out
#SBATCH --time=10:00:00            # Job duration
#SBATCH --partition=volta-x86      # GPU partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --cpus-per-task=16         # CPUs per task
#SBATCH --mem=64G                  # RAM
#SBATCH --array=0-2   # Run jobs oversplit 

# Check the number of parameters 
# Parse the arguments
DATA_SIZE=$1
if [ -z "$DATA_SIZE" ]; then
    echo "Usage: sbatch run.slurm <data_size>"
    exit 1
fi

DEFAULT_MODEL_NAME="HipHopnn"
if [ -z "$2" ]; then
    MODEL_NAME=$DEFAULT_MODEL_NAME
else
    MODEL_NAME=$2
fi


# Set single NVIDIA GPU
echo "Starting training job on $(hostname) with job ID: $SLURM_JOB_ID, partition: $SLURM_JOB_PARTITION"
echo "Running in directory: $(pwd)"
nvidia-smi
echo "Loading conda environment..."
module load miniconda3

# Check if conda environment exists
if conda info --envs | grep -q "hippynn"; then
    source ~/.bashrc
    echo "Activating conda environment 'hippynn'..."
    conda activate hippynn
else
    echo "Conda environment 'hippynn' not found. Please create it first"
    exit 1
fi

# Run the training script
echo "Running training script. Python=`which python`"
# Set PYTHONPATH to the project root
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 python ./examples/methane_configurations.py --data-split $SLURM_ARRAY_TASK_ID --data-size $DATA_SIZE