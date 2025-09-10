#!/bin/bash
#SBATCH --job-name=mito_just_train_han24
#SBATCH --partition=long
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --constraint="vr40g|vr80g"
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Change to the working directory
cd /projects/weilab/liupeng/banis

# Run the just command
srun just train_han24
discord train_han24 finished on $SLURM_JOB_ID
