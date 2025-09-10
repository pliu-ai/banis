#!/bin/bash
#SBATCH --job-name=mito_just_train_ribfrac
#SBATCH --partition=long
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --constraint="vr40g|vr80g"
#SBATCH --output=logs/%j_just_train_ribfrac.out
#SBATCH --error=logs/%j_just_train_ribfrac.err

# Change to the working directory
cd /projects/weilab/liupeng/banis

# Run the just command
srun just train_ribfrac
discord train_ribfrac finished on $SLURM_JOB_ID
