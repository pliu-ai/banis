#!/bin/bash
#SBATCH --job-name=mito_just_train_betaseg
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
srun just train_betaseg
discord train_betaseg finished on $SLURM_JOB_ID
