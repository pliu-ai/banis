import argparse
import subprocess
import sys
import os
from pathlib import Path

def create_slurm_script(
    command, 
    job_name, 
    partition, 
    time, 
    cpus, 
    mem, 
    gpus, 
    constraint
):
    """Dynamically create SLURM sbatch script content based on the provided parameters"""
    
    # If user doesn't provide job_name, auto-generate one based on command
    if not job_name:
        job_name = f"{command.replace(' ', '_')}"

    # Create logs directory (if it doesn't exist)
    Path("logs").mkdir(exist_ok=True)

    # Use f-string to build script template
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:{gpus}
#SBATCH --constraint="{constraint}"
#SBATCH --output=logs/%j_{job_name}.out
#SBATCH --error=logs/%j_{job_name}.err

# Change to working directory
cd {os.getcwd()}
echo "Current working directory: $(pwd)"

echo "=========================================================="
echo "Starting on $(hostname)"
echo "Time is $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running command: {command}"
echo "=========================================================="
source ~/miniconda3/bin/activate
conda activate sdt
# Run script
{command}

# Send discord notification after task completion
discord "command '{command}' finished on job $SLURM_JOB_ID"
"""
    return slurm_script

def submit_job(args):
    """Submit SLURM job"""
    
    # 1. Create SLURM script content
    script_content = create_slurm_script(
        command=args.command,
        job_name=args.job_name,
        partition=args.partition,
        time=args.time,
        cpus=args.cpus,
        mem=args.mem,
        gpus=args.gpus,
        constraint=args.constraint
    )
    
    print("--- SLURM script to be submitted ---")
    print(script_content)
    print("--------------------------")
    
    try:
        # 2. Pass script content to sbatch via stdin and execute
        result = subprocess.run(
            ['sbatch'],
            input=script_content,
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Job submitted successfully!")
        print(f"   {result.stdout.strip()}")

    except FileNotFoundError:
        print("❌ Error: 'sbatch' command not found. Please ensure Slurm environment is properly configured.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("❌ Error: Job submission failed.", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Submit just command as a SLURM job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values
    )
    
    # --- Required arguments ---
    parser.add_argument('command', help='Just command to run (e.g., train_model)')
    
    # --- Optional arguments (override SLURM default configuration) ---
    parser.add_argument('--job_name', '-n', help='Specify job name (default: auto-generated from command)')
    parser.add_argument('--time', '-t', default='120:00:00', help='Job runtime (D-HH:MM:SS)')
    parser.add_argument('--partition', '-p', default='long', help='Specify partition to submit to')
    parser.add_argument('--cpus', '-c', type=int, default=32, help='Number of CPU cores per task')
    parser.add_argument('--mem', '-m', default='240G', help='Requested memory size (e.g., 240G)')
    parser.add_argument('--gpus', '-g', type=int, default=1, help='Number of GPUs to request')
    parser.add_argument('--constraint', default='', help='GPU type constraint (e.g., "v100|a100")')
    
    args = parser.parse_args()
    
    # Check if justfile exists
    if not Path('justfile').exists():
        print("❌ Error: justfile not found in current directory.", file=sys.stderr)
        sys.exit(1)
        
    submit_job(args)

if __name__ == '__main__':
    main()