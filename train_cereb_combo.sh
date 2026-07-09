#!/usr/bin/env bash
set -euo pipefail

# Submit BANIS training on combined cerebellum datasets.
# Override paths with environment variables when running on a cluster:
#   BASE_DATA_PATH=/path/to/banis_ready SAVE_PATH=/path/to/runs bash train_cereb_combo.sh

BASE_DATA_PATH="${BASE_DATA_PATH:-data/cerebellum/banis_ready}"
SAVE_PATH="${SAVE_PATH:-runs/banis}"
EXP_NAME="${EXP_NAME:-cereb_combo_sdt}"

submit_slurm --command "python -u BANIS.py \
  --data_setting cereb_normalized cereb_p7_pc2 \
  --base_data_path ${BASE_DATA_PATH} \
  --save_path ${SAVE_PATH} \
  --devices 2 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --no-compile \
  --small_size 192 \
  --n_steps 100000 \
  --ensure_foreground \
  --max_sampling_attempts 10000 \
  --sdt \
  --no-wandb \
  --no-final_full_cube_inference \
  --exp_name ${EXP_NAME}" \
  -t 120:00:00 \
  -p long \
  -c 16 \
  -g 2 \
  -m 96G \
  -n tr_cereb_combo \
  -e sdt \
  --constraint "vr40g|vr80g|vr144g"
