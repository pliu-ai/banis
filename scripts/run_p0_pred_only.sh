#!/usr/bin/env bash
set -euo pipefail

CKPT="${CKPT:-checkpoints/model.ckpt}"
IN_DIR="${IN_DIR:-data/cerebellum/p0}"
OUT_ROOT="${OUT_ROOT:-outputs/p0_banis_pred}"

mkdir -p "$OUT_ROOT"

for f in "$IN_DIR"/*_im.tiff; do
  [ -e "$f" ] || continue
  base="$(basename "$f" .tiff)"
  out_dir="$OUT_ROOT/$base"
  mkdir -p "$out_dir"

  python src/inference/pred_and_eval.py \
    --checkpoint_path "$CKPT" \
    --input_path "$f" \
    --output_path "$out_dir" \
    --prediction_channels 6 \
    --small_size 192 \
    --batch_size 4 \
    --output_format tiff \
    --save_channels all
done
