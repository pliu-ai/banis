# Cerebellum BANIS Training and Testing Handoff

This document explains how to prepare cerebellum mitochondria instance-segmentation data, train BANIS, run inference/evaluation, and export predictions for ITK-SNAP.

The commands use placeholders such as `<DATA_ROOT>` and `<RUN_ROOT>`. Replace them with local paths on your machine or cluster. Do not commit private cluster paths, usernames, or dataset locations to this public repository.

## 1. Setup

Start from the repository root:

```bash
cd <BANIS_REPO>
```

Activate the environment used for BANIS:

```bash
micromamba activate sdt
```

Quick check:

```bash
python -c "import torch, zarr, tifffile, SimpleITK; print('env ok')"
```

For Slurm jobs, the examples use the helper command `submit_slurm` and pass `-e sdt`.

## 2. Data Layout

The cerebellum workflow uses two data sources:

1. Previous normalized cerebellum training data.
2. New p7 finished proofreading labels.

The data converter is:

```bash
src/data/convert_cerebellum_to_banis.py
```

### Normalized Dataset

Expected inputs:

```text
<NORMALIZED_ROOT>/
  cerebellum_mito_instance_dataset.json
  ...
```

The manifest JSON should contain relative image and mask paths:

```json
{
  "images": ["relative/path/to/image.tif"],
  "masks": ["relative/path/to/instance_mask.tif"]
}
```

Labels should use:

- `0`: background
- `1, 2, 3, ...`: mitochondria instance IDs
- `-1`: ignored/unlabeled voxels, if present

### p7 Finished Proofreading Dataset

Expected files:

```text
<P7_LABEL_ROOT>/<tile>_im_mask.tif
<P7_IMAGE_ROOT>/<tile>_im.tif or <tile>_im.h5
<P7_MASK_ROOT>/<tile>_mask_pc2.h5 or <tile>_mask_pc2.tif
```

Example naming pattern:

```text
0-192_10240-12288_2048-4096_im_mask.tif
0-192_10240-12288_2048-4096_im.h5
0-192_10240-12288_2048-4096_mask_pc2.h5
```

Important: p7 labels are only trusted inside the proofreading mask. The converter sets labels outside the region mask to `-1`, so BANIS ignores them during training loss.

Current mask search order:

```text
mask_pc2.h5
mask_pc9.h5
mask_pc2.tif
mask_pc9.tif
```

If your experiment must use only `mask_pc2`, verify every p7 tile has a matching `mask_pc2` file before conversion.

## 3. Convert Data to BANIS Format

BANIS expects this output layout:

```text
<BANIS_DATA_ROOT>/
  cereb_normalized/
    train/<sample>/data.zarr
    val/<sample>/data.zarr
    test/<sample>/data.zarr
  cereb_p7_pc2/
    train/<sample>/data.zarr
    val/<sample>/data.zarr
    test/<sample>/data.zarr
```

Each `data.zarr` contains:

```text
img  shape=(Z, Y, X, 1), uint8
seg  shape=(Z, Y, X), signed integer labels
```

Run conversion:

```bash
python src/data/convert_cerebellum_to_banis.py \
  --normalized-root <NORMALIZED_ROOT> \
  --normalized-manifest <NORMALIZED_ROOT>/cerebellum_mito_instance_dataset.json \
  --p7-root <P7_LABEL_ROOT> \
  --p7-image-root <P7_IMAGE_ROOT> \
  --p7-region-mask-root <P7_MASK_ROOT> \
  --output-root <BANIS_DATA_ROOT> \
  --overwrite
```

The script also supports environment-variable defaults:

```bash
export CEREB_NORMALIZED_ROOT=<NORMALIZED_ROOT>
export CEREB_NORMALIZED_MANIFEST=<NORMALIZED_ROOT>/cerebellum_mito_instance_dataset.json
export CEREB_P7_LABEL_ROOT=<P7_LABEL_ROOT>
export CEREB_P7_IMAGE_ROOT=<P7_IMAGE_ROOT>
export CEREB_P7_REGION_MASK_ROOT=<P7_MASK_ROOT>
```

Check converted samples:

```bash
python - <<'PY'
import zarr
from pathlib import Path

root = Path("<BANIS_DATA_ROOT>")
for dataset in ["cereb_normalized", "cereb_p7_pc2"]:
    for split in ["train", "val", "test"]:
        samples = sorted((root / dataset / split).glob("*/data.zarr"))
        print(dataset, split, len(samples))
        if samples:
            z = zarr.open(str(samples[0]), mode="r")
            print(" ", samples[0])
            print("  img", z["img"].shape, z["img"].dtype)
            seg = z["seg"][:]
            print("  seg", seg.shape, seg.dtype, "min", seg.min(), "max", seg.max())
PY
```

Optional check for p7 ignored regions:

```bash
python - <<'PY'
import zarr
from pathlib import Path

sample = Path("<BANIS_DATA_ROOT>/cereb_p7_pc2/train/<SAMPLE_ID>/data.zarr")
z = zarr.open(str(sample), mode="r")
seg = z["seg"][:]
print("ignored voxels:", int((seg == -1).sum()))
print("foreground voxels:", int((seg > 0).sum()))
PY
```

## 4. Train

The training wrapper is:

```bash
train_cereb_combo.sh
```

It trains on both datasets:

```bash
--data_setting cereb_normalized cereb_p7_pc2
```

Run with environment variables:

```bash
BASE_DATA_PATH=<BANIS_DATA_ROOT> \
SAVE_PATH=<RUN_ROOT> \
EXP_NAME=cereb_combo_sdt \
bash train_cereb_combo.sh
```

Equivalent direct command:

```bash
python -u BANIS.py \
  --data_setting cereb_normalized cereb_p7_pc2 \
  --base_data_path <BANIS_DATA_ROOT> \
  --save_path <RUN_ROOT> \
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
  --exp_name cereb_combo_sdt
```

The run directory is:

```text
<RUN_ROOT>/cereb_combo_sdt
```

Check Slurm status:

```bash
squeue -u $USER
squeue -j <JOB_ID> -o '%.18i %.9T %.12M %.20R %.40j'
sacct -j <JOB_ID> --format=JobID,JobName%25,State,Elapsed,ExitCode,NodeList -P
```

Check logs:

```bash
tail -100 logs/<JOB_ID>_tr_cereb_combo.out
tail -100 logs/<JOB_ID>_tr_cereb_combo.err
```

Check checkpoints:

```bash
find <RUN_ROOT>/cereb_combo_sdt/default/checkpoints \
  -maxdepth 1 -type f -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort
```

Check recent TensorBoard losses:

```bash
python - <<'PY'
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path("<RUN_ROOT>/cereb_combo_sdt/default")
files = sorted(root.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
tags = ["train_loss_epoch", "val_loss_epoch", "train_aff_loss_epoch", "val_aff_loss_epoch"]

merged = {tag: {} for tag in tags}
for f in files:
    ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        continue
    for tag in tags:
        if tag in ea.Tags().get("scalars", []):
            for x in ea.Scalars(tag):
                merged[tag][x.step] = x.value

for tag in tags:
    print("\n" + tag)
    for step, value in sorted(merged[tag].items())[-10:]:
        print(step, value)
PY
```

## 5. Evaluate Instance Segmentation

Use:

```bash
scripts/evaluate_cereb_train_instances.py
```

The script:

1. Loads a BANIS checkpoint.
2. Runs tiled inference.
3. Thresholds the first three affinity channels.
4. Runs connected components to produce instances.
5. Ignores ground-truth voxels where `seg == -1`.
6. Computes instance IoU metrics at `IoU >= 0.5`.
7. Optionally saves instance predictions and channel TIFFs.

Example:

```bash
python -u scripts/evaluate_cereb_train_instances.py \
  --checkpoint <RUN_ROOT>/cereb_combo_sdt/default/checkpoints/<CHECKPOINT>.ckpt \
  --base-data-path <BANIS_DATA_ROOT> \
  --datasets cereb_normalized cereb_p7_pc2 \
  --split train \
  --output-dir <EVAL_OUTPUT_ROOT>/train_instance_eval \
  --affinity-threshold 0.5 \
  --iou-threshold 0.5 \
  --prediction-channels 7 \
  --save-predictions \
  --save-tiffs \
  --save-channel-tiffs \
  --tiff-compression zlib
```

Submit to Slurm:

```bash
submit_slurm --command "python -u scripts/evaluate_cereb_train_instances.py \
  --checkpoint <RUN_ROOT>/cereb_combo_sdt/default/checkpoints/<CHECKPOINT>.ckpt \
  --base-data-path <BANIS_DATA_ROOT> \
  --datasets cereb_normalized cereb_p7_pc2 \
  --split train \
  --output-dir <EVAL_OUTPUT_ROOT>/train_instance_eval \
  --affinity-threshold 0.5 \
  --iou-threshold 0.5 \
  --prediction-channels 7 \
  --save-predictions --save-tiffs --save-channel-tiffs \
  --tiff-compression zlib" \
  -t 24:00:00 \
  -p long \
  -c 8 \
  -g 1 \
  -m 96G \
  -n eval_cereb_train \
  -e sdt
```

Evaluation output:

```text
<EVAL_OUTPUT_ROOT>/train_instance_eval/
  instance_metrics.json
  partial_results.json
  predictions_zarr/
  predictions_tiff/
  channels_tiff/<dataset>__<sample>/channel_0.tif ... channel_6.tif
```

Main metrics:

- `accuracy`: instance accuracy, `TP / (TP + FP + FN)`.
- `f1`: instance F1.
- `precision`: instance precision.
- `recall`: instance recall.
- `binary_f1`: foreground/background voxel-level F1, not instance-level.
- `n_true`: number of ground-truth instances.
- `n_pred`: number of predicted instances.

In this workflow, `accuracy@0.5` means instance IoU matching with threshold `0.5`.

## 6. Export for ITK-SNAP

The raw instance TIFFs are compressed BigTIFF `uint32` volumes. ITK-SNAP may show them incorrectly or appear blank. NIfTI is more reliable.

Convert instance TIFFs to NIfTI:

```bash
python scripts/export_instance_seg_for_itksnap.py \
  --input-dir <EVAL_OUTPUT_ROOT>/train_instance_eval/predictions_tiff \
  --output-dir <EVAL_OUTPUT_ROOT>/train_instance_eval/itksnap_nii
```

Open in ITK-SNAP:

1. Load the raw EM image as the main image.
2. Use `Segmentation -> Open Segmentation` for the `.nii.gz`.
3. Increase segmentation opacity if needed.

## 7. Filter Small Components and Remap Labels

Predicted connected components can contain many tiny fragments and large label IDs. For inspection, filter small components and remap labels to compact IDs.

```bash
python scripts/postprocess_instance_seg_for_itksnap.py \
  --input-dir <EVAL_OUTPUT_ROOT>/train_instance_eval/predictions_tiff \
  --output-dir <EVAL_OUTPUT_ROOT>/train_instance_eval/itksnap_nii_postprocessed_min100 \
  --min-voxels 100
```

This does:

- Removes predicted instances smaller than `--min-voxels`.
- Remaps remaining labels to `1..N`.
- Saves `.nii.gz` files for ITK-SNAP.
- Writes `summary.jsonl` with original/kept/removed label counts.

Use a smaller threshold such as `20` or `50` if `100` removes too much.

## 8. Inference on Raw Image Files

For raw image files not already in BANIS-ready Zarr format:

```bash
python src/inference/pred_and_eval.py \
  --checkpoint_path <CHECKPOINT>.ckpt \
  --input_path <IMAGE>.tif \
  --output_path <OUTPUT_DIR> \
  --prediction_channels 7 \
  --small_size 192 \
  --batch_size 4 \
  --output_format tiff \
  --save_channels all
```

If ground truth is available:

```bash
--gt_file <GROUND_TRUTH>.tif
```

Helper scripts:

```bash
scripts/run_p0_pred_only.sh
scripts/run_p7_pred_only.sh
```

Before using them, set:

```bash
CKPT=<CHECKPOINT>.ckpt
IN_DIR=<IMAGE_DIR>
OUT_ROOT=<OUTPUT_DIR>
```

## 9. Common Problems

### ITK-SNAP shows a blank segmentation

The prediction may not be empty. ITK-SNAP can have trouble with compressed BigTIFF `uint32` label volumes.

Fix:

```bash
python scripts/export_instance_seg_for_itksnap.py \
  --input-dir <EVAL_OUTPUT>/predictions_tiff \
  --output-dir <EVAL_OUTPUT>/itksnap_nii
```

Then open the `.nii.gz` as a segmentation.

### Very large label IDs

Connected components can create large label IDs. After masking/filtering, IDs may have gaps. Use:

```bash
python scripts/postprocess_instance_seg_for_itksnap.py \
  --input-dir <EVAL_OUTPUT>/predictions_tiff \
  --output-dir <EVAL_OUTPUT>/itksnap_nii_postprocessed_min100 \
  --min-voxels 100
```

### Many `Low loss mask mean` warnings

This is expected when p7 volumes contain large ignored regions outside proofreading masks. BANIS ignores those voxels during affinity/SDT loss.

### Validation only uses one dataset

The current combined training command trains on both `cereb_normalized` and `cereb_p7_pc2`, but validation is taken from the first dataset in the current training code path.

To check p7 validation separately, run:

```bash
python -u scripts/evaluate_cereb_train_instances.py \
  --checkpoint <CHECKPOINT>.ckpt \
  --base-data-path <BANIS_DATA_ROOT> \
  --datasets cereb_p7_pc2 \
  --split val \
  --output-dir <EVAL_OUTPUT_ROOT>/p7_val_eval \
  --prediction-channels 7
```

## 10. Recommended Student Workflow

1. Prepare finished labels, raw images, and proofreading masks.
2. Verify every p7 label has a matching raw image and region mask.
3. Convert data with `convert_cerebellum_to_banis.py`.
4. Inspect a few `data.zarr` files.
5. Confirm p7 outside-mask labels are `-1`.
6. Train with `train_cereb_combo.sh`.
7. Monitor Slurm, logs, checkpoints, and TensorBoard losses.
8. Pick a checkpoint based on validation loss or comparison experiments.
9. Evaluate with `evaluate_cereb_train_instances.py`.
10. Export predictions to NIfTI for ITK-SNAP.
11. Optionally filter/remap small components for visualization.
