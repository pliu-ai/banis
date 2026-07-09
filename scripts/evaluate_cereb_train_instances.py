#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import tifffile
import torch
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BANIS import BANIS
from src.inference.inference import compute_connected_component_segmentation, patched_inference


def binary_precision_recall_f1(pred_mask: np.ndarray, true_mask: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum(pred_mask & true_mask))
    fp = int(np.sum(pred_mask & ~true_mask))
    fn = int(np.sum(~pred_mask & true_mask))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return {
        "binary_tp": tp,
        "binary_fp": fp,
        "binary_fn": fn,
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,
        "binary_accuracy": accuracy,
    }


def instance_metrics_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresh: float = 0.5,
) -> Dict[str, float | int]:
    true_ids = np.unique(y_true[y_true > 0])
    pred_ids = np.unique(y_pred[y_pred > 0])
    n_true = int(len(true_ids))
    n_pred = int(len(pred_ids))
    if n_true == 0 and n_pred == 0:
        return {
            "criterion": "iou",
            "thresh": thresh,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "f1": 0.0,
            "n_true": 0,
            "n_pred": 0,
            "mean_true_score": 0.0,
            "mean_matched_score": 0.0,
            "panoptic_quality": 0.0,
        }

    true_map = np.zeros(int(true_ids.max()) + 1 if n_true else 1, dtype=np.int32)
    pred_map = np.zeros(int(pred_ids.max()) + 1 if n_pred else 1, dtype=np.int32)
    if n_true:
        true_map[true_ids] = np.arange(1, n_true + 1, dtype=np.int32)
    if n_pred:
        pred_map[pred_ids] = np.arange(1, n_pred + 1, dtype=np.int32)

    true_pos = y_true > 0
    pred_pos = y_pred > 0
    true_area = np.bincount(true_map[y_true[true_pos]], minlength=n_true + 1)
    pred_area = np.bincount(pred_map[y_pred[pred_pos]], minlength=n_pred + 1)

    overlap_mask = true_pos & pred_pos
    if np.any(overlap_mask):
        true_seq = true_map[y_true[overlap_mask]].astype(np.int64, copy=False)
        pred_seq = pred_map[y_pred[overlap_mask]].astype(np.int64, copy=False)
        pair_index = true_seq * (n_pred + 1) + pred_seq
        overlap_counts = np.bincount(pair_index)
        nonzero_pairs = np.nonzero(overlap_counts)[0]
        true_pair = nonzero_pairs // (n_pred + 1)
        pred_pair = nonzero_pairs % (n_pred + 1)
        valid = (true_pair > 0) & (pred_pair > 0)
        true_pair = true_pair[valid]
        pred_pair = pred_pair[valid]
        intersection = overlap_counts[nonzero_pairs[valid]].astype(np.float64)
        union = true_area[true_pair] + pred_area[pred_pair] - intersection
        ious = intersection / union
    else:
        true_pair = np.empty(0, dtype=np.int64)
        ious = np.empty(0, dtype=np.float64)

    matched = ious >= thresh
    tp = int(np.sum(matched))
    fp = int(n_pred - tp)
    fn = int(n_true - tp)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0

    best_true = np.zeros(n_true + 1, dtype=np.float64)
    if len(ious):
        np.maximum.at(best_true, true_pair, ious)
    mean_true_score = float(best_true[1:].mean()) if n_true else 0.0
    mean_matched_score = float(ious[matched].mean()) if tp else 0.0
    panoptic_quality = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn > 0 else 0.0

    return {
        "criterion": "iou",
        "thresh": thresh,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "n_true": n_true,
        "n_pred": n_pred,
        "mean_true_score": mean_true_score,
        "mean_matched_score": mean_matched_score,
        "panoptic_quality": panoptic_quality,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BANIS instance segmentation metrics on BANIS-ready cerebellum data."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--base-data-path",
        default=Path(os.environ.get("BANIS_CEREB_DATA_ROOT", "data/cerebellum/banis_ready")),
        type=Path,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cereb_normalized", "cereb_p7_pc2"],
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--affinity-threshold", default=0.5, type=float)
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    parser.add_argument("--small-size", default=None, type=int)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--prediction-channels", default=3, type=int, choices=range(3, 8))
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-tiffs", action="store_true")
    parser.add_argument("--save-channel-tiffs", action="store_true")
    parser.add_argument("--tiff-compression", default="zlib")
    return parser.parse_args()


def iter_sample_dirs(args: argparse.Namespace) -> List[Path]:
    sample_dirs: List[Path] = []
    selected = set(args.sample_id)
    for dataset in args.datasets:
        split_dir = args.base_data_path / dataset / args.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for sample_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if selected and sample_dir.name not in selected:
                continue
            sample_dirs.append(sample_dir)
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]
    if not sample_dirs:
        raise ValueError("No samples selected for evaluation")
    return sample_dirs


def to_jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def channel_to_uint8(channel: np.ndarray) -> np.ndarray:
    finite = np.isfinite(channel)
    if not np.any(finite):
        return np.zeros(channel.shape, dtype=np.uint8)
    cmin = float(channel[finite].min())
    cmax = float(channel[finite].max())
    if cmin >= 0.0 and cmax <= 1.0:
        scaled = np.clip(channel, 0.0, 1.0) * 255.0
    elif cmin >= -1.0 and cmax <= 1.0:
        scaled = (np.clip(channel, -1.0, 1.0) + 1.0) * 127.5
    else:
        scaled = (channel - cmin) / (cmax - cmin + 1e-8) * 255.0
    return scaled.astype(np.uint8)


def write_prediction_outputs(
    sample_dir: Path,
    pred_eval: np.ndarray,
    pred_aff: np.ndarray,
    args: argparse.Namespace,
    metrics: Dict,
) -> None:
    sample_key = f"{sample_dir.parent.parent.name}__{sample_dir.name}"

    if args.save_predictions:
        pred_dir = args.output_dir / "predictions_zarr"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"{sample_key}.zarr"
        zarr.array(pred_eval, dtype=np.uint32, store=str(pred_path), overwrite=True)
        metrics["pred_zarr_path"] = str(pred_path)

    if args.save_tiffs:
        pred_tiff_dir = args.output_dir / "predictions_tiff"
        pred_tiff_dir.mkdir(parents=True, exist_ok=True)
        pred_tiff_path = pred_tiff_dir / f"{sample_key}__instance_seg.tif"
        tifffile.imwrite(
            pred_tiff_path,
            pred_eval.astype(np.uint32, copy=False),
            bigtiff=True,
            compression=args.tiff_compression,
        )
        metrics["pred_tiff_path"] = str(pred_tiff_path)

    if args.save_channel_tiffs:
        channel_dir = args.output_dir / "channels_tiff" / sample_key
        channel_dir.mkdir(parents=True, exist_ok=True)
        channel_paths = []
        for channel_idx in range(pred_aff.shape[0]):
            channel_path = channel_dir / f"channel_{channel_idx}.tif"
            tifffile.imwrite(
                channel_path,
                channel_to_uint8(pred_aff[channel_idx]),
                bigtiff=True,
                compression=args.tiff_compression,
            )
            channel_paths.append(str(channel_path))
        metrics["channel_tiff_paths"] = channel_paths


def evaluate_sample(
    sample_dir: Path,
    model: BANIS,
    args: argparse.Namespace,
    small_size: int,
) -> Dict:
    data_path = sample_dir / "data.zarr"
    data = zarr.open(str(data_path), mode="r")
    img = data["img"]
    gt = data["seg"][:]

    print(f"\n=== Evaluating {sample_dir.parent.parent.name}/{sample_dir.name} ===", flush=True)
    print(f"image={img.shape} {img.dtype}, gt={gt.shape} {gt.dtype}", flush=True)

    pred_aff = patched_inference(
        img,
        model=model,
        small_size=small_size,
        do_overlap=True,
        prediction_channels=args.prediction_channels,
        divide=255,
    )
    pred_seg = compute_connected_component_segmentation(
        pred_aff[:3] > args.affinity_threshold
    )

    labeled_mask = gt >= 0
    gt_eval = gt.astype(np.int64, copy=True)
    gt_eval[~labeled_mask] = 0
    pred_eval = pred_seg.astype(np.uint32, copy=False)
    pred_eval[~labeled_mask] = 0

    metrics = instance_metrics_iou(
        gt_eval,
        pred_eval,
        thresh=args.iou_threshold,
    )
    binary_metrics = binary_precision_recall_f1(pred_eval > 0, gt_eval > 0)
    metrics.update(
        {
            "dataset": sample_dir.parent.parent.name,
            "split": sample_dir.parent.name,
            "sample_id": sample_dir.name,
            "data_path": str(data_path),
            "affinity_threshold": args.affinity_threshold,
            "iou_threshold": args.iou_threshold,
            **binary_metrics,
        }
    )

    write_prediction_outputs(sample_dir, pred_eval, pred_aff, args, metrics)

    print(
        "metrics: "
        f"accuracy@{args.iou_threshold}={metrics['accuracy']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']}",
        flush=True,
    )
    return {k: to_jsonable(v) for k, v in metrics.items()}


def summarize(results: List[Dict]) -> Dict:
    total_tp = int(sum(r.get("tp", 0) for r in results))
    total_fp = int(sum(r.get("fp", 0) for r in results))
    total_fn = int(sum(r.get("fn", 0) for r in results))
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn > 0 else 0.0

    metric_names = [
        "accuracy",
        "f1",
        "precision",
        "recall",
        "binary_accuracy",
        "binary_f1",
        "binary_precision",
        "binary_recall",
        "mean_true_score",
        "mean_matched_score",
        "panoptic_quality",
    ]
    summary = {
        "n_samples": len(results),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_true_instances": int(sum(r.get("n_true", 0) for r in results)),
        "total_pred_instances": int(sum(r.get("n_pred", 0) for r in results)),
        "overall_precision": precision,
        "overall_recall": recall,
        "overall_f1": f1,
        "overall_accuracy": accuracy,
    }
    for name in metric_names:
        values = [float(r[name]) for r in results if name in r]
        if values:
            summary[f"{name}_mean"] = float(np.mean(values))
            summary[f"{name}_std"] = float(np.std(values))
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = iter_sample_dirs(args)
    print(f"Selected {len(sample_dirs)} samples", flush=True)
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = BANIS.load_from_checkpoint(str(args.checkpoint))
    model.eval()
    model.cuda()
    torch.set_float32_matmul_precision("medium")
    small_size = args.small_size or int(model.hparams.small_size)

    results: List[Dict] = []
    for sample_dir in sample_dirs:
        result = evaluate_sample(sample_dir, model, args, small_size)
        results.append(result)
        with (args.output_dir / "partial_results.json").open("w") as f:
            json.dump({"results": results, "summary": summarize(results)}, f, indent=2)

    summary = summarize(results)
    output = {
        "checkpoint": str(args.checkpoint),
        "base_data_path": str(args.base_data_path),
        "datasets": args.datasets,
        "split": args.split,
        "affinity_threshold": args.affinity_threshold,
        "iou_threshold": args.iou_threshold,
        "prediction_channels": args.prediction_channels,
        "small_size": small_size,
        "results": results,
        "summary": summary,
    }
    with (args.output_dir / "instance_metrics.json").open("w") as f:
        json.dump(output, f, indent=2)
    print("\n=== Summary ===", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"Saved: {args.output_dir / 'instance_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
