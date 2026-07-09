#!/usr/bin/env python3
"""Filter small instance labels, remap IDs, and export for ITK-SNAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing '*__instance_seg.tif' files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for filtered, remapped NIfTI label volumes.",
    )
    parser.add_argument(
        "--pattern",
        default="*__instance_seg.tif",
        help="Input glob pattern under --input-dir.",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=100,
        help="Remove instance labels smaller than this many voxels.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(8.0, 8.0, 30.0),
        metavar=("X", "Y", "Z"),
        help="NIfTI voxel spacing in x y z order.",
    )
    parser.add_argument(
        "--force-uint32",
        action="store_true",
        help="Keep labels as uint32 instead of using uint16 when possible.",
    )
    parser.add_argument(
        "--save-tiff",
        action="store_true",
        help="Also save a compressed BigTIFF copy of the filtered label volume.",
    )
    return parser.parse_args()


def filter_and_remap(arr: np.ndarray, min_voxels: int) -> tuple[np.ndarray, dict[str, int]]:
    if min_voxels < 1:
        raise ValueError("--min-voxels must be >= 1")

    max_label = int(arr.max())
    sizes = np.bincount(arr.reshape(-1), minlength=max_label + 1)
    original_label_count = int(np.count_nonzero(sizes[1:]))
    original_nonzero = int(arr.size - sizes[0])

    keep = sizes >= min_voxels
    keep[0] = False
    kept_old_labels = np.flatnonzero(keep)
    kept_label_count = int(kept_old_labels.size)

    old_to_new = np.zeros(max_label + 1, dtype=np.uint32)
    old_to_new[kept_old_labels] = np.arange(1, kept_label_count + 1, dtype=np.uint32)
    remapped = old_to_new[arr]

    filtered_nonzero = int(np.count_nonzero(remapped))
    stats = {
        "min_voxels": int(min_voxels),
        "original_max_label": int(max_label),
        "original_label_count": original_label_count,
        "original_nonzero": original_nonzero,
        "kept_label_count": kept_label_count,
        "removed_label_count": int(original_label_count - kept_label_count),
        "filtered_nonzero": filtered_nonzero,
        "removed_nonzero": int(original_nonzero - filtered_nonzero),
    }
    return remapped, stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tiff_dir = args.output_dir / "tiff"
    if args.save_tiff:
        tiff_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(args.input_dir.glob(args.pattern))
    if not inputs:
        raise FileNotFoundError(f"No files matched {args.input_dir / args.pattern}")

    summary_path = args.output_dir / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as summary:
        for path in inputs:
            arr = tifffile.imread(path)
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D volume for {path}, got shape {arr.shape}")

            remapped, stats = filter_and_remap(arr, args.min_voxels)
            if not args.force_uint32 and stats["kept_label_count"] <= np.iinfo(np.uint16).max:
                output_arr = remapped.astype(np.uint16, copy=False)
            else:
                output_arr = remapped.astype(np.uint32, copy=False)

            image = sitk.GetImageFromArray(output_arr)
            image.SetSpacing(tuple(args.spacing))

            output_path = args.output_dir / f"{path.stem}_min{args.min_voxels}_remap.nii.gz"
            sitk.WriteImage(image, str(output_path), True)

            if args.save_tiff:
                tiff_path = tiff_dir / f"{path.stem}_min{args.min_voxels}_remap.tif"
                tifffile.imwrite(
                    tiff_path,
                    output_arr,
                    bigtiff=True,
                    compression="zlib",
                    photometric="minisblack",
                    metadata={"axes": "ZYX"},
                )

            record = {
                "input": str(path),
                "output": str(output_path),
                "shape": list(arr.shape),
                "dtype": str(output_arr.dtype),
                **stats,
            }
            summary.write(json.dumps(record) + "\n")
            summary.flush()
            print(
                f"wrote {output_path} kept={stats['kept_label_count']} "
                f"removed={stats['removed_label_count']} dtype={output_arr.dtype}",
                flush=True,
            )


if __name__ == "__main__":
    main()
