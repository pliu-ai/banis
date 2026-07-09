#!/usr/bin/env python3
"""Export saved instance segmentation TIFFs to ITK-SNAP friendly NIfTI files."""

from __future__ import annotations

import argparse
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
        help="Directory for ITK-SNAP friendly NIfTI label volumes.",
    )
    parser.add_argument(
        "--pattern",
        default="*__instance_seg.tif",
        help="Input glob pattern under --input-dir.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(args.input_dir.glob(args.pattern))
    if not inputs:
        raise FileNotFoundError(f"No files matched {args.input_dir / args.pattern}")

    for path in inputs:
        arr = tifffile.imread(path)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D volume for {path}, got shape {arr.shape}")

        max_label = int(arr.max())
        if not args.force_uint32 and max_label <= np.iinfo(np.uint16).max:
            arr = arr.astype(np.uint16, copy=False)
        else:
            arr = arr.astype(np.uint32, copy=False)

        image = sitk.GetImageFromArray(arr)
        image.SetSpacing(tuple(args.spacing))

        output_path = args.output_dir / f"{path.stem}.nii.gz"
        sitk.WriteImage(image, str(output_path), True)
        nonzero = int(np.count_nonzero(arr))
        print(
            f"wrote {output_path} shape={arr.shape} dtype={arr.dtype} "
            f"max_label={max_label} nonzero={nonzero}",
            flush=True,
        )


if __name__ == "__main__":
    main()
