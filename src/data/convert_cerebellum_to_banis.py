import argparse
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import tifffile
import zarr


@dataclass
class Sample:
    sample_id: str
    image_path: Path
    label_path: Path
    region_mask_path: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert cerebellum TIFF/H5 data into BANIS-ready zarr layout."
    )
    parser.add_argument(
        "--normalized-root",
        type=Path,
        default=Path(os.environ.get("CEREB_NORMALIZED_ROOT", "data/cerebellum/normalized")),
        help="Path to normalized dataset root.",
    )
    parser.add_argument(
        "--normalized-manifest",
        type=Path,
        default=Path(
            os.environ.get(
                "CEREB_NORMALIZED_MANIFEST",
                "data/cerebellum/normalized/cerebellum_mito_instance_dataset.json",
            )
        ),
        help="Manifest JSON containing normalized image/mask pairs.",
    )
    parser.add_argument(
        "--p7-root",
        type=Path,
        default=Path(os.environ.get("CEREB_P7_LABEL_ROOT", "data/cerebellum/p7_finished")),
        help="Path to p7 finished proofreading masks.",
    )
    parser.add_argument(
        "--p7-image-root",
        type=Path,
        default=Path(os.environ.get("CEREB_P7_IMAGE_ROOT", "data/cerebellum/p7_images")),
        help="Path with paired p7 *_im.tif/*.h5 image volumes.",
    )
    parser.add_argument(
        "--p7-region-mask-root",
        type=Path,
        default=Path(os.environ.get("CEREB_P7_REGION_MASK_ROOT", "data/cerebellum/p7_images")),
        help="Path with *_mask_pc2.h5 region masks for p7.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New folder where BANIS-ready dataset will be written.",
    )
    parser.add_argument(
        "--dataset-a-name",
        type=str,
        default="cereb_normalized",
        help="BANIS data_setting name for normalized dataset.",
    )
    parser.add_argument(
        "--dataset-b-name",
        type=str,
        default="cereb_p7_pc2",
        help="BANIS data_setting name for p7 dataset.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Train split fraction.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folder.",
    )
    return parser.parse_args()


def read_tiff(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D TIFF at {path}, got shape {arr.shape}")
    return arr


def read_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return read_tiff(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        arr = read_single_h5_dataset(path)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D H5 dataset at {path}, got shape {arr.shape}")
        return arr
    raise ValueError(f"Unsupported volume format: {path}")


def read_single_h5_dataset(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if len(keys) != 1:
            raise ValueError(f"Expected one dataset in {path}, found keys={keys}")
        return f[keys[0]][:]


def read_region_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return read_single_h5_dataset(path) > 0
    if path.suffix.lower() in {".tif", ".tiff"}:
        return read_tiff(path) > 0
    raise ValueError(f"Unsupported region mask format: {path}")


def align_to_shape(
    sample_id: str,
    arr: np.ndarray,
    target_shape: Tuple[int, int, int],
    fill_value: int | bool,
    role: str,
) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D {role} for {sample_id}, got shape={arr.shape}")

    print(
        f"Warning: {role} shape mismatch for {sample_id}: "
        f"{arr.shape} -> {target_shape}; padding/cropping unmatched voxels"
    )
    out_dtype = np.result_type(arr.dtype, np.asarray(fill_value).dtype)
    out = np.full(target_shape, fill_value, dtype=out_dtype)
    common_shape = tuple(min(src, dst) for src, dst in zip(arr.shape, target_shape))
    src_slices = tuple(slice(0, size) for size in common_shape)
    dst_slices = tuple(slice(0, size) for size in common_shape)
    out[dst_slices] = arr[src_slices]
    return out


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output path {path} already exists and is not empty. Use --overwrite or a new path."
        )
    path.mkdir(parents=True, exist_ok=True)


def split_samples(
    samples: Sequence[Sample], train_frac: float, val_frac: float
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0,1)")
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be in [0,1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    n = len(samples)
    if n < 3:
        raise ValueError(f"Need at least 3 samples to make train/val/test splits, got {n}")

    ordered = sorted(samples, key=lambda s: s.sample_id)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train = ordered[:n_train]
    val = ordered[n_train:n_train + n_val]
    test = ordered[n_train + n_val:]
    return train, val, test


def write_sample(output_sample_dir: Path, image: np.ndarray, seg: np.ndarray) -> None:
    output_sample_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_sample_dir / "data.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.create_array("img", data=image[..., None], overwrite=True)
    root.create_array("seg", data=seg, overwrite=True)

    # BANIS evaluation expects this file to exist.
    with (output_sample_dir / "skeleton.pkl").open("wb") as f:
        pickle.dump({}, f)


def convert_one_sample(sample: Sample) -> Tuple[np.ndarray, np.ndarray]:
    img = read_volume(sample.image_path)
    seg = read_volume(sample.label_path)

    if seg.shape != img.shape:
        seg = align_to_shape(sample.sample_id, seg, img.shape, fill_value=-1, role="label")

    # Keep full labels for volumes without region mask.
    if sample.region_mask_path is not None:
        region_mask = read_region_mask(sample.region_mask_path)
        if region_mask.shape != img.shape:
            region_mask = align_to_shape(
                sample.sample_id,
                region_mask,
                img.shape,
                fill_value=False,
                role="region mask",
            )
        if np.any(~region_mask):
            max_val = int(seg.max())
            seg_dtype = np.int32 if max_val <= np.iinfo(np.int32).max else np.int64
            seg = seg.astype(seg_dtype, copy=False)
            seg[~region_mask] = -1

    return img, seg


def load_normalized_samples(normalized_root: Path, manifest_path: Path) -> List[Sample]:
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    images = manifest["images"]
    masks = manifest["masks"]
    if len(images) != len(masks):
        raise ValueError("Manifest has different numbers of images and masks")

    samples: List[Sample] = []
    for image_rel, mask_rel in zip(images, masks):
        image_path = normalized_root / image_rel
        label_path = normalized_root / mask_rel
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")
        sample_id = image_path.stem
        samples.append(Sample(sample_id=sample_id, image_path=image_path, label_path=label_path))
    return samples


def load_p7_samples(p7_root: Path, p7_image_root: Path, p7_region_mask_root: Path) -> List[Sample]:
    label_paths = sorted(p7_root.glob("*_im_mask.tif"))
    if not label_paths:
        raise FileNotFoundError(f"No *_im_mask.tif labels found in {p7_root}")

    samples: List[Sample] = []
    for label_path in label_paths:
        sample_prefix = label_path.name.replace("_im_mask.tif", "")
        image_candidates = [
            p7_image_root / f"{sample_prefix}_im.tif",
            p7_image_root / f"{sample_prefix}_im.tiff",
            p7_image_root / f"{sample_prefix}_im.h5",
            p7_root / f"{sample_prefix}_im.tif",
            p7_root / f"{sample_prefix}_im.tiff",
            p7_root / f"{sample_prefix}_im.h5",
        ]
        image_path = next((p for p in image_candidates if p.exists()), None)
        if image_path is None:
            raise FileNotFoundError(
                f"Missing paired image for {label_path}: tried {', '.join(str(p) for p in image_candidates)}"
            )

        candidates = [
            p7_region_mask_root / f"{sample_prefix}_mask_pc2.h5",
            p7_region_mask_root / f"{sample_prefix}_mask_pc9.h5",
            p7_region_mask_root / f"{sample_prefix}_mask_pc2.tif",
            p7_region_mask_root / f"{sample_prefix}_mask_pc9.tif",
        ]
        region_mask_path = next((p for p in candidates if p.exists()), None)
        if region_mask_path is None:
            raise FileNotFoundError(
                f"Missing region mask for {label_path}: tried {', '.join(str(p) for p in candidates)}"
            )
        samples.append(
            Sample(
                sample_id=sample_prefix,
                image_path=image_path,
                label_path=label_path,
                region_mask_path=region_mask_path,
            )
        )
    return samples


def write_dataset(output_root: Path, dataset_name: str, samples: List[Sample], train_frac: float, val_frac: float) -> None:
    train, val, test = split_samples(samples, train_frac, val_frac)
    split_map = {"train": train, "val": val, "test": test}

    for split, split_samples_list in split_map.items():
        for sample in split_samples_list:
            img, seg = convert_one_sample(sample)
            out_sample_dir = output_root / dataset_name / split / sample.sample_id
            write_sample(out_sample_dir, img, seg)
            print(f"[{dataset_name}/{split}] wrote {sample.sample_id} -> {out_sample_dir / 'data.zarr'}")

    print(
        f"[{dataset_name}] split counts: train={len(train)}, val={len(val)}, test={len(test)}"
    )


def main() -> None:
    args = parse_args()
    ensure_clean_dir(args.output_root, args.overwrite)

    normalized_samples = load_normalized_samples(args.normalized_root, args.normalized_manifest)
    p7_samples = load_p7_samples(args.p7_root, args.p7_image_root, args.p7_region_mask_root)

    if args.overwrite:
        for dataset_name in (args.dataset_a_name, args.dataset_b_name):
            dataset_dir = args.output_root / dataset_name
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)

    write_dataset(
        output_root=args.output_root,
        dataset_name=args.dataset_a_name,
        samples=normalized_samples,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    write_dataset(
        output_root=args.output_root,
        dataset_name=args.dataset_b_name,
        samples=p7_samples,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    print(f"Done. BANIS-ready data written to: {args.output_root}")
    print("Use --data_setting with both dataset names to train together.")
    print(f"Example: --data_setting {args.dataset_a_name} {args.dataset_b_name}")


if __name__ == "__main__":
    main()
