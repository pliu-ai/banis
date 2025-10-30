#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Refine 3-D instance segmentation by combining an accurate binary mask with a
partial/coarse instance mask.

Usage
-----
python refine_instances_with_skeleton.py --binary_dir tiff_outputs \
                                         --instances_dir test_sdt_pred \
                                         --output_dir refined_output
"""
from typing import Union, Tuple, List
import argparse
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import ball, remove_small_objects
import tifffile as tiff
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom
import glob
import os

# ---------- I/O helpers ----------------------------------------------------- #
def find_matching_files(binary_dir: Path, instances_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching binary and instance file pairs based on naming pattern."""
    matching_pairs = []
    
    # Get all .tif files in binary directory
    binary_files = list(binary_dir.glob("*.tif"))
    
    for binary_file in binary_files:
        # Extract the case ID from binary filename (e.g., pred_aff_test_RibFrac590.tif -> RibFrac590)
        case_id = binary_file.stem.replace("pred_aff_test_", "")
        
        # Look for corresponding instance file
        instance_file = instances_dir / f"pred_aff_test_{case_id}" / "watershed_segmentation.tif"
        
        if instance_file.exists():
            matching_pairs.append((binary_file, instance_file))
            print(f"Found matching pair: {binary_file.name} <-> {instance_file}")
        else:
            print(f"Warning: No matching instance file found for {binary_file.name}")
    
    return matching_pairs


def read_volume(path: Path) -> Tuple[np.ndarray, Union[sitk.Image, None]]:
    """Read a 3-D volume from .tiff / .nii / .nii.gz."""
    suffixes = path.suffixes
    if suffixes[-2:] == [".nii", ".gz"] or suffixes[-1] == ".nii":
        img = nib.load(str(path))
        vol = img.get_fdata().astype(np.uint16)
        return vol, img.affine
    elif suffixes[-1].lower() in (".tif", ".tiff"):
        vol = tiff.imread(str(path)).astype(np.uint16)
        return vol, None
    else:
        raise ValueError(f"Unsupported file type: {path}")


def save_volume(vol: np.ndarray, ref_img: Union[sitk.Image, None], path: Path) -> None:
    """Save a 3-D volume to .tiff / .nii.gz (determined by extension)."""
    if path.suffixes[-2:] == [".nii", ".gz"] or path.suffix == ".nii":
        img = nib.Nifti1Image(vol.astype(np.uint16), affine if affine is not None else np.eye(4))
        nib.save(img, str(path))
    elif path.suffix.lower() in (".tif", ".tiff"):
        tiff.imwrite(str(path), vol.astype(np.uint16), compression="zlib")
    else:
        raise ValueError(f"Unsupported output type: {path}")


# ---------- Core algorithm -------------------------------------------------- #
def refine_instance_seg(
    binary_mask: np.ndarray,
    coarse_instances: np.ndarray,
    min_seed_vox: int = 50,
    seed_dilate_iter: int = 1,
    min_final_vox: int = 200,
) -> np.ndarray:
    """Combine binary and coarse masks to obtain refined instances."""
    # 1) Clean small seeds
    seeds_clean = remove_small_objects(coarse_instances, min_size=min_seed_vox)

    # 2) Optional dilation to fill tiny holes inside seeds
    if seed_dilate_iter > 0:
        structure = ball(1)
        for _ in range(seed_dilate_iter):
            seeds_clean = ndi.grey_dilation(seeds_clean, footprint=structure)

    # 3) Distance transform inside binary mask
    distance = ndi.distance_transform_edt(binary_mask.astype(bool))

    # 4) Marker-controlled watershed (negative distance => basins interior)
    refined = watershed(-distance, markers=seeds_clean, mask=binary_mask.astype(bool))

    # 5) Remove tiny final instances
    refined = remove_small_objects(refined, min_size=min_final_vox)

    # 6) Relabel instances to be 1..N (background=0)
    unique_vals = np.unique(refined)
    unique_vals = unique_vals[unique_vals != 0]  # exclude background
    relabeled = np.zeros_like(refined, dtype=np.uint16)
    for new_id, old_id in enumerate(unique_vals, start=1):
        relabeled[refined == old_id] = new_id

    return relabeled


def process_single_pair(binary_file: Path, instance_file: Path, output_file: Path, 
                       min_seed_vox: int = 50, seed_dilate_iter: int = 1, 
                       min_final_vox: int = 500) -> bool:
    """Process a single binary-instance file pair."""
    try:
        # Load volumes
        binary, affine_bin = read_volume(binary_file)
        if np.unique(binary).shape[0] > 2:
            # not the BC mask
            print(f"Not the BC mask: {binary_file}")
            binary[binary > 1] = 1
        else:
            print(f"The BC mask: {binary_file}")
            binary[binary > 1] = 0  # Ensure binary mask is boolean

        # Load coarse instance mask
        seeds, _ = read_volume(instance_file)
        # transpose seeds
        seeds = np.transpose(seeds, (2, 1, 0))
        if seeds.shape != binary.shape:
            print(f"Resizing seeds to binary shape: {seeds.shape} -> {binary.shape}")
            # resize seeds to binary shape
            zoom_factors = [t / s for t, s in zip(binary.shape, seeds.shape)]
            seeds = zoom(seeds, zoom_factors, order=0)
        
        if binary.shape != seeds.shape:
            raise ValueError("Binary mask and instance mask must have identical shapes.")

        # Run refinement
        refined = refine_instance_seg(
            binary_mask=binary,
            coarse_instances=seeds,
            min_seed_vox=min_seed_vox,
            seed_dilate_iter=seed_dilate_iter,
            min_final_vox=min_final_vox,
        )

        # Save result (use binary's affine if writing NIfTI)
        save_volume(refined, affine_bin, output_file)
        print(f"[✓] Refined instance segmentation saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"[✗] Error processing {binary_file.name}: {str(e)}")
        return False


def process_directories(binary_dir: Path, instances_dir: Path, output_dir: Path,
                       min_seed_vox: int = 50, seed_dilate_iter: int = 1, 
                       min_final_vox: int = 500) -> None:
    """Process all matching file pairs in the directories."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching file pairs
    matching_pairs = find_matching_files(binary_dir, instances_dir)
    
    if not matching_pairs:
        print("No matching file pairs found!")
        return
    
    print(f"Found {len(matching_pairs)} matching file pairs to process.")
    
    # Process each pair
    successful = 0
    failed = 0
    
    for binary_file, instance_file in matching_pairs:
        # Create output filename
        case_id = binary_file.stem.replace("pred_aff_test_", "")
        output_file = output_dir / f"refined_{case_id}.tif"
        
        if process_single_pair(binary_file, instance_file, output_file, 
                              min_seed_vox, seed_dilate_iter, min_final_vox):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete: {successful} successful, {failed} failed")


# ---------- CLI ------------------------------------------------------------- #
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refine 3-D instance segmentation using binary masks and coarse instance masks from directories."
    )
    p.add_argument("--binary_dir", required=True, type=Path, help="Directory containing binary mask files (.tif)")
    p.add_argument("--instances_dir", required=True, type=Path, help="Directory containing instance mask subdirectories")
    p.add_argument("--output_dir", required=True, type=Path, help="Output directory for refined segmentations")
    p.add_argument("--min_seed_vox", type=int, default=50, help="Minimum voxels to keep a seed label")
    p.add_argument("--seed_dilate_iter", type=int, default=1, help="Number of dilations applied to seeds")
    p.add_argument("--min_final_vox", type=int, default=500, help="Remove final labels smaller than this")
    return p.parse_args()


def main() -> None:
    args = get_args()

    # Validate input directories
    if not args.binary_dir.exists():
        raise ValueError(f"Binary directory does not exist: {args.binary_dir}")
    if not args.instances_dir.exists():
        raise ValueError(f"Instances directory does not exist: {args.instances_dir}")

    # Process all matching file pairs
    process_directories(
        binary_dir=args.binary_dir,
        instances_dir=args.instances_dir,
        output_dir=args.output_dir,
        min_seed_vox=args.min_seed_vox,
        seed_dilate_iter=args.seed_dilate_iter,
        min_final_vox=args.min_final_vox,
    )


if __name__ == "__main__":
    main()
