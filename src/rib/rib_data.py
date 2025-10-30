import h5py
import zarr
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from types import SimpleNamespace
import nibabel as nib


def get_conf(config_path=None):
    """
    Loads a YAML config file and returns it as a nested SimpleNamespace object.
    If config_path is None, it parses command-line arguments for a --config path.
    """
    if config_path is None:
        # Use a separate parser to not interfere with other argument parsers
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--config', default='config.yaml', help='Path to the config file')
        args, _ = parser.parse_known_args()
        config_path = args.config

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Recursively convert dict to SimpleNamespace
    def dict_to_sns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_sns(i) for i in d]
        return d

    return dict_to_sns(config_dict)


def convert_nii_to_zarr(nii_im_path, nii_seg_path, zarr_path):
    """Convert NIfTI files to zarr format with img and seg keys."""
    
    # Load NIfTI files
    img_nii = nib.load(nii_im_path)
    seg_nii = nib.load(nii_seg_path)
    
    # Get data arrays
    img_data = img_nii.get_fdata()
    seg_data = seg_nii.get_fdata()
    
    # Convert to numpy arrays and ensure proper data types
    img_data = np.asarray(img_data)
    seg_data = np.asarray(seg_data)
    
    # Ensure both are 3D
    assert len(img_data.shape) == 3, f"Expected img to be 3D, got shape {img_data.shape}"
    assert len(seg_data.shape) == 3, f"Expected seg to be 3D, got shape {seg_data.shape}"
    
    # Ensure both have the same shape
    assert img_data.shape == seg_data.shape, f"Image shape {img_data.shape} != Segmentation shape {seg_data.shape}"
    
    # Convert segmentation to integer type
    seg_data = seg_data.astype(np.int32)
    
    # Add trailing dimension for images (should be 4D: x,y,z,channel)
    img_data = img_data[..., None]
    
    # Create zarr store
    root = zarr.open(zarr_path, mode='w')
    
    # Save img and seg
    root.array('img', data=img_data)
    root.array('seg', data=seg_data)


def create_skeleton_pkl(path):
    """Create empty skeleton.pkl file."""
    with open(path, 'wb') as f:
        pickle.dump({}, f)


def process_split(samples, data_path, split_dir, split_name):
    """Process a data split (train/val/test)."""
    os.makedirs(split_dir, exist_ok=True)
    
    for sample in tqdm(samples, desc=f"Processing {split_name}"):
        sample_dir = os.path.join(split_dir, sample)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Convert NIfTI to zarr
        nii_im_path = os.path.join(data_path, 'img', f'{sample}-image.nii.gz')
        nii_seg_path = os.path.join(data_path, 'seg', f'{sample}-rib-seg.nii.gz')
        zarr_path = os.path.join(sample_dir, 'data.zarr')
        
        assert os.path.exists(nii_im_path), f"Image file not found: {nii_im_path}"
        assert os.path.exists(nii_seg_path), f"Segmentation file not found: {nii_seg_path}"
        
        convert_nii_to_zarr(nii_im_path, nii_seg_path, zarr_path)
        print(f"Converted {sample} to {zarr_path}")
        
        # Create skeleton.pkl
        skeleton_path = os.path.join(sample_dir, 'skeleton.pkl')
        create_skeleton_pkl(skeleton_path)


def process_rib_setting(setting_config, base_output_path):
    """Process a single rib setting configuration."""
    
    setting_name = setting_config.name
    data_path = setting_config.path
    output_dir = os.path.join(base_output_path, setting_name)
    
    # Process train/val/test splits
    process_split(setting_config.train, data_path, os.path.join(output_dir, 'train'), 'train')
    process_split(setting_config.val, data_path, os.path.join(output_dir, 'val'), 'val')
    process_split(setting_config.test, data_path, os.path.join(output_dir, 'test'), 'test')


if __name__ == "__main__":
    conf = get_conf()
    
    # Get rib configurations
    rib_configs = conf.rib.settings
    base_output_path = conf.rib.data_path
    
    # Process each rib setting
    for setting_config in tqdm(rib_configs, desc="Processing settings"):
        print(f"Processing setting: {setting_config.name}")
        process_rib_setting(setting_config, base_output_path)
    
    print("Data conversion completed!")
