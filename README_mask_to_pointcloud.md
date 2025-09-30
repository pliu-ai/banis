# BANIS Mask to Point Cloud Conversion

This directory contains scripts to convert BANIS segmentation outputs to point cloud format for training neuron shape reasoning models.

## Files

- `convert_banis_to_neuron_reasoning.py` - Basic conversion script
- `advanced_mask_to_pointcloud.py` - Advanced conversion with skeletonization
- `example_convert_to_neuron_reasoning.py` - Example usage script
- `mask_to_pointcloud.py` - Full-featured conversion with SWC support

## Quick Start

### 1. Basic Conversion

```bash
python convert_banis_to_neuron_reasoning.py \
    --input_dir /path/to/banis/outputs \
    --output_dir /path/to/pointclouds \
    --voxel_size 1.0 1.0 1.0 \
    --samples_per_neuron 1024 \
    --min_size 100
```

### 2. Advanced Conversion with Skeletonization

```bash
python advanced_mask_to_pointcloud.py \
    --input_dir /path/to/banis/outputs \
    --output_dir /path/to/pointclouds \
    --voxel_size 1.0 1.0 1.0 \
    --samples_per_neuron 1024 \
    --min_size 100 \
    --use_skeleton
```

### 3. Using the Example Script

```bash
python example_convert_to_neuron_reasoning.py
```

## Parameters

- `--input_dir`: Directory containing BANIS prediction files (zarr, tiff, npy)
- `--output_dir`: Output directory for point clouds and metadata
- `--voxel_size`: Voxel size in z, y, x order (default: 1.0 1.0 1.0)
- `--samples_per_neuron`: Number of points to sample per neuron (default: 1024)
- `--min_size`: Minimum instance size in voxels (default: 100)
- `--pattern`: File pattern to match (default: *.zarr)
- `--use_skeleton`: Use skeletonization for more realistic neuron structure

## Output Files

The conversion creates:

1. **Point Cloud Files**: `{filename}_pointcloud.pt` - PyTorch tensors containing:
   - `points`: Point coordinates (N, 3)
   - `labels`: Instance labels (N,)
   - `voxel_size`: Voxel size used
   - `samples_per_neuron`: Number of points sampled

2. **SWC Files**: `{filename}.swc` - Standard neuron morphology format

3. **Metadata Files**:
   - `family_to_id.json` - Family to ID mapping
   - `neuron_ids.csv` - Neuron ID and family information

## Using with Neuron Shape Reasoning

After conversion, update your neuron shape reasoning config:

```yaml
data_path: /path/to/pointclouds
neuron_id_path: /path/to/pointclouds/neuron_ids.csv
fam_to_id_mapping: /path/to/pointclouds/family_to_id.json
point_cloud_size: 1024
```

Then run training:

```bash
python train_affinity.py \
    --data_path /path/to/pointclouds \
    --neuron_id_path /path/to/pointclouds/neuron_ids.csv \
    --fam_to_id_mapping /path/to/pointclouds/family_to_id.json \
    --point_cloud_size 1024
```

## Conversion Methods

### Basic Method
- Converts all voxels in segmentation mask to points
- Simple and fast
- Good for dense structures

### Advanced Method with Skeletonization
- Uses 3D skeletonization to extract centerline points
- More realistic neuron structure
- Better for elongated structures like axons/dendrites
- Slower but more accurate

## Tips

1. **Voxel Size**: Adjust based on your data resolution
2. **Samples per Neuron**: Higher values give more detail but use more memory
3. **Min Size**: Filter out small noise instances
4. **Skeletonization**: Use for elongated structures, skip for dense objects

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `samples_per_neuron` or process files individually
2. **Empty Point Clouds**: Check `min_size` parameter and input data
3. **Skeletonization Fails**: Try without `--use_skeleton` flag

### File Format Support

- **Zarr**: `.zarr` files (recommended for large datasets)
- **TIFF**: `.tif`, `.tiff` files
- **NumPy**: `.npy` files

## Example Workflow

1. Run BANIS inference to get segmentation masks
2. Convert masks to point clouds:
   ```bash
   python advanced_mask_to_pointcloud.py \
       --input_dir ./rib_outputs \
       --output_dir ./rib_pointclouds \
       --voxel_size 1.0 1.0 1.0 \
       --samples_per_neuron 1024
   ```
3. Update neuron shape reasoning config
4. Train the model:
   ```bash
   python train_affinity.py --data_path ./rib_pointclouds --neuron_id_path ./rib_pointclouds/neuron_ids.csv --fam_to_id_mapping ./rib_pointclouds/family_to_id.json
   ```

## Dependencies

- torch
- numpy
- zarr
- tifffile
- scipy
- scikit-image
- pandas
- tqdm
- pathlib

