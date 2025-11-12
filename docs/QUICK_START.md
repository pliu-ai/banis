# Quick Start: BANIS to Neuron Shape Reasoning

## 1. 转换BANIS输出为点云

```bash
# 基本转换
python convert_banis_to_neuron_reasoning.py \
    --input_dir ./rib_outputs \
    --output_dir ./rib_pointclouds \
    --voxel_size 1.0 1.0 1.0 \
    --samples_per_neuron 1024

# 高级转换（使用骨架化）
python advanced_mask_to_pointcloud.py \
    --input_dir ./rib_outputs \
    --output_dir ./rib_pointclouds \
    --voxel_size 1.0 1.0 1.0 \
    --samples_per_neuron 1024 \
    --use_skeleton
```

## 2. 测试转换功能

```bash
python test_mask_conversion.py
```

## 3. 在Neuron Shape Reasoning中使用

更新配置文件 `config.yaml`:

```yaml
rib:
  data_path: ./rib_pointclouds
  settings:
    - name: ribFrac
      resolution: [1, 1, 1]
      path: ./rib_pointclouds
      train: ["RibFrac1", "RibFrac2", ...]  # 根据实际文件名
      test: ["RibFrac101", "RibFrac102", ...]
      val: ["RibFrac201", "RibFrac202", ...]
```

## 4. 训练模型

```bash
cd /projects/weilab/liupeng/neuron-shape-reasoning

python train_affinity.py \
    --data_path /projects/weilab/liupeng/banis/rib_pointclouds \
    --neuron_id_path /projects/weilab/liupeng/banis/rib_pointclouds/neuron_ids.csv \
    --fam_to_id_mapping /projects/weilab/liupeng/banis/rib_pointclouds/family_to_id.json \
    --point_cloud_size 1024
```

## 参数说明

- `--input_dir`: BANIS输出目录
- `--output_dir`: 点云输出目录  
- `--voxel_size`: 体素大小 (z, y, x)
- `--samples_per_neuron`: 每个神经元的点数
- `--min_size`: 最小实例大小
- `--use_skeleton`: 使用骨架化（更真实的神经元结构）

## 输出文件

- `{filename}_pointcloud.pt`: 点云数据
- `{filename}.swc`: SWC格式文件
- `neuron_ids.csv`: 神经元ID列表
- `family_to_id.json`: 家族映射

## 注意事项

1. 确保BANIS输出文件格式正确（zarr, tiff, npy）
2. 根据数据调整体素大小
3. 骨架化适合细长结构，密集结构可以不用
4. 内存不足时减少 `samples_per_neuron`

