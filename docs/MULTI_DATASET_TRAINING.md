# 多数据集联合训练功能

## 概述

该功能允许在训练时同时加载多个数据集，用于训练一个更通用的模型。所有指定的数据集会被合并到一个训练集中。

## 使用方法

### 1. 单数据集训练（原有功能，向后兼容）

```bash
uv run BANIS.py --seed 0 --batch_size 8 --n_steps 500000 \
    --data_setting betaSeg \
    --base_data_path /projects/weilab/liupeng/banis/mito_data \
    --save_path ./mito_outputs --devices=1 --sdt
```

### 2. 多数据集联合训练（新功能）

只需在 `--data_setting` 参数后指定多个数据集名称（用空格分隔）：

```bash
uv run BANIS.py --seed 0 --batch_size 4 --n_steps 500000 \
    --data_setting betaSeg han24 Jurkat \
    --base_data_path /projects/weilab/liupeng/banis/mito_data \
    --save_path ./mito_outputs --devices=1 --sdt
```

### 3. 使用 justfile 快捷命令

我们提供了一些预定义的训练命令：

```bash
# 训练3个数据集
just train_multi_mito

# 训练所有8个mito数据集
just train_all_mito
```

## 功能特性

### 自动处理
- ✅ **自动分辨率适配**: 每个数据集的分辨率会从 `config.yaml` 中自动读取
- ✅ **数据集合并**: 所有训练数据会自动合并为一个 `ConcatDataset`
- ✅ **验证数据**: 默认使用第一个数据集的验证集（加快验证速度）
- ✅ **实验命名**: 多数据集会自动在实验名中用 `+` 连接，例如 `ds_betaSeg+han24+Jurkat`

### 数据加载流程

1. **参数解析**: `--data_setting` 接受一个或多个数据集名称
2. **分辨率查找**: 从 config.yaml 的 `mito` 或 `rib` 配置中查找每个数据集的分辨率
3. **数据加载**: 
   - 对每个数据集分别调用 `get_train_data()` 和 `get_val_data()`
   - 所有训练数据通过 `ConcatDataset` 合并
   - 验证数据使用第一个数据集（可根据需要修改）
4. **训练**: 模型在合并后的数据集上训练

## 配置示例

在 `config.yaml` 中，确保你要使用的数据集都已正确配置：

```yaml
mito:
  data_path: ./mito_data
  settings:
    - name: betaSeg
      resolution: [16, 16, 16]
      path: /projects/weilab/dataset/MitoLE/betaSeg
      train: ["high_c3", "low_c1", "low_c2"]
      test: ["high_c2", "high_c4", "low_c3"]
      val: ["high_c1"]
    
    - name: han24
      resolution: [30, 8, 8]
      path: /projects/weilab/dataset/MitoLE/han24
      train: ["neg0", "pos0"]
      test: ["neg1", "pos1"]
      val: ["pos2"]
    
    # ... 更多数据集
```

## 注意事项

1. **内存使用**: 训练多个数据集会增加内存使用，建议根据可用资源调整 `--batch_size`
2. **数据路径**: 所有数据集必须在同一个 `base_data_path` 下，按标准目录结构组织：
   ```
   base_data_path/
   ├── dataset1/
   │   ├── train/
   │   ├── val/
   │   └── test/
   ├── dataset2/
   │   ├── train/
   │   ├── val/
   │   └── test/
   ```
3. **验证数据**: 当前实现使用第一个数据集的验证集，如需在所有数据集上验证，可修改 `data.py` 中的 `load_data()` 函数
4. **通道数**: 所有数据集应具有相同的输入通道数

## 实现细节

### 修改的文件
- `BANIS.py`: 修改了 `main()` 和 `parse_args()` 函数以支持多数据集
- `data.py`: 修改了 `load_data()`, `get_syn_train_data()`, `get_val_data()` 函数
- `justfile`: 添加了多数据集训练示例命令

### 关键函数
- `load_data()`: 检测单/多数据集，并相应地加载和合并数据
- 向后兼容性通过检查 `hasattr(args, 'data_settings')` 实现

## 示例

### 训练3个mito数据集
```bash
uv run BANIS.py \
    --seed 0 \
    --batch_size 4 \
    --n_steps 500000 \
    --data_setting betaSeg han24 Jurkat \
    --base_data_path /projects/weilab/liupeng/banis/mito_data \
    --save_path ./mito_outputs \
    --devices=1 \
    --sdt
```

### 训练所有mito数据集（更通用的模型）
```bash
uv run BANIS.py \
    --seed 0 \
    --batch_size 2 \
    --n_steps 1000000 \
    --data_setting betaSeg han24 Jurkat Cardiac Kidney Liver Sperm Macrophage \
    --base_data_path /projects/weilab/liupeng/banis/mito_data \
    --save_path ./mito_outputs \
    --devices=1 \
    --sdt
```

## 输出

训练日志会显示：
```
Loading multiple datasets: ['betaSeg', 'han24', 'Jurkat']
Loading dataset: betaSeg with resolution (16, 16, 16)
[betaSeg] image+segmentation paths: [...]
[betaSeg] Segmentation shapes: [...]
Loading dataset: han24 with resolution (30, 8, 8)
[han24] image+segmentation paths: [...]
[han24] Segmentation shapes: [...]
...
Using validation data from: betaSeg
```

实验名会自动生成，例如：
```
25-10-26_10-30-45-123456ds_betaSeg+han24+Jurkat_lrng10_s0_b4_mS_k3_...
```







