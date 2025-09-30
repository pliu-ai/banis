# 分割评估模块使用说明

本模块提供了完整的分割评估功能，包括标签匹配、Dice系数计算和批量评估。

## 功能特性

- **标签匹配**: 使用匈牙利算法基于IoU最大匹配预测标签与GT标签
- **Dice系数计算**: 计算每个标签的Dice系数和平均Dice
- **批量评估**: 支持批量处理多个文件
- **结果保存**: 支持JSON和文本格式保存结果
- **多格式支持**: 支持TIFF、Zarr、NIfTI (.nii.gz) 格式

## 文件说明

### 1. `extract_segmentation.py` (已更新)
- 添加了评估功能到主提取脚本
- 支持在提取分割时同时进行评估

### 2. `evaluate_segmentation.py`
- 独立的评估脚本
- 支持单个文件评估

### 3. `batch_evaluate.py`
- 批量评估脚本
- 自动匹配预测文件和GT文件

### 4. `example_evaluation.py`
- 示例脚本，展示如何使用评估功能

## 使用方法

### 1. 单个文件评估

```bash
# 基本使用
python evaluate_segmentation.py --pred prediction.tif --gt ground_truth.tif --output results.txt

# 支持Zarr格式
python evaluate_segmentation.py --pred prediction.zarr --gt ground_truth.zarr --output results.txt

# 支持NIfTI格式
python evaluate_segmentation.py --pred prediction.tif --gt ground_truth.nii.gz --output results.txt

# 自动调整GT尺寸
python evaluate_segmentation.py --pred prediction.tif --gt ground_truth.tif --output results.txt --resize-gt
```

### 2. 在提取时评估

```bash
# 提取分割并同时评估
python extract_segmentation.py --input pred.zarr --output result.tif --gt ground_truth.tif
```

### 3. 批量评估

```bash
# 批量评估
python batch_evaluate.py --pred-dir predictions/ --gt-dir ground_truths/ --output-dir results/

# 自定义文件模式
python batch_evaluate.py --pred-dir predictions/ --gt-dir ground_truths/ --output-dir results/ --pred-pattern "*_seg.tif" --gt-pattern "*.nii.gz"
```

### 4. 运行示例

```bash
# 运行示例脚本
python example_evaluation.py
```

## 评估算法

### 1. 标签匹配
- 计算预测标签与GT标签之间的IoU矩阵
- 使用匈牙利算法进行最优匹配
- 只匹配IoU > 0的标签对

### 2. Dice系数计算
```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```
其中：
- A: 预测标签的掩码
- B: GT标签的掩码
- |A ∩ B|: 交集大小
- |A| + |B|: 并集大小

### 3. 平均Dice
计算所有GT标签的Dice系数的平均值

## 输出结果

### 1. 控制台输出
- 匹配的标签对和IoU值
- 每个标签的Dice系数
- 平均Dice和匹配统计

### 2. 文件输出
- **JSON格式**: 包含完整的评估结果
- **文本格式**: 人类可读的结果摘要

### 3. 批量评估结果
- 每个文件的单独结果
- 整体统计信息（均值、标准差、最值等）

## 参数说明

### `evaluate_segmentation.py`
- `--pred, -p`: 预测分割文件路径
- `--gt, -g`:  ground truth文件路径
- `--output, -o`: 结果输出路径
- `--resize-gt`: 自动调整GT尺寸以匹配预测

### `batch_evaluate.py`
- `--pred-dir, -p`: 预测文件目录
- `--gt-dir, -g`: GT文件目录
- `--output-dir, -o`: 结果输出目录
- `--pred-pattern`: 预测文件模式（默认: *_segmentation.tif）
- `--gt-pattern`: GT文件模式（默认: *.tif）

## 示例结果

```
Evaluation Results
==================
Prediction file: prediction.tif
Ground truth file: ground_truth.tif
Prediction shape: (128, 128, 128)
Ground truth shape: (128, 128, 128)

Average Dice: 0.8234
Matched labels: 18/22

Individual Dice scores:
  Label 1: 0.9234
  Label 2: 0.8456
  Label 3: 0.7891
  ...

Label mapping (pred -> gt):
  10 -> 1
  20 -> 2
  30 -> 3
  ...
```

## 注意事项

1. **文件格式**: 支持TIFF、Zarr和NIfTI (.nii.gz) 格式
2. **尺寸匹配**: 如果GT和预测尺寸不同，会自动调整GT尺寸
3. **标签ID**: 预测和GT的标签ID不需要相同，会自动匹配
4. **内存使用**: 大文件可能需要较多内存，建议分批处理
5. **NIfTI格式**: 使用SimpleITK读取，支持压缩的.nii.gz文件

## 依赖包

```bash
pip install numpy scipy scikit-image tifffile zarr SimpleITK
```

## 故障排除

1. **内存不足**: 使用批量处理或减小文件尺寸
2. **文件格式错误**: 确保文件格式正确且可读
3. **尺寸不匹配**: 使用`--resize-gt`参数自动调整
4. **标签匹配失败**: 检查预测和GT是否包含有效的标签
