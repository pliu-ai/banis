# 多数据集训练功能 - 修改日志

## 修改概述

为BANIS项目添加了多数据集联合训练功能，允许同时加载多个数据集来训练一个更通用的模型。

## 修改文件

### 1. BANIS.py

**修改内容：**

- **parse_args() 函数**:
  - 修改 `--data_setting` 参数，使用 `nargs="+"` 支持接受多个数据集名称
  - 默认值改为列表格式 `["base"]` 以保持一致性

- **main() 函数**:
  - 添加对多数据集的处理逻辑
  - 为每个数据集从 config.yaml 中查找对应的 resolution
  - 将 `data_settings` (列表) 和 `resolutions` (列表) 存储到 args 中
  - 保持向后兼容性，将第一个数据集的 resolution 作为主 resolution
  - 修改实验名称生成逻辑，多个数据集用 `+` 连接 (例如: `ds_betaSeg+han24+Jurkat`)

**关键代码片段：**
```python
# 支持单个或多个数据集
data_settings = args.data_setting if isinstance(args.data_setting, list) else [args.data_setting]

# 为每个数据集获取resolution
resolutions = []
for ds in data_settings:
    resolution = None
    # 从mito或rib配置中查找
    for setting in conf.mito.settings:
        if setting.name == ds:
            resolution = tuple(setting.resolution)
            break
    ...
    resolutions.append(resolution)

args.data_settings = data_settings
args.resolutions = resolutions
```

### 2. data.py

**修改内容：**

- **load_data() 函数**:
  - 添加对单个和多个数据集的处理逻辑
  - 单个数据集时保持原有行为（向后兼容）
  - 多个数据集时，为每个数据集创建独立的 args 副本并分别加载
  - 使用 ConcatDataset 合并所有训练数据集
  - 验证数据使用第一个数据集（可根据需要修改）
  - 添加详细的日志输出

- **get_syn_train_data() 函数**:
  - 添加对 data_setting 参数类型的检查（支持字符串或列表）
  - 添加数据路径存在性检查
  - 添加带数据集名称的日志输出

- **get_val_data() 函数**:
  - 添加对 data_setting 参数类型的检查
  - 添加数据路径存在性检查
  - 添加带数据集名称的日志输出

**关键代码片段：**
```python
def load_data(args: argparse.Namespace):
    data_settings = args.data_settings if hasattr(args, 'data_settings') else [args.data_setting]
    resolutions = args.resolutions if hasattr(args, 'resolutions') else [args.resolution]
    
    if len(data_settings) == 1:
        # 单数据集 - 原有行为
        train_data = get_train_data(args)
        val_data = get_val_data(args)
    else:
        # 多数据集 - 加载并合并
        train_datasets = []
        val_datasets = []
        
        for ds, res in zip(data_settings, resolutions):
            ds_args = argparse.Namespace(**vars(args))
            ds_args.data_setting = ds
            ds_args.resolution = res
            
            train_ds = get_train_data(ds_args)
            val_ds = get_val_data(ds_args)
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
        
        train_data = ConcatDataset(train_datasets)
        val_data = val_datasets[0]  # 使用第一个数据集
    
    return train_data, val_data, val_data.img.shape[-1]
```

### 3. justfile

**添加内容：**

- `train_multi_mito`: 训练3个mito数据集的示例命令
- `train_all_mito`: 训练所有8个mito数据集的示例命令
- `launch_multi_mito`: 使用launcher提交多数据集训练任务

### 4. 新增文件

- **MULTI_DATASET_TRAINING.md**: 详细的使用文档和说明
- **CHANGELOG_MULTI_DATASET.md**: 本文件，修改日志

## 使用示例

### 单数据集（向后兼容）
```bash
uv run BANIS.py --data_setting betaSeg --base_data_path ./mito_data ...
```

### 多数据集
```bash
# 2个数据集
uv run BANIS.py --data_setting betaSeg han24 --base_data_path ./mito_data ...

# 3个数据集
uv run BANIS.py --data_setting betaSeg han24 Jurkat --base_data_path ./mito_data ...

# 使用justfile
just train_multi_mito
just train_all_mito
```

## 设计决策

1. **向后兼容性**: 保留了原有的单数据集训练功能，确保现有代码和脚本不受影响

2. **验证策略**: 默认使用第一个数据集的验证集，因为：
   - 在所有数据集上验证会显著增加训练时间
   - 第一个数据集通常是代表性最强的
   - 如需在所有数据集上验证，可以轻松修改 load_data() 函数

3. **数据合并**: 使用 ConcatDataset 直接合并所有训练数据，不使用加权采样，因为：
   - 简单直接，易于理解和调试
   - 每个数据集贡献的样本数与其大小成正比
   - 如需加权采样，可以使用已有的 WeightedConcatDataset

4. **参数设计**: 
   - 使用 nargs="+" 允许一个参数接受多个值
   - 保持参数名称 `--data_setting` 不变，避免破坏性修改
   - 默认值设为列表 `["base"]` 保持类型一致性

## 测试

- ✅ 参数解析测试通过
- ✅ 单数据集训练（向后兼容性）验证
- ✅ 多数据集参数处理验证
- ✅ 代码 linter 检查通过（仅外部依赖警告）

## 注意事项

1. 所有数据集必须在同一个 base_data_path 下
2. 所有数据集应具有相同的输入通道数
3. 训练多个数据集会增加内存使用，建议适当降低 batch_size
4. 确保 config.yaml 中已正确配置所有要使用的数据集

## 未来改进方向

1. 支持对不同数据集使用加权采样
2. 支持在所有数据集上进行验证（可选）
3. 支持混合不同分辨率的数据集（需要额外的预处理）
4. 添加数据集统计信息的自动日志记录







