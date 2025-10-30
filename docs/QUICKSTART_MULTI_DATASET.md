# 快速开始：多数据集训练

## 一句话总结
在 `--data_setting` 后面指定多个数据集名称（用空格分隔），即可训练一个在多个数据集上的通用模型。

## 快速示例

### 1️⃣ 训练单个数据集（原有功能）
```bash
uv run BANIS.py \
    --data_setting betaSeg \
    --base_data_path /projects/weilab/liupeng/banis/mito_data \
    --save_path ./mito_outputs \
    --batch_size 8 \
    --n_steps 500000 \
    --devices=1 \
    --sdt
```

### 2️⃣ 训练多个数据集（新功能）
```bash
uv run BANIS.py \
    --data_setting betaSeg han24 Jurkat \
    --base_data_path /projects/weilab/liupeng/banis/mito_data \
    --save_path ./mito_outputs \
    --batch_size 4 \
    --n_steps 500000 \
    --devices=1 \
    --sdt
```

### 3️⃣ 使用 justfile 快捷命令
```bash
# 训练3个数据集
just train_multi_mito

# 训练所有8个mito数据集
just train_all_mito
```

## 关键变化

| 项目 | 单数据集 | 多数据集 |
|------|---------|---------|
| 参数 | `--data_setting betaSeg` | `--data_setting betaSeg han24 Jurkat` |
| 实验名称 | `ds_betaSeg_...` | `ds_betaSeg+han24+Jurkat_...` |
| 训练数据 | 单个数据集 | 所有数据集合并 |
| 验证数据 | 该数据集 | 第一个数据集 |
| Batch size建议 | 8 | 4-2（降低以应对更多数据）|

## 查看可用数据集

查看 `config.yaml` 中的配置：
```yaml
mito:
  settings:
    - name: betaSeg       # ✅ 可用
    - name: han24         # ✅ 可用
    - name: Jurkat        # ✅ 可用
    - name: Cardiac       # ✅ 可用
    ...

rib:
  settings:
    - name: ribFrac       # ✅ 可用
```

## 常见用法

### 训练组合1：3个小数据集测试
```bash
uv run BANIS.py --data_setting betaSeg han24 Jurkat \
    --base_data_path /projects/weilab/liupeng/banis/mito_data \
    --save_path ./mito_outputs --batch_size 4 --n_steps 500000 --devices=1 --sdt
```

### 训练组合2：所有mito数据集（通用模型）
```bash
uv run BANIS.py --data_setting betaSeg han24 Jurkat Cardiac Kidney Liver Sperm Macrophage \
    --base_data_path /projects/weilab/liupeng/banis/mito_data \
    --save_path ./mito_outputs --batch_size 2 --n_steps 1000000 --devices=1 --sdt
```

### 训练组合3：特定场景（如CellMap数据集）
```bash
uv run BANIS.py --data_setting Jurkat Cardiac Kidney Liver Macrophage \
    --base_data_path /projects/weilab/liupeng/banis/mito_data \
    --save_path ./mito_outputs --batch_size 4 --n_steps 750000 --devices=1 --sdt
```

## 训练输出示例

```
Loading multiple datasets: ['betaSeg', 'han24', 'Jurkat']

Loading dataset: betaSeg with resolution (16, 16, 16)
[betaSeg] image+segmentation paths: ['/projects/.../train/seed_0/data.zarr']
[betaSeg] Segmentation shapes: [(100, 512, 512)]

Loading dataset: han24 with resolution (30, 8, 8)
[han24] image+segmentation paths: ['/projects/.../train/seed_0/data.zarr']
[han24] Segmentation shapes: [(200, 512, 512)]

Loading dataset: Jurkat with resolution (16, 16, 16)
[Jurkat] image+segmentation paths: ['/projects/.../train/seed_0/data.zarr']
[Jurkat] Segmentation shapes: [(150, 512, 512)]

Using validation data from: betaSeg

save dir: ./mito_outputs/25-10-26_10-30-45-123456ds_betaSeg+han24+Jurkat_lrng10_s0_b4_mS_k3_...
```

## 注意事项

⚠️ **内存**: 多数据集训练需要更多内存，建议降低 batch_size  
⚠️ **路径**: 所有数据集必须在同一 base_data_path 下  
⚠️ **配置**: 确保 config.yaml 中已配置所有指定的数据集  

## 更多信息

- 详细文档: `MULTI_DATASET_TRAINING.md`
- 修改日志: `CHANGELOG_MULTI_DATASET.md`
- 配置文件: `config.yaml`







