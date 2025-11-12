# 迁移指南：从旧版BANIS到重构版本

本指南帮助你从旧的BANIS代码库平滑迁移到新的重构版本。

## 📋 目录

- [主要变化](#主要变化)
- [配置系统迁移](#配置系统迁移)
- [代码迁移](#代码迁移)
- [训练脚本迁移](#训练脚本迁移)
- [推理代码迁移](#推理代码迁移)
- [常见问题](#常见问题)

---

## 🔄 主要变化

### 1. 配置管理

**旧版本**: 使用argparse + YAML混合方式
```python
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-3)
# ... 50多个参数
```

**新版本**: 使用dataclass + YAML
```python
from configs import TrainingConfig

config = TrainingConfig(
    batch_size=8,
    learning_rate=1e-3
)
# 类型安全，自动补全，验证
```

### 2. 项目结构

**旧版本**:
```
banis/
├── BANIS.py (555行)
├── data.py
├── inference.py
├── metrics.py
├── batch_*.py (多个重复脚本)
└── config.yaml
```

**新版本**:
```
banis/
├── configs/          # 配置类
├── banis/           # 核心包
│   ├── core/        # 模型和训练
│   ├── data/        # 数据处理
│   ├── inference/   # 推理
│   ├── evaluation/  # 评估
│   └── utils/       # 工具
├── scripts/         # 入口脚本
└── examples/        # 配置示例
```

### 3. 模块化

**旧版本**: 所有功能在单个文件中
- BANIS.py: 训练、验证、推理、评估都在一起

**新版本**: 清晰的模块边界
- `core/`: 模型定义和训练逻辑
- `data/`: 数据加载和处理
- `inference/`: 推理和后处理
- `evaluation/`: 评估指标

---

## ⚙️ 配置系统迁移

### 步骤1: 转换旧配置文件

使用提供的转换工具:

```python
from configs.config_loader import load_legacy_config, save_config

# 读取旧配置
config = load_legacy_config("config.yaml")

# 保存为新格式
save_config(config, "configs/new_config.yaml")
```

### 步骤2: 手动调整配置

新配置文件结构:

```yaml
# 旧版本 config.yaml
params:
  learning_rate: [1e-3]
  batch_size: [8]
  seed: [0]

# 新版本 (更清晰)
training:
  learning_rate: 0.001
  batch_size: 8
  seed: 0

model:
  model_id: S
  kernel_size: 3

data:
  datasets:
    - name: betaSeg
      resolution: [16, 16, 16]
      path: /path/to/data
      train_splits: ["vol1", "vol2"]
```

### 步骤3: 验证配置

```python
from configs.config_loader import load_config

# 加载并验证
config = load_config("configs/new_config.yaml")
config.validate()  # 自动检查配置是否合法
```

---

## 💻 代码迁移

### 训练代码

#### 旧版本
```python
# 旧的训练方式
from BANIS import BANIS, parse_args

args = parse_args()
model = BANIS(**vars(args))
trainer = pl.Trainer(...)
trainer.fit(model, train_loader, val_loader)
```

#### 新版本
```python
# 新的训练方式
from configs.config_loader import load_config
from banis.core.trainer import BANISTrainer

config = load_config("configs/train.yaml")
trainer = BANISTrainer(config, train_loader, val_loader)
trainer.train()
```

### 模型创建

#### 旧版本
```python
# 在BANIS类的__init__中创建
self.model = create_mednext_v1(
    num_input_channels=self.hparams.num_input_channels,
    num_classes=6,
    model_id=self.hparams.model_id,
    kernel_size=self.hparams.kernel_size,
)
```

#### 新版本
```python
# 使用工厂函数
from banis.core.model import create_model
from configs import ModelConfig

config = ModelConfig(model_id="S", kernel_size=3)
model = create_model(config)
```

### 数据加载

#### 旧版本
```python
from data import load_data

train_data, val_data, n_channels = load_data(args)
```

#### 新版本
```python
from banis.data.dataset import AffinityDataset
from configs import DataConfig

# 创建数据集
train_dataset = AffinityDataset(
    seg=train_seg,
    img=train_img,
    config=data_config
)

# 或使用辅助函数
from banis.data.dataloader import create_dataloaders

train_loader, val_loader = create_dataloaders(data_config)
```

### 推理

#### 旧版本
```python
from inference import patched_inference, compute_connected_component_segmentation

aff_pred = patched_inference(img_data, model=model, small_size=128)
seg = compute_connected_component_segmentation(aff_pred[:3] > threshold)
```

#### 新版本
```python
from banis.inference.predictor import Predictor

predictor = Predictor(model, patch_size=128, overlap=True)
predictions = predictor.predict(img_data)
segmentation = predictor.postprocess(predictions, threshold=0.5)
```

### 评估

#### 旧版本
```python
from metrics import compute_metrics

metrics = compute_metrics(pred_seg, skel_path)
```

#### 新版本
```python
from banis.evaluation.evaluator import Evaluator

evaluator = Evaluator()
metrics = evaluator.evaluate(pred_seg, skeleton_path=skel_path)
```

---

## 🎯 训练脚本迁移

### 旧的训练命令

```bash
python BANIS.py \
    --seed 0 \
    --batch_size 8 \
    --n_steps 50000 \
    --data_setting base \
    --base_data_path /path/to/data/ \
    --save_path /path/to/outputs/
```

### 新的训练命令

```bash
# 使用配置文件
python scripts/train.py --config examples/train_mito.yaml

# 使用命令行覆盖
python scripts/train.py --config examples/train_mito.yaml \
    --override \
    training.seed=0 \
    training.batch_size=8 \
    training.n_steps=50000
```

---

## 🔍 推理代码迁移

### 完整示例：旧 → 新

#### 旧版本推理脚本

```python
# old_inference.py
import argparse
import zarr
import torch
from BANIS import BANIS
from inference import patched_inference, compute_connected_component_segmentation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

# 加载模型
checkpoint = torch.load(args.checkpoint)
model = BANIS.load_from_checkpoint(args.checkpoint)
model.eval()
model.cuda()

# 加载数据
img = zarr.open(args.input)["img"]

# 推理
aff = patched_inference(img, model, small_size=128)

# 后处理
seg = compute_connected_component_segmentation(aff[:3] > 0.5)

# 保存
zarr.array(seg, store=args.output, overwrite=True)
```

#### 新版本推理脚本

```python
# scripts/inference.py
import argparse
from pathlib import Path
from banis.core.model import load_model_from_checkpoint
from banis.inference.predictor import Predictor
from banis.utils.io import load_zarr, save_zarr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    # 加载模型 (更简单，自动处理设备)
    model = load_model_from_checkpoint(args.checkpoint)
    
    # 创建预测器 (封装了所有推理逻辑)
    predictor = Predictor(
        model=model,
        patch_size=128,
        overlap=True,
        device="cuda"
    )
    
    # 加载数据 (带错误处理)
    img = load_zarr(args.input, key="img")
    
    # 推理 (一行代码)
    predictions = predictor.predict(img)
    
    # 后处理 (清晰的接口)
    segmentation = predictor.postprocess(
        predictions,
        threshold=args.threshold
    )
    
    # 保存 (自动处理路径和格式)
    save_zarr(segmentation, args.output, chunks=(512, 512, 512))
    
    print(f"Segmentation saved to {args.output}")

if __name__ == "__main__":
    main()
```

---

## 🛠️ 实用迁移工具

### 自动化迁移脚本

创建 `scripts/migrate_from_legacy.py`:

```python
#!/usr/bin/env python3
"""自动迁移旧代码到新结构的工具"""

import shutil
from pathlib import Path
from configs.config_loader import load_legacy_config, save_config

def migrate_config(old_config_path: Path, new_config_path: Path):
    """迁移配置文件"""
    print(f"迁移配置: {old_config_path} -> {new_config_path}")
    config = load_legacy_config(old_config_path)
    save_config(config, new_config_path)
    print("✓ 配置迁移完成")

def migrate_checkpoints(old_ckpt_dir: Path, new_ckpt_dir: Path):
    """复制和整理检查点"""
    print(f"迁移检查点: {old_ckpt_dir} -> {new_ckpt_dir}")
    new_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    for ckpt in old_ckpt_dir.glob("*.ckpt"):
        shutil.copy(ckpt, new_ckpt_dir / ckpt.name)
        print(f"  ✓ {ckpt.name}")

def main():
    # 迁移配置
    migrate_config(
        Path("config.yaml"),
        Path("configs/migrated_config.yaml")
    )
    
    # 迁移检查点
    migrate_checkpoints(
        Path("outputs/old_exp/checkpoints"),
        Path("outputs/migrated_exp/checkpoints")
    )
    
    print("\n" + "="*60)
    print("迁移完成! 下一步:")
    print("1. 检查 configs/migrated_config.yaml")
    print("2. 使用新脚本训练:")
    print("   python scripts/train.py --config configs/migrated_config.yaml")
    print("="*60)

if __name__ == "__main__":
    main()
```

---

## ❓ 常见问题

### Q1: 旧的检查点能在新代码中使用吗？

**A**: 可以! 新的 `load_model_from_checkpoint` 函数兼容旧格式:

```python
from banis.core.model import load_model_from_checkpoint

# 自动处理旧格式
model = load_model_from_checkpoint("old_checkpoint.ckpt")
```

### Q2: 数据格式需要改变吗？

**A**: 不需要。新代码完全兼容旧的数据格式（zarr, pkl等）。

### Q3: 能否渐进式迁移？

**A**: 可以! 建议步骤:
1. 先迁移配置文件
2. 使用新的训练脚本训练新模型
3. 逐步迁移推理和评估代码
4. 保持旧代码作为备份

### Q4: 性能会改变吗？

**A**: 不会。核心算法保持不变，只是组织方式更好。

### Q5: 需要重新训练模型吗？

**A**: 不需要。旧模型可以直接使用。但建议用新代码重新训练以确保最佳性能。

### Q6: 多久能完成迁移？

**A**: 
- 小项目: 1-2小时
- 中等项目: 半天
- 大项目: 1-2天

---

## 📊 迁移清单

使用此清单追踪迁移进度:

- [ ] **配置迁移**
  - [ ] 转换config.yaml到新格式
  - [ ] 验证所有配置参数
  - [ ] 测试配置加载

- [ ] **代码迁移**
  - [ ] 更新import语句
  - [ ] 迁移训练脚本
  - [ ] 迁移推理脚本
  - [ ] 迁移评估脚本

- [ ] **测试**
  - [ ] 训练小样本验证
  - [ ] 推理测试
  - [ ] 评估指标对比
  - [ ] 性能测试

- [ ] **文档**
  - [ ] 更新README
  - [ ] 更新团队文档
  - [ ] 添加使用示例

- [ ] **清理**
  - [ ] 备份旧代码
  - [ ] 删除重复文件
  - [ ] 更新.gitignore

---

## 🎓 学习资源

### 新代码库的关键概念

1. **Dataclass配置**: [configs/base_config.py](configs/base_config.py)
2. **模型工厂**: [banis/core/model.py](banis/core/model.py)
3. **训练器封装**: [banis/core/trainer.py](banis/core/trainer.py)
4. **预测器接口**: [banis/inference/predictor.py](banis/inference/predictor.py)

### 推荐阅读顺序

1. [README_NEW.md](README_NEW.md) - 整体概览
2. [examples/train_mito.yaml](examples/train_mito.yaml) - 配置示例
3. [scripts/train.py](scripts/train.py) - 训练脚本
4. [REFACTORING_PLAN.md](REFACTORING_PLAN.md) - 设计思路

---

## 💡 最佳实践

### 1. 保持向后兼容

```python
# 提供兼容层
def legacy_train_func(**kwargs):
    """兼容旧接口的训练函数"""
    # 转换旧参数到新配置
    config = convert_legacy_args(kwargs)
    # 使用新代码
    trainer = BANISTrainer(config)
    trainer.train()
```

### 2. 逐步迁移

不要一次性重写所有代码，而是:
1. 新功能用新代码
2. 旧功能保持不动
3. 逐步替换旧代码

### 3. 充分测试

```python
# 确保结果一致
old_result = old_train_function(args)
new_result = new_train_function(config)
assert torch.allclose(old_result, new_result)
```

---

## 🆘 需要帮助？

- 📖 查看完整文档: [docs/](docs/)
- 🐛 报告问题: [GitHub Issues](https://github.com/your-org/banis/issues)
- 💬 讨论: [Discussions](https://github.com/your-org/banis/discussions)
- 📧 邮件: your-email@example.com

---

**记住**: 迁移是一个渐进的过程。不要急于一次性完成，分步骤进行会更加顺利！

