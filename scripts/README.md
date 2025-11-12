# Scripts Directory

## 📋 目录说明

此目录包含所有可执行脚本，用于训练、推理、评估等任务。

## 📁 目录结构

```
scripts/
├── train.py              # 训练脚本
├── inference.py          # 推理脚本（待实现）
├── evaluate.py           # 评估脚本（待实现）
├── cleanup_project.sh    # 项目整理脚本
└── slurm/               # Slurm集群相关
    ├── submit_slurm.py
    ├── slurm_job_scheduler.py
    ├── validation_watcher.py
    ├── validation_watcher.sh
    └── aff_train.sh
```

## 🚀 主要脚本

### train.py - 训练脚本

训练BANIS模型的主要脚本。

**基本用法**:
```bash
# 使用配置文件训练
python scripts/train.py --config examples/configs/train_mito.yaml

# 使用命令行覆盖参数
python scripts/train.py --config examples/configs/train_mito.yaml \
    --override training.batch_size=16 training.learning_rate=0.002

# 从检查点恢复训练
python scripts/train.py --config examples/configs/train_mito.yaml \
    --override training.resume_from_checkpoint=outputs/exp/checkpoints/last.ckpt

# 调试模式（只运行10个batch）
python scripts/train.py --config examples/configs/train_mito.yaml \
    --override training.fast_dev_run=10
```

**参数说明**:
- `--config`: 配置文件路径
- `--override`: 覆盖配置参数（可多个）
- `--log-level`: 日志级别（DEBUG, INFO, WARNING, ERROR）

### inference.py - 推理脚本

对新数据进行推理预测。

**用法**:
```bash
python scripts/inference.py \
    --checkpoint outputs/exp/checkpoints/best.ckpt \
    --input data/test/image.zarr \
    --output predictions/pred.zarr
```

### evaluate.py - 评估脚本

评估模型预测结果。

**用法**:
```bash
python scripts/evaluate.py \
    --predictions predictions/pred.zarr \
    --ground-truth data/test/gt.zarr \
    --skeleton data/test/skeleton.pkl
```

## 🖥️ Slurm集群脚本

### submit_slurm.py - 提交Slurm作业

提交训练作业到Slurm集群。

**用法**:
```bash
python scripts/slurm/submit_slurm.py \
    --config examples/configs/train_mito.yaml \
    --job-name my_training \
    --time 24:00:00 \
    --mem 64G
```

### slurm_job_scheduler.py - 作业调度器

批量提交多个作业。

**用法**:
```bash
python scripts/slurm/slurm_job_scheduler.py
```

根据 `config.yaml` 中的参数网格自动生成并提交多个作业。

### validation_watcher.py - 验证监控

监控验证进程，用于长时间训练。

### aff_train.sh - Slurm批处理脚本

Slurm批处理作业脚本模板。

## 🛠️ 工具脚本

### cleanup_project.sh - 项目整理

整理项目文件结构。

**用法**:
```bash
bash scripts/cleanup_project.sh
```

## 📖 使用示例

### 示例1: 训练线粒体分割模型

```bash
# 1. 准备配置
cp examples/configs/train_mito.yaml my_config.yaml

# 2. 开始训练
python scripts/train.py --config my_config.yaml

# 3. 监控训练
tensorboard --logdir outputs/
```

### 示例2: 在Slurm上批量实验

```bash
# 1. 编辑config.yaml设置参数网格
vim config.yaml

# 2. 提交作业
python scripts/slurm/slurm_job_scheduler.py

# 3. 监控作业
squeue -u $USER
```

### 示例3: 快速调试

```bash
# 只运行5个batch验证代码
python scripts/train.py --config my_config.yaml \
    --override training.fast_dev_run=5 \
    --log-level DEBUG
```

## 🔧 开发新脚本

创建新脚本时，请遵循以下模板：

```python
#!/usr/bin/env python3
"""脚本简短描述。

详细描述脚本功能和用法。

Example:
    python scripts/my_script.py --arg1 value1
"""

import argparse
from pathlib import Path

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="脚本描述")
    parser.add_argument("--arg1", required=True, help="参数说明")
    return parser.parse_args()

def main():
    """主函数。"""
    args = parse_args()
    # 实现功能
    pass

if __name__ == "__main__":
    main()
```

## 📝 脚本规范

1. **使用argparse**: 所有脚本都应使用argparse解析参数
2. **添加帮助信息**: 每个参数都要有清晰的help说明
3. **错误处理**: 添加适当的try-except和错误提示
4. **日志记录**: 使用logging模块记录日志
5. **类型注解**: 添加类型提示提高代码可读性

## 🆘 故障排查

### 脚本无法执行

```bash
# 检查权限
ls -l scripts/*.py

# 添加执行权限
chmod +x scripts/*.py
```

### 导入错误

```bash
# 确保在项目根目录运行
cd /projects/weilab/liupeng/banis

# 或设置PYTHONPATH
export PYTHONPATH=/projects/weilab/liupeng/banis:$PYTHONPATH
```

### Slurm作业失败

```bash
# 查看错误日志
cat logs/slurm-JOBID.err

# 检查作业状态
scontrol show job JOBID
```

## 📚 更多信息

- 主文档: `../README.md`
- 快速开始: `../docs/QUICKSTART.md`
- API文档: `../docs/API.md`

---

**需要帮助？** 查看 `../docs/` 获取更多文档 📖

