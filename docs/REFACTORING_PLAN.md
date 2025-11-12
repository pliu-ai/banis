# BANIS 项目重构方案

## 📊 当前问题分析

### 1. 代码组织问题
- ❌ 核心逻辑(BANIS.py)过于庞大(555行)
- ❌ 数据加载、模型训练、推理混在一起
- ❌ 工具脚本散落在根目录
- ❌ 缺乏清晰的模块边界

### 2. 配置管理问题
- ❌ argparse和YAML配置混用
- ❌ 配置参数散落在多处
- ❌ 缺乏配置验证
- ❌ 没有使用dataclass进行类型安全

### 3. 代码质量问题
- ❌ 缺乏类型注解
- ❌ 文档不完整
- ❌ 错误处理不统一
- ❌ 日志记录不规范

### 4. 可维护性问题
- ❌ 重复代码多(batch_*.py文件)
- ❌ 硬编码路径和参数
- ❌ 测试覆盖率低
- ❌ 依赖关系不清晰

---

## 🎯 重构目标

### 1. 清晰的项目结构
```
banis/
├── configs/                  # 配置文件
│   ├── __init__.py
│   ├── base_config.py       # 基础配置dataclass
│   ├── training_config.py   # 训练配置
│   ├── data_config.py       # 数据配置
│   └── default.yaml         # 默认配置
├── banis/                   # 核心代码包
│   ├── __init__.py
│   ├── core/                # 核心模块
│   │   ├── __init__.py
│   │   ├── model.py        # 模型定义
│   │   ├── trainer.py      # 训练器
│   │   └── lightning_module.py  # PyTorch Lightning模块
│   ├── data/                # 数据处理
│   │   ├── __init__.py
│   │   ├── dataset.py      # 数据集类
│   │   ├── transforms.py   # 数据增强
│   │   ├── affinity.py     # 亲和力计算
│   │   └── dataloader.py   # 数据加载器
│   ├── inference/           # 推理模块
│   │   ├── __init__.py
│   │   ├── predictor.py    # 推理器
│   │   ├── postprocess.py  # 后处理
│   │   └── connected_components.py  # 连通组件
│   ├── evaluation/          # 评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py      # 指标计算
│   │   └── evaluator.py    # 评估器
│   └── utils/               # 工具函数
│       ├── __init__.py
│       ├── logging.py      # 日志配置
│       ├── checkpoint.py   # 检查点管理
│       ├── visualization.py # 可视化
│       └── io.py           # IO操作
├── scripts/                 # 脚本文件
│   ├── train.py            # 训练脚本
│   ├── inference.py        # 推理脚本
│   ├── evaluate.py         # 评估脚本
│   └── preprocess_data.py  # 数据预处理
├── tests/                   # 测试文件
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_inference.py
├── notebooks/               # Jupyter notebooks
│   └── visualization.ipynb
├── docs/                    # 文档
│   ├── INSTALLATION.md
│   ├── QUICKSTART.md
│   ├── API.md
│   └── CONTRIBUTING.md
├── examples/                # 示例代码
│   └── train_mito.yaml
├── environment.yaml         # Conda环境
├── pyproject.toml          # 项目配置
├── setup.py                # 安装脚本
├── README.md               # 项目说明
└── LICENSE                 # 许可证
```

### 2. 使用dataclass进行配置管理
- ✅ 类型安全
- ✅ 自动验证
- ✅ 清晰的配置结构
- ✅ 支持YAML序列化

### 3. 模块化设计
- ✅ 单一职责原则
- ✅ 清晰的接口定义
- ✅ 依赖注入
- ✅ 易于测试和扩展

### 4. 完善的文档和注释
- ✅ 所有公共API都有文档字符串
- ✅ 类型注解
- ✅ 使用示例
- ✅ API文档

---

## 🚀 实施步骤

### Phase 1: 配置系统重构 (1-2天)
1. 创建dataclass配置类
2. 实现YAML加载和验证
3. 迁移现有配置

### Phase 2: 核心模块重构 (3-4天)
1. 拆分BANIS.py
2. 重构数据加载模块
3. 重构模型和训练器
4. 重构推理模块

### Phase 3: 工具和脚本整理 (1-2天)
1. 合并重复的batch脚本
2. 创建统一的CLI接口
3. 添加日志和错误处理

### Phase 4: 测试和文档 (2-3天)
1. 编写单元测试
2. 编写集成测试
3. 完善文档
4. 代码审查

### Phase 5: 验证和部署 (1天)
1. 功能验证
2. 性能测试
3. 文档审查
4. 发布

---

## 📝 关键改进点

### 1. 配置管理
**Before:**
```python
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-3)
# ... 50+ arguments
```

**After:**
```python
@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    # ... with type hints and validation
```

### 2. 模型封装
**Before:**
- 555行的单一文件
- 训练、推理、评估混在一起

**After:**
- 模型定义独立
- 训练逻辑独立
- 推理逻辑独立
- 清晰的接口

### 3. 错误处理
**Before:**
```python
# 缺乏错误处理
img_data = zarr.open(os.path.join(seed_path, "data.zarr"))["img"]
```

**After:**
```python
try:
    img_data = load_zarr_data(seed_path, "data.zarr", "img")
except DataLoadError as e:
    logger.error(f"Failed to load data: {e}")
    raise
```

### 4. 代码复用
**Before:**
- batch_convert.py
- batch_evaluate.py
- batch_extract_example.py
(大量重复代码)

**After:**
```python
# 统一的批处理框架
batch_processor = BatchProcessor(config)
batch_processor.process(task_type="convert")
```

---

## 💡 最佳实践应用

### 1. 类型注解
```python
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch

def patched_inference(
    img: np.ndarray,
    model: torch.nn.Module,
    small_size: int = 128,
    do_overlap: bool = True,
) -> np.ndarray:
    """Perform patched inference with type safety."""
    ...
```

### 2. 依赖注入
```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        config: TrainingConfig,
    ):
        """Dependencies injected via constructor."""
        ...
```

### 3. 工厂模式
```python
class ModelFactory:
    @staticmethod
    def create(config: ModelConfig) -> nn.Module:
        """Create model based on config."""
        if config.architecture == "mednext":
            return create_mednext_v1(**config.to_dict())
        ...
```

### 4. 上下文管理器
```python
with CheckpointManager(config.save_dir) as ckpt:
    for epoch in range(config.epochs):
        train_one_epoch()
        ckpt.save_if_best(model, metrics)
```

---

## 🔍 代码质量保证

### 1. 静态检查
```bash
# Type checking
mypy banis/

# Linting
flake8 banis/
black banis/

# Import sorting
isort banis/
```

### 2. 单元测试
```python
# tests/test_affinity.py
def test_compute_affinities():
    seg = np.array([[[0, 1, 1], [1, 1, 2]]])
    aff, mask = compute_affinities(seg, long_range=1)
    assert aff.shape == (6, *seg.shape)
    assert mask.dtype == bool
```

### 3. 集成测试
```python
# tests/test_training.py
def test_full_training_pipeline():
    config = load_config("tests/fixtures/test_config.yaml")
    trainer = Trainer(config)
    metrics = trainer.train()
    assert metrics["val_loss"] < 1.0
```

---

## 📚 文档模板

### API文档
```python
def compute_connected_components(
    hard_aff: np.ndarray
) -> np.ndarray:
    """
    Compute connected components from affinities.
    
    Args:
        hard_aff: Boolean affinities with shape (3, X, Y, Z).
                 Channel 0: X-axis connections
                 Channel 1: Y-axis connections  
                 Channel 2: Z-axis connections
    
    Returns:
        Segmentation with shape (X, Y, Z), where each connected
        component has a unique integer ID starting from 1.
        Background voxels are labeled as 0.
    
    Raises:
        ValueError: If input shape is invalid.
        
    Example:
        >>> aff = np.random.rand(3, 64, 64, 64) > 0.5
        >>> seg = compute_connected_components(aff)
        >>> assert seg.shape == (64, 64, 64)
    """
    ...
```

---

## ⚠️ 注意事项

### 1. 向后兼容
- 保留旧的CLI接口
- 提供迁移脚本
- 文档说明变更

### 2. 性能
- 确保重构不影响性能
- 添加性能测试
- 必要时使用profiling

### 3. 测试覆盖
- 核心功能100%覆盖
- 关键路径集成测试
- 回归测试

### 4. 渐进式迁移
- 逐模块重构
- 保持CI/CD绿色
- 频繁的代码审查

---

## 📈 预期收益

### 1. 开发效率
- ✅ 减少50%的重复代码
- ✅ 新功能开发速度提升30%
- ✅ Bug修复时间减少40%

### 2. 代码质量
- ✅ 类型安全
- ✅ 更好的可测试性
- ✅ 降低技术债务

### 3. 团队协作
- ✅ 清晰的代码结构
- ✅ 完善的文档
- ✅ 易于上手

### 4. 可维护性
- ✅ 模块化设计
- ✅ 依赖关系清晰
- ✅ 易于扩展

---

## 🎓 参考资源

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [PyTorch Lightning Best Practices](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Clean Code in Python](https://github.com/zedr/clean-code-python)

---

**重构完成时间估计**: 7-12天  
**建议团队规模**: 1-2人  
**风险等级**: 中等 (可通过渐进式迁移降低风险)

