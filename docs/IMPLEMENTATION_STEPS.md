# BANIS 重构实施步骤

本文档提供详细的分步实施指南，帮助团队按计划完成重构。

## 📋 总览

- **预计时间**: 7-12个工作日
- **风险等级**: 中等（通过渐进式迁移可降低风险）
- **建议人员**: 1-2名开发者
- **优先级**: 高

---

## 🎯 Phase 1: 配置系统重构 (1-2天)

### Day 1: 创建配置类

#### 任务清单
- [x] 创建 `configs/base_config.py` - 基础配置类
- [x] 创建 `configs/data_config.py` - 数据配置
- [x] 创建 `configs/training_config.py` - 训练配置  
- [x] 创建 `configs/model_config.py` - 模型配置
- [x] 创建 `configs/config_loader.py` - 配置加载器

#### 验证步骤
```python
# test_config.py
from configs.config_loader import load_config

# 测试配置加载
config = load_config("examples/train_mito.yaml")
config.validate()
print("✓ 配置系统工作正常")

# 测试类型检查
assert isinstance(config.training.batch_size, int)
assert isinstance(config.training.learning_rate, float)
print("✓ 类型检查通过")
```

#### 交付物
- ✅ 完整的配置类实现
- ✅ 配置验证逻辑
- ✅ 配置加载和保存功能
- ✅ 示例配置文件 `examples/train_mito.yaml`

---

### Day 2: 配置系统集成

#### 任务清单
- [ ] 实现 `load_legacy_config` - 兼容旧格式
- [ ] 编写配置转换脚本
- [ ] 测试配置与旧系统的兼容性
- [ ] 编写配置系统文档

#### 验证步骤
```bash
# 转换旧配置
python scripts/convert_config.py config.yaml configs/new_config.yaml

# 验证新配置
python -c "
from configs.config_loader import load_config
config = load_config('configs/new_config.yaml')
config.validate()
print('✓ 配置转换成功')
"
```

#### 交付物
- ✅ 配置转换工具
- ✅ 向后兼容性支持
- ✅ 配置系统文档

---

## 🏗️ Phase 2: 核心模块重构 (3-4天)

### Day 3: 模型模块

#### 任务清单
- [x] 创建 `banis/core/model.py`
  - [x] `create_model()` 工厂函数
  - [x] `load_model_from_checkpoint()` 加载函数
  - [x] `count_parameters()` 工具函数
- [x] 测试模型创建和加载

#### 实现示例
```python
# banis/core/model.py
from configs.model_config import ModelConfig

def create_model(config: ModelConfig) -> nn.Module:
    """创建模型 - 已实现"""
    if config.architecture == "mednext":
        return create_mednext_v1(...)
    raise ValueError(f"Unknown architecture: {config.architecture}")
```

#### 验证步骤
```python
# 测试模型创建
from banis.core.model import create_model
from configs import ModelConfig

config = ModelConfig(model_id="S")
model = create_model(config)
assert model is not None
print(f"✓ 模型创建成功，参数量: {count_parameters(model):,}")
```

---

### Day 4: Lightning模块

#### 任务清单
- [x] 创建 `banis/core/lightning_module.py`
  - [x] `BANISLightningModule` 类
  - [x] 训练和验证步骤
  - [x] 优化器配置
  - [x] 日志和可视化
- [x] 单元测试

#### 验证步骤
```python
# 测试Lightning模块
from banis.core.lightning_module import BANISLightningModule
from configs import ModelConfig, TrainingConfig

module = BANISLightningModule(
    model_config=ModelConfig(),
    training_config=TrainingConfig()
)

# 测试前向传播
dummy_input = torch.randn(1, 1, 64, 64, 64)
output = module(dummy_input)
assert output.shape[1] == 6  # 6个输出通道
print("✓ Lightning模块工作正常")
```

---

### Day 5: 训练器和数据加载

#### 任务清单
- [x] 创建 `banis/core/trainer.py`
  - [x] `BANISTrainer` 类
  - [x] 训练流程封装
  - [x] 检查点管理
  
- [ ] 重构数据加载 (可选，先使用旧代码)
  - [ ] `banis/data/dataset.py`
  - [ ] `banis/data/transforms.py`
  - [ ] `banis/data/affinity.py`

#### 验证步骤
```python
# 集成测试
from configs.config_loader import load_config
from banis.core.trainer import BANISTrainer

config = load_config("examples/train_mito.yaml")
config.training.fast_dev_run = 10  # 快速测试

trainer = BANISTrainer(config, train_loader, val_loader)
trainer.train()
print("✓ 训练器集成成功")
```

---

### Day 6: 推理模块

#### 任务清单
- [ ] 创建 `banis/inference/predictor.py`
  - [ ] `Predictor` 类
  - [ ] Patched inference实现
  - [ ] 后处理逻辑
  
- [ ] 创建 `banis/inference/postprocess.py`
  - [ ] 连通组件分析
  - [ ] 阈值处理

#### 实现示例
```python
# banis/inference/predictor.py
class Predictor:
    def __init__(self, model, patch_size=128, overlap=True):
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
    
    def predict(self, img: np.ndarray) -> np.ndarray:
        """执行推理"""
        # 实现patched inference
        return predictions
    
    def postprocess(self, pred: np.ndarray, threshold: float) -> np.ndarray:
        """后处理得到分割"""
        # 实现连通组件
        return segmentation
```

#### 验证步骤
```python
# 测试推理
from banis.inference.predictor import Predictor

predictor = Predictor(model, patch_size=128)
img = np.random.rand(256, 256, 256, 1)
pred = predictor.predict(img)
seg = predictor.postprocess(pred, threshold=0.5)
print(f"✓ 推理成功，分割shape: {seg.shape}")
```

---

## 🧰 Phase 3: 工具和脚本整理 (1-2天)

### Day 7: 脚本和工具

#### 任务清单
- [x] 创建 `scripts/train.py` - 主训练脚本
- [ ] 创建 `scripts/inference.py` - 推理脚本
- [ ] 创建 `scripts/evaluate.py` - 评估脚本
- [ ] 创建 `scripts/convert_config.py` - 配置转换

#### 实现要点

**train.py**:
```python
#!/usr/bin/env python3
"""训练脚本 - 已实现"""
from configs.config_loader import load_config
from banis.core.trainer import BANISTrainer

def main():
    args = parse_args()
    config = load_config(args.config, **parse_overrides(args.override))
    # ... 加载数据
    trainer = BANISTrainer(config, train_loader, val_loader)
    trainer.train()
```

**inference.py**:
```python
#!/usr/bin/env python3
"""推理脚本"""
from banis.inference.predictor import Predictor
from banis.core.model import load_model_from_checkpoint

def main():
    args = parse_args()
    model = load_model_from_checkpoint(args.checkpoint)
    predictor = Predictor(model)
    predictions = predictor.predict(load_data(args.input))
    save_predictions(predictions, args.output)
```

#### 验证步骤
```bash
# 测试训练脚本
python scripts/train.py --config examples/train_mito.yaml \
    --override training.fast_dev_run=10

# 测试推理脚本
python scripts/inference.py \
    --checkpoint outputs/test/checkpoints/last.ckpt \
    --input test_data.zarr \
    --output test_pred.zarr
```

---

### Day 8: 工具函数

#### 任务清单
- [x] 创建 `banis/utils/logging.py` - 日志工具
- [x] 创建 `banis/utils/io.py` - IO工具
- [x] 创建 `banis/utils/visualization.py` - 可视化
- [ ] 创建 `banis/utils/checkpoint.py` - 检查点管理

#### 验证步骤
```python
# 测试工具函数
from banis.utils import setup_logger, load_zarr, save_zarr

logger = setup_logger("test")
logger.info("测试日志")

data = load_zarr("test.zarr")
save_zarr(data, "output.zarr")
print("✓ 工具函数正常")
```

---

## 📝 Phase 4: 测试和文档 (2-3天)

### Day 9-10: 单元测试

#### 任务清单
- [ ] `tests/test_config.py` - 配置系统测试
- [ ] `tests/test_model.py` - 模型测试
- [ ] `tests/test_data.py` - 数据加载测试
- [ ] `tests/test_inference.py` - 推理测试
- [ ] `tests/test_trainer.py` - 训练器测试

#### 测试模板
```python
# tests/test_model.py
import pytest
from banis.core.model import create_model
from configs import ModelConfig

def test_create_model():
    """测试模型创建"""
    config = ModelConfig(model_id="S")
    model = create_model(config)
    assert model is not None
    
def test_model_forward():
    """测试前向传播"""
    config = ModelConfig(model_id="S")
    model = create_model(config)
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    assert y.shape == (1, 6, 64, 64, 64)

def test_model_checkpoint():
    """测试检查点加载"""
    # ... 实现测试
```

#### 测试覆盖率目标
- 核心模块: 90%+
- 数据处理: 80%+
- 工具函数: 70%+

#### 运行测试
```bash
# 运行所有测试
pytest tests/ -v

# 生成覆盖率报告
pytest --cov=banis --cov-report=html tests/
```

---

### Day 11: 文档编写

#### 任务清单
- [x] `README_NEW.md` - 主文档
- [x] `REFACTORING_PLAN.md` - 重构方案
- [x] `MIGRATION_GUIDE.md` - 迁移指南
- [x] `IMPLEMENTATION_STEPS.md` - 实施步骤
- [ ] `docs/API.md` - API文档
- [ ] `docs/QUICKSTART.md` - 快速开始
- [ ] `docs/CONTRIBUTING.md` - 贡献指南

#### 文档清单
```markdown
docs/
├── API.md              # API参考
├── QUICKSTART.md       # 5分钟快速开始
├── ADVANCED.md         # 高级用法
├── TROUBLESHOOTING.md  # 问题排查
└── CONTRIBUTING.md     # 贡献指南
```

---

## ✅ Phase 5: 验证和部署 (1天)

### Day 12: 最终验证

#### 完整功能测试

```bash
# 1. 配置转换测试
python scripts/convert_config.py config.yaml configs/test_config.yaml

# 2. 训练测试 (短时间)
python scripts/train.py --config configs/test_config.yaml \
    --override training.n_steps=100

# 3. 推理测试
python scripts/inference.py \
    --checkpoint outputs/test_exp/checkpoints/last.ckpt \
    --input test_data/sample.zarr \
    --output predictions/test_pred.zarr

# 4. 评估测试
python scripts/evaluate.py \
    --predictions predictions/test_pred.zarr \
    --ground-truth test_data/sample_gt.zarr \
    --skeleton test_data/sample_skeleton.pkl
```

#### 性能对比测试

```python
# compare_old_new.py
"""对比旧版和新版的结果"""

# 旧版训练
old_model = train_with_old_code(config)
old_metrics = evaluate(old_model)

# 新版训练
new_model = train_with_new_code(config)  
new_metrics = evaluate(new_model)

# 对比结果
assert abs(old_metrics['nerl'] - new_metrics['nerl']) < 0.01
print("✓ 结果一致性验证通过")
```

#### 检查清单

- [ ] 所有单元测试通过
- [ ] 集成测试通过
- [ ] 性能测试通过
- [ ] 文档完整
- [ ] 代码风格检查通过 (flake8, black, mypy)
- [ ] 无明显的内存泄漏
- [ ] 向后兼容性验证

---

## 📊 进度追踪

### 使用项目看板

创建GitHub Projects或使用其他工具追踪进度:

```
📋 TODO
- [ ] Phase 1: 配置系统
- [ ] Phase 2: 核心模块
- [ ] Phase 3: 工具脚本
- [ ] Phase 4: 测试文档
- [ ] Phase 5: 验证部署

🏗️ In Progress
- [x] 配置类实现
- [ ] Lightning模块

✅ Done
- [x] 项目规划
- [x] 目录结构创建
```

### 每日站会

每天简短同步:
1. 昨天完成了什么？
2. 今天计划做什么？
3. 遇到什么阻碍？

---

## 🚨 风险管理

### 潜在风险

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 破坏现有功能 | 高 | 充分测试，保留旧代码备份 |
| 时间超期 | 中 | 分阶段交付，优先核心功能 |
| 团队学习曲线 | 中 | 提供培训，编写详细文档 |
| 性能下降 | 低 | 性能基准测试 |

### 回滚计划

如果重构出现严重问题:

1. **立即回滚**
   ```bash
   git checkout main
   git branch refactor-backup
   ```

2. **保留新配置系统**
   - 新配置系统可以独立使用
   - 不影响旧代码运行

3. **渐进式迁移**
   - 只迁移稳定的模块
   - 保持新旧代码共存

---

## 📈 成功指标

### 定量指标

- ✅ 代码行数减少 30%+
- ✅ 重复代码减少 50%+
- ✅ 测试覆盖率 80%+
- ✅ 文档完整度 90%+
- ✅ 性能保持或提升

### 定性指标

- ✅ 代码可读性提高
- ✅ 维护成本降低
- ✅ 新功能开发更快
- ✅ Bug修复更容易
- ✅ 团队满意度提升

---

## 🎓 团队培训

### 培训计划

#### Week 1: 新架构介绍
- 重构动机和目标
- 新项目结构讲解
- 配置系统使用
- 实践：创建配置文件

#### Week 2: 核心模块使用
- 模型创建和加载
- 训练器使用
- 推理接口
- 实践：训练简单模型

#### Week 3: 高级特性
- 自定义数据集
- 扩展模型
- 调试技巧
- 实践：添加新功能

### 培训材料
- [ ] PPT: BANIS架构概览
- [ ] 视频: 配置系统教程
- [ ] 文档: API参考手册
- [ ] 示例: 端到端训练代码

---

## 📞 支持和沟通

### 沟通渠道
- 📧 Email: team@example.com
- 💬 Slack: #banis-refactor
- 🐛 Issues: GitHub Issues
- 📅 会议: 每周三下午3点

### 问题升级
1. 首先查看文档和FAQ
2. 在Slack提问
3. 创建GitHub Issue
4. 紧急问题直接联系技术负责人

---

## 🎉 里程碑庆祝

完成各阶段后，记得庆祝：

- ✅ Phase 1完成: 团队午餐
- ✅ Phase 3完成: 代码审查会
- ✅ 全部完成: 团队聚餐 🎊

---

**记住**: 这是一个持续改进的过程。不要追求完美，先让代码运行起来，然后逐步优化！

Good luck! 🚀

