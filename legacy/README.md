# Legacy Code Archive

## 📋 说明

这个目录包含重构前的旧代码，保留用于：
- **参考对比**: 查看旧实现方式
- **向后兼容**: 过渡期间使用
- **测试验证**: 对比新旧结果

## 📁 内容

- `BANIS.py` - 旧的训练主程序 (555行)
- `data.py` - 旧的数据加载模块 (550行)
- `inference.py` - 旧的推理代码 (243行)
- `metrics.py` - 旧的评估指标 (304行)
- `batch_*.py` - 批处理脚本
- 其他工具脚本

## 🔧 使用旧代码

如果需要运行旧代码：

```bash
# 方法1: 从legacy目录运行
cd legacy/
python BANIS.py --help

# 方法2: 从根目录运行
python legacy/BANIS.py --config config.yaml

# 方法3: 在Python中导入
from legacy.BANIS import BANIS
model = BANIS(**args)
```

## 🚀 迁移到新代码

**强烈推荐使用新代码！** 查看：
- `../docs/MIGRATION_GUIDE.md` - 详细的迁移指南
- `../docs/QUICKSTART.md` - 快速开始教程
- `../README.md` - 主文档

### 新代码的优势

✅ **更清晰的结构**
- 模块化设计，职责分明
- 代码组织更合理

✅ **类型安全的配置**
- 使用dataclass管理配置
- 自动验证和类型检查

✅ **更好的文档**
- 完整的类型注解
- 详细的API文档

✅ **易于扩展**
- 清晰的接口定义
- 插件式架构

### 快速对比

| 特性 | 旧代码 | 新代码 |
|------|--------|--------|
| 配置管理 | 50+参数 | dataclass |
| 代码组织 | 单文件555行 | 模块化 |
| 类型注解 | 10% | 95% |
| 文档 | 基础 | 完善 |
| 可维护性 | 低 | 高 |

## 📊 新旧代码对比

### 训练命令对比

**旧代码**:
```bash
python BANIS.py --seed 0 --batch_size 8 --n_steps 50000 \
    --learning_rate 0.001 --weight_decay 0.01 --model_id S \
    --kernel_size 3 --data_setting base --base_data_path /path/ \
    # ... 另外40+个参数
```

**新代码**:
```bash
# 使用配置文件
python scripts/train.py --config examples/configs/train_mito.yaml

# 或使用命令行覆盖
python scripts/train.py --config examples/configs/train_mito.yaml \
    --override training.batch_size=16
```

### 代码使用对比

**旧代码**:
```python
from BANIS import BANIS
model = BANIS(**vars(args))  # 传入大量参数
```

**新代码**:
```python
from configs.config_loader import load_config
from banis.core.trainer import BANISTrainer

config = load_config("config.yaml")
trainer = BANISTrainer(config)
trainer.train()
```

## ⚠️ 重要提示

1. **此目录中的代码不再维护**
   - 仅用于参考和过渡
   - 不会修复bug或添加新功能

2. **建议尽快迁移**
   - 新代码更容易维护
   - 有更好的性能和功能

3. **兼容性保证**
   - 旧的检查点在新代码中可用
   - 数据格式保持兼容

## 🆘 需要帮助？

迁移过程中遇到问题？

1. 📖 查看迁移指南: `../docs/MIGRATION_GUIDE.md`
2. 💬 查看FAQ: `../docs/MIGRATION_GUIDE.md#常见问题`
3. 📧 联系团队获取支持

## 📅 时间线

- **2024-10**: 代码重构，旧代码归档
- **2024-11**: 过渡期（同时支持新旧代码）
- **2024-12**: 旧代码将被完全弃用

---

**开始迁移**: 查看 `../docs/QUICKSTART.md` 🚀

