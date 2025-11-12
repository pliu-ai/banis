# 🚀 快速开始：5分钟体验新BANIS

> 想立即尝试重构后的代码？跟随这个5分钟教程！

## ⚡ 3步开始

### Step 1: 测试配置系统 (1分钟)

```bash
# 进入项目目录
cd /projects/weilab/liupeng/banis

# 测试配置加载
python -c "
from configs.config_loader import load_config
config = load_config('examples/train_mito.yaml')
config.validate()
print('✅ 配置系统正常!')
print(f'  - 批大小: {config.training.batch_size}')
print(f'  - 学习率: {config.training.learning_rate}')
print(f'  - 模型: {config.model.model_id}')
print(f'  - 数据集: {[ds.name for ds in config.data.datasets]}')
"
```

### Step 2: 测试模型创建 (2分钟)

```bash
# 测试模型
python -c "
import torch
from configs.model_config import ModelConfig
from banis.core.model import create_model, count_parameters

# 创建配置
config = ModelConfig(model_id='S', kernel_size=3, num_input_channels=1)

# 创建模型
print('创建模型...')
model = create_model(config)

# 测试前向传播
print('测试前向传播...')
x = torch.randn(1, 1, 64, 64, 64)
y = model(x)

print(f'✅ 模型创建成功!')
print(f'  - 输入形状: {x.shape}')
print(f'  - 输出形状: {y.shape}')
print(f'  - 参数量: {count_parameters(model):,}')
"
```

### Step 3: 运行快速训练 (2分钟)

```bash
# 创建测试配置
cat > /tmp/test_config.yaml << 'EOF'
data:
  datasets:
    - name: betaSeg
      resolution: [16, 16, 16]
      path: /projects/weilab/dataset/MitoLE/betaSeg
      train_splits: ["high_c3"]
      val_splits: ["high_c1"]
  base_data_path: ./mito_data
  small_size: 64
  num_workers: 2
  augment: false

training:
  batch_size: 2
  n_steps: 10
  learning_rate: 0.001
  fast_dev_run: 5  # 只运行5个batch
  save_path: /tmp/banis_test

model:
  model_id: S
  kernel_size: 3
EOF

# 运行训练 (如果数据存在)
python scripts_new/train.py --config /tmp/test_config.yaml || echo "需要准备数据才能运行训练"
```

---

## 📖 查看文档

```bash
# 查看重构方案
less REFACTORING_PLAN.md

# 查看快速总结
less REFACTORING_SUMMARY.md

# 查看实施步骤
less IMPLEMENTATION_STEPS.md

# 查看迁移指南
less MIGRATION_GUIDE.md

# 查看新README
less README_NEW.md
```

---

## 🔄 转换现有配置

```bash
# 如果你有旧的config.yaml，转换它
python -c "
from pathlib import Path
from configs.config_loader import load_legacy_config, save_config

# 读取旧配置
old_config = Path('config.yaml')
if old_config.exists():
    print(f'转换配置: {old_config}')
    config = load_legacy_config(old_config)
    
    # 保存新配置
    new_config = Path('configs/converted_config.yaml')
    new_config.parent.mkdir(exist_ok=True)
    save_config(config, new_config)
    
    print(f'✅ 配置已转换并保存到: {new_config}')
    print('现在可以使用:')
    print(f'  python scripts/train.py --config {new_config}')
else:
    print('未找到config.yaml，跳过转换')
"
```

---

## 🎯 实际使用示例

### 示例1: 训练线粒体分割模型

```bash
# 1. 准备配置文件
cp examples/train_mito.yaml my_mito_config.yaml

# 2. 编辑配置(可选)
# 修改 my_mito_config.yaml 中的参数

# 3. 开始训练
python scripts_new/train.py --config my_mito_config.yaml

# 4. 监控训练
tensorboard --logdir outputs/
```

### 示例2: 快速实验不同参数

```bash
# 测试不同批大小
python scripts_new/train.py --config examples/train_mito.yaml \
    --override training.batch_size=16

# 测试不同学习率
python scripts_new/train.py --config examples/train_mito.yaml \
    --override training.learning_rate=0.0005

# 测试不同模型大小
python scripts_new/train.py --config examples/train_mito.yaml \
    --override model.model_id=L model.kernel_size=5

# 组合多个参数
python scripts_new/train.py --config examples/train_mito.yaml \
    --override \
    training.batch_size=16 \
    training.learning_rate=0.002 \
    model.model_id=M
```

### 示例3: 调试模式

```bash
# 快速验证代码能否运行 (只运行10个batch)
python scripts_new/train.py --config examples/train_mito.yaml \
    --override training.fast_dev_run=10

# 打开详细日志
python scripts_new/train.py --config examples/train_mito.yaml \
    --log-level DEBUG

# 打开异常检测 (找出NaN)
python scripts_new/train.py --config examples/train_mito.yaml \
    --override training.detect_anomaly=true
```

---

## 🧪 代码质量检查

```bash
# 运行类型检查 (需要安装mypy)
# pip install mypy
mypy configs/ || echo "类型检查完成"

# 运行代码格式检查 (需要安装flake8)
# pip install flake8
flake8 configs/ banis/ --max-line-length=120 || echo "代码检查完成"

# 运行代码格式化 (需要安装black)
# pip install black
black configs/ banis/ --line-length=120 || echo "代码格式化完成"
```

---

## 📊 查看项目结构

```bash
# 查看新的项目结构
tree -L 2 configs/ banis/ scripts_new/ examples/

# 或使用ls
echo "=== 配置系统 ==="
ls -lh configs/

echo "=== 核心代码 ==="
ls -lh banis/*/

echo "=== 脚本 ==="
ls -lh scripts_new/

echo "=== 示例 ==="
ls -lh examples/
```

---

## 💡 下一步做什么?

### 如果你是用户
1. ✅ 阅读 [README_NEW.md](README_NEW.md)
2. ✅ 尝试 [examples/train_mito.yaml](examples/train_mito.yaml)
3. ✅ 运行你的第一个训练
4. ✅ 查看TensorBoard结果

### 如果你是开发者
1. ✅ 阅读 [REFACTORING_PLAN.md](REFACTORING_PLAN.md)
2. ✅ 理解新的代码结构
3. ✅ 查看 [banis/core/](banis/core/) 实现
4. ✅ 开始贡献代码

### 如果你要迁移旧代码
1. ✅ 阅读 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
2. ✅ 转换配置文件
3. ✅ 逐步迁移代码
4. ✅ 测试验证

---

## ❓ 常见问题快速解答

### Q: 新代码兼容旧的检查点吗?
**A**: 是的! 使用 `load_model_from_checkpoint()` 自动处理。

### Q: 必须重写所有代码吗?
**A**: 不需要。可以渐进式迁移，新旧代码可以共存。

### Q: 配置文件格式改了很多吗?
**A**: 是的，但更清晰了。使用 `load_legacy_config()` 可以转换旧配置。

### Q: 性能会变差吗?
**A**: 不会。核心算法未变，只是组织方式更好。

### Q: 多久能学会新系统?
**A**: 
- 基本使用: 30分钟
- 熟练使用: 2-3小时  
- 深入理解: 1-2天

---

## 🆘 遇到问题?

### 报错: ModuleNotFoundError
```bash
# 确保在正确的目录
cd /projects/weilab/liupeng/banis

# 确保Python路径正确
export PYTHONPATH=/projects/weilab/liupeng/banis:$PYTHONPATH

# 或者安装包
pip install -e .
```

### 报错: Config validation failed
```bash
# 检查配置文件
python -c "
from configs.config_loader import load_config
config = load_config('your_config.yaml')
config.validate()  # 会显示具体错误
"
```

### 训练无法启动
```bash
# 使用快速模式检查
python scripts_new/train.py --config your_config.yaml \
    --override training.fast_dev_run=1
```

---

## 📚 推荐阅读顺序

**第一天** (30分钟):
1. [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - 快速了解改进
2. [README_NEW.md](README_NEW.md) - 新项目介绍
3. 运行上面的3个测试

**第二天** (2小时):
1. [examples/train_mito.yaml](examples/train_mito.yaml) - 配置示例
2. [scripts_new/train.py](scripts_new/train.py) - 训练脚本
3. 尝试训练一个模型

**第三天** (1天):
1. [REFACTORING_PLAN.md](REFACTORING_PLAN.md) - 详细设计
2. [banis/core/](banis/core/) - 核心实现
3. 开始使用或扩展

---

## ✨ 特性亮点

### 1. 类型安全的配置
```python
# ✅ 自动补全
config.training.batch_size  # IDE会提示
config.training.learning_rate  # 类型检查

# ✅ 验证
config.validate()  # 自动检查参数合法性
```

### 2. 清晰的错误信息
```python
# ❌ 旧版本
# Error: ...
# (在555行代码中找bug)

# ✅ 新版本
# DataLoadError: Failed to load data.zarr: File not found
# at banis/utils/io.py:45 in load_zarr()
# (精确定位)
```

### 3. 简单的扩展
```python
# 添加新模型只需:
# 1. 创建 banis/models/my_model.py
# 2. 在 model.py 注册
# 3. 更新 ModelConfig
# 完成!
```

---

## 🎉 完成!

恭喜你完成了快速开始教程！现在你可以:

- ✅ 使用新的配置系统
- ✅ 创建和训练模型
- ✅ 理解项目结构
- ✅ 开始实际使用

**下一步**: 
- 查看完整文档了解更多功能
- 尝试在实际项目中使用
- 给项目贡献代码

---

**问题或建议?** 
- 📧 Email: your-email@example.com
- 💬 GitHub: [Issues](https://github.com/your-org/banis/issues)

**Happy Coding! 🚀**

