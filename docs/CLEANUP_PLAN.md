# 🧹 BANIS 项目文件整理方案

## 📋 当前问题

根目录下有太多文件混在一起：
- ❌ 新旧代码混杂
- ❌ 脚本散落各处
- ❌ 文档不统一
- ❌ 配置文件重复
- ❌ 难以找到需要的文件

## 🎯 整理目标

创建清晰的目录结构，将文件按功能分类存放：
- ✅ 核心代码在 `banis/` 包中
- ✅ 旧代码归档到 `legacy/`
- ✅ 脚本统一在 `scripts/`
- ✅ 文档集中在 `docs/`
- ✅ 配置在 `configs/` 或根目录（主要配置）

---

## 📁 目标目录结构

```
banis/
├── 📄 README.md                 # 主文档
├── 📄 LICENSE
├── 📄 .gitignore
├── 📄 environment.yaml          # Conda环境
├── 📄 pyproject.toml            # 项目配置（新增）
│
├── 📁 banis/                    # 核心Python包
│   ├── core/                   # 模型和训练
│   ├── data/                   # 数据处理
│   ├── inference/              # 推理
│   ├── evaluation/             # 评估
│   └── utils/                  # 工具函数
│
├── 📁 configs/                  # 配置文件
│   ├── *.py                    # 配置类
│   └── *.yaml                  # YAML配置
│
├── 📁 scripts/                  # 所有脚本
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── batch_process.py
│   ├── convert_data.py
│   └── slurm/                  # Slurm相关
│       ├── submit_job.sh
│       └── job_scheduler.py
│
├── 📁 examples/                 # 示例和教程
│   ├── configs/
│   └── notebooks/
│
├── 📁 docs/                     # 所有文档
│   ├── README.md -> ../README.md
│   ├── QUICKSTART.md
│   ├── API.md
│   ├── MIGRATION.md
│   └── guides/
│
├── 📁 tests/                    # 测试文件
│   ├── test_model.py
│   └── test_data.py
│
├── 📁 legacy/                   # 旧代码归档
│   ├── BANIS.py
│   ├── data.py
│   ├── inference.py
│   ├── metrics.py
│   ├── batch_*.py
│   └── README.md               # 说明旧代码用途
│
├── 📁 outputs/                  # 输出目录
├── 📁 logs/                     # 日志目录
├── 📁 data/                     # 数据目录（软链接）
│   ├── mito_data -> ../mito_data/
│   └── rib_data -> ../rib_data/
│
└── 📁 notebooks/                # Jupyter notebooks
    └── visualization.ipynb
```

---

## 🔄 文件迁移计划

### Phase 1: 创建目录结构
```bash
mkdir -p legacy
mkdir -p scripts/slurm
mkdir -p docs/guides
mkdir -p notebooks
mkdir -p data
```

### Phase 2: 归档旧代码
```bash
# 旧的核心代码
mv BANIS.py legacy/
mv data.py legacy/
mv inference.py legacy/
mv metrics.py legacy/

# 旧的批处理脚本
mv batch_*.py legacy/

# 其他旧脚本
mv simple_convert.py legacy/
mv quick_process.py legacy/
mv processing_configs.py legacy/
mv mito_data.py legacy/
mv convert_predictions.py legacy/
mv benchmark.py legacy/
mv andromeda_launcher.py legacy/
mv show_data.py legacy/
```

### Phase 3: 整理脚本
```bash
# Slurm相关
mv submit_slurm.py scripts/slurm/
mv slurm_job_scheduler.py scripts/slurm/
mv validation_watcher.py scripts/slurm/
mv validation_watcher.sh scripts/slurm/
mv aff_train.sh scripts/slurm/

# 合并scripts_new到scripts
mv scripts_new/train.py scripts/
```

### Phase 4: 整理文档
```bash
# 重构文档移到docs
mv REFACTORING_*.md docs/
mv MIGRATION_GUIDE.md docs/
mv IMPLEMENTATION_STEPS.md docs/
mv QUICK_REFACTOR_START.md docs/

# 整理README
mv README_NEW.md README.md
mv QUICK_START.md docs/QUICKSTART.md
```

### Phase 5: 整理配置
```bash
# 示例配置
mv examples/*.yaml examples/configs/ 2>/dev/null || true

# 旧配置归档
cp config.yaml legacy/config.yaml.old
# 保留config.yaml在根目录作为向后兼容
```

---

## 📝 详细迁移清单

### ✅ 保留在根目录
- [x] README.md (使用新版)
- [x] LICENSE
- [x] .gitignore
- [x] environment.yaml
- [x] config.yaml (兼容旧代码)
- [x] pyproject.toml (新增)

### 📦 移动到 legacy/
- [ ] BANIS.py
- [ ] data.py  
- [ ] inference.py
- [ ] metrics.py
- [ ] batch_convert.py
- [ ] batch_evaluate.py
- [ ] batch_extract_example.py
- [ ] simple_convert.py
- [ ] quick_process.py
- [ ] processing_configs.py
- [ ] mito_data.py
- [ ] convert_predictions.py
- [ ] benchmark.py
- [ ] andromeda_launcher.py
- [ ] show_data.py

### 🔧 移动到 scripts/
- [ ] scripts_new/train.py → scripts/train.py

### 🔧 移动到 scripts/slurm/
- [ ] submit_slurm.py
- [ ] slurm_job_scheduler.py
- [ ] validation_watcher.py
- [ ] validation_watcher.sh
- [ ] aff_train.sh

### 📚 移动到 docs/
- [ ] REFACTORING_PLAN.md
- [ ] REFACTORING_SUMMARY.md
- [ ] IMPLEMENTATION_STEPS.md
- [ ] MIGRATION_GUIDE.md
- [ ] QUICK_REFACTOR_START.md
- [ ] REFACTORING_FILES_INDEX.md
- [ ] QUICK_START.md → QUICKSTART.md

### 🗂️ 整理数据目录
- [ ] mito_data/ (保持原位，创建软链接)
- [ ] rib_data/ (保持原位，创建软链接)
- [ ] mito_outputs/ → outputs/mito/
- [ ] rib_outputs/ → outputs/rib/
- [ ] rib_pointclouds/ → outputs/rib_pointclouds/

### 🗑️ 可以删除的目录
- [ ] __pycache__/ (Git忽略)
- [ ] torchinductor_liupen/ (编译缓存)
- [ ] scripts_new/ (合并到scripts后)
- [ ] docs_new/ (合并到docs后)

---

## 🚀 执行步骤

### Step 1: 备份（重要！）
```bash
# 创建完整备份
cd /projects/weilab/liupeng
tar -czf banis_backup_$(date +%Y%m%d_%H%M%S).tar.gz banis/
```

### Step 2: 执行自动整理脚本
```bash
cd /projects/weilab/liupeng/banis
bash scripts/cleanup_project.sh
```

### Step 3: 验证
```bash
# 检查新结构
tree -L 2 -d

# 确认重要文件存在
ls README.md LICENSE environment.yaml
ls -d banis/ configs/ scripts/ docs/ legacy/
```

### Step 4: 更新导入路径
```bash
# 如果有旧代码仍在使用，需要更新导入
# 从: from BANIS import ...
# 到: from legacy.BANIS import ...
# 或更好: from banis.core import ...
```

---

## 📋 清理后的根目录

整理后，根目录应该只有这些文件：

```
banis/
├── README.md              # 主文档
├── LICENSE               # 许可证
├── .gitignore           # Git忽略
├── environment.yaml     # Conda环境
├── pyproject.toml       # Python项目配置
├── config.yaml          # 主配置（向后兼容）
│
├── banis/               # Python包
├── configs/             # 配置
├── scripts/             # 脚本
├── examples/            # 示例
├── docs/                # 文档
├── tests/               # 测试
├── legacy/              # 旧代码
├── outputs/             # 输出
├── logs/                # 日志
├── data/                # 数据（软链接）
└── notebooks/           # Notebooks
```

---

## ⚠️ 注意事项

### 1. 保持向后兼容
```bash
# 在legacy/目录创建说明文件
cat > legacy/README.md << 'EOF'
# Legacy Code Archive

这个目录包含重构前的旧代码，用于：
- 参考和对比
- 向后兼容
- 过渡期间使用

## 旧代码使用方法

如果你需要使用旧代码：

```bash
# 方法1: 直接运行
python legacy/BANIS.py --help

# 方法2: 导入使用
from legacy.BANIS import BANIS
```

## 迁移到新代码

请参考：
- docs/MIGRATION_GUIDE.md
- docs/QUICKSTART.md

EOF
```

### 2. 更新.gitignore
```bash
# 添加忽略规则
echo "
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/

# Outputs
outputs/
logs/
*.log

# Data
mito_data/
rib_data/
mito_outputs/
rib_outputs/

# IDE
.vscode/
.idea/

# Temp
*.tmp
.DS_Store
torchinductor_*/
" >> .gitignore
```

### 3. 创建软链接
```bash
# 数据目录软链接
cd data/
ln -s ../mito_data .
ln -s ../rib_data .
```

---

## ✅ 验收标准

整理完成后应该满足：

- [ ] 根目录只有核心配置文件（<10个文件）
- [ ] 所有旧代码在 legacy/
- [ ] 所有脚本在 scripts/
- [ ] 所有文档在 docs/
- [ ] 核心代码在 banis/ 包中
- [ ] 目录结构清晰
- [ ] Git仓库整洁
- [ ] 旧代码仍可运行（向后兼容）

---

## 📊 整理前后对比

### 整理前
```
banis/ (根目录)
├── 30+ 个Python文件混在一起
├── 7个文档文件混乱
├── 配置文件重复
└── 难以找到需要的文件
```

### 整理后
```
banis/ (根目录)
├── 5个核心文件
├── banis/ (核心包)
├── scripts/ (所有脚本)
├── docs/ (所有文档)
├── legacy/ (旧代码归档)
└── 清晰的结构
```

---

## 🎓 最佳实践

### 1. 保持根目录整洁
```
✅ 好的根目录:
- README.md
- LICENSE
- .gitignore
- environment.yaml
- setup.py / pyproject.toml
- config.yaml (主配置)

❌ 避免:
- 30+ 个散落的脚本
- 重复的文档
- 临时文件
```

### 2. 按功能组织代码
```
✅ 功能清晰:
scripts/
├── train.py
├── inference.py
└── slurm/
    └── submit_job.sh

❌ 功能混乱:
根目录/
├── batch_convert.py
├── batch_evaluate.py
├── simple_convert.py
└── quick_process.py
```

### 3. 文档集中管理
```
✅ 文档整理:
docs/
├── README.md
├── QUICKSTART.md
├── API.md
└── guides/
    ├── migration.md
    └── development.md

❌ 文档散乱:
根目录/
├── README.md
├── README_NEW.md
├── QUICK_START.md
└── REFACTORING_*.md (7个)
```

---

## 🔧 自动化脚本

创建 `scripts/cleanup_project.sh`:

```bash
#!/bin/bash
# 项目清理脚本

echo "🧹 开始整理 BANIS 项目..."

# 创建目录
mkdir -p legacy scripts/slurm docs/guides notebooks data

# 归档旧代码
echo "📦 归档旧代码..."
mv BANIS.py data.py inference.py metrics.py legacy/ 2>/dev/null || true
mv batch_*.py legacy/ 2>/dev/null || true
mv simple_convert.py quick_process.py processing_configs.py legacy/ 2>/dev/null || true

# 整理脚本
echo "🔧 整理脚本..."
mv submit_slurm.py slurm_job_scheduler.py validation_watcher.* aff_train.sh scripts/slurm/ 2>/dev/null || true

# 整理文档
echo "📚 整理文档..."
mv REFACTORING_*.md MIGRATION_GUIDE.md IMPLEMENTATION_STEPS.md QUICK_REFACTOR_START.md docs/ 2>/dev/null || true
mv README_NEW.md README.md 2>/dev/null || true

echo "✅ 整理完成！"
echo "请检查新的目录结构: tree -L 2 -d"
```

---

## 📞 需要帮助？

如有问题：
1. 查看 docs/QUICKSTART.md
2. 查看 legacy/README.md 了解旧代码
3. 查看 docs/MIGRATION_GUIDE.md 了解迁移

---

**开始整理**: `bash scripts/cleanup_project.sh` 🚀

