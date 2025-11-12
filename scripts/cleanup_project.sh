#!/bin/bash
# BANIS 项目自动整理脚本
# 用途: 将散落的文件整理到合适的目录

set -e  # 遇到错误立即退出

PROJECT_ROOT="/projects/weilab/liupeng/banis"
cd "$PROJECT_ROOT"

echo "════════════════════════════════════════════════════════════"
echo "🧹 BANIS 项目文件整理脚本"
echo "════════════════════════════════════════════════════════════"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 询问是否继续
echo -e "${YELLOW}警告: 此脚本将移动大量文件！${NC}"
echo "建议先备份项目: tar -czf banis_backup.tar.gz ../banis/"
echo ""
read -p "是否继续? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📁 Phase 1: 创建目录结构"
echo "════════════════════════════════════════════════════════════"

# 创建目录
mkdir -p legacy
mkdir -p scripts/slurm
mkdir -p docs/guides
mkdir -p notebooks
mkdir -p data
mkdir -p tests

echo -e "${GREEN}✓${NC} 目录结构创建完成"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📦 Phase 2: 归档旧代码到 legacy/"
echo "════════════════════════════════════════════════════════════"

# 归档旧核心代码
files_to_archive=(
    "BANIS.py"
    "data.py"
    "inference.py"
    "metrics.py"
)

for file in "${files_to_archive[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" legacy/
        echo -e "${GREEN}✓${NC} $file → legacy/"
    fi
done

# 归档批处理脚本
for file in batch_*.py; do
    if [ -f "$file" ]; then
        mv "$file" legacy/
        echo -e "${GREEN}✓${NC} $file → legacy/"
    fi
done

# 归档其他旧脚本
other_scripts=(
    "simple_convert.py"
    "quick_process.py"
    "processing_configs.py"
    "mito_data.py"
    "convert_predictions.py"
    "benchmark.py"
    "andromeda_launcher.py"
    "show_data.py"
)

for file in "${other_scripts[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" legacy/
        echo -e "${GREEN}✓${NC} $file → legacy/"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🔧 Phase 3: 整理脚本到 scripts/"
echo "════════════════════════════════════════════════════════════"

# Slurm相关脚本
slurm_scripts=(
    "submit_slurm.py"
    "slurm_job_scheduler.py"
    "validation_watcher.py"
    "validation_watcher.sh"
    "aff_train.sh"
)

for file in "${slurm_scripts[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" scripts/slurm/
        echo -e "${GREEN}✓${NC} $file → scripts/slurm/"
    fi
done

# 移动scripts_new中的文件
if [ -d "scripts_new" ]; then
    for file in scripts_new/*.py; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            mv "$file" "scripts/$filename"
            echo -e "${GREEN}✓${NC} scripts_new/$filename → scripts/"
        fi
    done
    rmdir scripts_new 2>/dev/null || true
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📚 Phase 4: 整理文档到 docs/"
echo "════════════════════════════════════════════════════════════"

# 移动重构文档
refactor_docs=(
    "REFACTORING_PLAN.md"
    "REFACTORING_SUMMARY.md"
    "IMPLEMENTATION_STEPS.md"
    "MIGRATION_GUIDE.md"
    "QUICK_REFACTOR_START.md"
    "REFACTORING_FILES_INDEX.md"
    "CLEANUP_PLAN.md"
)

for file in "${refactor_docs[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" docs/
        echo -e "${GREEN}✓${NC} $file → docs/"
    fi
done

# 整理README
if [ -f "README_NEW.md" ]; then
    # 备份旧README
    if [ -f "README.md" ]; then
        mv README.md docs/README.old.md
        echo -e "${BLUE}ℹ${NC} 旧README.md → docs/README.old.md"
    fi
    mv README_NEW.md README.md
    echo -e "${GREEN}✓${NC} README_NEW.md → README.md"
fi

# 移动QUICK_START
if [ -f "QUICK_START.md" ]; then
    mv QUICK_START.md docs/QUICKSTART.md
    echo -e "${GREEN}✓${NC} QUICK_START.md → docs/QUICKSTART.md"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📝 Phase 5: 整理配置和示例"
echo "════════════════════════════════════════════════════════════"

# 备份旧配置
if [ -f "config.yaml" ]; then
    cp config.yaml legacy/config.yaml.old
    echo -e "${BLUE}ℹ${NC} 备份 config.yaml → legacy/config.yaml.old"
fi

# 整理示例配置
mkdir -p examples/configs
if [ -d "examples" ]; then
    for file in examples/*.yaml; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            if [ ! -d "examples/configs" ]; then
                mkdir -p examples/configs
            fi
            # 如果文件不在configs子目录，移动它
            if [[ "$file" != "examples/configs/"* ]]; then
                mv "$file" "examples/configs/$filename" 2>/dev/null || true
            fi
        fi
    done
fi

echo -e "${GREEN}✓${NC} 配置文件整理完成"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🗂️  Phase 6: 整理输出目录"
echo "════════════════════════════════════════════════════════════"

# 整理输出目录
mkdir -p outputs/mito outputs/rib

if [ -d "mito_outputs" ] && [ "$(ls -A mito_outputs 2>/dev/null)" ]; then
    mv mito_outputs/* outputs/mito/ 2>/dev/null || true
    rmdir mito_outputs 2>/dev/null || true
    echo -e "${GREEN}✓${NC} mito_outputs → outputs/mito/"
fi

if [ -d "rib_outputs" ] && [ "$(ls -A rib_outputs 2>/dev/null)" ]; then
    mv rib_outputs/* outputs/rib/ 2>/dev/null || true
    rmdir rib_outputs 2>/dev/null || true
    echo -e "${GREEN}✓${NC} rib_outputs → outputs/rib/"
fi

if [ -d "rib_pointclouds" ]; then
    mv rib_pointclouds outputs/
    echo -e "${GREEN}✓${NC} rib_pointclouds → outputs/"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🔗 Phase 7: 创建软链接"
echo "════════════════════════════════════════════════════════════"

# 创建数据目录软链接
cd data/
if [ -d "../mito_data" ] && [ ! -e "mito_data" ]; then
    ln -s ../mito_data .
    echo -e "${GREEN}✓${NC} 创建软链接: data/mito_data → ../mito_data"
fi

if [ -d "../rib_data" ] && [ ! -e "rib_data" ]; then
    ln -s ../rib_data .
    echo -e "${GREEN}✓${NC} 创建软链接: data/rib_data → ../rib_data"
fi
cd ..

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🧹 Phase 8: 清理临时文件"
echo "════════════════════════════════════════════════════════════"

# 清理__pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓${NC} 清理 __pycache__"

# 清理编译缓存
if [ -d "torchinductor_liupen" ]; then
    rm -rf torchinductor_liupen
    echo -e "${GREEN}✓${NC} 清理 torchinductor 缓存"
fi

# 删除空的scripts_new和docs_new
rmdir scripts_new 2>/dev/null || true
rmdir docs_new 2>/dev/null || true

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📝 Phase 9: 创建说明文件"
echo "════════════════════════════════════════════════════════════"

# 创建legacy/README.md
cat > legacy/README.md << 'EOF'
# Legacy Code Archive

## 📋 说明

这个目录包含重构前的旧代码，保留用于：
- **参考对比**: 查看旧实现方式
- **向后兼容**: 过渡期间使用
- **测试验证**: 对比新旧结果

## 📁 内容

- `BANIS.py` - 旧的训练主程序
- `data.py` - 旧的数据加载
- `inference.py` - 旧的推理代码
- `metrics.py` - 旧的评估指标
- `batch_*.py` - 批处理脚本
- 其他工具脚本

## 🔧 使用旧代码

如果需要运行旧代码：

```bash
# 方法1: 直接运行
cd legacy/
python BANIS.py --help

# 方法2: 从根目录运行
python legacy/BANIS.py --config legacy/config.yaml.old
```

## 🚀 迁移到新代码

**推荐使用新代码！** 查看：
- `../docs/MIGRATION_GUIDE.md` - 迁移指南
- `../docs/QUICKSTART.md` - 快速开始
- `../README.md` - 主文档

新代码优势：
- ✅ 更清晰的结构
- ✅ 类型安全的配置
- ✅ 更好的文档
- ✅ 易于扩展

## ⚠️ 注意

此目录中的代码**不再维护**，仅用于参考。
请尽快迁移到新代码。
EOF

echo -e "${GREEN}✓${NC} 创建 legacy/README.md"

# 创建scripts/README.md
cat > scripts/README.md << 'EOF'
# Scripts Directory

## 📋 脚本说明

此目录包含所有可执行脚本。

### 主要脚本

- `train.py` - 训练脚本
- `inference.py` - 推理脚本
- `evaluate.py` - 评估脚本

### Slurm作业

- `slurm/` - Slurm集群相关脚本
  - `submit_job.sh` - 提交作业
  - `job_scheduler.py` - 作业调度器

### 工具脚本

- `cleanup_project.sh` - 项目整理脚本

## 🚀 使用方法

```bash
# 训练
python scripts/train.py --config examples/configs/train_mito.yaml

# 推理
python scripts/inference.py --checkpoint path/to/model.ckpt

# 提交Slurm作业
bash scripts/slurm/submit_job.sh
```
EOF

echo -e "${GREEN}✓${NC} 创建 scripts/README.md"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ 整理完成！"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 新的项目结构："
echo ""

# 显示新结构
if command -v tree &> /dev/null; then
    tree -L 2 -d --dirsfirst
else
    # 如果没有tree命令，使用ls
    echo "主要目录："
    ls -d */ 2>/dev/null | head -20
fi

echo ""
echo "📝 后续步骤："
echo ""
echo "1. 检查新结构: tree -L 2 -d"
echo "2. 查看主文档: cat README.md"
echo "3. 查看旧代码: cat legacy/README.md"
echo "4. 测试新代码: python -c 'from configs.config_loader import load_config'"
echo "5. 提交更改: git add . && git commit -m 'Refactor: Reorganize project structure'"
echo ""
echo -e "${GREEN}✨ 项目整理完成！结构更清晰了！${NC}"

