#!/bin/bash
# 快速训练脚本 - 从现有PCAP提取特征并训练模型

set -e  # 遇到错误立即退出

# Fix encoding
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "=========================================="
echo "视频QoE快速训练流水线"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -d "experiments" ]; then
    echo "错误: 未找到 experiments 目录"
    echo "请在项目根目录运行此脚本"
    exit 1
fi

# 统计实验数量
exp_count=$(find experiments -maxdepth 1 -type d -name "exp_*" | wc -l)
echo "找到 $exp_count 个实验"
echo ""

if [ "$exp_count" -eq 0 ]; then
    echo "错误: 没有实验数据"
    echo "请先运行: sudo python3 scripts/auto_train_pipeline.py --quick"
    exit 1
fi

# 步骤1: 批量提取特征
echo "步骤 1/2: 从PCAP提取特征..."
echo "----------------------------------------"
python3 scripts/extract_features_from_pcap.py || {
    echo "警告: 特征提取失败，尝试继续..."
}
echo ""

# 检查是否生成了features.csv
features_count=$(find experiments -maxdepth 2 -name "features.csv" | wc -l)
echo "生成了 $features_count 个 features.csv 文件"
echo ""

if [ "$features_count" -eq 0 ]; then
    echo "错误: 未能生成任何 features.csv"
    echo "请检查 PCAP 文件是否有效"
    exit 1
fi

# 步骤2: 训练模型
echo "步骤 2/2: 训练模型..."
echo "----------------------------------------"
python3 scripts/train_model.py \
    --experiments-dir experiments \
    --output-dir models \
    --model-type xgboost \
    --class-names 480p 720p 1080p

echo ""
echo "=========================================="
echo "✓ 训练完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - models/xgboost_model.pkl"
echo "  - models/preprocessor.pkl"
echo "  - models/confusion_matrix.png"
echo ""
echo "下一步: 在宿主机运行"
echo "  sudo python3 scripts/realtime_capture_host.py"

