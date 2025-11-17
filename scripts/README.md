# Scripts Directory
脚本目录说明

本目录包含各种自动化脚本和工具。

---

## 🎯 核心生产脚本

### 1. `auto_train_pipeline.py` ⭐
**VM端自动化训练流水线**

在Mininet VM中自动生成训练数据并训练机器学习模型。

```bash
# 完整训练（推荐）
sudo python3 scripts/auto_train_pipeline.py --samples 10 --duration 60

# 快速测试
sudo python3 scripts/auto_train_pipeline.py --quick

# 仅训练模型
python3 scripts/auto_train_pipeline.py --train-only
```

**功能**:
- ✅ 自动运行多个Mininet实验
- ✅ 收集训练数据（features + ground truth）
- ✅ 训练XGBoost和Random Forest模型
- ✅ 生成评估报告和可视化
- ✅ 保存预训练模型

**输出**:
- `experiments/` - 实验数据目录
- `models/xgboost_model.pkl` - XGBoost模型
- `models/random_forest_model.pkl` - Random Forest模型
- `models/preprocessor.pkl` - 特征预处理器
- `models/training_report.md` - 训练报告

### 2. `realtime_capture_host.py` ⭐
**宿主机实时捕获和识别脚本**

在真实环境（宿主机）中捕获视频网站流量并实时识别质量。

```bash
# 基本使用
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --interface en0

# 监测特定IP
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --target-ip 142.250.185.78

# 保存捕获数据
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --save-pcap capture.pcap
```

**功能**:
- ✅ 在宿主机网卡上实时捕获流量
- ✅ 自动识别视频流量
- ✅ 使用ML模型实时预测质量
- ✅ Beautiful终端UI（Rich库）
- ✅ 支持保存PCAP文件

**适用场景**:
- 监测YouTube、Netflix等真实视频网站
- 分析用户实际观看体验
- 网络质量评估
- ISP性能监控

---

## 🧪 训练和测试脚本

### 3. `train_model.py`
训练单个机器学习模型。

### 4. `evaluate_model.py` ⭐ **NEW**
**详细模型评估工具**

生成完整的模型性能评估报告，包括训练/验证/测试集分析。

```bash
# 基础评估
python3 scripts/evaluate_model.py

# 详细评估（含分类报告）
python3 scripts/evaluate_model.py --detailed

# 指定模型
python3 scripts/evaluate_model.py \
  --model-path models/xgboost_model.pkl \
  --experiments-dir experiments
```

**输出内容**:
- ✅ 数据集统计（总样本数、各类别分布）
- ✅ 数据集划分（训练70% / 验证15% / 测试15%）
- ✅ 总体性能指标（准确率、精确率、召回率、F1）
- ✅ 各类别性能（每个分辨率的详细指标）
- ✅ 混淆矩阵（可视化预测错误）
- ✅ 详细分类报告（sklearn格式）

**示例输出**:
```
┌───────────────────┬─────────┐
│ 指标              │ 数值    │
├───────────────────┼─────────┤
│ 准确率 (Accuracy) │ 0.9630  │
│ 精确率 (Precision)│ 0.9667  │
│ 召回率 (Recall)   │ 0.9630  │
│ F1-Score          │ 0.9630  │
└───────────────────┴─────────┘

┌──────────┬────────┬─────────┬─────────┬──────────┐
│ 分辨率   │ 样本数 │ 精确率  │ 召回率  │ F1-Score │
├──────────┼────────┼─────────┼─────────┼──────────┤
│ 480p     │ 9      │ 1.0000  │ 0.8889  │ 0.9412   │
│ 720p     │ 9      │ 0.9000  │ 1.0000  │ 0.9474   │
│ 1080p    │ 9      │ 1.0000  │ 1.0000  │ 1.0000   │
└──────────┴────────┴─────────┴─────────┴──────────┘
```

**适用场景**:
- 训练完成后评估模型性能
- 对比不同模型的效果
- 分析各分辨率识别准确率
- 发现模型弱点并针对性改进

```bash
python scripts/train_model.py \
  --experiments-dir experiments/ \
  --output-dir models/ \
  --model-type xgboost \
  --class-names 480p 720p 1080p
```

**由 `auto_train_pipeline.py` 内部调用**。

### 4. `collect_training_data.py`
批量收集训练数据（框架脚本）。

```bash
sudo python3 scripts/collect_training_data.py \
  --samples 10 \
  --duration 60 \
  --scenarios low-bandwidth mobile-4g wifi
```

**注意**: 推荐使用 `auto_train_pipeline.py`，它提供完整的端到端流程。

### 5. `test_story_7_*.py`
Story 7相关组件的单元测试脚本。

```bash
# 测试数据加载器
python scripts/test_story_7_1.py

# 测试预处理器
python scripts/test_story_7_2.py

# 测试训练脚本
python scripts/test_story_7_5.py
```

---

## 🎬 演示脚本

### 6. `demo_mininet_pipeline.py`
在Mininet中演示实时监测流水线。

```bash
sudo python3 scripts/demo_mininet_pipeline.py \
  --scenario high-bandwidth \
  --duration 30
```

### 7. `demo_realtime_monitor.py`
使用模拟数据演示实时监测。

```bash
python3 scripts/demo_realtime_monitor.py
```

### 8. `demo_simple_test.py`
简单的Mininet网络测试。

```bash
sudo python3 scripts/demo_simple_test.py
```

---

## 📊 数据分析脚本

### 9. `analyze_experiments.py`
分析实验结果并生成报告（如果存在）。

```bash
python scripts/analyze_experiments.py --experiments-dir experiments/
```

---

## 🔧 工具脚本

### 10. `setup_environment.sh`
环境设置脚本（如果存在）。

```bash
bash scripts/setup_environment.sh
```

---

## 使用场景速查

### 场景1: 首次使用 - 训练模型

```bash
# 在VM中
cd /home/mininet/cn
sudo python3 scripts/auto_train_pipeline.py --quick

# 传输到宿主机
scp models/*.pkl user@host:/path/to/cn/models/
```

### 场景2: 实时监测真实流量

```bash
# 在宿主机
# 1. 打开YouTube视频
# 2. 运行监测
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl
```

### 场景3: 更新和改进模型

```bash
# 在VM中收集更多数据
sudo python3 scripts/auto_train_pipeline.py \
  --samples 20 \
  --duration 90 \
  --scenarios wifi high-quality mobile-4g

# 仅重新训练
python3 scripts/auto_train_pipeline.py --train-only
```

### 场景4: 测试和开发

```bash
# 运行单元测试
python scripts/test_story_7_1.py

# 运行演示
sudo python3 scripts/demo_mininet_pipeline.py
```

### 场景5: 性能评估

```bash
# 监测并保存数据
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --duration 3600 \
  --save-pcap evaluation.pcap

# 后续分析PCAP文件
```

---

## 🚨 常见问题

### Q: 哪个脚本需要sudo？

**A**: 
- ✅ 需要sudo: 
  - `auto_train_pipeline.py` (VM中运行Mininet)
  - `realtime_capture_host.py` (宿主机捕获包)
  - `demo_*.py` (涉及Mininet)
- ❌ 不需要sudo:
  - `train_model.py`
  - `test_story_*.py`

### Q: 如何选择使用哪个脚本？

**A**:
- **训练模型**: 使用 `auto_train_pipeline.py`
- **实时监测**: 使用 `realtime_capture_host.py`
- **测试功能**: 使用 `test_story_*.py` 或 `demo_*.py`

### Q: 脚本执行时间？

**A**:
- `auto_train_pipeline.py --quick`: ~30分钟
- `auto_train_pipeline.py` (完整): 2-4小时
- `realtime_capture_host.py`: 持续运行（可指定duration）
- `demo_*.py`: 1-5分钟
- `test_*.py`: 几秒到几分钟

---

## 📦 依赖关系

```
auto_train_pipeline.py
├── ExperimentManager (video_qoe.experiment)
├── RealTimePipeline (video_qoe.monitoring)
└── train_model.py (subprocess)

realtime_capture_host.py
├── pyshark (外部)
├── joblib (外部)
└── rich (外部)

train_model.py
├── ExperimentDataLoader (video_qoe.training)
├── FeaturePreprocessor (video_qoe.training)
├── XGBoostTrainer / RandomForestTrainer (video_qoe.training)
└── ModelEvaluator (video_qoe.training)
```

---

## 📝 最佳实践

### 1. 训练前

- 确保VM有足够磁盘空间（10GB+）
- 检查Python环境和依赖
- 使用 `--dry-run` 预览执行计划

### 2. 训练中

- 使用 `screen` 或 `tmux` 避免SSH断开
- 监控磁盘使用情况
- 保存训练日志

### 3. 部署时

- 验证模型文件完整性
- 测试预测准确性
- 记录模型版本和性能

### 4. 监测时

- 使用合适的窗口大小（1-2秒）
- 定期保存监测数据
- 监控系统资源使用

---

## 🔗 相关文档

- [快速开始指南](../QUICKSTART.md)
- [生产环境部署](../docs/guides/PRODUCTION_DEPLOYMENT.md)
- [模型训练指南](../models/README.md)
- [系统架构](../docs/architecture.md)

---

**最后更新**: 2025-11-15  
**文档版本**: 1.0.0


