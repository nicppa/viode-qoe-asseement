# 生产环境部署指南
Production Deployment Guide

本指南介绍如何在生产环境中使用视频质量评估系统。

---

## 📋 目录

1. [概述](#概述)
2. [VM端：训练模型](#vm端训练模型)
3. [宿主机端：实时监测](#宿主机端实时监测)
4. [完整工作流](#完整工作流)
5. [故障排除](#故障排除)

---

## 概述

系统分为两个部分：

### VM端（Mininet虚拟机）
- **用途**: 生成训练数据并训练机器学习模型
- **环境**: Mininet VM with Python 3.7+
- **输出**: 训练好的模型文件 (`xgboost_model.pkl`, `preprocessor.pkl`)

### 宿主机端（真实环境）
- **用途**: 捕获真实视频流量并实时识别质量
- **环境**: macOS/Linux/Windows with Python 3.7+
- **输入**: 训练好的模型文件
- **输出**: 实时视频质量预测

---

## VM端：训练模型

### 1. 环境准备

在Mininet VM中：

```bash
# 确保在项目目录
cd /home/mininet/cn

# 激活虚拟环境（如果有）
source venv/bin/activate

# 确认依赖已安装
pip list | grep -E "xgboost|scikit-learn|pandas"
```

### 2. 运行自动化训练流水线

#### 方式A: 完整训练（推荐）

生成大量数据并训练高质量模型（需要2-4小时）：

```bash
sudo python3 scripts/auto_train_pipeline.py \
  --samples 10 \
  --duration 60 \
  --model-types xgboost random_forest
```

参数说明：
- `--samples 10`: 每个场景/分辨率组合运行10次
- `--duration 60`: 每次实验持续60秒
- `--model-types xgboost random_forest`: 训练两种模型

总实验数: 3分辨率 × 6场景 × 10样本 = **180个实验**

#### 方式B: 快速测试（开发用）

快速生成少量数据进行测试（约30分钟）：

```bash
sudo python3 scripts/auto_train_pipeline.py \
  --samples 2 \
  --duration 30 \
  --quick
```

#### 方式C: 仅训练模型（使用已有数据）

如果已经有实验数据，只需要训练模型：

```bash
python3 scripts/auto_train_pipeline.py \
  --train-only \
  --experiments-dir experiments/ \
  --models-dir models/ \
  --model-types xgboost
```

### 3. 验证训练结果

训练完成后，检查输出：

```bash
# 查看模型文件
ls -lh models/

# 应该看到:
# - xgboost_model.pkl          # XGBoost模型
# - random_forest_model.pkl    # Random Forest模型
# - preprocessor.pkl           # 特征预处理器
# - training_report.md         # 训练报告
# - confusion_matrix.png       # 混淆矩阵
# - feature_importance.png     # 特征重要性
```

### 4. 传输模型到宿主机

使用SCP或共享文件夹将模型文件传输到宿主机：

```bash
# 方法1: 使用SCP
scp models/*.pkl user@host:/path/to/models/

# 方法2: 使用共享文件夹（VirtualBox）
# 在VM中将模型复制到共享目录
cp models/*.pkl /mnt/shared/

# 方法3: 使用U盘或其他存储介质
```

---

## 宿主机端：实时监测

### 1. 环境准备

在宿主机（macOS/Linux/Windows）上：

```bash
# 安装依赖
pip install pyshark pandas numpy scikit-learn xgboost rich joblib netifaces

# macOS可能还需要安装Wireshark（提供tshark）
brew install wireshark

# Linux
sudo apt-get install tshark

# Windows
# 从 https://www.wireshark.org/download.html 下载安装
```

### 2. 检查网卡

列出可用的网络接口：

```bash
python scripts/realtime_capture_host.py --list-interfaces
```

输出示例：
```
可用网卡:
  1. en0 (192.168.1.100)      # Wi-Fi
  2. en1 (N/A)                # Thunderbolt
  3. lo0 (127.0.0.1)          # Loopback
```

### 3. 开始实时监测

#### 基本用法

自动检测网卡并开始监测：

```bash
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl
```

#### 指定网卡

```bash
sudo python3 scripts/realtime_capture_host.py \
  --interface en0 \
  --model models/xgboost_model.pkl
```

#### 监测特定视频网站

获取视频网站IP后监测：

```bash
# 先获取目标网站IP
ping youtube.com  # 或 nslookup youtube.com

# 监测该IP
sudo python3 scripts/realtime_capture_host.py \
  --interface en0 \
  --model models/xgboost_model.pkl \
  --target-ip 142.250.185.78
```

#### 保存捕获数据

```bash
sudo python3 scripts/realtime_capture_host.py \
  --interface en0 \
  --model models/xgboost_model.pkl \
  --save-pcap capture.pcap
```

### 4. 实时输出示例

监测运行时的输出：

```
┌─────────────────────────────────────────────────────────┐
│           🎥 实时视频质量监测                             │
├────────────────────┬────────────────────────────────────┤
│ 指标               │ 数值                                │
├────────────────────┼────────────────────────────────────┤
│ 监测时长           │ 45 秒                               │
│ 捕获包数           │ 1,234                               │
│ TCP包数            │ 1,180                               │
│ 视频包数           │ 856                                 │
│ 总流量             │ 12.34 MB                            │
│ 预测次数           │ 42                                  │
│ 当前质量           │ 1080p (89.5%)                       │
│ 当前吞吐           │ 8.56 Mbps                           │
└────────────────────┴────────────────────────────────────┘

┌───────────────── 📊 最近预测 ──────────────────┐
│ 1080p (89.5%) | 8.56 Mbps                      │
│ 1080p (91.2%) | 9.12 Mbps                      │
│ 720p (78.3%) | 5.43 Mbps                       │
│ 1080p (88.7%) | 8.91 Mbps                      │
│ 1080p (90.1%) | 9.05 Mbps                      │
└─────────────────────────────────────────────────┘
```

---

## 完整工作流

### 端到端示例

```bash
# ===== 第一步: 在VM中训练模型 =====

# 1. SSH到VM
ssh mininet@192.168.56.101

# 2. 运行自动化训练流水线
cd /home/mininet/cn
sudo python3 scripts/auto_train_pipeline.py --samples 10 --duration 60

# 3. 等待完成（2-4小时）
# ✓ 收集数据完成
# ✓ 训练模型完成
# ✓ 模型保存到 models/

# 4. 退出VM
exit

# ===== 第二步: 传输模型到宿主机 =====

# 使用SCP传输
scp mininet@192.168.56.101:/home/mininet/cn/models/*.pkl ./models/

# ===== 第三步: 在宿主机上实时监测 =====

# 1. 打开YouTube视频开始播放

# 2. 运行监测脚本
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --interface en0

# 3. 观察实时输出
# 可以看到视频质量随时间变化
# 1080p -> 720p -> 480p -> 1080p ...

# 4. 停止监测（Ctrl+C）
# 查看统计摘要
```

---

## 高级用法

### 1. 持续监测并记录日志

```bash
# 监测1小时并保存结果
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --interface en0 \
  --duration 3600 \
  --save-pcap long_capture.pcap \
  2>&1 | tee monitoring.log
```

### 2. 监测多个视频平台

```bash
# 终端1: YouTube
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --target-ip 142.250.185.78

# 终端2: Netflix
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --target-ip 52.85.84.116
```

### 3. 批量分析历史数据

如果已有PCAP文件，可以离线分析：

```python
# analyze_pcap.py
from video_qoe.monitoring import RealTimePipeline

pipeline = RealTimePipeline(
    pcap_path='historical_capture.pcap',
    client_ip='192.168.1.100',
    capture_mode=False,
    predictor_type='ml_model',
    model_path='models/xgboost_model.pkl'
)

with pipeline:
    stats = pipeline.run()
    
print(f"Total predictions: {stats.predictions_made}")
```

### 4. 自定义特征和模型

```bash
# 使用自定义配置训练
sudo python3 scripts/auto_train_pipeline.py \
  --samples 20 \
  --duration 90 \
  --scenarios wifi high-quality mobile-4g \
  --model-types xgboost
```

---

## 故障排除

### 问题1: 权限错误

```
PermissionError: Operation not permitted
```

**解决方案**:
- 使用 `sudo` 运行脚本
- macOS可能需要在"安全性与隐私"中授权终端访问网络

### 问题2: 找不到网卡

```
ValueError: Interface 'en0' not found
```

**解决方案**:
```bash
# 列出所有网卡
python scripts/realtime_capture_host.py --list-interfaces

# 使用正确的网卡名称
sudo python3 scripts/realtime_capture_host.py --interface <正确的名称>
```

### 问题3: 模型加载失败

```
FileNotFoundError: models/xgboost_model.pkl not found
```

**解决方案**:
```bash
# 检查模型文件是否存在
ls -l models/

# 如果不存在，在VM中重新训练
sudo python3 scripts/auto_train_pipeline.py --train-only

# 传输到宿主机
scp mininet@<VM-IP>:/home/mininet/cn/models/*.pkl ./models/
```

### 问题4: 没有捕获到视频包

```
监测时长: 30秒
视频包数: 0
```

**可能原因**:
1. 没有视频流量（确保正在播放视频）
2. 网卡选择错误（检查是否为正确的Wi-Fi/以太网接口）
3. 视频使用了不常见的端口

**解决方案**:
```bash
# 1. 确认视频正在播放
# 2. 不指定target-ip，捕获所有视频流量
sudo python3 scripts/realtime_capture_host.py \
  --interface en0 \
  --model models/xgboost_model.pkl
  
# 3. 查看原始流量确认
sudo tcpdump -i en0 -c 100 tcp port 443
```

### 问题5: 预测结果不准确

**解决方案**:
1. **收集更多训练数据**:
   ```bash
   sudo python3 scripts/auto_train_pipeline.py --samples 20 --duration 90
   ```

2. **确保场景覆盖全面**: 包含所有网络条件（低带宽、高带宽、丢包等）

3. **检查特征完整性**: 确保预处理器与模型匹配

4. **使用不同模型**: 尝试Random Forest而不是XGBoost

### 问题6: VM实验失败

```
Failed to setup experiment
```

**解决方案**:
```bash
# 清理Mininet
sudo mn -c

# 重启网络
sudo systemctl restart networking  # Linux
# 或
sudo /etc/init.d/networking restart

# 重新运行
sudo python3 scripts/auto_train_pipeline.py --quick
```

---

## 性能优化

### 1. 提高预测准确率

- 增加训练样本数: `--samples 20`
- 延长实验时长: `--duration 90`
- 覆盖更多场景
- 收集真实场景数据

### 2. 减少资源消耗

- 增大窗口大小: `--window-size 2.0` （减少预测频率）
- 使用更轻量的模型
- 限制捕获包大小

### 3. 提高监测效率

- 使用BPF过滤器精确捕获
- 指定target-ip减少处理包数
- 使用专用网卡

---

## 生产环境建议

### 1. 安全性

- 不要在生产环境中保存PCAP文件（可能包含敏感信息）
- 限制模型访问权限
- 使用安全的模型传输方式

### 2. 可靠性

- 实现错误恢复机制
- 添加健康检查
- 配置日志轮转

### 3. 可扩展性

- 使用消息队列分发预测任务
- 部署多个监测节点
- 实现模型热更新

---

## 参考资料

- [系统架构文档](../architecture.md)
- [模型训练指南](../training-guide.md)
- [特征工程文档](../features.md)
- [API参考](../api-reference.md)

---

**最后更新**: 2025-11-15  
**文档版本**: 1.0.0


