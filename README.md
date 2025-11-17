# 快速开始指南 - 视频质量评估系统
Quick Start Guide - Video QoE Assessment System

从零到实时监测，只需三步！

---

## 🚀 三步开始

### 第一步：在VM中生成训练数据并训练模型

在Mininet VM中运行：

```bash
cd /home/mininet/cn

# 快速测试（约30分钟）
sudo python3 scripts/auto_train_pipeline.py --quick

# 或完整训练（推荐，约2-4小时）
sudo python3 scripts/auto_train_pipeline.py --samples 10 --duration 60
```

完成后会得到：
- ✅ `models/xgboost_model.pkl` - 训练好的模型
- ✅ `models/preprocessor.pkl` - 特征预处理器
- ✅ `models/training_report.md` - 训练报告

### 第二步：将模型传输到宿主机

```bash
# 从宿主机执行（假设VM IP为192.168.56.101）
scp mininet@192.168.56.101:/home/mininet/cn/models/*.pkl ./models/
```

或使用共享文件夹/U盘等方式。

### 第三步：在宿主机上实时监测

在宿主机上：

```bash
# 1. 安装依赖（首次运行）
pip install pyshark pandas numpy scikit-learn xgboost rich joblib netifaces

# 2. 打开YouTube等视频网站开始播放

# 3. 运行监测脚本
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --interface en0

# 4. 观察实时输出，查看视频质量变化！
```

---

## 📊 效果展示

运行后会看到类似这样的实时输出：

```
┌─────────────────────────────────────────┐
│     🎥 实时视频质量监测                   │
├──────────────┬──────────────────────────┤
│ 监测时长     │ 45 秒                     │
│ 捕获包数     │ 1,234                     │
│ 视频包数     │ 856                       │
│ 总流量       │ 12.34 MB                  │
│ 预测次数     │ 42                        │
│ 当前质量     │ 1080p (89.5%)             │
│ 当前吞吐     │ 8.56 Mbps                 │
└──────────────┴──────────────────────────┘

┌────────────── 📊 最近预测 ──────────────┐
│ 1080p (89.5%) | 8.56 Mbps               │
│ 1080p (91.2%) | 9.12 Mbps               │
│ 720p (78.3%) | 5.43 Mbps                │
│ 1080p (88.7%) | 8.91 Mbps               │
└──────────────────────────────────────────┘
```

---

## 🛠️ 常见问题

### Q1: 没有sudo权限怎么办？

**A**: VM端训练必须使用sudo（Mininet需要）。宿主机端也需要sudo才能捕获网络包。

### Q2: 如何知道使用哪个网卡？

**A**: 运行以下命令查看：

```bash
python scripts/realtime_capture_host.py --list-interfaces
```

常见网卡：
- macOS: `en0` (Wi-Fi), `en1` (以太网)
- Linux: `eth0`, `wlan0`
- Windows: 在网络连接中查看

### Q3: 训练需要多长时间？

**A**: 
- 快速模式 (`--quick`): 约30分钟
- 推荐模式 (`--samples 10`): 约2-4小时
- 仅训练模式 (`--train-only`): 约5-15分钟

### Q4: 捕获不到视频流量？

**A**: 确保：
1. ✅ 视频正在播放
2. ✅ 使用正确的网卡（运行视频的网卡）
3. ✅ 没有使用VPN（会干扰捕获）

### Q5: 预测结果不准确？

**A**: 
1. 收集更多训练数据（增加`--samples`）
2. 确保训练数据覆盖多种网络条件
3. 尝试不同的模型类型

---

## 📚 详细文档

- **完整部署指南**: [docs/guides/PRODUCTION_DEPLOYMENT.md](docs/guides/PRODUCTION_DEPLOYMENT.md)
- **系统架构**: [docs/architecture.md](docs/architecture.md)
- **训练指南**: [models/README.md](models/README.md)
- **故障排除**: [docs/guides/TROUBLESHOOTING.md](docs/guides/TROUBLESHOOTING.md)

---

## 🎯 下一步

### 提高准确性

```bash
# 收集更多训练数据
sudo python3 scripts/auto_train_pipeline.py --samples 20 --duration 90
```

### 监测特定网站

```bash
# 获取YouTube IP
ping youtube.com

# 监测该IP
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --target-ip <YouTube的IP>
```

### 保存监测数据

```bash
sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --save-pcap monitoring_$(date +%Y%m%d_%H%M%S).pcap
```

---

## 💡 使用技巧

### 1. 后台运行监测

```bash
nohup sudo python3 scripts/realtime_capture_host.py \
  --model models/xgboost_model.pkl \
  --duration 3600 \
  > monitoring.log 2>&1 &
```

### 2. 定时训练更新模型

```bash
# 添加到crontab
0 2 * * 0 cd /home/mininet/cn && sudo python3 scripts/auto_train_pipeline.py --train-only
```

### 3. 比较不同模型

```bash
# 训练多个模型
sudo python3 scripts/auto_train_pipeline.py \
  --model-types xgboost random_forest

# 分别测试效果
sudo python3 scripts/realtime_capture_host.py --model models/xgboost_model.pkl
sudo python3 scripts/realtime_capture_host.py --model models/random_forest_model.pkl
```

---

## 🆘 获取帮助

```bash
# 查看训练脚本帮助
python scripts/auto_train_pipeline.py --help

# 查看监测脚本帮助
python scripts/realtime_capture_host.py --help
```

---

## ✅ 系统要求

### VM端（Mininet）
- Ubuntu 14.04+ or Debian
- Python 3.7+
- Mininet 2.3+
- 2GB+ RAM
- 10GB+ 磁盘空间

### 宿主机端
- macOS 10.14+ / Linux / Windows 10+
- Python 3.7+
- 网卡访问权限（sudo/admin）
- 500MB+ 磁盘空间（模型文件）

---

**开始您的视频质量监测之旅吧！** 🚀

有问题？查看 [完整文档](docs/guides/PRODUCTION_DEPLOYMENT.md) 或提交 Issue。


