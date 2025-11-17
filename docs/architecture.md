# 系统架构文档
## 视频质量评估系统 - Mininet版

**项目名称：** 应用质量评估  
**版本：** 1.0 (MVP)  
**日期：** 2025-11-08  
**架构师：** BMad  
**状态：** ✅ 已完成

---

## 📋 文档概述

本文档定义了"视频质量评估系统"的完整技术架构，包括系统设计、模块划分、关键技术选型、数据流设计和V2扩展规划。

**架构目标：**
- ✅ 满足实时监测要求（延迟 < 10秒）
- ✅ 准确计算35个特征（性能优化）
- ✅ 模块化设计，便于测试和维护
- ✅ 为V2真实浏览器场景预留扩展点

**相关文档：**
- [PRD产品需求文档](PRD.md) - 功能需求定义
- [头脑风暴会话](brainstorming-session-2025-11-08.md) - 创新特征工程方法
- [领域研究报告](research-comprehensive-2025-11-08.md) - 技术调研

---

## 🎯 架构设计原则

### 核心原则

1. **实时性优先**：流式处理，边捕获边分析，不等完整PCAP
2. **性能优化**：原生Python + numpy向量化，确保 < 10秒延迟
3. **模块解耦**：7大核心模块，独立开发测试
4. **配置驱动**：场景、模型、输出可灵活配置
5. **可扩展性**：接口抽象，为V2真实浏览器预留扩展点
6. **研究友好**：代码可读性与性能平衡，便于论文复现

### 设计约束

**性能约束：**
- 监测延迟 < 10秒（端到端）
- 内存占用 < 2GB
- CPU使用 < 50%（单核）
- 启动时间 < 30秒

**功能约束：**
- MVP仅支持Mininet环境
- 单机运行（非分布式）
- 单流量捕获（一次一个实验）

**技术约束：**
- Python 3.8+
- Mininet 2.3+
- Linux环境（Ubuntu 20.04+）
- 需要root权限（网络捕获）

---

## 🏗️ 系统整体架构

### 架构模式

采用 **Pipeline架构模式**（流水线处理）+ **分层架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                    用户接口层                                 │
│              CLI命令行 (monitor.py / train_model.py)          │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    应用逻辑层                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ 实验管理器    │  │ 监测流水线    │  │ 训练流水线    │     │
│  │ (Experiment  │  │ (Monitoring  │  │ (Training    │     │
│  │  Manager)    │  │  Pipeline)   │  │  Pipeline)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    核心服务层                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ 流量捕获    │  │ 特征提取    │  │ 预测引擎    │           │
│  │ (Capturer) │→│ (Feature   │→│ (Predictor)│           │
│  │            │  │  Extractor)│  │            │           │
│  └────────────┘  └────────────┘  └────────────┘           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ 模型训练    │  │ 输出管理    │  │ 配置管理    │           │
│  │ (Trainer)  │  │ (Output)   │  │ (Config)   │           │
│  └────────────┘  └────────────┘  └────────────┘           │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    基础设施层                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ Mininet    │  │ tcpdump/   │  │ 文件系统    │           │
│  │ 网络仿真    │  │ pyshark    │  │ PCAP/CSV   │           │
│  └────────────┘  └────────────┘  └────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### 数据流架构

**实时监测数据流：**

```
1. 实验启动
   ↓
2. Mininet环境初始化
   ├─ 创建网络拓扑 (h1 ←→ s1 ←→ h2)
   ├─ 配置网络条件 (带宽/延迟/丢包)
   └─ 启动HTTP Server (h1)
   ↓
3. 流量捕获启动
   ├─ tcpdump后台捕获 → capture.pcap
   └─ pyshark实时读取
   ↓
4. 实时处理流水线 (每秒循环)
   ├─ 包预处理: Packet → PacketInfo (轻量级)
   ├─ 滑动窗口: 维护最近1秒的数据包
   ├─ 特征计算: 35个特征 (numpy向量化)
   ├─ 模型推理: XGBoost预测分辨率
   └─ 输出显示: CLI实时输出 + CSV写入
   ↓
5. 实验结束
   ├─ 保存完整PCAP
   ├─ 保存特征CSV
   ├─ 保存Ground Truth
   └─ 生成实验报告
```

**模型训练数据流：**

```
1. 加载实验数据
   ├─ 扫描experiments/目录
   ├─ 加载PCAP文件
   └─ 加载Ground Truth
   ↓
2. 批量特征提取
   ├─ 遍历PCAP (离线分析)
   ├─ 计算35个特征 (复用监测代码)
   └─ 生成完整特征DataFrame
   ↓
3. 数据预处理
   ├─ 缺失值处理
   ├─ 特征归一化
   ├─ 特征选择 (可选)
   └─ 划分训练/验证/测试集
   ↓
4. 模型训练
   ├─ 训练XGBoost
   ├─ 训练Random Forest
   ├─ 训练LSTM
   └─ 超参数调优
   ↓
5. 模型评估
   ├─ 准确率/精确率/召回率
   ├─ 混淆矩阵
   ├─ 特征重要性分析
   └─ 生成评估报告
   ↓
6. 保存模型
   └─ 序列化为.pkl文件
```

---

## 🧩 核心模块设计

### 模块1: 实验管理器 (ExperimentManager)

**职责：**
- 解析命令行参数和配置文件
- 初始化Mininet环境
- 协调各模块的启动和停止
- 生成实验ID和目录结构
- 记录Ground Truth

**关键组件：**

```python
class ExperimentManager:
    """实验生命周期管理"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.exp_id = self._generate_exp_id()
        self.exp_dir = Path('experiments') / self.exp_id
        self.mininet_topo = None
        self.ground_truth = GroundTruth()
    
    def setup(self) -> ExperimentContext:
        """实验环境准备"""
        # 1. 创建实验目录
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 保存配置
        self._save_config()
        
        # 3. 初始化Mininet
        self.mininet_topo = self._setup_mininet()
        
        # 4. 记录Ground Truth
        self.ground_truth.record_config(self.config)
        
        return ExperimentContext(
            exp_id=self.exp_id,
            exp_dir=self.exp_dir,
            capture_interface='h2-eth0',
            pcap_path=self.exp_dir / 'capture.pcap'
        )
    
    def _setup_mininet(self) -> Mininet:
        """Mininet拓扑初始化"""
        # 创建拓扑: h1 (server) ←→ s1 ←→ h2 (client)
        topo = SingleSwitchTopo(n=2)
        net = Mininet(topo=topo, link=TCLink, controller=OVSController)
        net.start()
        
        # 获取主机
        h1, h2 = net.get('h1', 'h2')
        
        # 配置网络条件（在h2的link上）
        self._configure_network_conditions(net, h2)
        
        # 启动HTTP Server（h1）
        self._start_http_server(h1)
        
        return net
    
    def _configure_network_conditions(self, net, host):
        """配置带宽、延迟、丢包"""
        link = net.linksBetween(net.get('s1'), host)[0]
        link.intf1.config(
            bw=self._parse_bandwidth(self.config.network.bandwidth),
            delay=self.config.network.delay,
            loss=self.config.network.loss,
            jitter=self.config.network.jitter
        )
        
        # 记录到Ground Truth
        self.ground_truth.record_network_config(self.config.network)
    
    def cleanup(self):
        """清理实验环境"""
        if self.mininet_topo:
            self.mininet_topo.stop()
        
        # 保存Ground Truth
        self.ground_truth.save(self.exp_dir / 'ground_truth.json')
```

**接口定义：**

```python
@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment: Dict  # name, description
    network: NetworkConfig  # bandwidth, delay, loss, jitter
    video: VideoConfig  # file, expected_resolution
    model: ModelConfig  # path, confidence_threshold
    monitoring: MonitoringConfig  # update_interval, enable_color
    output: OutputConfig  # save_pcap, save_features

@dataclass
class ExperimentContext:
    """实验运行上下文"""
    exp_id: str
    exp_dir: Path
    capture_interface: str
    pcap_path: Path
```

---

### 模块2: 流量捕获器 (PacketCapturer)

**职责：**
- 使用tcpdump后台捕获PCAP
- 使用pyshark实时读取流式数据包
- 同时保存完整PCAP文件用于离线分析

**技术选型：** tcpdump + pyshark读取（稳定可靠）

**设计：**

```python
class PacketCapturer:
    """流量捕获器（tcpdump + pyshark）"""
    
    def __init__(self, interface: str, pcap_path: Path):
        self.interface = interface
        self.pcap_path = pcap_path
        self.tcpdump_process = None
        self.capture = None
    
    def start(self):
        """启动捕获"""
        # 1. 启动tcpdump后台进程（保存PCAP）
        self.tcpdump_process = subprocess.Popen([
            'sudo', 'tcpdump',
            '-i', self.interface,
            '-w', str(self.pcap_path),
            '-s', '0',  # 完整包
            'tcp'  # 只捕获TCP
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. 等待PCAP文件创建
        time.sleep(1)
        
        # 3. 启动pyshark实时读取
        self.capture = pyshark.FileCapture(
            str(self.pcap_path),
            keep_packets=False,  # 不保留在内存（重要！）
            display_filter='tcp'
        )
    
    def get_packet_stream(self) -> Iterator[Packet]:
        """获取实时数据包流"""
        return self.capture.sniff_continuously()
    
    def stop(self):
        """停止捕获"""
        if self.tcpdump_process:
            self.tcpdump_process.terminate()
            self.tcpdump_process.wait()
        
        if self.capture:
            self.capture.close()
```

**关键设计点：**
- `keep_packets=False`: pyshark不在内存中保留数据包，避免内存泄漏
- `sniff_continuously()`: 流式迭代器，边读边处理
- tcpdump在后台持续写入，pyshark实时尾随读取

---

### 模块3: 特征提取引擎 (FeatureExtractor)

**职责：**
- 维护1秒滑动窗口
- 计算35个TCP/IP特征（性能优化重点）
- 输出特征向量供模型使用

**技术选型：** 原生Python + numpy向量化（性能优先）

**三层架构：**

#### 层1: 包级预处理

```python
@dataclass
class PacketInfo:
    """轻量级数据包信息（避免保留完整包对象）"""
    timestamp: float
    size: int
    tcp_seq: Optional[int]
    tcp_ack: Optional[int]
    tcp_flags: Optional[str]
    tcp_window: Optional[int]
    rtt: Optional[float]
    is_retrans: bool
    direction: str  # 'up' or 'down'

class PacketPreprocessor:
    """包预处理器"""
    
    def __init__(self, client_ip: str):
        self.client_ip = client_ip
    
    def extract_packet_info(self, pkt) -> Optional[PacketInfo]:
        """从pyshark Packet提取关键信息"""
        try:
            # 只处理TCP包
            if not hasattr(pkt, 'tcp'):
                return None
            
            # 提取关键字段
            return PacketInfo(
                timestamp=float(pkt.sniff_timestamp),
                size=int(pkt.length),
                tcp_seq=int(pkt.tcp.seq) if hasattr(pkt.tcp, 'seq') else None,
                tcp_ack=int(pkt.tcp.ack) if hasattr(pkt.tcp, 'ack') else None,
                tcp_flags=pkt.tcp.flags if hasattr(pkt.tcp, 'flags') else None,
                tcp_window=int(pkt.tcp.window_size_value) if hasattr(pkt.tcp, 'window_size_value') else None,
                rtt=float(pkt.tcp.analysis.ack_rtt) if hasattr(pkt.tcp, 'analysis') and hasattr(pkt.tcp.analysis, 'ack_rtt') else None,
                is_retrans=(hasattr(pkt.tcp, 'analysis') and hasattr(pkt.tcp.analysis, 'retransmission')),
                direction='up' if pkt.ip.src == self.client_ip else 'down'
            )
        except Exception as e:
            logger.warning(f"Failed to extract packet info: {e}")
            return None
```

#### 层2: 滑动窗口缓冲

```python
class SlidingWindowBuffer:
    """1秒滑动窗口（高效实现）"""
    
    def __init__(self, window_size: float = 1.0):
        self.packets = deque()  # 双端队列，O(1)插入删除
        self.window_size = window_size
    
    def add_packet(self, pkt_info: PacketInfo):
        """添加数据包"""
        self.packets.append(pkt_info)
        self._cleanup_old_packets()
    
    def _cleanup_old_packets(self):
        """移除窗口外的老数据包"""
        if not self.packets:
            return
        
        now = self.packets[-1].timestamp
        while self.packets and (now - self.packets[0].timestamp) > self.window_size:
            self.packets.popleft()
    
    def get_window_data(self) -> List[PacketInfo]:
        """获取当前窗口的所有数据包"""
        return list(self.packets)
    
    def is_ready(self) -> bool:
        """窗口是否已积累足够数据"""
        if len(self.packets) < 10:  # 至少10个包
            return False
        
        if not self.packets:
            return False
        
        duration = self.packets[-1].timestamp - self.packets[0].timestamp
        return duration >= self.window_size * 0.8  # 至少0.8秒
```

#### 层3: 高性能特征计算

```python
class OptimizedFeatureCalculator:
    """35个特征的高性能计算（原生Python + numpy）"""
    
    def compute_all_features(self, packets: List[PacketInfo]) -> np.ndarray:
        """
        计算35个特征
        性能目标: < 3秒（1秒窗口数据）
        """
        if not packets:
            return np.zeros(35)
        
        # 一次性转numpy数组（后续所有计算复用）
        timestamps = np.array([p.timestamp for p in packets])
        sizes = np.array([p.size for p in packets])
        rtts = np.array([p.rtt for p in packets if p.rtt is not None])
        windows = np.array([p.tcp_window for p in packets if p.tcp_window is not None])
        
        # 并行计算三组特征
        tcp_features = self._compute_tcp_features(packets, rtts, windows)
        traffic_features = self._compute_traffic_features(packets, timestamps, sizes)
        temporal_features = self._compute_temporal_features(timestamps, sizes)
        
        return np.concatenate([tcp_features, traffic_features, temporal_features])
    
    def _compute_tcp_features(self, packets, rtts, windows) -> np.ndarray:
        """TCP层特征（10个）"""
        features = np.zeros(10)
        
        # 1. 重传率
        retrans_count = sum(1 for p in packets if p.is_retrans)
        features[0] = retrans_count / len(packets) if packets else 0
        
        # 2-4. RTT统计（numpy向量化）
        if len(rtts) > 0:
            features[1] = np.mean(rtts)  # avg_rtt
            features[2] = np.std(rtts)   # rtt_std
            features[3] = np.max(rtts)   # max_rtt
        
        # 5-6. TCP窗口统计
        if len(windows) > 0:
            features[4] = np.mean(windows)  # avg_window
            features[5] = np.var(windows)   # window_var
        
        # 7. 慢启动计数（窗口快速增长）
        features[6] = self._detect_slow_start(windows)
        
        # 8. 拥塞事件（窗口突降）
        features[7] = self._detect_congestion_events(windows)
        
        # 9. ACK延迟
        features[8] = self._compute_ack_delay(packets)
        
        # 10. 连接建立时间
        features[9] = self._get_conn_setup_time(packets)
        
        return features
    
    def _compute_traffic_features(self, packets, timestamps, sizes) -> np.ndarray:
        """流量统计特征（15个）"""
        features = np.zeros(15)
        
        # 吞吐量计算（向量化）
        if len(timestamps) > 1:
            duration = timestamps[-1] - timestamps[0]
            if duration > 0:
                total_bytes = np.sum(sizes)
                throughput_mbps = (total_bytes * 8) / (duration * 1e6)
                
                # 细粒度吞吐量（10个100ms窗口）
                mini_windows = np.array_split(sizes, 10)
                mini_throughputs = [
                    np.sum(w) * 8 / 0.1 / 1e6 
                    for w in mini_windows if len(w) > 0
                ]
                
                features[0] = throughput_mbps  # avg_throughput
                if mini_throughputs:
                    features[1] = np.std(mini_throughputs)   # throughput_std
                    features[2] = np.min(mini_throughputs)   # throughput_min
                    features[3] = np.max(mini_throughputs)   # throughput_max
                    mean_tp = np.mean(mini_throughputs)
                    features[4] = np.std(mini_throughputs) / mean_tp if mean_tp > 0 else 0  # throughput_cv
        
        # 包大小统计
        features[5] = np.mean(sizes)              # avg_packet_size
        features[6] = np.std(sizes)               # packet_size_std
        features[7] = np.sum(sizes > 1200) / len(sizes)  # large_packet_ratio (MTU)
        features[8] = self._compute_entropy(sizes)       # packet_size_entropy
        
        # 上下行比例
        up_bytes = sum(p.size for p in packets if p.direction == 'up')
        down_bytes = sum(p.size for p in packets if p.direction == 'down')
        features[9] = up_bytes / down_bytes if down_bytes > 0 else 0
        
        # 总量统计
        features[10] = np.sum(sizes)                    # total_bytes
        features[11] = len(sizes)                       # total_packets
        features[12] = timestamps[-1] - timestamps[0]   # conn_duration
        features[13] = np.var(np.diff(sizes)) if len(sizes) > 1 else 0  # byte_rate_var
        features[14] = 1  # flow_count (Mininet单流)
        
        return features
    
    def _compute_temporal_features(self, timestamps, sizes) -> np.ndarray:
        """时序特征（10个）"""
        features = np.zeros(10)
        
        # 包间隔统计
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            features[0] = np.mean(intervals)  # interval_mean
            features[1] = np.std(intervals)   # interval_std
            mean_iv = np.mean(intervals)
            features[2] = np.std(intervals) / mean_iv if mean_iv > 0 else 0  # interval_cv
        else:
            intervals = np.array([])
        
        # 周期性检测（FFT）
        features[3] = self._compute_periodicity_fft(intervals)
        
        # 空窗期检测
        if len(intervals) > 0:
            gap_threshold = 0.5  # 500ms
            gaps = intervals > gap_threshold
            features[4] = np.sum(gaps)  # num_gaps
            features[5] = np.mean(intervals[gaps]) if np.sum(gaps) > 0 else 0  # gap_duration_avg
        
        # 突发检测
        features[6], features[7] = self._detect_bursts(timestamps, sizes)
        
        # 自相关
        features[8] = self._compute_autocorrelation(sizes)
        
        # 趋势斜率
        features[9] = self._compute_trend_slope(timestamps, sizes)
        
        return features
    
    # 辅助方法
    def _compute_entropy(self, values: np.ndarray) -> float:
        """计算香农熵"""
        hist, _ = np.histogram(values, bins=20)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))
    
    def _compute_periodicity_fft(self, intervals: np.ndarray) -> float:
        """FFT检测周期性"""
        if len(intervals) < 4:
            return 0
        fft = np.fft.fft(intervals)
        power = np.abs(fft) ** 2
        return np.max(power[1:]) / np.sum(power) if np.sum(power) > 0 else 0
    
    def _detect_bursts(self, timestamps, sizes) -> Tuple[int, float]:
        """突发检测（滑动窗口）"""
        burst_threshold = np.mean(sizes) * 2
        burst_count = 0
        burst_intensity = 0
        
        # 100ms滑动窗口
        window_size = 0.1
        i = 0
        while i < len(timestamps):
            window_start = timestamps[i]
            window_bytes = 0
            j = i
            while j < len(timestamps) and (timestamps[j] - window_start) < window_size:
                window_bytes += sizes[j]
                j += 1
            
            if window_bytes > burst_threshold:
                burst_count += 1
                burst_intensity += window_bytes
            
            i = j if j > i else i + 1
        
        return burst_count, burst_intensity / burst_count if burst_count > 0 else 0
    
    def _compute_autocorrelation(self, values: np.ndarray, lag: int = 1) -> float:
        """自相关系数"""
        if len(values) < lag + 1:
            return 0
        return np.corrcoef(values[:-lag], values[lag:])[0, 1] if len(values) > lag else 0
    
    def _compute_trend_slope(self, timestamps, sizes) -> float:
        """线性拟合趋势"""
        if len(timestamps) < 2:
            return 0
        coeffs = np.polyfit(timestamps - timestamps[0], sizes, 1)
        return coeffs[0]  # 斜率
    
    def _detect_slow_start(self, windows: np.ndarray) -> int:
        """检测慢启动（窗口指数增长）"""
        if len(windows) < 3:
            return 0
        
        growth_rate = np.diff(windows) / (windows[:-1] + 1)
        return np.sum(growth_rate > 0.5)  # 增长50%以上
    
    def _detect_congestion_events(self, windows: np.ndarray) -> int:
        """检测拥塞事件（窗口突降）"""
        if len(windows) < 2:
            return 0
        
        drop_rate = np.diff(windows) / (windows[:-1] + 1)
        return np.sum(drop_rate < -0.5)  # 下降50%以上
    
    def _compute_ack_delay(self, packets: List[PacketInfo]) -> float:
        """平均ACK延迟"""
        ack_packets = [p for p in packets if p.tcp_flags and 'A' in p.tcp_flags]
        if not ack_packets or len(ack_packets) < 2:
            return 0
        
        timestamps = np.array([p.timestamp for p in ack_packets])
        delays = np.diff(timestamps)
        return np.mean(delays)
    
    def _get_conn_setup_time(self, packets: List[PacketInfo]) -> float:
        """连接建立时间（SYN → SYN-ACK → ACK）"""
        # 查找SYN, SYN-ACK, ACK
        syn_time = None
        synack_time = None
        
        for p in packets:
            if not p.tcp_flags:
                continue
            if 'S' in p.tcp_flags and 'A' not in p.tcp_flags:  # SYN
                syn_time = p.timestamp
            elif 'S' in p.tcp_flags and 'A' in p.tcp_flags:  # SYN-ACK
                synack_time = p.timestamp
            elif syn_time and synack_time and 'A' in p.tcp_flags:  # ACK
                return p.timestamp - syn_time
        
        return synack_time - syn_time if (syn_time and synack_time) else 0
```

**性能优化关键点：**
1. ✅ 单次遍历：数据转numpy后，所有计算基于向量操作
2. ✅ 轻量级对象：PacketInfo只保留必要字段
3. ✅ 避免Python循环：尽可能使用numpy函数
4. ✅ 增量计算：滑动窗口deque高效
5. ✅ 预估性能：2-3秒（1秒窗口数据）

---

### 模块4: 预测引擎 (PredictionEngine)

**职责：**
- 加载训练好的模型
- 特征向量 → 分辨率预测
- 计算置信度

**设计：**

```python
@dataclass
class Prediction:
    """预测结果"""
    resolution: str  # '480p', '720p', '1080p'
    confidence: float  # 0-1
    probabilities: np.ndarray  # [prob_480p, prob_720p, prob_1080p]
    timestamp: float

class PredictionEngine:
    """预测引擎（单例模型）"""
    
    def __init__(self, model_path: Path):
        self.model = self._load_model(model_path)
        self.classes = ['480p', '720p', '1080p']
    
    def _load_model(self, model_path: Path):
        """加载预训练模型"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        return joblib.load(model_path)
    
    def predict(self, features: np.ndarray) -> Prediction:
        """
        预测分辨率
        Args:
            features: shape (1, 35) 或 (35,)
        Returns:
            Prediction对象
        """
        # 确保shape正确
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 模型推理
        probabilities = self.model.predict_proba(features)[0]
        pred_class = np.argmax(probabilities)
        
        return Prediction(
            resolution=self.classes[pred_class],
            confidence=probabilities[pred_class],
            probabilities=probabilities,
            timestamp=time.time()
        )
    
    def batch_predict(self, features: np.ndarray) -> List[Prediction]:
        """批量预测（用于离线分析）"""
        probabilities = self.model.predict_proba(features)
        pred_classes = np.argmax(probabilities, axis=1)
        
        return [
            Prediction(
                resolution=self.classes[pred_classes[i]],
                confidence=probabilities[i, pred_classes[i]],
                probabilities=probabilities[i],
                timestamp=time.time()
            )
            for i in range(len(features))
        ]
```

---

### 模块5: 输出管理器 (OutputManager)

**职责：**
- CLI实时输出（rich库，颜色高亮）
- 数据持久化（CSV、JSON、PCAP）
- 实验总结报告

**技术选型：** rich库（现代化CLI体验）

**设计：**

```python
class OutputManager:
    """输出管理器"""
    
    def __init__(self, exp_dir: Path, enable_color: bool = True):
        self.exp_dir = exp_dir
        self.cli_writer = CLIWriter(enable_color)
        self.data_writer = DataWriter(exp_dir)
        self.start_time = time.time()
        self.event_count = {'quality_up': 0, 'quality_down': 0, 'stall': 0}
    
    def output_realtime(self, prediction: Prediction, metrics: NetworkMetrics):
        """实时输出（每秒调用一次）"""
        # 1. CLI输出
        elapsed = int(time.time() - self.start_time)
        self.cli_writer.write_line(elapsed, prediction, metrics)
        
        # 2. 同步写入CSV
        self.data_writer.append_csv(elapsed, prediction, metrics)
        
        # 3. 检测事件
        self._detect_and_log_events(prediction, metrics)
    
    def finalize(self):
        """实验结束，生成总结"""
        self.data_writer.close()
        self._generate_summary()

class CLIWriter:
    """CLI输出（rich库）"""
    
    def __init__(self, enable_color: bool):
        self.console = Console() if enable_color else Console(no_color=True)
        self.prev_resolution = None
    
    def write_header(self, exp_id: str, scenario: str, video: str):
        """输出头部信息"""
        self.console.print("\n[bold]=== 视频质量实时监测 ===[/bold]")
        self.console.print(f"实验ID: {exp_id}")
        self.console.print(f"场景: {scenario}")
        self.console.print(f"视频: {video}")
        self.console.print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.console.print("━" * 80)
        self.console.print()
    
    def write_line(self, elapsed: int, prediction: Prediction, metrics: NetworkMetrics):
        """输出单行监测数据"""
        # 格式化时间
        time_str = f"[{elapsed:03d}]"
        
        # 分辨率（颜色编码）
        res_str = self._format_resolution(prediction.resolution, prediction.confidence)
        
        # 网络指标
        metrics_str = (
            f"吞吐量: {metrics.throughput:.1f} Mbps | "
            f"丢包: {metrics.loss_rate:.1f}% | "
            f"RTT: {metrics.rtt:.0f}ms"
        )
        
        # 事件标注
        event_str = self._check_events(prediction)
        
        # 组合输出
        line = f"{time_str} {res_str} | {metrics_str}{event_str}"
        self.console.print(line)
    
    def _format_resolution(self, resolution: str, confidence: float) -> str:
        """分辨率颜色编码"""
        conf_str = f"({confidence*100:.0f}%)"
        
        # 低置信度警告
        if confidence < 0.7:
            return f"[yellow]{resolution}? {conf_str}[/yellow]"
        
        # 正常置信度，按分辨率着色
        color_map = {
            '480p': 'red',
            '720p': 'yellow',
            '1080p': 'green',
            '4K': 'bright_green'
        }
        color = color_map.get(resolution, 'white')
        return f"[{color}]{resolution} {conf_str}[/{color}]"
    
    def _check_events(self, prediction: Prediction) -> str:
        """检测质量变化事件"""
        if self.prev_resolution is None:
            self.prev_resolution = prediction.resolution
            return ""
        
        event = ""
        if prediction.resolution != self.prev_resolution:
            if self._resolution_to_num(prediction.resolution) > self._resolution_to_num(self.prev_resolution):
                event = " [green]⚠️ 质量提升[/green]"
            else:
                event = " [red]🔴 质量下降[/red]"
        
        self.prev_resolution = prediction.resolution
        return event
    
    @staticmethod
    def _resolution_to_num(res: str) -> int:
        """分辨率转数字"""
        return {'480p': 1, '720p': 2, '1080p': 3, '4K': 4}.get(res, 0)

class DataWriter:
    """数据持久化"""
    
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.features_csv = exp_dir / 'features.csv'
        self.timeline_json = exp_dir / 'timeline.json'
        
        # 初始化CSV
        self.csv_file = open(self.features_csv, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self._write_csv_header()
        
        # 时间线数据
        self.timeline_events = []
    
    def _write_csv_header(self):
        """写入CSV表头（35个特征 + 预测 + 置信度）"""
        headers = [
            'timestamp',
            # TCP特征（10个）
            'retrans_rate', 'avg_rtt', 'rtt_std', 'max_rtt', 'avg_window', 
            'window_var', 'slow_start_count', 'congestion_events', 'ack_delay', 'conn_setup_time',
            # 流量统计特征（15个）
            'avg_throughput', 'throughput_std', 'throughput_min', 'throughput_max', 'throughput_cv',
            'avg_packet_size', 'packet_size_std', 'large_packet_ratio', 'packet_size_entropy',
            'uplink_downlink_ratio', 'total_bytes', 'total_packets', 'conn_duration', 'byte_rate_var', 'flow_count',
            # 时序特征（10个）
            'interval_mean', 'interval_std', 'interval_cv', 'periodicity_score', 'num_gaps',
            'gap_duration_avg', 'burst_count', 'burst_intensity', 'autocorrelation', 'trend_slope',
            # 预测结果
            'predicted_resolution', 'confidence'
        ]
        self.csv_writer.writerow(headers)
    
    def append_csv(self, elapsed: int, prediction: Prediction, metrics: NetworkMetrics):
        """追加一行数据（特征 + 预测）"""
        # 这里features已经在前面计算过，需要传入
        # 简化起见，从metrics中提取
        row = [elapsed] + list(metrics.features) + [prediction.resolution, prediction.confidence]
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # 立即写入
    
    def close(self):
        """关闭文件"""
        if self.csv_file:
            self.csv_file.close()
```

---

## ⏱️ 性能分析

### 延迟分解

| 阶段 | 预估延迟 | 说明 | 优化措施 |
|-----|---------|------|---------|
| PCAP捕获 | 1-2秒 | tcpdump写入延迟 | 后台异步，无法避免 |
| 包解析 | 0.5秒 | pyshark读取1秒窗口 | 流式读取，已优化 |
| **特征计算** | **2-3秒** | **35个特征计算（关键路径）** | **numpy向量化，单次遍历** |
| 模型推理 | 0.1秒 | XGBoost C++后端 | 已最优 |
| CLI输出 | 0.1秒 | rich输出 | 异步写入 |
| **总延迟** | **4-7秒** | **✅ 满足<10秒要求** | - |

### 内存占用

| 组件 | 预估内存 | 说明 |
|-----|---------|------|
| Mininet | ~200MB | 网络仿真 |
| tcpdump | ~50MB | PCAP捕获 |
| pyshark | ~100MB | 包解析 |
| 滑动窗口 | ~10MB | 1秒轻量级PacketInfo |
| 模型 | ~50MB | XGBoost模型 |
| Python运行时 | ~100MB | 基础开销 |
| **总计** | **~500MB-1GB** | **✅ 远小于2GB限制** |

### CPU使用

- **正常负载：** 30-40%（单核）
- **峰值负载：** 50-60%（特征计算时）
- **空闲期：** <10%（等待数据包）

### 扩展性能优化方案（如需）

如果实际测试发现延迟>7秒，可采用：

1. **多线程流水线：**
   ```
   线程1: 捕获 + 解析
   线程2: 特征计算
   线程3: 模型推理 + 输出
   ```

2. **Cython编译热点：**
   - 将`_compute_tcp_features`等编译为C扩展
   - 预计提速2-3倍

3. **更激进的增量计算：**
   - 缓存上一秒的中间结果
   - 只计算增量变化

---

## 📊 模型训练框架架构

### 设计原则

**训练与监测解耦，但共享特征计算核心**

```
监测工具 (monitor.py)       训练框架 (train_model.py)
      ↓                              ↓
  实时特征提取                  批量特征提取
      ↓                              ↓
      └────────→  共享特征库  ←────────┘
           (video_qoe.features)
```

### 训练流水线

```python
class ModelTrainingPipeline:
    """端到端训练流程"""
    
    def __init__(self, config: TrainingConfig):
        self.data_loader = ExperimentDataLoader()
        self.feature_extractor = FeatureExtractor()  # 复用监测代码
        self.preprocessor = FeaturePreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
    
    def run(self, experiments_dir: Path, output_dir: Path):
        """完整训练流程"""
        # 1. 加载所有实验
        logger.info("Loading experiments...")
        experiments = self.data_loader.load_experiments(experiments_dir)
        logger.info(f"Found {len(experiments)} experiments")
        
        # 2. 批量特征提取
        logger.info("Extracting features...")
        features_df = self._batch_extract_features(experiments)
        
        # 3. 数据预处理
        logger.info("Preprocessing...")
        X, y = self.preprocessor.prepare_dataset(features_df)
        
        # 4. 划分数据集
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self._split_dataset(X, y)
        
        # 5. 训练多种模型
        logger.info("Training models...")
        models = {}
        for model_type in ['xgboost', 'random_forest', 'lstm']:
            logger.info(f"  Training {model_type}...")
            model = self.trainer.train(
                model_type, X_train, y_train, X_val, y_val
            )
            models[model_type] = model
        
        # 6. 评估对比
        logger.info("Evaluating models...")
        results = self.evaluator.evaluate_all(models, X_test, y_test)
        
        # 7. 保存最佳模型
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_model = models[best_model_name]
        model_path = output_dir / f'{best_model_name}_v1.0.pkl'
        joblib.dump(best_model, model_path)
        logger.info(f"Best model ({best_model_name}) saved to {model_path}")
        
        # 8. 保存预处理器
        self.preprocessor.save(output_dir / 'preprocessor_v1.0.pkl')
        
        # 9. 生成报告
        self.evaluator.generate_report(results, output_dir)
        logger.info(f"Evaluation report saved to {output_dir}/evaluation_report.md")
```

---

## 📁 项目代码结构

```
video-qoe-assessment/
├── video_qoe/                      # 核心库（pip安装包）
│   ├── __init__.py
│   ├── features/                   # 特征计算（共享核心）
│   │   ├── __init__.py
│   │   ├── extractor.py           # 统一接口
│   │   ├── tcp_features.py        # TCP计算器
│   │   ├── traffic_features.py    # 流量计算器
│   │   ├── temporal_features.py   # 时序计算器
│   │   └── packet_info.py         # PacketInfo类
│   │
│   ├── monitoring/                 # 实时监测
│   │   ├── pipeline.py
│   │   ├── capturer.py
│   │   ├── preprocessor.py
│   │   ├── window_buffer.py
│   │   └── predictor.py
│   │
│   ├── training/                   # 模型训练
│   │   ├── pipeline.py
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   │
│   ├── experiment/                 # 实验管理
│   │   ├── manager.py
│   │   ├── topology.py
│   │   └── scenarios.py
│   │
│   ├── output/                     # 输出管理
│   │   ├── cli_writer.py
│   │   └── data_writer.py
│   │
│   └── models/                     # 模型相关
│       ├── base.py
│       └── model_loader.py
│
├── scripts/                        # CLI工具
│   ├── monitor.py                 # 实时监测
│   ├── train_model.py             # 模型训练
│   └── extract_features.py        # 特征提取
│
├── configs/                        # 配置
│   ├── scenarios/                 # 场景模板
│   │   ├── low-bandwidth.yaml
│   │   ├── high-quality.yaml
│   │   └── ...
│   └── default_config.yaml
│
├── models/                         # 预训练模型
│   ├── xgboost_v1.0.pkl
│   └── preprocessor_v1.0.pkl
│
├── experiments/                    # 实验数据
├── tests/                          # 单元测试
├── docs/                           # 文档
└── README.md
```

---

## ⚙️ 配置管理

### 多层配置系统

**优先级（从高到低）：**
1. 命令行参数
2. 用户配置文件（`--config`）
3. 场景模板（`--scenario`）
4. 默认配置

### 场景模板示例

```yaml
# configs/scenarios/low-bandwidth.yaml
name: "Low Bandwidth Scenario"
network:
  bandwidth: 2 Mbps
  delay: 50 ms
  loss: 1%
video:
  expected_resolution: "720p"
monitoring:
  update_interval: 1
  confidence_threshold: 0.7
```

---

## 🔌 V2扩展点设计

### 为真实浏览器场景预留接口

#### 1. 流量来源抽象

```python
class ITrafficSource(ABC):
    @abstractmethod
    def start_capture(self, interface: str) -> None: pass
    
    @abstractmethod
    def get_packet_stream(self) -> Iterator[Packet]: pass

# MVP: MininetTrafficSource
# V2: BrowserTrafficSource (with流量过滤)
```

#### 2. 流量识别模块（V2新增）

```python
class VideoTrafficClassifier:
    def is_video_flow(self, flow: Flow) -> bool:
        """判断是否视频流量"""
        pass
    
    def identify_platform(self, flow: Flow) -> str:
        """识别平台（YouTube/Netflix/爱奇艺）"""
        pass
```

#### 3. 可插拔特征计算器

```python
class IFeatureCalculator(ABC):
    @abstractmethod
    def compute(self, packets) -> np.ndarray: pass

# 允许研究人员扩展新特征
```

---

## 📊 关键架构决策记录

| 决策点 | 选择 | 理由 | 权衡 |
|-------|------|------|------|
| **流量捕获** | tcpdump + pyshark | 稳定可靠 | 略有延迟但可接受 |
| **特征计算** | 原生Python + numpy | 性能优先 | 代码复杂度 ↑ |
| **模型管理** | 单例模型 | MVP简洁 | V2需扩展 |
| **CLI输出** | rich | 现代体验 | 依赖 ↑ |
| **代码复用** | 共享特征库 | 避免重复 | 接口设计 |

---

## ✅ 架构验证

### 满足的PRD需求

**性能需求：**
- ✅ 监测延迟 4-7秒 < 10秒要求
- ✅ 内存占用 ~1GB < 2GB限制
- ✅ CPU使用 30-50% < 50%要求
- ✅ 启动时间 < 30秒（Mininet启动）

**功能需求：**
- ✅ 7大核心模块职责清晰
- ✅ 35个特征准确计算（numpy优化）
- ✅ 实时监测 + 模型训练双流水线
- ✅ 配置驱动（场景模板）
- ✅ V2扩展点预留

**非功能需求：**
- ✅ 模块解耦，便于测试
- ✅ 接口抽象，易于扩展
- ✅ 代码复用（共享特征库）
- ✅ 可观测性（日志/监控）

---

## 📈 后续工作

**开发阶段（参考PRD）：**

1. **Week 1-2: 基础框架**
   - Mininet实验管理器
   - 流量捕获器
   - 基础特征提取（10个核心特征）

2. **Week 3-5: 核心功能**
   - 完整特征工程（35个特征）
   - 实时监测流水线
   - CLI输出

3. **Week 6-7: 模型训练**
   - 数据采集（30+场景）
   - 训练流水线
   - 模型评估

4. **Week 8-9: 集成测试**
   - 端到端测试
   - 性能优化
   - 文档完善

5. **Week 10: 发布**
   - 用户验收
   - 论文实验支持

---

## 📚 参考文档

- [PRD产品需求文档](PRD.md)
- [头脑风暴会话](brainstorming-session-2025-11-08.md) - 第一性原理特征工程
- [综合领域研究](research-comprehensive-2025-11-08.md) - 技术调研

---

**架构文档状态：** ✅ 已完成  
**版本：** 1.0  
**批准日期：** 2025-11-08

---

## 🎯 架构总结

本架构设计了一个**高性能、模块化、可扩展**的视频质量评估系统。核心特点：

1. **Pipeline架构** - 流式处理，实时监测
2. **性能优化** - 原生Python + numpy，满足 < 10秒延迟
3. **模块解耦** - 7大模块，独立开发测试
4. **代码复用** - 监测和训练共享特征计算
5. **扩展性** - 接口抽象，为V2预留扩展点

**架构已就绪，可开始实施开发！** 🚀




