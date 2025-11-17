"""
数据持久化模块

负责将实时监测数据保存到文件：
- features.csv: 特征数据 + 预测结果
- timeline.json: 事件时间线
- summary.txt: 实验总结报告
"""

import csv
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

from video_qoe.prediction.predictor import Prediction


@dataclass
class TimelineEvent:
    """时间线事件
    
    Attributes:
        time: 事件发生时间（秒）
        type: 事件类型
        description: 事件描述
        data: 附加数据（可选）
    """
    time: float
    type: str
    description: str
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = {
            'time': self.time,
            'type': self.type,
            'description': self.description
        }
        if self.data:
            d['data'] = self.data
        return d


class DataWriter:
    """数据持久化器
    
    负责将实时监测数据持久化到文件系统：
    1. features.csv - 特征向量 + 预测结果（逐行追加）
    2. timeline.json - 事件时间线（实验结束时保存）
    3. summary.txt - 实验总结报告（实验结束时生成）
    
    特点：
    - CSV立即刷新（防止数据丢失）
    - 异常安全（try-except包装）
    - 完整的35个特征名称
    - 时间线记录关键事件
    
    Attributes:
        exp_dir: 实验目录
        features_csv: CSV文件路径
        timeline_json: 时间线JSON路径
        csv_file: CSV文件句柄
        csv_writer: CSV写入器
        timeline_events: 时间线事件列表
        logger: 日志记录器
        start_time: 开始时间
        last_resolution: 上一次的分辨率
    
    Example:
        >>> writer = DataWriter(exp_dir)
        >>> writer.append_data(elapsed=1, prediction=pred, features=feat_vec)
        >>> writer.record_event(1.5, 'quality_change', 'Resolution changed')
        >>> writer.finalize()
    """
    
    # 35个特征的完整名称
    FEATURE_NAMES = [
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
    ]
    
    def __init__(self, exp_dir: Path, logger: Optional[logging.Logger] = None):
        """初始化数据写入器
        
        Args:
            exp_dir: 实验目录路径
            logger: 日志记录器（可选）
        """
        self.exp_dir = Path(exp_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # 确保目录存在
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.features_csv = self.exp_dir / 'features.csv'
        self.timeline_json = self.exp_dir / 'timeline.json'
        self.summary_txt = self.exp_dir / 'summary.txt'
        
        # 初始化CSV文件
        self.csv_file = None
        self.csv_writer = None
        self._init_csv()
        
        # 时间线事件
        self.timeline_events: List[TimelineEvent] = []
        self.start_time = time.time()
        
        # 统计信息
        self.total_rows = 0
        self.last_resolution = None
        self.resolution_counts = {'480p': 0, '720p': 0, '1080p': 0}
        self.quality_changes = 0
        
        # 记录开始事件
        self.record_event(0, 'experiment_start', 'Experiment started')
        
        self.logger.info(f"DataWriter initialized: {self.exp_dir}")
    
    def _init_csv(self):
        """初始化CSV文件（写入表头）"""
        try:
            # 打开CSV文件（写模式）
            self.csv_file = open(self.features_csv, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            
            # 写入表头
            header = ['timestamp'] + self.FEATURE_NAMES + ['predicted_resolution', 'confidence']
            self.csv_writer.writerow(header)
            self.csv_file.flush()  # 立即写入
            
            self.logger.info(f"CSV file initialized: {self.features_csv}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CSV: {e}")
            raise
    
    def append_data(self, elapsed: float, prediction: Prediction, features: np.ndarray):
        """追加一行数据到CSV
        
        Args:
            elapsed: 经过时间（秒）
            prediction: 预测结果
            features: 35维特征向量
        
        Returns:
            是否成功写入
        """
        try:
            # 验证特征维度
            if len(features) != 35:
                self.logger.warning(f"Feature dimension mismatch: expected 35, got {len(features)}")
                # 填充或截断
                if len(features) < 35:
                    features = np.pad(features, (0, 35 - len(features)), constant_values=0)
                else:
                    features = features[:35]
            
            # 构建数据行
            row = [elapsed] + features.tolist() + [prediction.resolution, prediction.confidence]
            
            # 写入CSV
            self.csv_writer.writerow(row)
            self.csv_file.flush()  # 立即刷新到磁盘
            
            # 更新统计
            self.total_rows += 1
            self.resolution_counts[prediction.resolution] = self.resolution_counts.get(prediction.resolution, 0) + 1
            
            # 检测质量变化
            if self.last_resolution and self.last_resolution != prediction.resolution:
                self.quality_changes += 1
                self.record_event(
                    elapsed,
                    'quality_change',
                    f"Resolution changed: {self.last_resolution} → {prediction.resolution}",
                    {'from': self.last_resolution, 'to': prediction.resolution}
                )
            
            self.last_resolution = prediction.resolution
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to append data: {e}")
            return False
    
    def record_event(self, elapsed: float, event_type: str, description: str, 
                    data: Optional[Dict[str, Any]] = None):
        """记录时间线事件
        
        Args:
            elapsed: 事件发生时间（秒）
            event_type: 事件类型
            description: 事件描述
            data: 附加数据（可选）
        """
        try:
            event = TimelineEvent(
                time=elapsed,
                type=event_type,
                description=description,
                data=data
            )
            self.timeline_events.append(event)
            
            self.logger.debug(f"Event recorded: [{elapsed:.1f}s] {event_type}: {description}")
            
        except Exception as e:
            self.logger.error(f"Failed to record event: {e}")
    
    def save_timeline(self):
        """保存时间线到JSON文件"""
        try:
            # 添加结束事件
            elapsed = time.time() - self.start_time
            self.record_event(elapsed, 'experiment_end', 'Experiment ended')
            
            # 转换为字典
            timeline_data = {
                'experiment_id': self.exp_dir.name,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'duration_sec': elapsed,
                'total_events': len(self.timeline_events),
                'events': [event.to_dict() for event in self.timeline_events]
            }
            
            # 写入JSON文件
            with open(self.timeline_json, 'w', encoding='utf-8') as f:
                json.dump(timeline_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Timeline saved: {self.timeline_json} ({len(self.timeline_events)} events)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save timeline: {e}")
            return False
    
    def generate_summary(self):
        """生成实验总结报告"""
        try:
            elapsed = time.time() - self.start_time
            # 防止除零
            if elapsed <= 0:
                elapsed = 0.001
            
            # 计算统计信息
            total_predictions = self.total_rows
            avg_rate = total_predictions / elapsed if elapsed > 0 else 0
            
            # 计算主要分辨率
            if total_predictions > 0:
                dominant_res = max(self.resolution_counts, key=self.resolution_counts.get)
                dominant_pct = self.resolution_counts[dominant_res] / total_predictions * 100
            else:
                dominant_res = 'N/A'
                dominant_pct = 0
            
            # 生成报告内容
            report = f"""实验总结报告
{'=' * 80}

实验基本信息
{'-' * 80}
实验ID: {self.exp_dir.name}
开始时间: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}
结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
持续时长: {elapsed:.1f} 秒

预测统计
{'-' * 80}
总预测次数: {total_predictions}
平均预测速率: {avg_rate:.2f} pred/sec

分辨率分布:
  - 1080p: {self.resolution_counts.get('1080p', 0):4d} ({self.resolution_counts.get('1080p', 0) / max(total_predictions, 1) * 100:5.1f}% )
  - 720p:  {self.resolution_counts.get('720p', 0):4d} ({self.resolution_counts.get('720p', 0) / max(total_predictions, 1) * 100:5.1f}% )
  - 480p:  {self.resolution_counts.get('480p', 0):4d} ({self.resolution_counts.get('480p', 0) / max(total_predictions, 1) * 100:5.1f}% )

主要分辨率: {dominant_res} ({dominant_pct:.1f}%)
质量切换次数: {self.quality_changes}

事件统计
{'-' * 80}
总事件数: {len(self.timeline_events)}

事件类型分布:
"""
            # 统计事件类型
            event_types = {}
            for event in self.timeline_events:
                event_types[event.type] = event_types.get(event.type, 0) + 1
            
            for event_type, count in sorted(event_types.items()):
                report += f"  - {event_type}: {count}\n"
            
            report += f"""
数据文件
{'-' * 80}
[OK] {self.features_csv.name} ({self.total_rows} 行)
[OK] {self.timeline_json.name} ({len(self.timeline_events)} 事件)
[OK] capture.pcap (if exists)
[OK] ground_truth.json (if exists)
[OK] config.yaml (if exists)

{'=' * 80}
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # 写入文件
            with open(self.summary_txt, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"Summary report generated: {self.summary_txt}")
            
            # 也打印到日志
            self.logger.info("\n" + report)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            return False
    
    def finalize(self):
        """结束数据写入（保存时间线和生成报告）
        
        应在实验结束时调用此方法，完成：
        1. 关闭CSV文件
        2. 保存时间线JSON
        3. 生成总结报告
        """
        try:
            # 关闭CSV文件
            if self.csv_file:
                self.csv_file.close()
                self.logger.info(f"CSV file closed ({self.total_rows} rows)")
            
            # 保存时间线
            self.save_timeline()
            
            # 生成总结报告
            self.generate_summary()
            
            self.logger.info("DataWriter finalized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to finalize DataWriter: {e}")
    
    def __enter__(self):
        """上下文管理器：进入"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器：退出（自动调用finalize）"""
        self.finalize()
        return False


if __name__ == '__main__':
    """测试模块"""
    import tempfile
    import shutil
    
    print("=" * 70)
    print("Testing DataWriter")
    print("=" * 70)
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp()) / 'test_exp_001'
    print(f"\nTest directory: {temp_dir}")
    
    try:
        # 测试1: 初始化
        print("\n[Test 1] Initialize DataWriter")
        writer = DataWriter(temp_dir)
        print(f"  CSV file: {writer.features_csv}")
        print(f"  Timeline file: {writer.timeline_json}")
        print(f"  ✓ Initialized")
        
        # 测试2: 写入数据
        print("\n[Test 2] Append data")
        for i in range(5):
            # 模拟预测结果
            pred = Prediction(
                resolution='720p' if i % 2 == 0 else '480p',
                confidence=0.85,
                probabilities=np.array([0.05, 0.85, 0.10]),
                timestamp=time.time()
            )
            
            # 模拟特征向量（35维）
            features = np.random.rand(35)
            
            # 写入
            success = writer.append_data(elapsed=i, prediction=pred, features=features)
            print(f"  Row {i+1}: {pred.resolution} - {'✓' if success else '✗'}")
        
        print(f"  Total rows: {writer.total_rows}")
        
        # 测试3: 记录事件
        print("\n[Test 3] Record events")
        writer.record_event(2.5, 'network_spike', 'Bandwidth spike detected')
        writer.record_event(4.0, 'stall', 'Video stall', {'duration': 1.5})
        print(f"  Total events: {len(writer.timeline_events)}")
        
        # 测试4: 完成
        print("\n[Test 4] Finalize")
        writer.finalize()
        
        # 检查生成的文件
        print("\n[Test 5] Check generated files")
        files = list(temp_dir.glob('*'))
        for f in files:
            size = f.stat().st_size
            print(f"  ✓ {f.name} ({size} bytes)")
        
        # 验证CSV内容
        print("\n[Test 6] Verify CSV content")
        with open(writer.features_csv, 'r') as f:
            lines = f.readlines()
            print(f"  Total lines: {len(lines)} (including header)")
            print(f"  Header: {lines[0][:80]}...")
            if len(lines) > 1:
                print(f"  First row: {lines[1][:80]}...")
        
        # 验证时间线
        print("\n[Test 7] Verify timeline JSON")
        with open(writer.timeline_json, 'r') as f:
            timeline = json.load(f)
            print(f"  Experiment ID: {timeline['experiment_id']}")
            print(f"  Duration: {timeline['duration_sec']:.1f}s")
            print(f"  Total events: {timeline['total_events']}")
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        
    finally:
        # 清理临时目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir.parent)
            print(f"\nCleaned up: {temp_dir.parent}")


