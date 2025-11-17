"""
事件检测和标注模块

自动检测视频质量变化和网络异常事件。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import time

from video_qoe.prediction.predictor import Prediction


class EventType(Enum):
    """事件类型枚举"""
    QUALITY_IMPROVE = "quality_improve"      # 质量提升
    QUALITY_DECREASE = "quality_decrease"    # 质量下降
    HIGH_LOSS = "high_loss"                  # 高丢包
    HIGH_RTT = "high_rtt"                    # 高延迟
    LOW_CONFIDENCE = "low_confidence"        # 低置信度
    NETWORK_SPIKE = "network_spike"          # 网络波动
    STALL_DETECTED = "stall_detected"        # 卡顿检测


@dataclass
class Event:
    """事件数据类"""
    event_type: EventType
    timestamp: float
    description: str
    severity: str = "INFO"  # INFO, WARN, ERROR
    data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'type': self.event_type.value,
            'timestamp': self.timestamp,
            'description': self.description,
            'severity': self.severity,
            'data': self.data
        }
    
    def format_for_display(self) -> str:
        """格式化为显示字符串"""
        severity_prefix = {
            'INFO': '[OK]',
            'WARN': '[WARN]',
            'ERROR': '[ERROR]'
        }
        prefix = severity_prefix.get(self.severity, '[INFO]')
        return f"{prefix} {self.description}"


class EventDetector:
    """事件检测器
    
    检测视频质量变化和网络异常事件。
    
    支持的事件类型：
    - 质量提升/下降（分辨率变化）
    - 高丢包（丢包率超过阈值）
    - 高延迟（RTT超过阈值）
    - 低置信度（预测置信度低）
    - 网络波动（指标突变）
    - 卡顿检测（长时间无数据）
    
    Attributes:
        thresholds: 事件检测阈值配置
        history: 历史记录（用于趋势分析）
        last_prediction: 上一次预测结果
        last_metrics: 上一次网络指标
        last_update_time: 上一次更新时间
        events_history: 事件历史记录
    
    Example:
        >>> detector = EventDetector()
        >>> events = detector.detect(prediction, current_time)
        >>> for event in events:
        ...     print(event.format_for_display())
    """
    
    def __init__(self, 
                 loss_threshold: float = 3.0,
                 rtt_threshold: float = 200.0,
                 confidence_threshold: float = 0.6,
                 spike_multiplier: float = 3.0,
                 stall_timeout: float = 5.0):
        """初始化事件检测器
        
        Args:
            loss_threshold: 高丢包阈值（百分比）
            rtt_threshold: 高RTT阈值（毫秒）
            confidence_threshold: 低置信度阈值
            spike_multiplier: 网络波动倍数
            stall_timeout: 卡顿超时（秒）
        """
        self.thresholds = {
            'loss': loss_threshold,
            'rtt': rtt_threshold,
            'confidence': confidence_threshold,
            'spike_multiplier': spike_multiplier,
            'stall_timeout': stall_timeout
        }
        
        # 历史状态
        self.last_prediction: Optional[Prediction] = None
        self.last_metrics: Dict = {}
        self.last_update_time: float = 0
        
        # 事件历史
        self.events_history: List[Event] = []
        
        # 分辨率排序（用于质量比较）
        self._resolution_rank = {
            '480p': 1,
            '720p': 2,
            '1080p': 3,
            '1440p': 4,
            '2160p': 5,
            'unknown': 0
        }
    
    def detect(self, prediction: Prediction, current_time: Optional[float] = None) -> List[Event]:
        """检测事件
        
        Args:
            prediction: 当前预测结果
            current_time: 当前时间戳（如果None则使用当前时间）
            
        Returns:
            检测到的事件列表
        """
        if current_time is None:
            current_time = time.time()
        
        events = []
        
        # 提取当前指标
        current_metrics = prediction.metrics or {}
        
        # 1. 质量变化检测
        quality_events = self._detect_quality_change(prediction, current_time)
        events.extend(quality_events)
        
        # 2. 高丢包检测
        loss_event = self._detect_high_loss(current_metrics, current_time)
        if loss_event:
            events.append(loss_event)
        
        # 3. 高延迟检测
        rtt_event = self._detect_high_rtt(current_metrics, current_time)
        if rtt_event:
            events.append(rtt_event)
        
        # 4. 低置信度检测
        conf_event = self._detect_low_confidence(prediction, current_time)
        if conf_event:
            events.append(conf_event)
        
        # 5. 网络波动检测
        spike_events = self._detect_network_spike(current_metrics, current_time)
        events.extend(spike_events)
        
        # 6. 卡顿检测
        stall_event = self._detect_stall(current_time)
        if stall_event:
            events.append(stall_event)
        
        # 更新历史状态
        self.last_prediction = prediction
        self.last_metrics = current_metrics
        self.last_update_time = current_time
        
        # 保存事件历史
        self.events_history.extend(events)
        
        return events
    
    def _detect_quality_change(self, prediction: Prediction, timestamp: float) -> List[Event]:
        """检测质量变化"""
        events = []
        
        if self.last_prediction is None:
            return events
        
        curr_res = prediction.resolution
        prev_res = self.last_prediction.resolution
        
        if curr_res == prev_res:
            return events
        
        curr_rank = self._resolution_rank.get(curr_res, 0)
        prev_rank = self._resolution_rank.get(prev_res, 0)
        
        if curr_rank > prev_rank:
            # 质量提升
            events.append(Event(
                event_type=EventType.QUALITY_IMPROVE,
                timestamp=timestamp,
                description=f"Quality improved: {prev_res} -> {curr_res}",
                severity="INFO",
                data={
                    'from': prev_res,
                    'to': curr_res,
                    'confidence': prediction.confidence
                }
            ))
        elif curr_rank < prev_rank:
            # 质量下降
            events.append(Event(
                event_type=EventType.QUALITY_DECREASE,
                timestamp=timestamp,
                description=f"Quality decreased: {prev_res} -> {curr_res}",
                severity="WARN",
                data={
                    'from': prev_res,
                    'to': curr_res,
                    'confidence': prediction.confidence
                }
            ))
        
        return events
    
    def _detect_high_loss(self, metrics: Dict, timestamp: float) -> Optional[Event]:
        """检测高丢包"""
        loss_rate = metrics.get('loss_rate', 0.0)
        
        if loss_rate > self.thresholds['loss']:
            return Event(
                event_type=EventType.HIGH_LOSS,
                timestamp=timestamp,
                description=f"High packet loss: {loss_rate:.1f}%",
                severity="WARN",
                data={'loss_rate': loss_rate, 'threshold': self.thresholds['loss']}
            )
        
        return None
    
    def _detect_high_rtt(self, metrics: Dict, timestamp: float) -> Optional[Event]:
        """检测高延迟"""
        rtt = metrics.get('rtt', 0.0)
        
        if rtt > self.thresholds['rtt']:
            return Event(
                event_type=EventType.HIGH_RTT,
                timestamp=timestamp,
                description=f"High RTT: {rtt:.1f} ms",
                severity="WARN",
                data={'rtt': rtt, 'threshold': self.thresholds['rtt']}
            )
        
        return None
    
    def _detect_low_confidence(self, prediction: Prediction, timestamp: float) -> Optional[Event]:
        """检测低置信度"""
        if prediction.confidence < self.thresholds['confidence']:
            return Event(
                event_type=EventType.LOW_CONFIDENCE,
                timestamp=timestamp,
                description=f"Low confidence: {prediction.confidence:.1%}",
                severity="WARN",
                data={'confidence': prediction.confidence, 'resolution': prediction.resolution}
            )
        
        return None
    
    def _detect_network_spike(self, metrics: Dict, timestamp: float) -> List[Event]:
        """检测网络波动（指标突变）"""
        events = []
        
        if not self.last_metrics:
            return events
        
        # 丢包率突变
        curr_loss = metrics.get('loss_rate', 0.0)
        prev_loss = self.last_metrics.get('loss_rate', 0.0)
        
        if prev_loss > 0 and curr_loss > prev_loss * self.thresholds['spike_multiplier'] and curr_loss > 2.0:
            events.append(Event(
                event_type=EventType.NETWORK_SPIKE,
                timestamp=timestamp,
                description=f"Loss spike: {prev_loss:.1f}% -> {curr_loss:.1f}%",
                severity="WARN",
                data={'from': prev_loss, 'to': curr_loss, 'metric': 'loss'}
            ))
        
        # RTT突变
        curr_rtt = metrics.get('rtt', 0.0)
        prev_rtt = self.last_metrics.get('rtt', 0.0)
        
        if prev_rtt > 0 and curr_rtt > prev_rtt * self.thresholds['spike_multiplier'] and curr_rtt > 100:
            events.append(Event(
                event_type=EventType.NETWORK_SPIKE,
                timestamp=timestamp,
                description=f"RTT spike: {prev_rtt:.1f}ms -> {curr_rtt:.1f}ms",
                severity="WARN",
                data={'from': prev_rtt, 'to': curr_rtt, 'metric': 'rtt'}
            ))
        
        return events
    
    def _detect_stall(self, timestamp: float) -> Optional[Event]:
        """检测卡顿（长时间无更新）"""
        if self.last_update_time == 0:
            return None
        
        time_since_update = timestamp - self.last_update_time
        
        if time_since_update > self.thresholds['stall_timeout']:
            return Event(
                event_type=EventType.STALL_DETECTED,
                timestamp=timestamp,
                description=f"Stall detected: {time_since_update:.1f}s no update",
                severity="ERROR",
                data={'duration': time_since_update, 'threshold': self.thresholds['stall_timeout']}
            )
        
        return None
    
    def get_recent_events(self, count: int = 10) -> List[Event]:
        """获取最近的事件
        
        Args:
            count: 返回事件数量
            
        Returns:
            最近的事件列表
        """
        return self.events_history[-count:] if self.events_history else []
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """按类型获取事件
        
        Args:
            event_type: 事件类型
            
        Returns:
            该类型的所有事件
        """
        return [e for e in self.events_history if e.event_type == event_type]
    
    def get_summary(self) -> Dict:
        """获取事件统计摘要
        
        Returns:
            事件统计字典
        """
        summary = {
            'total_events': len(self.events_history),
            'by_type': {},
            'by_severity': {'INFO': 0, 'WARN': 0, 'ERROR': 0}
        }
        
        for event in self.events_history:
            # 按类型统计
            event_type_name = event.event_type.value
            summary['by_type'][event_type_name] = summary['by_type'].get(event_type_name, 0) + 1
            
            # 按严重程度统计
            summary['by_severity'][event.severity] += 1
        
        return summary
    
    def clear_history(self):
        """清除事件历史"""
        self.events_history.clear()
    
    def __len__(self) -> int:
        """返回事件历史总数"""
        return len(self.events_history)
    
    def __str__(self) -> str:
        summary = self.get_summary()
        return f"EventDetector (events: {summary['total_events']}, WARN: {summary['by_severity']['WARN']}, ERROR: {summary['by_severity']['ERROR']})"


if __name__ == '__main__':
    """测试模块"""
    import numpy as np
    
    print("=" * 70)
    print("Testing EventDetector")
    print("=" * 70)
    
    detector = EventDetector()
    
    # 测试1: 质量下降
    print("\n[Test 1] Quality Decrease")
    pred1 = Prediction(
        resolution='1080p',
        confidence=0.85,
        probabilities=np.array([0.05, 0.10, 0.85]),
        timestamp=time.time(),
        metrics={'throughput': 10.0, 'loss_rate': 0.5, 'rtt': 30.0}
    )
    events = detector.detect(pred1)
    print(f"  Events: {len(events)}")
    
    time.sleep(0.1)
    
    pred2 = Prediction(
        resolution='720p',
        confidence=0.75,
        probabilities=np.array([0.10, 0.75, 0.15]),
        timestamp=time.time(),
        metrics={'throughput': 4.0, 'loss_rate': 1.5, 'rtt': 80.0}
    )
    events = detector.detect(pred2)
    print(f"  Events: {len(events)}")
    for event in events:
        print(f"    {event.format_for_display()}")
    
    # 测试2: 高丢包
    print("\n[Test 2] High Loss")
    pred3 = Prediction(
        resolution='480p',
        confidence=0.70,
        probabilities=np.array([0.70, 0.20, 0.10]),
        timestamp=time.time(),
        metrics={'throughput': 1.5, 'loss_rate': 4.5, 'rtt': 150.0}
    )
    events = detector.detect(pred3)
    print(f"  Events: {len(events)}")
    for event in events:
        print(f"    {event.format_for_display()}")
    
    # 测试3: 网络波动
    print("\n[Test 3] Network Spike")
    time.sleep(0.1)
    pred4 = Prediction(
        resolution='480p',
        confidence=0.65,
        probabilities=np.array([0.65, 0.25, 0.10]),
        timestamp=time.time(),
        metrics={'throughput': 1.2, 'loss_rate': 15.0, 'rtt': 450.0}
    )
    events = detector.detect(pred4)
    print(f"  Events: {len(events)}")
    for event in events:
        print(f"    {event.format_for_display()}")
    
    # 测试4: 获取统计
    print("\n[Test 4] Summary")
    summary = detector.get_summary()
    print(f"  Total events: {summary['total_events']}")
    print(f"  By severity: {summary['by_severity']}")
    print(f"  By type: {summary['by_type']}")
    
    print("\n" + "=" * 70)
    print("Basic tests completed!")
    print("=" * 70)



