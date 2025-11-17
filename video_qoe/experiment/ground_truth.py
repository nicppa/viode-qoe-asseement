"""
Ground Truth记录模块

提供实验Ground Truth数据的记录、管理和序列化功能。
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import logging

from video_qoe.experiment.network_config import NetworkConfig


@dataclass
class VideoInfo:
    """视频信息"""
    file: str = ""
    actual_resolution: str = ""
    actual_bitrate_kbps: float = 0.0
    duration_sec: float = 0.0
    codec: str = ""
    framerate: float = 0.0


@dataclass
class NetworkInfo:
    """网络信息"""
    configured: Dict[str, Any] = field(default_factory=dict)
    measured: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentEvent:
    """实验事件"""
    time: float
    event: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'time': self.time,
            'event': self.event
        }
        if self.details:
            result['details'] = self.details
        return result


class GroundTruth:
    """Ground Truth记录器
    
    记录实验的Ground Truth数据，包括：
    - 实验配置（网络、视频）
    - 实验事件时间线
    - 实验元数据
    
    用于后续验证模型预测准确性。
    """
    
    def __init__(self, experiment_id: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """初始化Ground Truth记录器
        
        Args:
            experiment_id: 实验ID，默认自动生成
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # 实验基本信息
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.timestamp = datetime.now().isoformat()
        self.scenario_name: Optional[str] = None
        
        # 网络和视频配置
        self.video = VideoInfo()
        self.network = NetworkInfo()
        
        # 事件时间线
        self.events: List[ExperimentEvent] = []
        
        # 额外元数据
        self.metadata: Dict[str, Any] = {}
        
        # 开始时间（用于相对时间计算）
        self.start_time: Optional[float] = None
        
        self.logger.info(f"GroundTruth initialized: {self.experiment_id}")
    
    def _generate_experiment_id(self) -> str:
        """生成实验ID
        
        格式: exp_YYYYMMDD_HHMMSS
        """
        now = datetime.now()
        return now.strftime("exp_%Y%m%d_%H%M%S")
    
    def set_scenario(self, scenario_name: str):
        """设置实验场景名称
        
        Args:
            scenario_name: 场景名称
        """
        self.scenario_name = scenario_name
        self.logger.debug(f"Scenario set: {scenario_name}")
    
    def record_network_config(self, config: NetworkConfig, measured: Optional[Dict[str, Any]] = None):
        """记录网络配置
        
        Args:
            config: 配置的网络条件
            measured: 实测的网络条件（可选）
        """
        self.network.configured = {
            'bandwidth': config.bandwidth,
            'delay': config.delay,
            'loss': config.loss,
            'jitter': config.jitter,
            'bandwidth_mbps': config._bandwidth_mbps,
            'delay_ms': config._delay_ms,
            'loss_percent': config._loss_percent,
            'jitter_ms': config._jitter_ms if config._jitter_ms is not None else 0.0,
        }
        
        if measured:
            self.network.measured = measured
        
        self.logger.debug(f"Network config recorded: {config}")
    
    def record_video_info(self, 
                         file: str, 
                         resolution: str = "", 
                         bitrate_kbps: float = 0.0,
                         duration_sec: float = 0.0,
                         codec: str = "",
                         framerate: float = 0.0):
        """记录视频信息
        
        Args:
            file: 视频文件路径
            resolution: 实际分辨率（如"1280x720"）
            bitrate_kbps: 实际比特率（kbps）
            duration_sec: 视频时长（秒）
            codec: 视频编码
            framerate: 帧率
        """
        self.video.file = file
        self.video.actual_resolution = resolution
        self.video.actual_bitrate_kbps = bitrate_kbps
        self.video.duration_sec = duration_sec
        self.video.codec = codec
        self.video.framerate = framerate
        
        self.logger.debug(f"Video info recorded: {file}")
    
    def add_event(self, event: str, details: Optional[Dict[str, Any]] = None):
        """添加事件到时间线
        
        Args:
            event: 事件名称
            details: 事件详细信息（可选）
        """
        if self.start_time is None:
            # 第一个事件，设置开始时间
            self.start_time = datetime.now().timestamp()
            relative_time = 0.0
        else:
            # 计算相对于实验开始的时间
            relative_time = datetime.now().timestamp() - self.start_time
        
        event_obj = ExperimentEvent(
            time=relative_time,
            event=event,
            details=details
        )
        self.events.append(event_obj)
        
        self.logger.debug(f"Event added: {event} at {relative_time:.2f}s")
    
    def set_metadata(self, key: str, value: Any):
        """设置元数据
        
        Args:
            key: 键
            value: 值
        """
        self.metadata[key] = value
        self.logger.debug(f"Metadata set: {key} = {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Ground Truth数据字典
        """
        data = {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'scenario': self.scenario_name,
            'video': asdict(self.video),
            'network': {
                'configured': self.network.configured,
                'measured': self.network.measured
            },
            'events': [event.to_dict() for event in self.events],
            'metadata': self.metadata
        }
        
        return data
    
    def save(self, output_path: Path) -> bool:
        """保存Ground Truth到JSON文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            是否成功保存
        """
        try:
            # 确保目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为字典
            data = self.to_dict()
            
            # 写入JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Ground Truth saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save Ground Truth: {e}")
            return False
    
    @classmethod
    def load(cls, input_path: Path, logger: Optional[logging.Logger] = None) -> 'GroundTruth':
        """从JSON文件加载Ground Truth
        
        Args:
            input_path: 输入文件路径
            logger: 日志记录器
            
        Returns:
            GroundTruth对象
        """
        logger = logger or logging.getLogger(__name__)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 创建对象
            gt = cls(experiment_id=data.get('experiment_id'), logger=logger)
            gt.timestamp = data.get('timestamp', '')
            gt.scenario_name = data.get('scenario')
            
            # 恢复视频信息
            video_data = data.get('video', {})
            gt.video = VideoInfo(**video_data)
            
            # 恢复网络信息
            network_data = data.get('network', {})
            gt.network.configured = network_data.get('configured', {})
            gt.network.measured = network_data.get('measured')
            
            # 恢复事件
            for event_data in data.get('events', []):
                event = ExperimentEvent(
                    time=event_data['time'],
                    event=event_data['event'],
                    details=event_data.get('details')
                )
                gt.events.append(event)
            
            # 恢复元数据
            gt.metadata = data.get('metadata', {})
            
            logger.info(f"Ground Truth loaded from: {input_path}")
            return gt
            
        except Exception as e:
            logger.error(f"Failed to load Ground Truth: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """获取Ground Truth摘要
        
        Returns:
            摘要字典
        """
        return {
            'experiment_id': self.experiment_id,
            'scenario': self.scenario_name,
            'timestamp': self.timestamp,
            'event_count': len(self.events),
            'duration': self.events[-1].time if self.events else 0.0,
            'network_configured': self.network.configured.get('bandwidth', 'N/A'),
            'video_file': self.video.file or 'N/A'
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"GroundTruth(exp_id={self.experiment_id}, scenario={self.scenario_name}, events={len(self.events)})"


if __name__ == '__main__':
    """测试模块"""
    import tempfile
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('ground_truth_test')
    
    print("=" * 70)
    print("Testing GroundTruth")
    print("=" * 70)
    
    # 测试1: 创建Ground Truth
    print("\n[Test 1] Create GroundTruth")
    gt = GroundTruth(logger=logger)
    print(f"Experiment ID: {gt.experiment_id}")
    print(f"Timestamp: {gt.timestamp}")
    
    # 测试2: 记录配置
    print("\n[Test 2] Record Configuration")
    from video_qoe.experiment.network_config import NetworkConfig
    
    gt.set_scenario('low-bandwidth')
    
    network_config = NetworkConfig(bandwidth="2Mbps", delay="50ms", loss="1%")
    gt.record_network_config(network_config)
    
    gt.record_video_info(
        file="test_video_720p.mp4",
        resolution="1280x720",
        bitrate_kbps=2500,
        duration_sec=180
    )
    
    print(f"Scenario: {gt.scenario_name}")
    print(f"Network: {gt.network.configured}")
    print(f"Video: {gt.video}")
    
    # 测试3: 记录事件
    print("\n[Test 3] Record Events")
    gt.add_event('experiment_start')
    gt.add_event('video_playback_start', {'url': 'http://10.0.0.1:8000/video.mp4'})
    gt.add_event('quality_change', {'from': '720p', 'to': '480p'})
    gt.add_event('experiment_end')
    
    print(f"Events recorded: {len(gt.events)}")
    for event in gt.events:
        print(f"  {event.time:.2f}s: {event.event}")
    
    # 测试4: 保存和加载
    print("\n[Test 4] Save and Load")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'ground_truth.json'
        
        # 保存
        success = gt.save(output_path)
        print(f"Save success: {success}")
        print(f"File size: {output_path.stat().st_size} bytes")
        
        # 加载
        gt_loaded = GroundTruth.load(output_path, logger)
        print(f"Loaded: {gt_loaded}")
        print(f"Events: {len(gt_loaded.events)}")
    
    # 测试5: 摘要
    print("\n[Test 5] Summary")
    summary = gt.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)

