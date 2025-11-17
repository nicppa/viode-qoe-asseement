"""
网络配置模块
Network Configuration Module

定义网络条件配置的数据结构和验证逻辑。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from video_qoe.utils.helpers import parse_bandwidth, parse_delay, parse_percentage


@dataclass
class NetworkConfig:
    """网络配置类
    
    定义网络条件参数，包括带宽、延迟、丢包和抖动。
    
    Attributes:
        bandwidth: 带宽（字符串格式，如 "2Mbps", "100Kbps"）
        delay: 延迟（字符串格式，如 "50ms", "0.1s"）
        loss: 丢包率（字符串格式，如 "1%", 或浮点数 0-100）
        jitter: 抖动（可选，字符串格式，如 "10ms"）
        
    Example:
        >>> config = NetworkConfig(
        ...     bandwidth="2Mbps",
        ...     delay="50ms",
        ...     loss="1%"
        ... )
        >>> config.validate()
        True
    """
    
    bandwidth: str = "10Mbps"  # 默认10Mbps
    delay: str = "0ms"          # 默认无延迟
    loss: str = "0%"            # 默认无丢包
    jitter: Optional[str] = None  # 默认无抖动
    
    # 解析后的数值（Mbps, ms, %）
    _bandwidth_mbps: float = field(init=False, repr=False, default=0.0)
    _delay_ms: float = field(init=False, repr=False, default=0.0)
    _loss_percent: float = field(init=False, repr=False, default=0.0)
    _jitter_ms: Optional[float] = field(init=False, repr=False, default=None)
    
    def __post_init__(self):
        """初始化后解析参数"""
        self._parse_parameters()
    
    def _parse_parameters(self):
        """解析字符串参数为数值"""
        try:
            self._bandwidth_mbps = parse_bandwidth(self.bandwidth)
            self._delay_ms = parse_delay(self.delay)
            self._loss_percent = parse_percentage(self.loss)
            
            if self.jitter:
                self._jitter_ms = parse_delay(self.jitter)
        except Exception as e:
            raise ValueError(f"Failed to parse network parameters: {e}")
    
    def validate(self) -> bool:
        """验证参数有效性
        
        Returns:
            是否有效
            
        Raises:
            ValueError: 参数无效时抛出异常
        """
        errors = []
        
        # 验证带宽
        if self._bandwidth_mbps <= 0:
            errors.append(f"Bandwidth must be > 0, got {self.bandwidth}")
        if self._bandwidth_mbps > 10000:  # 10Gbps上限
            errors.append(f"Bandwidth too high (>10Gbps): {self.bandwidth}")
        
        # 验证延迟
        if self._delay_ms < 0:
            errors.append(f"Delay must be >= 0, got {self.delay}")
        if self._delay_ms > 10000:  # 10秒上限
            errors.append(f"Delay too high (>10s): {self.delay}")
        
        # 验证丢包率
        if self._loss_percent < 0 or self._loss_percent > 100:
            errors.append(f"Loss must be 0-100%, got {self.loss}")
        
        # 验证抖动
        if self._jitter_ms is not None:
            if self._jitter_ms < 0:
                errors.append(f"Jitter must be >= 0, got {self.jitter}")
            if self._jitter_ms > 1000:  # 1秒上限
                errors.append(f"Jitter too high (>1s): {self.jitter}")
        
        if errors:
            raise ValueError("Network configuration validation failed:\n" + "\n".join(errors))
        
        return True
    
    def get_tc_params(self) -> Dict[str, Any]:
        """获取tc命令参数
        
        Returns:
            适用于TCLink.config()的参数字典
        """
        params = {
            'bw': self._bandwidth_mbps,  # Mbps
            'delay': self.delay,          # 字符串格式
            'loss': self._loss_percent,   # 百分比
        }
        
        if self.jitter:
            params['jitter'] = self.jitter
        
        return params
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            包含所有参数的字典
        """
        return {
            'bandwidth': self.bandwidth,
            'bandwidth_mbps': self._bandwidth_mbps,
            'delay': self.delay,
            'delay_ms': self._delay_ms,
            'loss': self.loss,
            'loss_percent': self._loss_percent,
            'jitter': self.jitter,
            'jitter_ms': self._jitter_ms,
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        parts = [
            f"BW={self.bandwidth}",
            f"Delay={self.delay}",
            f"Loss={self.loss}",
        ]
        if self.jitter:
            parts.append(f"Jitter={self.jitter}")
        return f"NetworkConfig({', '.join(parts)})"


@dataclass
class MeasuredNetworkConditions:
    """实测网络条件
    
    记录实际测量的网络性能指标，用于验证配置是否生效。
    
    Attributes:
        actual_bandwidth_mbps: 实测带宽（Mbps）
        actual_rtt_ms: 实测RTT（毫秒）
        actual_loss_percent: 实测丢包率（百分比）
        total_packets: 总包数
        lost_packets: 丢失包数
        measurement_method: 测量方法（'iperf', 'ping'等）
    """
    
    actual_bandwidth_mbps: Optional[float] = None
    actual_rtt_ms: Optional[float] = None
    actual_loss_percent: Optional[float] = None
    total_packets: int = 0
    lost_packets: int = 0
    measurement_method: str = "unknown"
    measurement_time: Optional[str] = None
    
    def compare_with_config(self, config: NetworkConfig,
                            tolerance_bandwidth: float = 10.0,
                            tolerance_delay: float = 5.0,
                            tolerance_loss: float = 1.5) -> Dict[str, Any]:
        """与配置的网络条件对比
        
        Args:
            config: 配置的网络条件
            tolerance_bandwidth: 带宽容差（百分比）
            tolerance_delay: 延迟容差（毫秒）
            tolerance_loss: 丢包容差（百分比）
            
        Returns:
            对比结果字典，包含误差信息
        """
        results = {
            'bandwidth': {},
            'delay': {},
            'loss': {},
        }
        
        # 带宽对比
        if self.actual_bandwidth_mbps is not None:
            expected = config._bandwidth_mbps
            actual = self.actual_bandwidth_mbps
            error_percent = abs(actual - expected) / expected * 100 if expected > 0 else 0
            
            results['bandwidth'] = {
                'expected': expected,
                'actual': actual,
                'error_percent': error_percent,
                'within_tolerance': error_percent <= tolerance_bandwidth,
            }
        
        # 延迟对比（RTT约为delay*2）
        if self.actual_rtt_ms is not None:
            expected_rtt = config._delay_ms * 2  # 往返时延
            actual = self.actual_rtt_ms
            error_ms = abs(actual - expected_rtt)
            
            results['delay'] = {
                'expected_rtt': expected_rtt,
                'actual_rtt': actual,
                'error_ms': error_ms,
                'within_tolerance': error_ms <= tolerance_delay,
            }
        
        # 丢包率对比
        if self.actual_loss_percent is not None:
            expected = config._loss_percent
            actual = self.actual_loss_percent
            error_percent = abs(actual - expected)
            
            results['loss'] = {
                'expected': expected,
                'actual': actual,
                'error_percent': error_percent,
                'within_tolerance': error_percent <= tolerance_loss,
            }
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'actual_bandwidth_mbps': self.actual_bandwidth_mbps,
            'actual_rtt_ms': self.actual_rtt_ms,
            'actual_loss_percent': self.actual_loss_percent,
            'total_packets': self.total_packets,
            'lost_packets': self.lost_packets,
            'measurement_method': self.measurement_method,
            'measurement_time': self.measurement_time,
        }


def create_network_config_from_dict(data: Dict[str, Any]) -> NetworkConfig:
    """从字典创建NetworkConfig
    
    Args:
        data: 包含网络配置的字典
        
    Returns:
        NetworkConfig实例
    """
    return NetworkConfig(
        bandwidth=data.get('bandwidth', '10Mbps'),
        delay=data.get('delay', '0ms'),
        loss=data.get('loss', '0%'),
        jitter=data.get('jitter'),
    )


if __name__ == '__main__':
    """测试模块"""
    print("=" * 70)
    print("Testing NetworkConfig")
    print("=" * 70)
    
    # 测试1: 基本配置
    print("\n[Test 1] Basic configuration")
    config = NetworkConfig(
        bandwidth="2Mbps",
        delay="50ms",
        loss="1%"
    )
    print(f"Config: {config}")
    print(f"Valid: {config.validate()}")
    print(f"TC params: {config.get_tc_params()}")
    print(f"Dict: {config.to_dict()}")
    
    # 测试2: 带抖动
    print("\n[Test 2] Configuration with jitter")
    config2 = NetworkConfig(
        bandwidth="5Mbps",
        delay="100ms",
        loss="2%",
        jitter="10ms"
    )
    print(f"Config: {config2}")
    print(f"Valid: {config2.validate()}")
    
    # 测试3: 无效配置
    print("\n[Test 3] Invalid configuration")
    try:
        bad_config = NetworkConfig(
            bandwidth="-1Mbps",
            delay="50ms",
            loss="150%"  # 无效：超过100%
        )
        bad_config.validate()
    except ValueError as e:
        print(f"✓ Caught validation error: {e}")
    
    # 测试4: 实测对比
    print("\n[Test 4] Measured vs Configured")
    config3 = NetworkConfig(bandwidth="10Mbps", delay="50ms", loss="1%")
    measured = MeasuredNetworkConditions(
        actual_bandwidth_mbps=9.8,
        actual_rtt_ms=102,  # RTT约为delay*2
        actual_loss_percent=1.2,
        measurement_method="iperf+ping"
    )
    comparison = measured.compare_with_config(config3)
    print(f"Comparison results:")
    for metric, data in comparison.items():
        if data:
            print(f"  {metric}: {data}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)

