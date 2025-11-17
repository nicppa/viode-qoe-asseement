"""
TCP层特征计算器

实现10个TCP相关的网络特征。
"""

import numpy as np
from typing import List, Optional, Dict
import logging

from video_qoe.capture.packet_info import PacketInfo


class TCPFeatureCalculator:
    """TCP特征计算器
    
    从PacketInfo列表中计算10个TCP层特征：
    1. retrans_rate - 重传率（基于序列号检测）
    2. avg_rtt - 平均RTT（基于SYN-ACK估算）
    3. rtt_std - RTT标准差
    4. max_rtt - 最大RTT
    5. avg_window - 平均TCP窗口大小
    6. window_var - TCP窗口方差
    7. slow_start_count - 慢启动检测次数
    8. congestion_events - 拥塞事件数（窗口突降）
    9. ack_delay - ACK延迟（平均）
    10. conn_setup_time - 连接建立时间
    
    Attributes:
        logger: 日志记录器
    
    Example:
        >>> calculator = TCPFeatureCalculator()
        >>> features = calculator.compute_tcp_features(packet_list)
        >>> print(features)  # numpy array of 10 features
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """初始化TCP特征计算器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_tcp_features(self, packets: List[PacketInfo]) -> np.ndarray:
        """计算TCP特征
        
        Args:
            packets: PacketInfo列表（来自滑动窗口）
            
        Returns:
            10个TCP特征的numpy数组
            
        Example:
            >>> features = calculator.compute_tcp_features(window_packets)
            >>> assert len(features) == 10
        """
        features = np.zeros(10, dtype=np.float64)
        
        if not packets:
            self.logger.warning("No packets provided for TCP feature calculation")
            return features
        
        # 过滤出TCP包
        tcp_packets = [p for p in packets if p.is_tcp()]
        
        if not tcp_packets:
            self.logger.debug(f"No TCP packets in {len(packets)} packets")
            return features
        
        try:
            # 1. 重传率（基于序列号）
            features[0] = self._compute_retrans_rate(tcp_packets)
            
            # 2-4. RTT统计（估算）
            rtt_stats = self._compute_rtt_stats(tcp_packets)
            features[1] = rtt_stats['mean']
            features[2] = rtt_stats['std']
            features[3] = rtt_stats['max']
            
            # 5-6. TCP窗口统计
            window_stats = self._compute_window_stats(tcp_packets)
            features[4] = window_stats['mean']
            features[5] = window_stats['var']
            
            # 7. 慢启动检测
            features[6] = self._detect_slow_start(tcp_packets)
            
            # 8. 拥塞事件检测
            features[7] = self._detect_congestion_events(tcp_packets)
            
            # 9. ACK延迟
            features[8] = self._compute_ack_delay(tcp_packets)
            
            # 10. 连接建立时间
            features[9] = self._get_conn_setup_time(tcp_packets)
            
        except Exception as e:
            self.logger.error(f"Error computing TCP features: {e}", exc_info=True)
        
        return features
    
    def _compute_retrans_rate(self, tcp_packets: List[PacketInfo]) -> float:
        """计算重传率
        
        检测重复的序列号，认为是重传。
        
        Args:
            tcp_packets: TCP包列表
            
        Returns:
            重传率（0-1之间）
        """
        if len(tcp_packets) < 2:
            return 0.0
        
        # 按流分组（简化：只看序列号）
        seq_numbers = []
        for pkt in tcp_packets:
            if pkt.tcp_seq is not None and pkt.tcp_seq > 0:
                seq_numbers.append(pkt.tcp_seq)
        
        if len(seq_numbers) < 2:
            return 0.0
        
        # 检测重复序列号
        seen = set()
        retrans_count = 0
        for seq in seq_numbers:
            if seq in seen:
                retrans_count += 1
            seen.add(seq)
        
        return retrans_count / len(seq_numbers)
    
    def _compute_rtt_stats(self, tcp_packets: List[PacketInfo]) -> Dict[str, float]:
        """计算RTT统计
        
        估算RTT：使用相邻ACK包的时间差作为近似。
        
        Args:
            tcp_packets: TCP包列表
            
        Returns:
            RTT统计字典（mean, std, max）
        """
        stats = {'mean': 0.0, 'std': 0.0, 'max': 0.0}
        
        # 收集带ACK的包
        ack_packets = [p for p in tcp_packets if p.tcp_ack is not None and p.tcp_ack > 0]
        
        if len(ack_packets) < 2:
            return stats
        
        # 估算RTT：相邻ACK包的时间差
        rtt_estimates = []
        for i in range(1, len(ack_packets)):
            time_diff = ack_packets[i].timestamp - ack_packets[i-1].timestamp
            if 0 < time_diff < 1.0:  # 合理范围（0-1秒）
                rtt_estimates.append(time_diff)
        
        if rtt_estimates:
            rtt_array = np.array(rtt_estimates)
            stats['mean'] = float(np.mean(rtt_array))
            stats['std'] = float(np.std(rtt_array))
            stats['max'] = float(np.max(rtt_array))
        
        return stats
    
    def _compute_window_stats(self, tcp_packets: List[PacketInfo]) -> Dict[str, float]:
        """计算TCP窗口统计
        
        Args:
            tcp_packets: TCP包列表
            
        Returns:
            窗口统计字典（mean, var）
        """
        stats = {'mean': 0.0, 'var': 0.0}
        
        windows = [p.tcp_window for p in tcp_packets if p.tcp_window is not None and p.tcp_window > 0]
        
        if windows:
            window_array = np.array(windows, dtype=np.float64)
            stats['mean'] = float(np.mean(window_array))
            stats['var'] = float(np.var(window_array))
        
        return stats
    
    def _detect_slow_start(self, tcp_packets: List[PacketInfo]) -> float:
        """检测慢启动次数
        
        检测窗口快速增长的阶段（窗口翻倍）。
        
        Args:
            tcp_packets: TCP包列表
            
        Returns:
            慢启动次数
        """
        windows = [p.tcp_window for p in tcp_packets if p.tcp_window is not None and p.tcp_window > 0]
        
        if len(windows) < 3:
            return 0.0
        
        slow_start_count = 0
        for i in range(2, len(windows)):
            # 检测窗口快速增长（增长 > 50%）
            if windows[i-1] > 0:
                growth_rate = (windows[i] - windows[i-1]) / windows[i-1]
                if growth_rate > 0.5:
                    slow_start_count += 1
        
        return float(slow_start_count)
    
    def _detect_congestion_events(self, tcp_packets: List[PacketInfo]) -> float:
        """检测拥塞事件
        
        检测窗口突然减小的事件（拥塞指示）。
        
        Args:
            tcp_packets: TCP包列表
            
        Returns:
            拥塞事件数
        """
        windows = [p.tcp_window for p in tcp_packets if p.tcp_window is not None and p.tcp_window > 0]
        
        if len(windows) < 2:
            return 0.0
        
        congestion_count = 0
        for i in range(1, len(windows)):
            # 检测窗口突降（减少 > 30%）
            if windows[i-1] > 0:
                decrease_rate = (windows[i-1] - windows[i]) / windows[i-1]
                if decrease_rate > 0.3:
                    congestion_count += 1
        
        return float(congestion_count)
    
    def _compute_ack_delay(self, tcp_packets: List[PacketInfo]) -> float:
        """计算ACK延迟
        
        计算数据包和对应ACK之间的平均时间差。
        简化版本：计算相邻ACK的平均间隔。
        
        Args:
            tcp_packets: TCP包列表
            
        Returns:
            平均ACK延迟（秒）
        """
        ack_packets = [p for p in tcp_packets if p.tcp_ack is not None and p.tcp_ack > 0]
        
        if len(ack_packets) < 2:
            return 0.0
        
        delays = []
        for i in range(1, len(ack_packets)):
            delay = ack_packets[i].timestamp - ack_packets[i-1].timestamp
            if 0 < delay < 0.5:  # 合理范围
                delays.append(delay)
        
        return float(np.mean(delays)) if delays else 0.0
    
    def _get_conn_setup_time(self, tcp_packets: List[PacketInfo]) -> float:
        """获取连接建立时间
        
        检测SYN, SYN-ACK, ACK三次握手的时间。
        简化版本：检测第一个SYN到第一个确认ACK的时间。
        
        Args:
            tcp_packets: TCP包列表
            
        Returns:
            连接建立时间（秒）
        """
        if len(tcp_packets) < 3:
            return 0.0
        
        # 查找SYN包（tcp_flags & 0x02 == SYN）
        syn_packets = [p for p in tcp_packets if p.tcp_flags is not None and (p.tcp_flags & 0x02)]
        
        # 查找ACK包（tcp_flags & 0x10 == ACK）
        ack_packets = [p for p in tcp_packets if p.tcp_flags is not None and (p.tcp_flags & 0x10)]
        
        if syn_packets and ack_packets:
            # 第一个SYN到第一个ACK的时间
            conn_time = ack_packets[0].timestamp - syn_packets[0].timestamp
            if 0 < conn_time < 5.0:  # 合理范围（0-5秒）
                return float(conn_time)
        
        return 0.0
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表
        
        Returns:
            10个TCP特征的名称
        """
        return [
            'tcp_retrans_rate',      # 1. 重传率
            'tcp_avg_rtt',           # 2. 平均RTT
            'tcp_rtt_std',           # 3. RTT标准差
            'tcp_max_rtt',           # 4. 最大RTT
            'tcp_avg_window',        # 5. 平均窗口
            'tcp_window_var',        # 6. 窗口方差
            'tcp_slow_start_count',  # 7. 慢启动次数
            'tcp_congestion_events', # 8. 拥塞事件数
            'tcp_ack_delay',         # 9. ACK延迟
            'tcp_conn_setup_time',   # 10. 连接建立时间
        ]


if __name__ == '__main__':
    """测试模块"""
    import time
    
    print("=" * 70)
    print("Testing TCPFeatureCalculator")
    print("=" * 70)
    
    # 创建测试数据
    print("\n[Test 1] Create test TCP packets")
    base_time = time.time()
    test_packets = []
    
    for i in range(20):
        pkt = PacketInfo(
            timestamp=base_time + i * 0.05,
            number=i,
            src_ip='10.0.0.1',
            dst_ip='10.0.0.2',
            src_port=8000,
            dst_port=54321,
            ip_proto=6,  # TCP
            tcp_seq=1000 + i * 100,
            tcp_ack=2000 + i * 100,
            tcp_window=65535 - i * 1000,  # 窗口递减
            tcp_flags=0x018,  # PSH+ACK
        )
        test_packets.append(pkt)
    
    print(f"  Created {len(test_packets)} test packets")
    
    # 测试特征计算
    print("\n[Test 2] Compute TCP features")
    calculator = TCPFeatureCalculator()
    features = calculator.compute_tcp_features(test_packets)
    
    print(f"  Features shape: {features.shape}")
    print(f"  Features: {features}")
    
    # 显示特征名称
    print("\n[Test 3] Feature names")
    names = calculator.get_feature_names()
    for i, name in enumerate(names):
        print(f"  [{i+1}] {name}: {features[i]:.6f}")
    
    print("\n" + "=" * 70)
    print("Basic tests completed!")
    print("For full testing, run scripts/test_story_4_1.py")
    print("=" * 70)



