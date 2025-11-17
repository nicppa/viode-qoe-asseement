"""
流量统计特征计算器

实现15个流量统计相关的网络特征。
"""

import numpy as np
from typing import List, Optional, Dict
import logging
from collections import Counter

from video_qoe.capture.packet_info import PacketInfo


class TrafficFeatureCalculator:
    """流量统计特征计算器
    
    从PacketInfo列表中计算15个流量统计特征：
    11-15. 吞吐量相关（平均/标准差/最小/最大/变异系数）
    16-19. 包大小相关（平均/标准差/大包比例/熵）
    20. 上下行比例
    21-25. 总量统计（字节/包数/时长/速率方差/流数）
    
    Attributes:
        logger: 日志记录器
        large_packet_threshold: 大包阈值（默认1200字节）
    
    Example:
        >>> calculator = TrafficFeatureCalculator()
        >>> features = calculator.compute_traffic_features(packet_list)
        >>> print(features)  # numpy array of 15 features
    """
    
    def __init__(self, large_packet_threshold: int = 1200, logger: Optional[logging.Logger] = None):
        """初始化流量特征计算器
        
        Args:
            large_packet_threshold: 大包阈值（字节），默认1200
            logger: 日志记录器
        """
        self.large_packet_threshold = large_packet_threshold
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_traffic_features(self, packets: List[PacketInfo], client_ip: Optional[str] = None) -> np.ndarray:
        """计算流量统计特征
        
        Args:
            packets: PacketInfo列表（来自滑动窗口）
            client_ip: 客户端IP地址（用于区分上下行），可选
            
        Returns:
            15个流量特征的numpy数组
            
        Example:
            >>> features = calculator.compute_traffic_features(window_packets, client_ip='10.0.0.2')
            >>> assert len(features) == 15
        """
        features = np.zeros(15, dtype=np.float64)
        
        if not packets:
            self.logger.warning("No packets provided for traffic feature calculation")
            return features
        
        try:
            # 11-15. 吞吐量统计
            throughput_stats = self._compute_throughput_stats(packets)
            features[0] = throughput_stats['mean']      # 11. 平均吞吐量
            features[1] = throughput_stats['std']       # 12. 吞吐量标准差
            features[2] = throughput_stats['min']       # 13. 最小吞吐量
            features[3] = throughput_stats['max']       # 14. 最大吞吐量
            features[4] = throughput_stats['cv']        # 15. 变异系数 (CV)
            
            # 16-19. 包大小统计
            packet_size_stats = self._compute_packet_size_stats(packets)
            features[5] = packet_size_stats['mean']         # 16. 平均包大小
            features[6] = packet_size_stats['std']          # 17. 包大小标准差
            features[7] = packet_size_stats['large_ratio']  # 18. 大包比例
            features[8] = packet_size_stats['entropy']      # 19. 包大小熵
            
            # 20. 上下行比例
            features[9] = self._compute_uplink_ratio(packets, client_ip)
            
            # 21-25. 总量统计
            aggregate_stats = self._compute_aggregate_stats(packets)
            features[10] = aggregate_stats['total_bytes']   # 21. 总字节数
            features[11] = aggregate_stats['total_packets'] # 22. 总包数
            features[12] = aggregate_stats['duration']      # 23. 时长
            features[13] = aggregate_stats['rate_var']      # 24. 速率方差
            features[14] = aggregate_stats['flow_count']    # 25. 流数
            
        except Exception as e:
            self.logger.error(f"Error computing traffic features: {e}", exc_info=True)
        
        return features
    
    def _compute_throughput_stats(self, packets: List[PacketInfo]) -> Dict[str, float]:
        """计算吞吐量统计
        
        将时间窗口分成多个子区间，计算每个区间的吞吐量。
        
        Args:
            packets: 数据包列表
            
        Returns:
            吞吐量统计字典（mean, std, min, max, cv）
        """
        stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'cv': 0.0}
        
        if len(packets) < 2:
            return stats
        
        # 计算总时长
        duration = packets[-1].timestamp - packets[0].timestamp
        if duration <= 0:
            return stats
        
        # 将时间窗口分成10个子区间
        num_bins = 10
        bin_size = duration / num_bins
        throughputs = []
        
        for i in range(num_bins):
            bin_start = packets[0].timestamp + i * bin_size
            bin_end = bin_start + bin_size
            
            # 计算这个区间内的字节数
            bin_bytes = sum(p.length for p in packets 
                          if p.length and bin_start <= p.timestamp < bin_end)
            
            # 转换为Mbps
            throughput_mbps = (bin_bytes * 8) / (bin_size * 1e6) if bin_size > 0 else 0
            throughputs.append(throughput_mbps)
        
        if throughputs:
            throughputs_array = np.array(throughputs)
            stats['mean'] = float(np.mean(throughputs_array))
            stats['std'] = float(np.std(throughputs_array))
            stats['min'] = float(np.min(throughputs_array))
            stats['max'] = float(np.max(throughputs_array))
            
            # 变异系数 (Coefficient of Variation)
            if stats['mean'] > 0:
                stats['cv'] = stats['std'] / stats['mean']
        
        return stats
    
    def _compute_packet_size_stats(self, packets: List[PacketInfo]) -> Dict[str, float]:
        """计算包大小统计
        
        Args:
            packets: 数据包列表
            
        Returns:
            包大小统计字典（mean, std, large_ratio, entropy）
        """
        stats = {'mean': 0.0, 'std': 0.0, 'large_ratio': 0.0, 'entropy': 0.0}
        
        sizes = [p.length for p in packets if p.length is not None and p.length > 0]
        
        if not sizes:
            return stats
        
        sizes_array = np.array(sizes, dtype=np.float64)
        
        # 平均值和标准差
        stats['mean'] = float(np.mean(sizes_array))
        stats['std'] = float(np.std(sizes_array))
        
        # 大包比例
        large_count = sum(1 for s in sizes if s >= self.large_packet_threshold)
        stats['large_ratio'] = large_count / len(sizes)
        
        # 包大小熵 (Shannon entropy)
        # 将包大小分成bins，计算分布的熵
        stats['entropy'] = self._compute_entropy(sizes)
        
        return stats
    
    def _compute_entropy(self, values: List[int]) -> float:
        """计算Shannon熵
        
        Args:
            values: 数值列表
            
        Returns:
            熵值
        """
        if not values:
            return 0.0
        
        # 计算频率分布
        counter = Counter(values)
        total = len(values)
        
        # 计算熵: H = -sum(p * log2(p))
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return float(entropy)
    
    def _compute_uplink_ratio(self, packets: List[PacketInfo], client_ip: Optional[str]) -> float:
        """计算上行流量比例
        
        Args:
            packets: 数据包列表
            client_ip: 客户端IP地址
            
        Returns:
            上行流量比例（0-1之间）
        """
        if not client_ip or not packets:
            return 0.5  # 默认值：假设上下行均衡
        
        uplink_bytes = 0
        total_bytes = 0
        
        for p in packets:
            if p.length and p.length > 0:
                total_bytes += p.length
                # 如果源IP是客户端，则是上行流量
                if p.src_ip == client_ip:
                    uplink_bytes += p.length
        
        if total_bytes == 0:
            return 0.5
        
        return uplink_bytes / total_bytes
    
    def _compute_aggregate_stats(self, packets: List[PacketInfo]) -> Dict[str, float]:
        """计算聚合统计量
        
        Args:
            packets: 数据包列表
            
        Returns:
            聚合统计字典（total_bytes, total_packets, duration, rate_var, flow_count）
        """
        stats = {
            'total_bytes': 0.0,
            'total_packets': 0.0,
            'duration': 0.0,
            'rate_var': 0.0,
            'flow_count': 0.0
        }
        
        if not packets:
            return stats
        
        # 总字节数
        stats['total_bytes'] = float(sum(p.length for p in packets if p.length))
        
        # 总包数
        stats['total_packets'] = float(len(packets))
        
        # 时长（秒）
        if len(packets) >= 2:
            stats['duration'] = packets[-1].timestamp - packets[0].timestamp
        
        # 速率方差：计算瞬时速率的方差
        if len(packets) >= 2:
            rates = []
            for i in range(1, len(packets)):
                time_diff = packets[i].timestamp - packets[i-1].timestamp
                if time_diff > 0 and packets[i].length:
                    # 瞬时速率 (Mbps)
                    rate = (packets[i].length * 8) / (time_diff * 1e6)
                    rates.append(rate)
            
            if rates:
                stats['rate_var'] = float(np.var(rates))
        
        # 流数：根据五元组计数
        flow_keys = set()
        for p in packets:
            flow_key = p.get_flow_key()
            if flow_key:
                flow_keys.add(flow_key)
        
        stats['flow_count'] = float(len(flow_keys))
        
        return stats
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表
        
        Returns:
            15个流量特征的名称
        """
        return [
            'traffic_avg_throughput',      # 11. 平均吞吐量 (Mbps)
            'traffic_throughput_std',      # 12. 吞吐量标准差
            'traffic_min_throughput',      # 13. 最小吞吐量
            'traffic_max_throughput',      # 14. 最大吞吐量
            'traffic_throughput_cv',       # 15. 吞吐量变异系数
            'traffic_avg_packet_size',     # 16. 平均包大小 (bytes)
            'traffic_packet_size_std',     # 17. 包大小标准差
            'traffic_large_packet_ratio',  # 18. 大包比例
            'traffic_packet_size_entropy', # 19. 包大小熵
            'traffic_uplink_ratio',        # 20. 上行流量比例
            'traffic_total_bytes',         # 21. 总字节数
            'traffic_total_packets',       # 22. 总包数
            'traffic_duration',            # 23. 时长 (秒)
            'traffic_rate_variance',       # 24. 速率方差
            'traffic_flow_count',          # 25. 流数
        ]


if __name__ == '__main__':
    """测试模块"""
    import time
    
    print("=" * 70)
    print("Testing TrafficFeatureCalculator")
    print("=" * 70)
    
    # 创建测试数据
    print("\n[Test 1] Create test packets")
    base_time = time.time()
    test_packets = []
    
    for i in range(50):
        pkt = PacketInfo(
            timestamp=base_time + i * 0.02,  # 每20ms一个包
            number=i,
            src_ip='10.0.0.1' if i % 2 == 0 else '10.0.0.2',
            dst_ip='10.0.0.2' if i % 2 == 0 else '10.0.0.1',
            src_port=8000,
            dst_port=54321,
            ip_proto=6,  # TCP
            length=1500 if i % 3 == 0 else 500,  # 混合包大小
        )
        test_packets.append(pkt)
    
    print(f"  Created {len(test_packets)} test packets")
    
    # 测试特征计算
    print("\n[Test 2] Compute traffic features")
    calculator = TrafficFeatureCalculator()
    features = calculator.compute_traffic_features(test_packets, client_ip='10.0.0.2')
    
    print(f"  Features shape: {features.shape}")
    
    # 显示特征名称
    print("\n[Test 3] Feature names and values")
    names = calculator.get_feature_names()
    for i, name in enumerate(names):
        print(f"  [{i+11:2d}] {name:30s}: {features[i]:.6f}")
    
    print("\n" + "=" * 70)
    print("Basic tests completed!")
    print("For full testing, run scripts/test_story_4_2.py")
    print("=" * 70)



