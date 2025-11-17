"""
时序特征计算器

实现10个时序相关的网络特征。
"""

import numpy as np
from typing import List, Optional, Dict
import logging
from scipy import signal, stats

from video_qoe.capture.packet_info import PacketInfo


class TemporalFeatureCalculator:
    """时序特征计算器
    
    从PacketInfo列表中计算10个时序特征：
    26-28. 包间隔统计（均值/标准差/变异系数）
    29. 周期性得分（FFT）
    30-31. 空窗期检测（数量/平均时长）
    32-33. 突发检测（次数/强度）
    34. 自相关系数
    35. 趋势斜率
    
    Attributes:
        logger: 日志记录器
        idle_threshold: 空窗期阈值（秒），默认0.1秒
        burst_threshold: 突发阈值（包数/秒），默认100
    
    Example:
        >>> calculator = TemporalFeatureCalculator()
        >>> features = calculator.compute_temporal_features(packet_list)
        >>> print(features)  # numpy array of 10 features
    """
    
    def __init__(self, 
                 idle_threshold: float = 0.1, 
                 burst_threshold: float = 100.0,
                 logger: Optional[logging.Logger] = None):
        """初始化时序特征计算器
        
        Args:
            idle_threshold: 空窗期阈值（秒），默认0.1秒
            burst_threshold: 突发阈值（包数/秒），默认100
            logger: 日志记录器
        """
        self.idle_threshold = idle_threshold
        self.burst_threshold = burst_threshold
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_temporal_features(self, packets: List[PacketInfo]) -> np.ndarray:
        """计算时序特征
        
        Args:
            packets: PacketInfo列表（来自滑动窗口）
            
        Returns:
            10个时序特征的numpy数组
            
        Example:
            >>> features = calculator.compute_temporal_features(window_packets)
            >>> assert len(features) == 10
        """
        features = np.zeros(10, dtype=np.float64)
        
        if not packets:
            self.logger.warning("No packets provided for temporal feature calculation")
            return features
        
        try:
            # 26-28. 包间隔统计
            iat_stats = self._compute_iat_stats(packets)
            features[0] = iat_stats['mean']   # 26. 平均包间隔
            features[1] = iat_stats['std']    # 27. 包间隔标准差
            features[2] = iat_stats['cv']     # 28. 包间隔变异系数
            
            # 29. 周期性得分（FFT）
            features[3] = self._compute_periodicity_score(packets)
            
            # 30-31. 空窗期检测
            idle_stats = self._detect_idle_periods(packets)
            features[4] = idle_stats['count']     # 30. 空窗期数量
            features[5] = idle_stats['avg_duration']  # 31. 平均空窗期时长
            
            # 32-33. 突发检测
            burst_stats = self._detect_bursts(packets)
            features[6] = burst_stats['count']    # 32. 突发次数
            features[7] = burst_stats['intensity']  # 33. 突发强度
            
            # 34. 自相关系数
            features[8] = self._compute_autocorrelation(packets)
            
            # 35. 趋势斜率
            features[9] = self._compute_trend_slope(packets)
            
        except Exception as e:
            self.logger.error(f"Error computing temporal features: {e}", exc_info=True)
        
        return features
    
    def _compute_iat_stats(self, packets: List[PacketInfo]) -> Dict[str, float]:
        """计算包间隔时间（Inter-Arrival Time）统计
        
        Args:
            packets: 数据包列表
            
        Returns:
            IAT统计字典（mean, std, cv）
        """
        stats_dict = {'mean': 0.0, 'std': 0.0, 'cv': 0.0}
        
        if len(packets) < 2:
            return stats_dict
        
        # 计算相邻包的时间间隔
        iats = []
        for i in range(1, len(packets)):
            iat = packets[i].timestamp - packets[i-1].timestamp
            if iat > 0:  # 确保时间递增
                iats.append(iat)
        
        if iats:
            iats_array = np.array(iats)
            stats_dict['mean'] = float(np.mean(iats_array))
            stats_dict['std'] = float(np.std(iats_array))
            
            # 变异系数
            if stats_dict['mean'] > 0:
                stats_dict['cv'] = stats_dict['std'] / stats_dict['mean']
        
        return stats_dict
    
    def _compute_periodicity_score(self, packets: List[PacketInfo]) -> float:
        """计算周期性得分（使用FFT）
        
        检测流量是否呈现周期性模式。
        
        Args:
            packets: 数据包列表
            
        Returns:
            周期性得分（0-1之间，越高越周期）
        """
        if len(packets) < 10:
            return 0.0
        
        try:
            # 计算包间隔
            iats = []
            for i in range(1, len(packets)):
                iat = packets[i].timestamp - packets[i-1].timestamp
                if iat > 0:
                    iats.append(iat)
            
            if len(iats) < 10:
                return 0.0
            
            # 应用FFT
            iats_array = np.array(iats)
            fft_result = np.fft.fft(iats_array)
            fft_magnitude = np.abs(fft_result)
            
            # 排除直流分量（第一个）
            fft_magnitude = fft_magnitude[1:len(fft_magnitude)//2]
            
            if len(fft_magnitude) == 0:
                return 0.0
            
            # 周期性得分：最大峰值与平均值的比率
            max_peak = np.max(fft_magnitude)
            mean_magnitude = np.mean(fft_magnitude)
            
            if mean_magnitude > 0:
                periodicity_score = (max_peak - mean_magnitude) / (max_peak + mean_magnitude)
                # 归一化到0-1
                periodicity_score = np.clip(periodicity_score, 0.0, 1.0)
                return float(periodicity_score)
            
        except Exception as e:
            self.logger.debug(f"FFT computation failed: {e}")
        
        return 0.0
    
    def _detect_idle_periods(self, packets: List[PacketInfo]) -> Dict[str, float]:
        """检测空窗期（idle periods）
        
        空窗期：两个连续包之间的间隔超过阈值。
        
        Args:
            packets: 数据包列表
            
        Returns:
            空窗期统计（count, avg_duration）
        """
        idle_stats = {'count': 0.0, 'avg_duration': 0.0}
        
        if len(packets) < 2:
            return idle_stats
        
        idle_durations = []
        for i in range(1, len(packets)):
            iat = packets[i].timestamp - packets[i-1].timestamp
            if iat > self.idle_threshold:
                idle_durations.append(iat)
        
        if idle_durations:
            idle_stats['count'] = float(len(idle_durations))
            idle_stats['avg_duration'] = float(np.mean(idle_durations))
        
        return idle_stats
    
    def _detect_bursts(self, packets: List[PacketInfo]) -> Dict[str, float]:
        """检测突发（bursts）
        
        突发：短时间内大量包到达（速率超过阈值）。
        
        Args:
            packets: 数据包列表
            
        Returns:
            突发统计（count, intensity）
        """
        burst_stats = {'count': 0.0, 'intensity': 0.0}
        
        if len(packets) < 10:
            return burst_stats
        
        # 使用滑动窗口检测突发（窗口大小=0.1秒）
        window_size = 0.1
        burst_count = 0
        max_burst_rate = 0.0
        
        i = 0
        while i < len(packets):
            window_start = packets[i].timestamp
            window_end = window_start + window_size
            
            # 计算窗口内的包数
            packets_in_window = 0
            j = i
            while j < len(packets) and packets[j].timestamp < window_end:
                packets_in_window += 1
                j += 1
            
            # 计算速率（包数/秒）
            rate = packets_in_window / window_size
            
            # 检测是否为突发
            if rate > self.burst_threshold:
                burst_count += 1
                max_burst_rate = max(max_burst_rate, rate)
            
            # 移动到下一个窗口
            i = max(i + 1, j)
        
        burst_stats['count'] = float(burst_count)
        burst_stats['intensity'] = float(max_burst_rate)
        
        return burst_stats
    
    def _compute_autocorrelation(self, packets: List[PacketInfo]) -> float:
        """计算自相关系数
        
        衡量时间序列与其滞后版本的相关性（lag=1）。
        
        Args:
            packets: 数据包列表
            
        Returns:
            自相关系数（-1到1之间）
        """
        if len(packets) < 3:
            return 0.0
        
        try:
            # 使用包间隔作为时间序列
            iats = []
            for i in range(1, len(packets)):
                iat = packets[i].timestamp - packets[i-1].timestamp
                if iat > 0:
                    iats.append(iat)
            
            if len(iats) < 3:
                return 0.0
            
            # 计算lag=1的自相关
            iats_array = np.array(iats)
            
            # 标准化
            mean = np.mean(iats_array)
            std = np.std(iats_array)
            
            if std == 0:
                return 0.0
            
            normalized = (iats_array - mean) / std
            
            # 计算自相关
            autocorr = np.correlate(normalized[:-1], normalized[1:], mode='valid')[0] / (len(normalized) - 1)
            
            return float(np.clip(autocorr, -1.0, 1.0))
            
        except Exception as e:
            self.logger.debug(f"Autocorrelation computation failed: {e}")
            return 0.0
    
    def _compute_trend_slope(self, packets: List[PacketInfo]) -> float:
        """计算趋势斜率
        
        使用线性回归计算流量速率的趋势（增长/下降）。
        
        Args:
            packets: 数据包列表
            
        Returns:
            趋势斜率（正值=增长，负值=下降）
        """
        if len(packets) < 10:
            return 0.0
        
        try:
            # 将时间窗口分成10个子区间，计算每个区间的包数
            duration = packets[-1].timestamp - packets[0].timestamp
            if duration <= 0:
                return 0.0
            
            num_bins = 10
            bin_size = duration / num_bins
            bin_counts = []
            
            for i in range(num_bins):
                bin_start = packets[0].timestamp + i * bin_size
                bin_end = bin_start + bin_size
                
                count = sum(1 for p in packets if bin_start <= p.timestamp < bin_end)
                bin_counts.append(count)
            
            # 线性回归
            x = np.arange(num_bins)
            y = np.array(bin_counts)
            
            # 使用最小二乘法计算斜率
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            return float(slope)
            
        except Exception as e:
            self.logger.debug(f"Trend slope computation failed: {e}")
            return 0.0
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表
        
        Returns:
            10个时序特征的名称
        """
        return [
            'temporal_avg_iat',           # 26. 平均包间隔 (秒)
            'temporal_iat_std',           # 27. 包间隔标准差
            'temporal_iat_cv',            # 28. 包间隔变异系数
            'temporal_periodicity',       # 29. 周期性得分 (0-1)
            'temporal_idle_count',        # 30. 空窗期数量
            'temporal_idle_avg_duration', # 31. 平均空窗期时长 (秒)
            'temporal_burst_count',       # 32. 突发次数
            'temporal_burst_intensity',   # 33. 突发强度 (包数/秒)
            'temporal_autocorr',          # 34. 自相关系数 (-1 to 1)
            'temporal_trend_slope',       # 35. 趋势斜率
        ]


if __name__ == '__main__':
    """测试模块"""
    import time
    
    print("=" * 70)
    print("Testing TemporalFeatureCalculator")
    print("=" * 70)
    
    # 创建测试数据
    print("\n[Test 1] Create test packets with pattern")
    base_time = time.time()
    test_packets = []
    
    # 创建周期性流量模式
    for i in range(50):
        # 周期性间隔：快-慢-快-慢
        if i % 10 < 5:
            interval = 0.01  # 快速
        else:
            interval = 0.05  # 慢速
        
        timestamp = base_time + sum([0.01 if j % 10 < 5 else 0.05 for j in range(i)])
        
        pkt = PacketInfo(
            timestamp=timestamp,
            number=i,
            src_ip='10.0.0.1',
            dst_ip='10.0.0.2',
            src_port=8000,
            dst_port=54321,
            ip_proto=6,
            length=1000,
        )
        test_packets.append(pkt)
    
    print(f"  Created {len(test_packets)} test packets")
    
    # 测试特征计算
    print("\n[Test 2] Compute temporal features")
    calculator = TemporalFeatureCalculator()
    features = calculator.compute_temporal_features(test_packets)
    
    print(f"  Features shape: {features.shape}")
    
    # 显示特征名称
    print("\n[Test 3] Feature names and values")
    names = calculator.get_feature_names()
    for i, name in enumerate(names):
        print(f"  [{i+26:2d}] {name:30s}: {features[i]:.6f}")
    
    print("\n" + "=" * 70)
    print("Basic tests completed!")
    print("For full testing, run scripts/test_story_4_3.py")
    print("=" * 70)



