"""
特征计算器聚合器

集成TCP、流量、时序三个特征计算器，提供统一的特征计算接口。
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import logging

from video_qoe.capture.packet_info import PacketInfo
from video_qoe.features.tcp_features import TCPFeatureCalculator
from video_qoe.features.traffic_features import TrafficFeatureCalculator
from video_qoe.features.temporal_features import TemporalFeatureCalculator


class FeatureCalculator:
    """特征计算器聚合器
    
    集成三个特征计算器，提供统一接口计算所有35个特征。
    
    Attributes:
        tcp_calculator: TCP特征计算器
        traffic_calculator: 流量特征计算器
        temporal_calculator: 时序特征计算器
        logger: 日志记录器
    
    Example:
        >>> calculator = FeatureCalculator()
        >>> features, names = calculator.compute_all_features(packets, client_ip='10.0.0.2')
        >>> print(f"Total features: {len(features)}")  # 35
    """
    
    def __init__(self, 
                 large_packet_threshold: int = 1200,
                 idle_threshold: float = 0.1,
                 burst_threshold: float = 100.0,
                 logger: Optional[logging.Logger] = None):
        """初始化特征计算器聚合器
        
        Args:
            large_packet_threshold: 大包阈值（字节），默认1200
            idle_threshold: 空窗期阈值（秒），默认0.1
            burst_threshold: 突发阈值（包数/秒），默认100
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化三个子计算器
        self.tcp_calculator = TCPFeatureCalculator(logger=self.logger)
        self.traffic_calculator = TrafficFeatureCalculator(
            large_packet_threshold=large_packet_threshold,
            logger=self.logger
        )
        self.temporal_calculator = TemporalFeatureCalculator(
            idle_threshold=idle_threshold,
            burst_threshold=burst_threshold,
            logger=self.logger
        )
        
        self.logger.info("FeatureCalculator initialized with all 3 sub-calculators")
    
    def compute_all_features(self, 
                            packets: List[PacketInfo], 
                            client_ip: Optional[str] = None) -> np.ndarray:
        """计算所有35个特征
        
        Args:
            packets: PacketInfo列表（来自滑动窗口）
            client_ip: 客户端IP地址（用于区分上下行），可选
            
        Returns:
            35个特征的numpy数组
            
        Example:
            >>> features = calculator.compute_all_features(window_packets, client_ip='10.0.0.2')
            >>> assert len(features) == 35
        """
        if not packets:
            self.logger.warning("No packets provided for feature calculation")
            return np.zeros(35, dtype=np.float64)
        
        try:
            # 计算三组特征
            tcp_features = self.tcp_calculator.compute_tcp_features(packets)  # 10个
            traffic_features = self.traffic_calculator.compute_traffic_features(packets, client_ip)  # 15个
            temporal_features = self.temporal_calculator.compute_temporal_features(packets)  # 10个
            
            # 合并为35维特征向量
            all_features = np.concatenate([
                tcp_features,      # 1-10
                traffic_features,  # 11-25
                temporal_features  # 26-35
            ])
            
            self.logger.debug(f"Computed {len(all_features)} features from {len(packets)} packets")
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"Error computing features: {e}", exc_info=True)
            return np.zeros(35, dtype=np.float64)
    
    def compute_features_dict(self, 
                             packets: List[PacketInfo], 
                             client_ip: Optional[str] = None) -> Dict[str, float]:
        """计算所有35个特征并返回字典
        
        Args:
            packets: PacketInfo列表
            client_ip: 客户端IP地址，可选
            
        Returns:
            特征名称到特征值的字典
            
        Example:
            >>> features_dict = calculator.compute_features_dict(packets, client_ip='10.0.0.2')
            >>> print(features_dict['tcp_retrans_rate'])
            >>> print(features_dict['traffic_avg_throughput'])
        """
        features = self.compute_all_features(packets, client_ip)
        names = self.get_feature_names()
        
        return dict(zip(names, features))
    
    def compute_features_with_names(self, 
                                    packets: List[PacketInfo], 
                                    client_ip: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """计算所有35个特征并返回特征值和特征名称
        
        Args:
            packets: PacketInfo列表
            client_ip: 客户端IP地址，可选
            
        Returns:
            (features, feature_names) 元组
            
        Example:
            >>> features, names = calculator.compute_features_with_names(packets)
            >>> for name, value in zip(names, features):
            ...     print(f"{name}: {value:.6f}")
        """
        features = self.compute_all_features(packets, client_ip)
        names = self.get_feature_names()
        
        return features, names
    
    def get_feature_names(self) -> List[str]:
        """获取所有35个特征的名称
        
        Returns:
            35个特征名称的列表
        """
        return (
            self.tcp_calculator.get_feature_names() +       # 1-10
            self.traffic_calculator.get_feature_names() +   # 11-25
            self.temporal_calculator.get_feature_names()    # 26-35
        )
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """获取特征分组
        
        Returns:
            特征分组字典
            
        Example:
            >>> groups = calculator.get_feature_groups()
            >>> print(groups['tcp'])  # TCP特征列表
            >>> print(groups['traffic'])  # 流量特征列表
            >>> print(groups['temporal'])  # 时序特征列表
        """
        return {
            'tcp': self.tcp_calculator.get_feature_names(),
            'traffic': self.traffic_calculator.get_feature_names(),
            'temporal': self.temporal_calculator.get_feature_names()
        }
    
    def get_feature_count(self) -> Dict[str, int]:
        """获取各组特征数量
        
        Returns:
            特征数量字典
            
        Example:
            >>> counts = calculator.get_feature_count()
            >>> print(counts)  # {'tcp': 10, 'traffic': 15, 'temporal': 10, 'total': 35}
        """
        return {
            'tcp': 10,
            'traffic': 15,
            'temporal': 10,
            'total': 35
        }
    
    def validate_features(self, features: np.ndarray) -> Dict[str, bool]:
        """验证特征向量的有效性
        
        检查特征向量是否包含NaN、Inf等无效值。
        
        Args:
            features: 特征向量
            
        Returns:
            验证结果字典
            
        Example:
            >>> features = calculator.compute_all_features(packets)
            >>> validation = calculator.validate_features(features)
            >>> if validation['is_valid']:
            ...     print("All features are valid!")
        """
        validation = {
            'is_valid': True,
            'has_nan': False,
            'has_inf': False,
            'shape_correct': False,
            'all_zeros': False
        }
        
        # 检查shape
        if features.shape == (35,):
            validation['shape_correct'] = True
        else:
            validation['is_valid'] = False
            self.logger.warning(f"Invalid feature shape: {features.shape}, expected (35,)")
        
        # 检查NaN
        if np.any(np.isnan(features)):
            validation['has_nan'] = True
            validation['is_valid'] = False
            self.logger.warning("Features contain NaN values")
        
        # 检查Inf
        if np.any(np.isinf(features)):
            validation['has_inf'] = True
            validation['is_valid'] = False
            self.logger.warning("Features contain Inf values")
        
        # 检查是否全为0（可能表示计算失败）
        if np.all(features == 0):
            validation['all_zeros'] = True
            self.logger.warning("All features are zero (possible computation failure)")
        
        return validation
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"FeatureCalculator(tcp={self.tcp_calculator.__class__.__name__}, "
                f"traffic={self.traffic_calculator.__class__.__name__}, "
                f"temporal={self.temporal_calculator.__class__.__name__})")
    
    def __repr__(self) -> str:
        return self.__str__()


if __name__ == '__main__':
    """测试模块"""
    import time
    
    print("=" * 70)
    print("Testing FeatureCalculator (Aggregator)")
    print("=" * 70)
    
    # 创建测试数据
    print("\n[Test 1] Create test packets")
    base_time = time.time()
    test_packets = []
    
    for i in range(50):
        pkt = PacketInfo(
            timestamp=base_time + i * 0.02,
            number=i,
            src_ip='10.0.0.1' if i % 2 == 0 else '10.0.0.2',
            dst_ip='10.0.0.2' if i % 2 == 0 else '10.0.0.1',
            src_port=8000,
            dst_port=54321,
            ip_proto=6,  # TCP
            tcp_seq=1000 + i * 100,
            tcp_ack=2000 + i * 100,
            tcp_window=65535 - i * 100,
            tcp_flags=0x018,
            length=1000,
        )
        test_packets.append(pkt)
    
    print(f"  Created {len(test_packets)} test packets")
    
    # 测试聚合器
    print("\n[Test 2] Initialize FeatureCalculator")
    calculator = FeatureCalculator()
    print(f"  Calculator: {calculator}")
    
    # 计算所有特征
    print("\n[Test 3] Compute all 35 features")
    features = calculator.compute_all_features(test_packets, client_ip='10.0.0.2')
    print(f"  Features shape: {features.shape}")
    print(f"  Features dtype: {features.dtype}")
    
    # 获取特征名称
    print("\n[Test 4] Get feature names")
    names = calculator.get_feature_names()
    print(f"  Total feature names: {len(names)}")
    
    # 显示部分特征
    print("\n[Test 5] Display sample features")
    for i in [0, 10, 25]:
        print(f"  [{i+1:2d}] {names[i]:30s}: {features[i]:.6f}")
    
    # 获取特征分组
    print("\n[Test 6] Get feature groups")
    groups = calculator.get_feature_groups()
    for group_name, group_features in groups.items():
        print(f"  {group_name}: {len(group_features)} features")
    
    # 验证特征
    print("\n[Test 7] Validate features")
    validation = calculator.validate_features(features)
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Basic tests completed!")
    print("For full testing, run scripts/test_story_4_4.py")
    print("=" * 70)



