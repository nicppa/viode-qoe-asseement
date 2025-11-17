"""
特征工程模块

实现35个TCP/IP网络特征的计算。
"""

from video_qoe.features.tcp_features import TCPFeatureCalculator
from video_qoe.features.traffic_features import TrafficFeatureCalculator
from video_qoe.features.temporal_features import TemporalFeatureCalculator
from video_qoe.features.feature_calculator import FeatureCalculator

__all__ = [
    'TCPFeatureCalculator',
    'TrafficFeatureCalculator',
    'TemporalFeatureCalculator',
    'FeatureCalculator',  # 推荐使用的统一接口
]

__version__ = '0.4.0'  # Epic 4: Feature Engineering Core - Story 4.1, 4.2, 4.3, 4.4

