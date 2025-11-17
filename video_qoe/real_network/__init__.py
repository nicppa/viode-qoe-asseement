"""
真实网络监测模块

支持在真实网络环境中进行流量捕获和视频质量监测。
"""

from video_qoe.real_network.capturer import RealNetworkCapturer
from video_qoe.real_network.detector import VideoTrafficDetector

__all__ = [
    'RealNetworkCapturer',
    'VideoTrafficDetector',
]

__version__ = '0.1.0'  # Epic 8: Real Video Stream Monitoring - Story 8.1



