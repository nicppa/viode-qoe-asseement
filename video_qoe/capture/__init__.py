"""
流量捕获模块

提供流量捕获和预处理功能。
"""

from .capturer import PacketCapturer
from .reader import PCAPReader, check_pyshark_installed, get_pyshark_version
from .packet_info import PacketInfo
from .sliding_window import SlidingWindowBuffer

__all__ = [
    'PacketCapturer',
    'PCAPReader',
    'PacketInfo',
    'SlidingWindowBuffer',
    'check_pyshark_installed',
    'get_pyshark_version',
]

__version__ = '0.4.0'  # Story 3.4: 滑动窗口缓冲实现

