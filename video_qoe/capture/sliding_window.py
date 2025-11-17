"""
滑动窗口缓冲实现

高效维护最近1秒的数据包，用于实时特征计算。
"""

from collections import deque
from typing import List, Optional
import logging

from .packet_info import PacketInfo


class SlidingWindowBuffer:
    """滑动窗口缓冲
    
    使用双端队列(deque)高效实现滑动窗口，维护最近window_size秒的数据包。
    
    特点:
    - O(1) 添加操作
    - O(k) 清理操作（k=过期包数量，通常很小）
    - 内存占用稳定（自动移除老数据）
    - 支持实时流量处理
    
    Attributes:
        window_size: 窗口大小（秒），默认1.0秒
        packets: 双端队列，存储PacketInfo对象
        min_packets: 窗口准备就绪的最小包数
        min_fill_ratio: 窗口填充率阈值（0.8 = 至少0.8秒数据）
    
    Example:
        >>> buffer = SlidingWindowBuffer(window_size=1.0)
        >>> for packet_info in packet_stream:
        ...     buffer.add_packet(packet_info)
        ...     if buffer.is_ready():
        ...         features = calculate_features(buffer.get_window_data())
    """
    
    def __init__(self, 
                 window_size: float = 1.0,
                 min_packets: int = 10,
                 min_fill_ratio: float = 0.8,
                 logger: Optional[logging.Logger] = None):
        """初始化滑动窗口
        
        Args:
            window_size: 窗口大小（秒）
            min_packets: 窗口准备就绪的最小包数
            min_fill_ratio: 窗口填充率阈值（0-1）
            logger: 日志记录器
        """
        self.window_size = window_size
        self.min_packets = min_packets
        self.min_fill_ratio = min_fill_ratio
        self.logger = logger or logging.getLogger(__name__)
        
        self.packets = deque()  # 双端队列，O(1)添加和弹出
        self._total_added = 0  # 累计添加的包数（用于统计）
        self._total_removed = 0  # 累计移除的包数
    
    def add_packet(self, packet_info: PacketInfo):
        """添加数据包到窗口
        
        自动移除超出窗口的老数据包。
        
        Args:
            packet_info: PacketInfo对象
            
        Complexity: 
            O(1) for append + O(k) for cleanup, where k is number of expired packets
            k is usually small (1-10), so amortized O(1)
        """
        self.packets.append(packet_info)
        self._total_added += 1
        self._cleanup_old_packets()
    
    def _cleanup_old_packets(self):
        """移除窗口外的老数据包
        
        从队列头部移除时间戳超出窗口的包。
        
        Complexity: O(k) where k is number of expired packets
        """
        if not self.packets:
            return
        
        # 使用最新包的时间戳作为参考
        now = self.packets[-1].timestamp
        
        # 从头部移除老包
        removed_count = 0
        while self.packets and (now - self.packets[0].timestamp) > self.window_size:
            self.packets.popleft()  # O(1)
            removed_count += 1
            self._total_removed += 1
        
        if removed_count > 0:
            self.logger.debug(f"Removed {removed_count} expired packets from window")
    
    def get_window_data(self) -> List[PacketInfo]:
        """获取当前窗口内的所有数据包
        
        Returns:
            PacketInfo对象列表，按时间戳升序
            
        Note:
            返回列表的副本，不影响内部deque
        """
        return list(self.packets)
    
    def get_packet_count(self) -> int:
        """获取当前窗口中的包数量
        
        Returns:
            包数量
        """
        return len(self.packets)
    
    def get_window_duration(self) -> float:
        """获取当前窗口的时间跨度
        
        Returns:
            时间跨度（秒），如果窗口为空则返回0
        """
        if len(self.packets) < 2:
            return 0.0
        
        return self.packets[-1].timestamp - self.packets[0].timestamp
    
    def is_ready(self) -> bool:
        """检查窗口是否准备就绪
        
        窗口准备就绪的条件：
        1. 至少有min_packets个包
        2. 时间跨度至少达到window_size * min_fill_ratio
        
        Returns:
            是否准备就绪
            
        Example:
            >>> buffer = SlidingWindowBuffer(window_size=1.0)
            >>> buffer.is_ready()  # False (empty)
            >>> # ... add packets ...
            >>> buffer.is_ready()  # True (enough packets and duration)
        """
        # 检查包数量
        if len(self.packets) < self.min_packets:
            return False
        
        # 检查时间跨度
        duration = self.get_window_duration()
        min_duration = self.window_size * self.min_fill_ratio
        
        return duration >= min_duration
    
    def get_stats(self) -> dict:
        """获取窗口统计信息
        
        Returns:
            包含统计信息的字典
        """
        return {
            'current_size': len(self.packets),
            'window_duration': self.get_window_duration(),
            'window_size': self.window_size,
            'is_ready': self.is_ready(),
            'total_added': self._total_added,
            'total_removed': self._total_removed,
            'min_packets': self.min_packets,
            'min_fill_ratio': self.min_fill_ratio,
        }
    
    def clear(self):
        """清空窗口"""
        self.packets.clear()
        self.logger.debug("Window cleared")
    
    def __len__(self) -> int:
        """支持len()函数"""
        return len(self.packets)
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"SlidingWindowBuffer(size={len(self.packets)}, "
                f"duration={self.get_window_duration():.3f}s, "
                f"ready={self.is_ready()})")


if __name__ == '__main__':
    """测试模块"""
    import time
    
    print("=" * 70)
    print("Testing SlidingWindowBuffer")
    print("=" * 70)
    
    # 测试1: 创建窗口
    print("\n[Test 1] Create SlidingWindowBuffer")
    buffer = SlidingWindowBuffer(window_size=1.0)
    print(f"  Created: {buffer}")
    print(f"  Window size: {buffer.window_size}s")
    print(f"  Min packets: {buffer.min_packets}")
    
    # 测试2: 添加数据包
    print("\n[Test 2] Add packets")
    base_time = time.time()
    
    for i in range(20):
        packet_info = PacketInfo(
            timestamp=base_time + i * 0.05,  # 每50ms一个包
            number=i,
            length=100,
            src_ip='10.0.0.1',
            dst_ip='10.0.0.2'
        )
        buffer.add_packet(packet_info)
    
    print(f"  Added 20 packets")
    print(f"  Current size: {len(buffer)}")
    print(f"  Duration: {buffer.get_window_duration():.3f}s")
    print(f"  Is ready: {buffer.is_ready()}")
    
    # 测试3: 窗口滑动
    print("\n[Test 3] Window sliding (add packets over 2 seconds)")
    for i in range(20, 60):
        packet_info = PacketInfo(
            timestamp=base_time + i * 0.05,
            number=i,
            length=100
        )
        buffer.add_packet(packet_info)
    
    print(f"  Added 40 more packets (60 total)")
    print(f"  Current size: {len(buffer)} (should be ~20, kept recent 1s)")
    print(f"  Duration: {buffer.get_window_duration():.3f}s")
    
    # 测试4: 统计信息
    print("\n[Test 4] Statistics")
    stats = buffer.get_stats()
    print(f"  Current size: {stats['current_size']}")
    print(f"  Total added: {stats['total_added']}")
    print(f"  Total removed: {stats['total_removed']}")
    print(f"  Is ready: {stats['is_ready']}")
    
    # 测试5: 获取窗口数据
    print("\n[Test 5] Get window data")
    window_data = buffer.get_window_data()
    print(f"  Retrieved {len(window_data)} packets")
    if len(window_data) >= 3:
        print(f"  First packet: #{window_data[0].number} at {window_data[0].timestamp:.3f}")
        print(f"  Last packet: #{window_data[-1].number} at {window_data[-1].timestamp:.3f}")
    
    print("\n" + "=" * 70)
    print("Basic tests completed!")
    print("For full testing, run scripts/test_story_3_4.py")
    print("=" * 70)



