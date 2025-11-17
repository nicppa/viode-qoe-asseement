"""
数据包信息轻量级数据类

从pyshark包中提取关键信息到轻量级dataclass。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging


@dataclass
class PacketInfo:
    """轻量级数据包信息
    
    从pyshark packet对象中提取的关键信息。
    比完整的pyshark packet对象轻量得多，便于存储和处理。
    
    Attributes:
        timestamp: 捕获时间戳（秒）
        number: 数据包编号
        length: 数据包长度（字节）
        
        # IP层
        src_ip: 源IP地址
        dst_ip: 目的IP地址
        ip_proto: IP协议号（6=TCP, 17=UDP, 1=ICMP）
        
        # TCP层
        src_port: 源端口
        dst_port: 目的端口
        tcp_flags: TCP标志位（如0x018=PSH+ACK）
        tcp_seq: TCP序列号
        tcp_ack: TCP确认号
        tcp_window: TCP窗口大小
        tcp_len: TCP payload长度
        
        # HTTP层（如果有）
        http_method: HTTP请求方法（GET, POST等）
        http_uri: HTTP请求URI
        http_status: HTTP响应状态码
        http_content_type: HTTP Content-Type
        http_content_length: HTTP Content-Length
        
        # 原始数据（可选）
        raw_data: 其他需要保留的原始数据
    
    Example:
        >>> info = PacketInfo(
        ...     timestamp=1699423456.123,
        ...     src_ip='10.0.0.1',
        ...     dst_ip='10.0.0.2',
        ...     src_port=8000,
        ...     dst_port=54321
        ... )
    """
    
    # 基本信息
    timestamp: float
    number: Optional[int] = None
    length: Optional[int] = None
    
    # IP层
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    ip_proto: Optional[int] = None
    
    # TCP层
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    tcp_flags: Optional[int] = None
    tcp_seq: Optional[int] = None
    tcp_ack: Optional[int] = None
    tcp_window: Optional[int] = None
    tcp_len: Optional[int] = None
    
    # HTTP层
    http_method: Optional[str] = None
    http_uri: Optional[str] = None
    http_status: Optional[int] = None
    http_content_type: Optional[str] = None
    http_content_length: Optional[int] = None
    
    # 其他
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_pyshark_packet(cls, packet, logger: Optional[logging.Logger] = None) -> 'PacketInfo':
        """从pyshark packet创建PacketInfo
        
        Args:
            packet: pyshark packet对象
            logger: 日志记录器
            
        Returns:
            PacketInfo实例
            
        Example:
            >>> import pyshark
            >>> cap = pyshark.FileCapture('test.pcap')
            >>> for packet in cap:
            ...     info = PacketInfo.from_pyshark_packet(packet)
            ...     print(f"{info.src_ip} -> {info.dst_ip}")
        """
        logger = logger or logging.getLogger(__name__)
        
        # 基本信息
        timestamp = float(getattr(packet, 'sniff_timestamp', 0))
        number = int(getattr(packet, 'number', 0)) if hasattr(packet, 'number') else None
        length = int(getattr(packet, 'length', 0)) if hasattr(packet, 'length') else None
        
        # 初始化PacketInfo
        info = cls(timestamp=timestamp, number=number, length=length)
        
        try:
            # IP层信息
            if hasattr(packet, 'ip'):
                info.src_ip = str(getattr(packet.ip, 'src', None))
                info.dst_ip = str(getattr(packet.ip, 'dst', None))
                proto = getattr(packet.ip, 'proto', None)
                info.ip_proto = int(proto) if proto is not None else None
            
            # TCP层信息
            if hasattr(packet, 'tcp'):
                src_port = getattr(packet.tcp, 'srcport', None)
                dst_port = getattr(packet.tcp, 'dstport', None)
                info.src_port = int(src_port) if src_port is not None else None
                info.dst_port = int(dst_port) if dst_port is not None else None
                
                # TCP flags (转换为整数)
                flags = getattr(packet.tcp, 'flags', None)
                if flags is not None:
                    try:
                        info.tcp_flags = int(flags, 16) if isinstance(flags, str) else int(flags)
                    except (ValueError, TypeError):
                        pass
                
                # TCP序列号和确认号
                seq = getattr(packet.tcp, 'seq', None)
                ack = getattr(packet.tcp, 'ack', None)
                info.tcp_seq = int(seq) if seq is not None else None
                info.tcp_ack = int(ack) if ack is not None else None
                
                # TCP窗口和payload长度
                window = getattr(packet.tcp, 'window_size_value', None)
                info.tcp_window = int(window) if window is not None else None
                
                tcp_len = getattr(packet.tcp, 'len', None)
                info.tcp_len = int(tcp_len) if tcp_len is not None else None
            
            # HTTP层信息
            if hasattr(packet, 'http'):
                # HTTP请求
                if hasattr(packet.http, 'request_method'):
                    info.http_method = str(packet.http.request_method)
                if hasattr(packet.http, 'request_uri'):
                    info.http_uri = str(packet.http.request_uri)
                
                # HTTP响应
                if hasattr(packet.http, 'response_code'):
                    try:
                        info.http_status = int(packet.http.response_code)
                    except (ValueError, TypeError):
                        pass
                
                # HTTP头部
                if hasattr(packet.http, 'content_type'):
                    info.http_content_type = str(packet.http.content_type)
                if hasattr(packet.http, 'content_length'):
                    try:
                        info.http_content_length = int(packet.http.content_length)
                    except (ValueError, TypeError):
                        pass
        
        except Exception as e:
            logger.debug(f"Error extracting packet info: {e}")
        
        return info
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            包含所有非None字段的字典
        """
        result = {
            'timestamp': self.timestamp,
        }
        
        # 只包含非None的字段
        if self.number is not None:
            result['number'] = self.number
        if self.length is not None:
            result['length'] = self.length
        
        # IP层
        if self.src_ip:
            result['src_ip'] = self.src_ip
        if self.dst_ip:
            result['dst_ip'] = self.dst_ip
        if self.ip_proto is not None:
            result['ip_proto'] = self.ip_proto
        
        # TCP层
        if self.src_port is not None:
            result['src_port'] = self.src_port
        if self.dst_port is not None:
            result['dst_port'] = self.dst_port
        if self.tcp_flags is not None:
            result['tcp_flags'] = self.tcp_flags
        if self.tcp_seq is not None:
            result['tcp_seq'] = self.tcp_seq
        if self.tcp_ack is not None:
            result['tcp_ack'] = self.tcp_ack
        if self.tcp_window is not None:
            result['tcp_window'] = self.tcp_window
        if self.tcp_len is not None:
            result['tcp_len'] = self.tcp_len
        
        # HTTP层
        if self.http_method:
            result['http_method'] = self.http_method
        if self.http_uri:
            result['http_uri'] = self.http_uri
        if self.http_status is not None:
            result['http_status'] = self.http_status
        if self.http_content_type:
            result['http_content_type'] = self.http_content_type
        if self.http_content_length is not None:
            result['http_content_length'] = self.http_content_length
        
        # 原始数据
        if self.raw_data:
            result['raw_data'] = self.raw_data
        
        return result
    
    def is_tcp(self) -> bool:
        """是否为TCP包"""
        return self.ip_proto == 6
    
    def is_http(self) -> bool:
        """是否为HTTP包"""
        return self.http_method is not None or self.http_status is not None
    
    def get_flow_key(self) -> Optional[str]:
        """获取流标识（五元组的哈希）
        
        Returns:
            流标识字符串，格式: "src_ip:src_port->dst_ip:dst_port:proto"
            如果信息不完整则返回None
        """
        if self.src_ip and self.dst_ip and self.ip_proto is not None:
            src_port = self.src_port if self.src_port is not None else 0
            dst_port = self.dst_port if self.dst_port is not None else 0
            return f"{self.src_ip}:{src_port}->{self.dst_ip}:{dst_port}:{self.ip_proto}"
        return None
    
    def __str__(self) -> str:
        """字符串表示"""
        parts = [f"Packet #{self.number or '?'}"]
        
        if self.src_ip and self.dst_ip:
            if self.src_port and self.dst_port:
                parts.append(f"{self.src_ip}:{self.src_port} -> {self.dst_ip}:{self.dst_port}")
            else:
                parts.append(f"{self.src_ip} -> {self.dst_ip}")
        
        if self.length:
            parts.append(f"{self.length}B")
        
        if self.http_method:
            parts.append(f"HTTP {self.http_method}")
        elif self.http_status:
            parts.append(f"HTTP {self.http_status}")
        
        return " ".join(parts)


if __name__ == '__main__':
    """测试模块"""
    print("=" * 70)
    print("Testing PacketInfo")
    print("=" * 70)
    
    # 测试1: 创建PacketInfo
    print("\n[Test 1] Create PacketInfo")
    info = PacketInfo(
        timestamp=1699423456.123,
        number=1,
        length=100,
        src_ip='10.0.0.1',
        dst_ip='10.0.0.2',
        src_port=8000,
        dst_port=54321,
        tcp_flags=0x018  # PSH+ACK
    )
    print(f"  Created: {info}")
    
    # 测试2: 转换为字典
    print("\n[Test 2] Convert to dict")
    data = info.to_dict()
    print(f"  Dict keys: {list(data.keys())}")
    print(f"  src_ip: {data.get('src_ip')}")
    print(f"  dst_ip: {data.get('dst_ip')}")
    
    # 测试3: 流标识
    print("\n[Test 3] Flow key")
    info.ip_proto = 6  # TCP
    flow_key = info.get_flow_key()
    print(f"  Flow key: {flow_key}")
    
    # 测试4: 类型检查
    print("\n[Test 4] Type checks")
    print(f"  is_tcp: {info.is_tcp()}")
    print(f"  is_http: {info.is_http()}")
    
    print("\n" + "=" * 70)
    print("Basic tests completed!")
    print("For full testing with pyshark, run scripts/test_story_3_3.py")
    print("=" * 70)



