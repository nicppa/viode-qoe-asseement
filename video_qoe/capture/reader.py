"""
PCAP读取器模块

使用pyshark实时读取PCAP文件中的数据包。
"""

import time
import asyncio
from pathlib import Path
from typing import Optional, Iterator, List, Dict, Any
import logging

try:
    import pyshark
except ImportError:
    pyshark = None


class PCAPReader:
    """PCAP文件实时读取器
    
    使用pyshark库实时读取tcpdump生成的PCAP文件。
    支持实时跟踪模式（tail -f模式）和批量读取模式。
    
    Attributes:
        pcap_path: PCAP文件路径
        display_filter: Wireshark显示过滤器
        logger: 日志记录器
        _capture: pyshark capture对象
        _packets_read: 已读取的数据包数量
    
    Example:
        >>> # 实时读取模式
        >>> reader = PCAPReader('capture.pcap')
        >>> for packet in reader.read_live(timeout=30):
        ...     print(f"Packet #{packet.number}")
        
        >>> # 批量读取模式
        >>> packets = reader.read_all()
        >>> print(f"Total: {len(packets)} packets")
    """
    
    def __init__(self, 
                 pcap_path: Path,
                 display_filter: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """初始化PCAP读取器
        
        Args:
            pcap_path: PCAP文件路径
            display_filter: Wireshark显示过滤器（如'tcp.port == 80'）
            logger: 日志记录器
            
        Raises:
            ImportError: pyshark未安装
            FileNotFoundError: PCAP文件不存在（批量模式）
        """
        if pyshark is None:
            raise ImportError(
                "pyshark is not installed. "
                "Install it with: pip install pyshark"
            )
        
        self.pcap_path = Path(pcap_path)
        self.display_filter = display_filter
        self.logger = logger or logging.getLogger(__name__)
        
        self._capture = None
        self._packets_read = 0
    
    def read_live(self, 
                  timeout: Optional[float] = None,
                  packet_count: Optional[int] = None,
                  ignore_truncation: bool = True) -> Iterator:
        """实时读取PCAP文件（tail -f模式）
        
        使用pyshark的FileCapture with keep_packets=False模式，
        实时跟踪正在写入的PCAP文件。
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
            packet_count: 最多读取的包数，None表示无限
            ignore_truncation: 是否忽略文件截断错误（默认True），在实时模式下推荐开启
            
        Yields:
            pyshark.packet.packet.Packet对象
            
        Example:
            >>> reader = PCAPReader('capture.pcap')
            >>> for packet in reader.read_live(timeout=30):
            ...     if hasattr(packet, 'tcp'):
            ...         print(f"TCP packet: {packet.tcp.srcport} -> {packet.tcp.dstport}")
        """
        start_time = time.time()
        packets_yielded = 0
        
        try:
            # 等待PCAP文件创建
            wait_timeout = 10
            wait_start = time.time()
            while not self.pcap_path.exists():
                if time.time() - wait_start > wait_timeout:
                    raise FileNotFoundError(
                        f"PCAP file not found after {wait_timeout}s: {self.pcap_path}"
                    )
                time.sleep(0.1)
            
            self.logger.info(f"Starting live PCAP read: {self.pcap_path}")
            
            # Fix for asyncio in Python 3.8+
            # Set event loop policy to avoid "child watcher not activated" error
            try:
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            except Exception as e:
                self.logger.debug(f"Could not set event loop policy: {e}")
            
            # 创建FileCapture对象
            # keep_packets=False: 实时模式，不保留已处理的包
            # only_summaries=False: 获取完整包信息
            self._capture = pyshark.FileCapture(
                str(self.pcap_path),
                display_filter=self.display_filter,
                keep_packets=False,  # 实时模式
                only_summaries=False
            )
            
            # 实时读取数据包
            try:
                for packet in self._capture:
                    # 超时检查
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        if elapsed > timeout:
                            self.logger.info(f"Read timeout after {elapsed:.1f}s")
                            break
                    
                    # 包数限制检查
                    if packet_count is not None and packets_yielded >= packet_count:
                        self.logger.info(f"Reached packet limit: {packet_count}")
                        break
                    
                    self._packets_read += 1
                    packets_yielded += 1
                    yield packet
            
            except Exception as e:
                # 处理TShark崩溃（通常是因为读取正在写入的PCAP文件）
                error_msg = str(e)
                if "cut short" in error_msg or "crashed" in error_msg:
                    self.logger.warning(f"TShark stopped (file truncation), processed {packets_yielded} packets")
                    # 优雅地结束，不抛出异常
                    return
                else:
                    # 其他异常仍然抛出
                    self.logger.error(f"Error in live read: {e}")
                    raise
            
            self.logger.info(f"Live read completed: {packets_yielded} packets")
            
        except Exception as e:
            # 特殊处理TShark截断错误（实时读取正在写入的PCAP时常见）
            if ignore_truncation and ("cut short" in str(e) or "truncated" in str(e).lower()):
                self.logger.warning(f"PCAP file truncation detected (normal in live mode): {e}")
                self.logger.info(f"Successfully read {packets_yielded} packets before truncation")
                # 不抛出异常，正常结束迭代
                return
            else:
                self.logger.error(f"Error in live read: {e}", exc_info=True)
                raise
        finally:
            self.close()
    
    def read_all(self, max_packets: Optional[int] = None) -> List:
        """批量读取PCAP文件中的所有数据包
        
        读取完整的PCAP文件，返回所有数据包列表。
        适用于已完成捕获的PCAP文件。
        
        Args:
            max_packets: 最多读取的包数，None表示全部读取
            
        Returns:
            数据包列表
            
        Raises:
            FileNotFoundError: PCAP文件不存在
            
        Example:
            >>> reader = PCAPReader('capture.pcap')
            >>> packets = reader.read_all()
            >>> print(f"Read {len(packets)} packets")
        """
        if not self.pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {self.pcap_path}")
        
        self.logger.info(f"Reading PCAP file: {self.pcap_path}")
        
        try:
            # Fix for asyncio in Python 3.8+
            try:
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            except Exception as e:
                self.logger.debug(f"Could not set event loop policy: {e}")
            
            # 创建FileCapture对象
            # keep_packets=True: 批量模式，保留所有包
            self._capture = pyshark.FileCapture(
                str(self.pcap_path),
                display_filter=self.display_filter,
                keep_packets=True
            )
            
            packets = []
            for i, packet in enumerate(self._capture):
                packets.append(packet)
                self._packets_read += 1
                
                if max_packets is not None and i + 1 >= max_packets:
                    self.logger.info(f"Reached max_packets limit: {max_packets}")
                    break
            
            self.logger.info(f"Read {len(packets)} packets from {self.pcap_path}")
            return packets
            
        except Exception as e:
            self.logger.error(f"Error reading PCAP: {e}", exc_info=True)
            raise
        finally:
            self.close()
    
    def get_packet_summary(self, packet) -> Dict[str, Any]:
        """获取数据包的简要信息
        
        提取数据包的关键字段，用于调试和日志。
        
        Args:
            packet: pyshark packet对象
            
        Returns:
            包含关键信息的字典
            
        Example:
            >>> reader = PCAPReader('capture.pcap')
            >>> for packet in reader.read_live(packet_count=1):
            ...     summary = reader.get_packet_summary(packet)
            ...     print(summary)
        """
        summary = {
            'number': getattr(packet, 'number', None),
            'timestamp': getattr(packet, 'sniff_timestamp', None),
            'length': getattr(packet, 'length', None),
        }
        
        # IP层信息
        if hasattr(packet, 'ip'):
            summary['src_ip'] = getattr(packet.ip, 'src', None)
            summary['dst_ip'] = getattr(packet.ip, 'dst', None)
            summary['protocol'] = getattr(packet.ip, 'proto', None)
        
        # TCP层信息
        if hasattr(packet, 'tcp'):
            summary['src_port'] = getattr(packet.tcp, 'srcport', None)
            summary['dst_port'] = getattr(packet.tcp, 'dstport', None)
            summary['tcp_flags'] = getattr(packet.tcp, 'flags', None)
            summary['seq'] = getattr(packet.tcp, 'seq', None)
            summary['ack'] = getattr(packet.tcp, 'ack', None)
        
        # HTTP层信息（如果有）
        if hasattr(packet, 'http'):
            summary['http_request'] = hasattr(packet.http, 'request')
            summary['http_response'] = hasattr(packet.http, 'response')
            if hasattr(packet.http, 'request_method'):
                summary['http_method'] = packet.http.request_method
            if hasattr(packet.http, 'response_code'):
                summary['http_status'] = packet.http.response_code
        
        return summary
    
    def close(self):
        """关闭capture对象"""
        if self._capture is not None:
            try:
                self._capture.close()
                self.logger.debug("PCAP capture closed")
            except Exception as e:
                self.logger.warning(f"Error closing capture: {e}")
            finally:
                self._capture = None
    
    @property
    def packets_read(self) -> int:
        """已读取的数据包总数"""
        return self._packets_read
    
    def __enter__(self):
        """Context manager支持"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager自动清理"""
        self.close()
        return False
    
    def __del__(self):
        """析构时确保关闭"""
        try:
            self.close()
        except:
            pass


def check_pyshark_installed() -> bool:
    """检查pyshark是否已安装
    
    Returns:
        是否已安装pyshark
    """
    return pyshark is not None


def get_pyshark_version() -> Optional[str]:
    """获取pyshark版本
    
    Returns:
        版本字符串，如果未安装则返回None
    """
    if pyshark is None:
        return None
    
    try:
        return pyshark.__version__
    except AttributeError:
        return "unknown"


if __name__ == '__main__':
    """测试模块"""
    import sys
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('reader_test')
    
    print("=" * 70)
    print("Testing PCAPReader")
    print("=" * 70)
    
    # 测试1: 检查pyshark
    print("\n[Test 1] Check pyshark")
    if check_pyshark_installed():
        version = get_pyshark_version()
        print(f"[OK] pyshark installed: version {version}")
    else:
        print("[FAIL] pyshark not installed")
        print("\nInstall with: pip install pyshark")
        sys.exit(1)
    
    # 测试2: 创建PCAPReader
    print("\n[Test 2] Create PCAPReader")
    test_pcap = Path('/tmp/test_reader.pcap')
    
    try:
        reader = PCAPReader(test_pcap, logger=logger)
        print(f"[OK] Reader created: {reader.pcap_path}")
    except Exception as e:
        print(f"[FAIL] Failed: {e}")
    
    print("\n" + "=" * 70)
    print("Basic tests completed!")
    print("For full testing, run scripts/test_story_3_2.py")
    print("=" * 70)

