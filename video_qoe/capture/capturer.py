"""
数据包捕获器模块

使用tcpdump捕获网络流量到PCAP文件。
"""

import os
import time
import subprocess
import signal
from pathlib import Path
from typing import Optional
import logging


class PacketCapturer:
    """数据包捕获器
    
    使用tcpdump在Mininet节点上捕获TCP流量。
    
    Attributes:
        interface: 捕获接口名称（如'h2-eth0'）
        pcap_path: PCAP文件保存路径
        filter_expr: tcpdump过滤表达式
        logger: 日志记录器
        tcpdump_process: tcpdump进程对象
        node: Mininet节点对象（可选）
    
    Example:
        >>> # 在Mininet节点上捕获
        >>> capturer = PacketCapturer(
        ...     interface='h2-eth0',
        ...     pcap_path=Path('capture.pcap'),
        ...     node=h2
        ... )
        >>> capturer.start_capture()
        >>> # ... 运行实验 ...
        >>> capturer.stop_capture()
    """
    
    def __init__(self, 
                 interface: str,
                 pcap_path: Path,
                 filter_expr: str = 'tcp',
                 node=None,
                 logger: Optional[logging.Logger] = None):
        """初始化数据包捕获器
        
        Args:
            interface: 网络接口名称
            pcap_path: PCAP文件路径
            filter_expr: tcpdump过滤表达式，默认'tcp'
            node: Mininet节点对象（如果在Mininet中使用）
            logger: 日志记录器
        """
        self.interface = interface
        self.pcap_path = Path(pcap_path)
        self.filter_expr = filter_expr
        self.node = node
        self.logger = logger or logging.getLogger(__name__)
        
        self.tcpdump_process = None
        self._is_running = False
        
        # 确保PCAP目录存在
        self.pcap_path.parent.mkdir(parents=True, exist_ok=True)
    
    def start_capture(self, snaplen: int = 0, buffer_size: int = 4096) -> bool:
        """启动数据包捕获
        
        Args:
            snaplen: 捕获包的最大长度，0表示捕获完整包
            buffer_size: 缓冲区大小（KB）
            
        Returns:
            是否成功启动
            
        Raises:
            RuntimeError: tcpdump启动失败
            
        Example:
            >>> capturer.start_capture()
            True
        """
        if self._is_running:
            self.logger.warning("Capture already running")
            return False
        
        # 删除旧的PCAP文件
        if self.pcap_path.exists():
            self.logger.debug(f"Removing old PCAP file: {self.pcap_path}")
            self.pcap_path.unlink()
        
        # 构建tcpdump命令
        cmd = [
            'tcpdump',
            '-i', self.interface,
            '-w', str(self.pcap_path),
            '-s', str(snaplen),  # Snaplen (0 = 完整包)
            '-B', str(buffer_size),  # 缓冲区大小
        ]
        
        # 添加过滤表达式
        if self.filter_expr:
            cmd.append(self.filter_expr)
        
        self.logger.info(f"Starting tcpdump: {' '.join(cmd)}")
        
        try:
            if self.node:
                # 在Mininet节点上运行
                cmd_str = ' '.join(cmd)
                # 使用后台运行，重定向stderr到/dev/null
                self.node.cmd(f'{cmd_str} 2>/dev/null &')
                
                # 等待PCAP文件创建
                timeout = 5
                for i in range(timeout * 10):
                    if self.pcap_path.exists():
                        self._is_running = True
                        self.logger.info(f"[OK] tcpdump started on {self.interface}")
                        self.logger.info(f"  Capturing to: {self.pcap_path}")
                        return True
                    time.sleep(0.1)
                
                raise RuntimeError(f"PCAP file not created after {timeout}s")
            else:
                # 在本地运行（需要sudo）
                self.tcpdump_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setpgrp  # 创建新进程组
                )
                
                # 等待PCAP文件创建
                timeout = 5
                for i in range(timeout * 10):
                    if self.pcap_path.exists():
                        self._is_running = True
                        self.logger.info(f"[OK] tcpdump started (PID: {self.tcpdump_process.pid})")
                        self.logger.info(f"  Interface: {self.interface}")
                        self.logger.info(f"  PCAP: {self.pcap_path}")
                        return True
                    time.sleep(0.1)
                
                # 超时，终止进程
                self.tcpdump_process.terminate()
                self.tcpdump_process.wait()
                raise RuntimeError(f"PCAP file not created after {timeout}s")
                
        except Exception as e:
            self.logger.error(f"Failed to start tcpdump: {e}")
            self._is_running = False
            raise
    
    def stop_capture(self) -> bool:
        """停止数据包捕获
        
        Returns:
            是否成功停止
            
        Example:
            >>> capturer.stop_capture()
            True
        """
        if not self._is_running:
            self.logger.warning("Capture not running")
            return False
        
        try:
            if self.node:
                # 在Mininet节点上停止
                # 查找tcpdump进程并终止
                result = self.node.cmd(f"pgrep -f 'tcpdump.*{self.interface}'")
                pids = result.strip().split('\n')
                
                for pid in pids:
                    if pid:
                        self.logger.debug(f"Killing tcpdump process: {pid}")
                        self.node.cmd(f"kill -TERM {pid}")
                
                # 等待一下让tcpdump写入完成
                time.sleep(0.5)
                
            else:
                # 本地进程
                if self.tcpdump_process:
                    self.logger.debug(f"Terminating tcpdump (PID: {self.tcpdump_process.pid})")
                    self.tcpdump_process.terminate()
                    
                    # 等待进程结束
                    try:
                        self.tcpdump_process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.logger.warning("tcpdump did not terminate, killing...")
                        self.tcpdump_process.kill()
                        self.tcpdump_process.wait()
            
            self._is_running = False
            
            # 验证PCAP文件
            if self.pcap_path.exists():
                file_size = self.pcap_path.stat().st_size
                self.logger.info(f"[OK] tcpdump stopped")
                self.logger.info(f"  PCAP file: {self.pcap_path}")
                self.logger.info(f"  Size: {file_size} bytes")
                return True
            else:
                self.logger.warning("PCAP file not found after stopping")
                return False
                
        except Exception as e:
            self.logger.error(f"Error stopping tcpdump: {e}")
            self._is_running = False
            return False
    
    def is_running(self) -> bool:
        """检查捕获是否正在运行
        
        Returns:
            是否正在运行
        """
        return self._is_running
    
    def get_packet_count(self) -> int:
        """获取已捕获的数据包数量（估算）
        
        Returns:
            数据包数量估算值
            
        Note:
            这是基于文件大小的粗略估算
        """
        if not self.pcap_path.exists():
            return 0
        
        file_size = self.pcap_path.stat().st_size
        # 粗略估算：PCAP文件头24字节 + 每包约100字节（平均）
        if file_size < 24:
            return 0
        return max(0, (file_size - 24) // 100)
    
    def __enter__(self):
        """Context manager支持"""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager自动清理"""
        self.stop_capture()
        return False
    
    def __del__(self):
        """析构时确保停止捕获"""
        try:
            if self._is_running:
                self.stop_capture()
        except:
            pass


if __name__ == '__main__':
    """测试模块"""
    import sys
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('capturer_test')
    
    print("=" * 70)
    print("Testing PacketCapturer")
    print("=" * 70)
    
    # 测试1: 基本创建
    print("\n[Test 1] Create PacketCapturer")
    capturer = PacketCapturer(
        interface='lo',  # 使用loopback接口测试
        pcap_path=Path('/tmp/test_capture.pcap'),
        logger=logger
    )
    print(f"[OK] Capturer created: {capturer.interface}")
    
    # 测试2: 启动和停止（需要sudo）
    print("\n[Test 2] Start and Stop Capture (requires sudo)")
    if os.geteuid() != 0:
        print("[SKIP] Skipping (need sudo)")
    else:
        try:
            capturer.start_capture()
            print(f"[OK] Capture started")
            
            # 生成一些流量
            subprocess.run(['ping', '-c', '3', '127.0.0.1'], 
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
            
            time.sleep(1)
            
            capturer.stop_capture()
            print(f"[OK] Capture stopped")
            print(f"  Packets: ~{capturer.get_packet_count()}")
            
        except Exception as e:
            print(f"[FAIL] Failed: {e}")
    
    print("\n" + "=" * 70)
    print("Tests completed!")
    print("=" * 70)

