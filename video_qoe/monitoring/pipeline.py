"""
实时监测流水线

整合流量捕获、特征计算、预测和输出的端到端流水线。
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from video_qoe.capture import PacketCapturer, PCAPReader, PacketInfo, SlidingWindowBuffer
from video_qoe.features import FeatureCalculator
from video_qoe.prediction import create_predictor
from video_qoe.output import ConsoleWriter


@dataclass
class PipelineStats:
    """流水线统计信息"""
    packets_processed: int = 0
    predictions_made: int = 0
    windows_processed: int = 0
    errors_count: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def elapsed_time(self) -> float:
        """计算运行时长（秒）"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def packets_per_second(self) -> float:
        """计算每秒处理的包数"""
        elapsed = self.elapsed_time()
        return self.packets_processed / elapsed if elapsed > 0 else 0.0
    
    def predictions_per_second(self) -> float:
        """计算每秒预测次数"""
        elapsed = self.elapsed_time()
        return self.predictions_made / elapsed if elapsed > 0 else 0.0


class RealTimePipeline:
    """实时监测流水线
    
    整合所有组件，提供端到端的实时视频质量监测功能。
    
    支持两种模式：
    1. 实时捕获模式：使用PacketCapturer实时捕获流量
    2. PCAP文件模式：读取已存在的PCAP文件
    
    Attributes:
        interface: 网络接口（实时捕获模式）
        pcap_path: PCAP文件路径
        client_ip: 客户端IP地址
        window_size: 滑动窗口大小（秒）
        predictor_type: 预测器类型（'rule_based'或'ml_model'）
        model_path: ML模型路径（可选）
        output_color: 是否启用彩色输出
        logger: 日志记录器
        
    Example:
        >>> # 实时捕获模式
        >>> pipeline = RealTimePipeline(
        ...     interface='h2-eth0',
        ...     pcap_path='capture.pcap',
        ...     client_ip='10.0.0.2',
        ...     predictor_type='rule_based'
        ... )
        >>> with pipeline:
        ...     pipeline.run(duration=60)
        
        >>> # PCAP文件模式
        >>> pipeline = RealTimePipeline(
        ...     pcap_path='existing.pcap',
        ...     client_ip='10.0.0.2',
        ...     capture_mode=False
        ... )
        >>> with pipeline:
        ...     pipeline.run()
    """
    
    def __init__(self,
                 pcap_path: str,
                 client_ip: str,
                 interface: Optional[str] = None,
                 node = None,
                 window_size: float = 1.0,
                 predictor_type: str = 'rule_based',
                 model_path: Optional[str] = None,
                 output_color: bool = True,
                 capture_mode: bool = True,
                 snaplen: int = 96,
                 logger: Optional[logging.Logger] = None):
        """初始化实时监测流水线
        
        Args:
            pcap_path: PCAP文件路径
            client_ip: 客户端IP地址（用于区分上下行流量）
            interface: 网络接口（实时捕获模式必需）
            node: Mininet节点对象（可选）
            window_size: 滑动窗口大小（秒），默认1.0
            predictor_type: 预测器类型，'rule_based'或'ml_model'
            model_path: ML模型路径（predictor_type='ml_model'时必需）
            output_color: 是否启用彩色输出
            capture_mode: 是否启用实时捕获模式
            snaplen: 捕获包的最大长度（字节），96=只捕获头部，0=完整包
            logger: 日志记录器
        """
        self.pcap_path = Path(pcap_path)
        self.client_ip = client_ip
        self.interface = interface
        self.node = node
        self.window_size = window_size
        self.predictor_type = predictor_type
        self.model_path = model_path
        self.output_color = output_color
        self.capture_mode = capture_mode
        self.snaplen = snaplen
        self.logger = logger or logging.getLogger(__name__)
        
        # 验证参数
        if capture_mode and not interface:
            raise ValueError("interface is required when capture_mode=True")
        
        if predictor_type == 'ml_model' and not model_path:
            raise ValueError("model_path is required when predictor_type='ml_model'")
        
        # 初始化组件
        self.capturer: Optional[PacketCapturer] = None
        self.reader: Optional[PCAPReader] = None
        self.window: Optional[SlidingWindowBuffer] = None
        self.feature_calc: Optional[FeatureCalculator] = None
        self.predictor = None
        self.writer: Optional[ConsoleWriter] = None
        
        # 统计信息
        self.stats = PipelineStats()
        
        # 运行状态
        self._is_running = False
        self._is_setup = False
    
    def setup(self):
        """初始化所有组件"""
        if self._is_setup:
            self.logger.warning("Pipeline already setup")
            return
        
        try:
            self.logger.info("Setting up pipeline components...")
            
            # 1. 初始化捕获器（实时模式）
            if self.capture_mode:
                self.logger.info(f"Initializing packet capturer on {self.interface}")
                self.capturer = PacketCapturer(
                    interface=self.interface,
                    pcap_path=self.pcap_path,
                    filter_expr='tcp',
                    node=self.node,
                    logger=self.logger
                )
            
            # 2. 初始化PCAP读取器
            self.logger.info(f"Initializing PCAP reader for {self.pcap_path}")
            self.reader = PCAPReader(
                pcap_path=self.pcap_path,
                display_filter='tcp',
                logger=self.logger
            )
            
            # 3. 初始化滑动窗口
            self.logger.info(f"Initializing sliding window (size={self.window_size}s)")
            self.window = SlidingWindowBuffer(window_size=self.window_size)
            
            # 4. 初始化特征计算器
            self.logger.info("Initializing feature calculator")
            self.feature_calc = FeatureCalculator()
            
            # 5. 初始化预测器
            use_ml = self.predictor_type == 'ml_model'
            self.logger.info(f"Initializing predictor (type={self.predictor_type})")
            self.predictor = create_predictor(
                use_ml_model=use_ml,
                model_path=self.model_path if use_ml else None
            )
            
            # 6. 初始化输出器
            self.logger.info(f"Initializing console writer (color={self.output_color})")
            self.writer = ConsoleWriter(enable_color=self.output_color)
            
            self._is_setup = True
            self.logger.info("Pipeline setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup pipeline: {e}")
            self.cleanup()
            raise
    
    def start_capture(self, snaplen: int = 96):
        """启动流量捕获（实时模式）
        
        Args:
            snaplen: 捕获包的最大长度（字节）
                    - 96: 足够捕获TCP/IP头（推荐，节省空间）
                    - 128: 保守值
                    - 0: 完整包（文件会很大！）
        """
        if not self.capture_mode:
            return
        
        if not self.capturer:
            raise RuntimeError("Capturer not initialized. Call setup() first.")
        
        self.logger.info(f"Starting packet capture (snaplen={snaplen})...")
        success = self.capturer.start_capture(snaplen=snaplen)
        
        if not success:
            raise RuntimeError("Failed to start packet capture")
        
        # 等待PCAP文件创建
        wait_timeout = 10
        wait_start = time.time()
        while not self.pcap_path.exists():
            if time.time() - wait_start > wait_timeout:
                raise RuntimeError(f"PCAP file not created within {wait_timeout}s")
            time.sleep(0.1)
        
        self.logger.info(f"Packet capture started, writing to {self.pcap_path}")
    
    def stop_capture(self):
        """停止流量捕获（实时模式）"""
        if not self.capture_mode or not self.capturer:
            return
        
        self.logger.info("Stopping packet capture...")
        self.capturer.stop_capture()
        self.logger.info("Packet capture stopped")
    
    def run(self, 
            duration: Optional[float] = None,
            max_predictions: Optional[int] = None) -> PipelineStats:
        """运行流水线
        
        Args:
            duration: 运行时长（秒），None表示处理完所有包
            max_predictions: 最大预测次数，None表示不限制
            
        Returns:
            流水线统计信息
            
        Example:
            >>> pipeline = RealTimePipeline(...)
            >>> pipeline.setup()
            >>> pipeline.start_capture()
            >>> stats = pipeline.run(duration=60)
            >>> print(f"Processed {stats.packets_processed} packets")
        """
        if not self._is_setup:
            raise RuntimeError("Pipeline not setup. Call setup() first.")
        
        if self._is_running:
            raise RuntimeError("Pipeline already running")
        
        self._is_running = True
        self.stats = PipelineStats()  # 重置统计
        
        try:
            self.logger.info("Starting pipeline execution...")
            
            # 写入输出表头
            self.writer.write_header()
            
            # 确定读取模式
            if self.capture_mode:
                # 实时读取模式
                packet_iterator = self.reader.read_live(
                    timeout=duration,
                    packet_count=None
                )
            else:
                # 批量读取模式
                packet_iterator = self.reader.read_all()
            
            # 主处理循环
            start_time = time.time()
            
            try:
                packet_iter = iter(packet_iterator)
            except Exception as e:
                self.logger.error(f"Failed to create packet iterator: {e}")
                raise
            
            while True:
                # 检查超时
                if duration and (time.time() - start_time) >= duration:
                    self.logger.info(f"Duration limit reached ({duration}s)")
                    break
                
                # 检查预测次数限制
                if max_predictions and self.stats.predictions_made >= max_predictions:
                    self.logger.info(f"Prediction limit reached ({max_predictions})")
                    break
                
                # 获取下一个包
                try:
                    pyshark_packet = next(packet_iter)
                except StopIteration:
                    self.logger.info("No more packets to process")
                    break
                except Exception as e:
                    # TShark可能在读取正在写入的PCAP文件时遇到截断的包
                    if "cut short" in str(e) or "TShark" in str(e):
                        self.logger.warning(f"TShark encountered truncated packet, stopping iteration")
                        break
                    self.logger.error(f"Error getting next packet: {e}")
                    self.stats.errors_count += 1
                    continue
                
                try:
                    # 转换为PacketInfo
                    pkt_info = PacketInfo.from_pyshark_packet(
                        pyshark_packet,
                        logger=self.logger
                    )
                    
                    # 添加到滑动窗口
                    self.window.add_packet(pkt_info)
                    self.stats.packets_processed += 1
                    
                    # 如果窗口准备好，进行预测
                    if self.window.is_ready():
                        window_packets = self.window.get_window_data()
                        self.stats.windows_processed += 1
                        
                        # 计算特征
                        features = self.feature_calc.compute_all_features(
                            window_packets,
                            client_ip=self.client_ip
                        )
                        
                        # 预测
                        prediction = self.predictor.predict(features)
                        self.stats.predictions_made += 1
                        
                        # 输出
                        elapsed = int(time.time() - start_time)
                        self.writer.write_line(prediction, elapsed=elapsed)
                
                except Exception as e:
                    self.stats.errors_count += 1
                    self.logger.error(f"Error processing packet: {e}")
                    # 继续处理下一个包
                    continue
            
            self.stats.end_time = time.time()
            
            # 输出总结
            self.writer.write_summary(total_predictions=self.stats.predictions_made)
            
            self.logger.info(f"Pipeline execution complete. Processed {self.stats.packets_processed} packets, "
                           f"made {self.stats.predictions_made} predictions")
            
            return self.stats
        
        except KeyboardInterrupt:
            self.logger.info("Pipeline interrupted by user")
            self.writer.write_info("Monitoring interrupted by user")
            self.stats.end_time = time.time()
            return self.stats
        
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            self.stats.end_time = time.time()
            raise
        
        finally:
            self._is_running = False
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("Cleaning up pipeline resources...")
        
        try:
            # 停止捕获
            if self.capturer and self.capture_mode:
                self.stop_capture()
            
            # 清理读取器
            if self.reader:
                try:
                    self.reader.close()
                except:
                    pass
            
            self._is_setup = False
            self._is_running = False
            
            self.logger.info("Pipeline cleanup complete")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.setup()
        if self.capture_mode:
            self.start_capture(snaplen=self.snaplen)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()
        return False  # 不抑制异常
    
    def get_stats(self) -> Dict[str, Any]:
        """获取流水线统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'packets_processed': self.stats.packets_processed,
            'predictions_made': self.stats.predictions_made,
            'windows_processed': self.stats.windows_processed,
            'errors_count': self.stats.errors_count,
            'elapsed_time': self.stats.elapsed_time(),
            'packets_per_second': self.stats.packets_per_second(),
            'predictions_per_second': self.stats.predictions_per_second(),
        }

