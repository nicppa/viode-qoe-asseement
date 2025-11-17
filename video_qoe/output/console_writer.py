"""
控制台输出格式化

使用rich库实现美观的CLI输出。
"""

import time
from typing import Optional, Dict
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from video_qoe.prediction.predictor import Prediction


class ConsoleWriter:
    """控制台输出器
    
    使用rich库实现美观的实时输出，如果rich不可用则降级到普通输出。
    
    输出格式：
    - 时间戳
    - 预测分辨率（带置信度）
    - 网络指标（吞吐量/丢包/RTT）
    - 事件标注
    
    颜色编码：
    - 1080p: 绿色（高质量）
    - 720p: 黄色（中等质量）
    - 480p: 红色（低质量）
    
    Attributes:
        console: Rich Console对象
        enable_color: 是否启用颜色
        line_count: 输出行数计数器
        start_time: 开始时间
    
    Example:
        >>> writer = ConsoleWriter()
        >>> writer.write_header()
        >>> writer.write_line(prediction, metrics)
    """
    
    def __init__(self, enable_color: bool = True):
        """初始化输出器
        
        Args:
            enable_color: 是否启用颜色（默认True）
        """
        self.enable_color = enable_color and RICH_AVAILABLE
        
        if self.enable_color:
            self.console = Console()
        else:
            self.console = None
        
        self.line_count = 0
        self.start_time = time.time()
        self.last_resolution = None
    
    def write_header(self):
        """输出表头"""
        if self.enable_color and self.console:
            # Rich格式
            header = Panel(
                Text("Video QoE Real-time Monitor", style="bold cyan"),
                subtitle=f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                border_style="cyan"
            )
            self.console.print(header)
            self.console.print()
        else:
            # 普通格式
            print("=" * 80)
            print(" " * 25 + "Video QoE Real-time Monitor")
            print(f" Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print()
            print(f"{'Time':>6} | {'Resolution':>12} | {'Throughput':>12} | {'Loss':>8} | {'RTT':>8} | Events")
            print("-" * 80)
    
    def write_line(self, prediction: Prediction, elapsed: Optional[int] = None):
        """输出一行监测数据
        
        Args:
            prediction: 预测结果
            elapsed: 经过时间（秒），如果None则自动计算
        """
        if elapsed is None:
            elapsed = int(time.time() - self.start_time)
        
        self.line_count += 1
        
        # 提取指标
        metrics = prediction.metrics or {}
        throughput = metrics.get('throughput', 0.0)
        loss_rate = metrics.get('loss_rate', 0.0)
        rtt = metrics.get('rtt', 0.0)
        
        # 检测事件
        events = self._detect_events(prediction)
        
        if self.enable_color and self.console:
            self._write_line_rich(elapsed, prediction, throughput, loss_rate, rtt, events)
        else:
            self._write_line_plain(elapsed, prediction, throughput, loss_rate, rtt, events)
        
        self.last_resolution = prediction.resolution
    
    def _write_line_rich(self, elapsed: int, prediction: Prediction, 
                        throughput: float, loss_rate: float, rtt: float, events: str):
        """Rich格式输出"""
        # 分辨率颜色
        res_color = self._get_resolution_color(prediction.resolution)
        
        # 置信度颜色
        conf = prediction.confidence
        conf_color = "green" if conf >= 0.8 else "yellow" if conf >= 0.6 else "red"
        
        # 格式化时间
        time_str = f"{elapsed:3d}s"
        
        # 格式化分辨率和置信度
        res_str = Text(f"{prediction.resolution:>4s}", style=f"bold {res_color}")
        conf_str = Text(f"({conf:5.1%})", style=conf_color)
        
        # 输出行
        line = Text()
        line.append(f"[{time_str:>5s}] ", style="dim")
        line.append(res_str)
        line.append(" ")
        line.append(conf_str)
        line.append(f" | {throughput:5.2f} Mbps | {loss_rate:4.1f}% | {rtt:5.1f} ms")
        
        if events:
            line.append(f" | {events}")
        
        self.console.print(line)
    
    def _write_line_plain(self, elapsed: int, prediction: Prediction,
                         throughput: float, loss_rate: float, rtt: float, events: str):
        """普通格式输出"""
        time_str = f"{elapsed:3d}s"
        res_conf = f"{prediction.resolution} ({prediction.confidence:5.1%})"
        
        print(f"[{time_str:>5s}] | {res_conf:>12s} | {throughput:6.2f} Mbps | {loss_rate:5.1f}% | {rtt:6.1f} ms | {events}")
    
    def _get_resolution_color(self, resolution: str) -> str:
        """获取分辨率对应的颜色"""
        colors = {
            '1080p': 'green',
            '720p': 'yellow',
            '480p': 'red'
        }
        return colors.get(resolution, 'white')
    
    def _detect_events(self, prediction: Prediction) -> str:
        """检测特殊事件
        
        Args:
            prediction: 预测结果
            
        Returns:
            事件描述字符串
        """
        events = []
        
        # 检测质量变化
        if self.last_resolution and self.last_resolution != prediction.resolution:
            if self._is_quality_decrease(self.last_resolution, prediction.resolution):
                events.append("[WARN] Quality decrease")
            else:
                events.append("[OK] Quality improve")
        
        # 检测低置信度
        if prediction.confidence < 0.6:
            events.append("[WARN] Low confidence")
        
        # 检测网络问题
        metrics = prediction.metrics or {}
        loss_rate = metrics.get('loss_rate', 0.0)
        rtt = metrics.get('rtt', 0.0)
        
        if loss_rate > 3.0:
            events.append("[WARN] High loss")
        
        if rtt > 200:
            events.append("[WARN] High RTT")
        
        return " ".join(events) if events else ""
    
    def _is_quality_decrease(self, old_res: str, new_res: str) -> bool:
        """判断是否质量下降"""
        quality_order = {'1080p': 3, '720p': 2, '480p': 1}
        return quality_order.get(new_res, 0) < quality_order.get(old_res, 0)
    
    def write_summary(self, total_predictions: int):
        """输出总结
        
        Args:
            total_predictions: 总预测次数
        """
        elapsed = int(time.time() - self.start_time)
        
        # 避免除零错误
        if elapsed == 0:
            elapsed = 1
        
        if self.enable_color and self.console:
            self.console.print()
            summary = Panel(
                f"Total predictions: {total_predictions}\n"
                f"Duration: {elapsed}s\n"
                f"Avg rate: {total_predictions/elapsed:.1f} pred/sec",
                title="[bold]Summary[/bold]",
                border_style="cyan"
            )
            self.console.print(summary)
        else:
            print()
            print("=" * 80)
            print(f" Summary:")
            print(f"   Total predictions: {total_predictions}")
            print(f"   Duration: {elapsed}s")
            print(f"   Avg rate: {total_predictions/elapsed:.1f} pred/sec")
            print("=" * 80)
    
    def write_error(self, message: str):
        """输出错误信息
        
        Args:
            message: 错误消息
        """
        if self.enable_color and self.console:
            self.console.print(f"[bold red][ERROR][/bold red] {message}")
        else:
            print(f"[ERROR] {message}")
    
    def write_info(self, message: str):
        """输出信息
        
        Args:
            message: 信息内容
        """
        if self.enable_color and self.console:
            self.console.print(f"[cyan][INFO][/cyan] {message}")
        else:
            print(f"[INFO] {message}")


if __name__ == '__main__':
    """测试模块"""
    import numpy as np
    
    print("=" * 70)
    print("Testing ConsoleWriter")
    print("=" * 70)
    
    # 测试初始化
    print("\n[Test 1] Initialize writer")
    writer = ConsoleWriter(enable_color=True)
    print(f"  Rich available: {RICH_AVAILABLE}")
    print(f"  Color enabled: {writer.enable_color}")
    
    # 输出表头
    print("\n[Test 2] Write header")
    writer.write_header()
    
    # 模拟一些预测结果
    print("\n[Test 3] Write predictions")
    
    # 高质量
    pred1 = Prediction(
        resolution='1080p',
        confidence=0.85,
        probabilities=np.array([0.05, 0.10, 0.85]),
        timestamp=time.time(),
        metrics={'throughput': 8.5, 'loss_rate': 0.2, 'rtt': 35.0}
    )
    writer.write_line(pred1, elapsed=1)
    time.sleep(0.5)
    
    # 中等质量
    pred2 = Prediction(
        resolution='720p',
        confidence=0.78,
        probabilities=np.array([0.10, 0.78, 0.12]),
        timestamp=time.time(),
        metrics={'throughput': 3.5, 'loss_rate': 1.2, 'rtt': 80.0}
    )
    writer.write_line(pred2, elapsed=2)
    time.sleep(0.5)
    
    # 低质量（触发事件）
    pred3 = Prediction(
        resolution='480p',
        confidence=0.65,
        probabilities=np.array([0.65, 0.25, 0.10]),
        timestamp=time.time(),
        metrics={'throughput': 1.5, 'loss_rate': 4.5, 'rtt': 220.0}
    )
    writer.write_line(pred3, elapsed=3)
    
    # 输出总结
    print("\n[Test 4] Write summary")
    writer.write_summary(total_predictions=3)
    
    print("\n" + "=" * 70)
    print("Basic tests completed!")
    print("=" * 70)

