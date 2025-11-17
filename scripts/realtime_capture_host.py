#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Video QoE Monitoring on Host Machine
宿主机实时视频质量监测 - 捕获真实视频网站流量并识别

This script runs on the HOST machine (not VM) to capture and analyze
real-world video streaming traffic from YouTube, Netflix, etc.

Features:
- Captures traffic on specified network interface
- Filters video streaming traffic automatically
- Extracts TCP/IP features in real-time
- Uses trained ML model to predict video quality
- Beautiful terminal output with rich

Requirements (Host machine):
    pip install pyshark scapy rich pandas numpy scikit-learn xgboost

Usage:
    # 自动检测网卡并监测
    python scripts/realtime_capture_host.py --model models/xgboost_model.pkl
    
    # 指定网卡
    python scripts/realtime_capture_host.py --interface en0 --model models/xgboost_model.pkl
    
    # 指定要监测的视频网站IP
    python scripts/realtime_capture_host.py --target-ip 142.250.185.78 --model models/xgboost_model.pkl
    
    # 保存捕获的数据
    python scripts/realtime_capture_host.py --model models/xgboost_model.pkl --save-pcap capture.pcap

Author: Video QoE Assessment System  
Date: 2025-11-15
"""

import argparse
import sys
import time
import signal
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from collections import deque
import threading

try:
    import pyshark
    import joblib
    import pandas as pd
    import numpy as np
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich import box
    from rich.text import Text
except ImportError as e:
    print(f"错误: 缺少必需的库 - {e}")
    print("\n请在宿主机上安装依赖:")
    print("  pip install pyshark pandas numpy scikit-learn xgboost rich joblib")
    sys.exit(1)

console = Console()

# 视频网站常见端口
VIDEO_PORTS = {80, 443, 8080, 1935}  # HTTP, HTTPS, Alt HTTP, RTMP

# 常见视频网站域名/IP（示例）
VIDEO_DOMAINS = {
    'youtube', 'googlevideo', 'ytimg',
    'netflix', 'nflxvideo', 'nflximg',
    'twitch', 'ttvnw',
    'bilibili', 'hdslb',
    'vimeo', 'vimeocdn'
}


@dataclass
class PacketStats:
    """数据包统计"""
    total_packets: int = 0
    tcp_packets: int = 0
    video_packets: int = 0
    bytes_total: int = 0
    predictions_made: int = 0
    start_time: float = 0
    
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0


class SlidingWindow:
    """滑动窗口 - 简化版"""
    def __init__(self, window_size: float = 1.0):
        self.window_size = window_size
        self.packets = deque()
        self.window_start = None
    
    def add_packet(self, packet_info: Dict):
        """添加数据包"""
        current_time = time.time()
        
        if self.window_start is None:
            self.window_start = current_time
        
        # 移除过期数据包
        while self.packets and (current_time - self.packets[0]['timestamp']) > self.window_size:
            self.packets.popleft()
            if not self.packets:
                self.window_start = current_time
        
        self.packets.append(packet_info)
    
    def is_ready(self) -> bool:
        """窗口是否准备好"""
        if not self.packets:
            return False
        current_time = time.time()
        return (current_time - self.window_start) >= self.window_size
    
    def get_packets(self) -> List[Dict]:
        """获取窗口内的数据包"""
        return list(self.packets)
    
    def clear(self):
        """清空窗口"""
        self.packets.clear()
        self.window_start = None


class SimpleFeatureExtractor:
    """简化的特征提取器"""
    
    @staticmethod
    def extract_features(packets: List[Dict]) -> Dict[str, float]:
        """从数据包列表提取特征
        
        Returns:
            包含35个特征的字典
        """
        if not packets:
            return SimpleFeatureExtractor._get_default_features()
        
        # 基本统计
        total_packets = len(packets)
        total_bytes = sum(p['length'] for p in packets)
        
        # TCP标志统计
        syn_count = sum(1 for p in packets if p.get('tcp_syn', False))
        ack_count = sum(1 for p in packets if p.get('tcp_ack', False))
        fin_count = sum(1 for p in packets if p.get('tcp_fin', False))
        rst_count = sum(1 for p in packets if p.get('tcp_rst', False))
        psh_count = sum(1 for p in packets if p.get('tcp_psh', False))
        
        # 时间统计
        if len(packets) > 1:
            time_span = packets[-1]['timestamp'] - packets[0]['timestamp']
            throughput = total_bytes / time_span if time_span > 0 else 0
        else:
            time_span = 0
            throughput = 0
        
        # 包大小统计
        sizes = [p['length'] for p in packets]
        mean_size = np.mean(sizes) if sizes else 0
        std_size = np.std(sizes) if len(sizes) > 1 else 0
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        
        # 构建35个特征（与训练时保持一致）
        features = {
            # TCP特征 (15个)
            'tcp_syn_count': syn_count,
            'tcp_syn_ratio': syn_count / total_packets if total_packets > 0 else 0,
            'tcp_ack_count': ack_count,
            'tcp_ack_ratio': ack_count / total_packets if total_packets > 0 else 0,
            'tcp_fin_count': fin_count,
            'tcp_fin_ratio': fin_count / total_packets if total_packets > 0 else 0,
            'tcp_rst_count': rst_count,
            'tcp_rst_ratio': rst_count / total_packets if total_packets > 0 else 0,
            'tcp_psh_count': psh_count,
            'tcp_psh_ratio': psh_count / total_packets if total_packets > 0 else 0,
            'tcp_retransmissions': 0,  # 简化版不计算重传
            'tcp_out_of_order': 0,
            'tcp_window_size_avg': 65535,  # 默认值
            'tcp_window_size_std': 0,
            'tcp_window_updates': 0,
            
            # 流量特征 (10个)
            'packet_count': total_packets,
            'total_bytes': total_bytes,
            'avg_packet_size': mean_size,
            'std_packet_size': std_size,
            'min_packet_size': min_size,
            'max_packet_size': max_size,
            'throughput_bps': throughput * 8,  # bits per second
            'throughput_mbps': throughput * 8 / 1_000_000,
            'packets_per_second': total_packets / time_span if time_span > 0 else 0,
            'bytes_per_second': throughput,
            
            # 时序特征 (10个)
            'duration': time_span,
            'inter_arrival_mean': time_span / total_packets if total_packets > 1 else 0,
            'inter_arrival_std': 0,  # 简化
            'inter_arrival_min': 0,
            'inter_arrival_max': 0,
            'burstiness': 0,  # 简化
            'flow_activity_ratio': 1.0,
            'idle_time_ratio': 0,
            'active_periods': 1,
            'idle_periods': 0,
        }
        
        return features
    
    @staticmethod
    def _get_default_features() -> Dict[str, float]:
        """获取默认特征值"""
        return {f'feature_{i}': 0.0 for i in range(35)}


class HostMonitor:
    """宿主机监测器"""
    
    def __init__(self, interface: str, model_path: str, preprocessor_path: str,
                 window_size: float = 1.0, target_ip: Optional[str] = None,
                 save_pcap: Optional[str] = None):
        self.interface = interface
        self.window_size = window_size
        self.target_ip = target_ip
        self.save_pcap = save_pcap
        
        # 加载模型
        console.print(f"[cyan]加载模型: {model_path}[/cyan]")
        self.model = joblib.load(model_path)
        
        console.print(f"[cyan]加载预处理器: {preprocessor_path}[/cyan]")
        self.preprocessor = joblib.load(preprocessor_path)
        
        # 初始化组件
        self.window = SlidingWindow(window_size)
        self.feature_extractor = SimpleFeatureExtractor()
        self.stats = PacketStats()
        self.running = False
        
        # 预测历史（用于显示）
        self.recent_predictions = deque(maxlen=10)
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """处理Ctrl+C"""
        console.print("\n[yellow]接收到中断信号，正在停止...[/yellow]")
        self.running = False
    
    def _is_video_packet(self, packet) -> bool:
        """判断是否为视频流量包
        
        简单启发式规则:
        1. TCP协议
        2. 端口443或80（HTTPS/HTTP）
        3. 包大小较大（> 100字节）
        """
        try:
            if not hasattr(packet, 'tcp'):
                return False
            
            # 检查端口
            src_port = int(packet.tcp.srcport)
            dst_port = int(packet.tcp.dstport)
            
            if not (src_port in VIDEO_PORTS or dst_port in VIDEO_PORTS):
                return False
            
            # 检查包大小
            if int(packet.length) < 100:
                return False
            
            # 如果指定了目标IP，只处理该IP的流量
            if self.target_ip:
                if hasattr(packet, 'ip'):
                    if packet.ip.src != self.target_ip and packet.ip.dst != self.target_ip:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def _extract_packet_info(self, packet) -> Dict:
        """提取数据包信息"""
        try:
            info = {
                'timestamp': time.time(),
                'length': int(packet.length),
                'tcp_syn': hasattr(packet.tcp, 'flags_syn') and packet.tcp.flags_syn == '1',
                'tcp_ack': hasattr(packet.tcp, 'flags_ack') and packet.tcp.flags_ack == '1',
                'tcp_fin': hasattr(packet.tcp, 'flags_fin') and packet.tcp.flags_fin == '1',
                'tcp_rst': hasattr(packet.tcp, 'flags_reset') and packet.tcp.flags_reset == '1',
                'tcp_psh': hasattr(packet.tcp, 'flags_push') and packet.tcp.flags_push == '1',
            }
            return info
        except Exception as e:
            console.print(f"[red]提取包信息错误: {e}[/red]")
            return None
    
    def _make_prediction(self):
        """执行预测"""
        try:
            # 获取窗口数据包
            packets = self.window.get_packets()
            if not packets:
                return
            
            # 提取特征
            features_dict = self.feature_extractor.extract_features(packets)
            
            # 转换为DataFrame（与训练时格式一致）
            features_df = pd.DataFrame([features_dict])
            
            # 预处理
            # 注意：preprocessor可能是sklearn的或者自定义的
            if hasattr(self.preprocessor, 'transform'):
                # 如果是FeaturePreprocessor，需要特殊处理
                if hasattr(self.preprocessor, 'scaler'):
                    # 自定义的FeaturePreprocessor
                    X_scaled = self.preprocessor.scaler.transform(features_df.values)
                else:
                    # sklearn的StandardScaler
                    X_scaled = self.preprocessor.transform(features_df.values)
            else:
                # 如果没有transform方法，直接使用原始特征
                X_scaled = features_df.values
            
            # 预测
            prediction = self.model.predict(X_scaled)[0]
            
            # 解码标签
            if hasattr(self.preprocessor, 'inverse_transform_labels'):
                # 自定义的FeaturePreprocessor
                resolution = self.preprocessor.inverse_transform_labels([prediction])[0]
            elif hasattr(self.preprocessor, 'label_encoder'):
                # 有label_encoder属性
                resolution = self.preprocessor.label_encoder.inverse_transform([prediction])[0]
            else:
                # 直接映射
                label_map = {0: '480p', 1: '720p', 2: '1080p'}
                resolution = label_map.get(prediction, 'unknown')
            
            # 获取置信度（如果模型支持）
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)[0]
                confidence = max(probabilities)
            else:
                confidence = 1.0
            
            # 保存预测结果
            self.recent_predictions.append({
                'resolution': resolution,
                'confidence': confidence,
                'timestamp': time.time(),
                'throughput_mbps': features_dict['throughput_mbps'],
                'packet_count': features_dict['packet_count']
            })
            
            self.stats.predictions_made += 1
            
            # 清空窗口，准备下一个
            self.window.clear()
            
        except Exception as e:
            console.print(f"[red]预测错误: {e}[/red]")
    
    def _create_display_table(self) -> Table:
        """创建显示表格"""
        table = Table(title="🎥 实时视频质量监测", box=box.ROUNDED, show_header=True)
        
        table.add_column("指标", style="cyan", width=20)
        table.add_column("数值", style="green", width=30)
        
        # 统计信息
        elapsed = self.stats.elapsed()
        table.add_row("监测时长", f"{elapsed:.0f} 秒")
        table.add_row("捕获包数", f"{self.stats.total_packets:,}")
        table.add_row("TCP包数", f"{self.stats.tcp_packets:,}")
        table.add_row("视频包数", f"{self.stats.video_packets:,}")
        table.add_row("总流量", f"{self.stats.bytes_total / 1_000_000:.2f} MB")
        table.add_row("预测次数", f"{self.stats.predictions_made}")
        
        # 最近预测
        if self.recent_predictions:
            latest = self.recent_predictions[-1]
            resolution = latest['resolution']
            confidence = latest['confidence']
            
            # 根据分辨率设置颜色
            if resolution == '1080p':
                color = 'green'
            elif resolution == '720p':
                color = 'yellow'
            else:
                color = 'red'
            
            table.add_row(
                "当前质量",
                f"[{color}]{resolution}[/{color}] ({confidence:.1%})"
            )
            table.add_row(
                "当前吞吐",
                f"{latest['throughput_mbps']:.2f} Mbps"
            )
        
        return table
    
    def _create_history_panel(self) -> Panel:
        """创建预测历史面板"""
        if not self.recent_predictions:
            return Panel("暂无预测", title="预测历史", border_style="dim")
        
        history_text = ""
        for i, pred in enumerate(reversed(list(self.recent_predictions))):
            resolution = pred['resolution']
            confidence = pred['confidence']
            throughput = pred['throughput_mbps']
            
            # 选择颜色
            if resolution == '1080p':
                color = 'green'
            elif resolution == '720p':
                color = 'yellow'
            else:
                color = 'red'
            
            history_text += f"[{color}]{resolution}[/{color}] " \
                          f"({confidence:.1%}) | " \
                          f"{throughput:.1f} Mbps\n"
            
            if i >= 4:  # 只显示最近5条
                break
        
        return Panel(history_text.strip(), title="📊 最近预测", border_style="cyan")
    
    def start(self, duration: Optional[int] = None):
        """开始监测"""
        console.print(Panel.fit(
            f"[bold cyan]开始监测视频流量[/bold cyan]\n"
            f"网卡: {self.interface}\n"
            f"窗口: {self.window_size}秒\n"
            f"{'目标IP: ' + self.target_ip if self.target_ip else '所有视频流量'}",
            border_style="cyan"
        ))
        
        self.running = True
        self.stats.start_time = time.time()
        
        # 创建捕获过滤器
        capture_filter = 'tcp'  # 只捕获TCP包
        
        try:
            # 开始捕获
            console.print(f"\n[yellow]正在初始化数据包捕获...[/yellow]")
            console.print(f"[dim]使用过滤器: {capture_filter}[/dim]\n")
            
            capture = pyshark.LiveCapture(
                interface=self.interface,
                bpf_filter=capture_filter,
                output_file=self.save_pcap if self.save_pcap else None
            )
            
            # 使用Rich Live显示
            with Live(self._create_display_table(), refresh_per_second=2, console=console) as live:
                for packet in capture.sniff_continuously():
                    if not self.running:
                        break
                    
                    # 检查是否超时
                    if duration and self.stats.elapsed() >= duration:
                        break
                    
                    self.stats.total_packets += 1
                    
                    # 检查是否为TCP包
                    if hasattr(packet, 'tcp'):
                        self.stats.tcp_packets += 1
                        
                        # 检查是否为视频包
                        if self._is_video_packet(packet):
                            self.stats.video_packets += 1
                            self.stats.bytes_total += int(packet.length)
                            
                            # 提取包信息并添加到窗口
                            packet_info = self._extract_packet_info(packet)
                            if packet_info:
                                self.window.add_packet(packet_info)
                            
                            # 如果窗口准备好，进行预测
                            if self.window.is_ready():
                                self._make_prediction()
                    
                    # 更新显示
                    if self.stats.total_packets % 10 == 0:  # 每10个包更新一次显示
                        live.update(self._create_display_table())
            
            # 打印最终统计
            self._print_summary()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]监测被用户中断[/yellow]")
        except Exception as e:
            console.print(f"\n[red]监测错误: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
    
    def _print_summary(self):
        """打印监测总结"""
        console.print("\n" + "=" * 60)
        console.print("[bold]监测总结[/bold]")
        console.print("=" * 60)
        console.print(f"监测时长: {self.stats.elapsed():.1f} 秒")
        console.print(f"总包数: {self.stats.total_packets:,}")
        console.print(f"视频包数: {self.stats.video_packets:,}")
        console.print(f"总流量: {self.stats.bytes_total / 1_000_000:.2f} MB")
        console.print(f"预测次数: {self.stats.predictions_made}")
        
        if self.recent_predictions:
            console.print("\n[cyan]质量分布:[/cyan]")
            resolutions = [p['resolution'] for p in self.recent_predictions]
            for res in ['1080p', '720p', '480p']:
                count = resolutions.count(res)
                if count > 0:
                    console.print(f"  {res}: {count} 次 ({count/len(resolutions):.1%})")
        
        if self.save_pcap:
            console.print(f"\n[green]PCAP已保存: {self.save_pcap}[/green]")
        
        console.print("=" * 60)


def list_interfaces():
    """列出可用网卡"""
    try:
        import netifaces
        interfaces = netifaces.interfaces()
        
        console.print("\n[cyan]可用网卡:[/cyan]")
        for i, iface in enumerate(interfaces, 1):
            addrs = netifaces.ifaddresses(iface)
            ip = addrs[netifaces.AF_INET][0]['addr'] if netifaces.AF_INET in addrs else 'N/A'
            console.print(f"  {i}. {iface} ({ip})")
        console.print()
        
    except ImportError:
        console.print("[yellow]提示: 安装 netifaces 可显示更多信息 (pip install netifaces)[/yellow]")
        console.print("\n常见网卡名称:")
        console.print("  macOS: en0, en1")
        console.print("  Linux: eth0, wlan0")
        console.print("  Windows: 使用 '网络连接' 中显示的名称\n")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='宿主机实时视频质量监测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测并监测
  python scripts/realtime_capture_host.py --model models/xgboost_model.pkl
  
  # 指定网卡
  python scripts/realtime_capture_host.py --interface en0 --model models/xgboost_model.pkl
  
  # 监测特定IP的视频流量
  python scripts/realtime_capture_host.py --target-ip 142.250.185.78 --model models/xgboost_model.pkl
  
  # 保存捕获数据
  python scripts/realtime_capture_host.py --model models/xgboost_model.pkl --save-pcap capture.pcap
  
  # 列出可用网卡
  python scripts/realtime_capture_host.py --list-interfaces
"""
    )
    
    parser.add_argument(
        '--interface', '-i',
        type=str,
        help='网络接口名称（如: en0, eth0）'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='训练好的模型文件路径 (.pkl)'
    )
    parser.add_argument(
        '--preprocessor', '-p',
        type=str,
        default='models/preprocessor.pkl',
        help='预处理器文件路径 (默认: models/preprocessor.pkl)'
    )
    parser.add_argument(
        '--window-size', '-w',
        type=float,
        default=1.0,
        help='滑动窗口大小（秒）(默认: 1.0)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        help='监测时长（秒），不指定则持续监测'
    )
    parser.add_argument(
        '--target-ip',
        type=str,
        help='目标IP地址（只监测该IP的流量）'
    )
    parser.add_argument(
        '--save-pcap',
        type=str,
        help='保存捕获的PCAP文件'
    )
    parser.add_argument(
        '--list-interfaces',
        action='store_true',
        help='列出可用的网络接口'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 列出网卡
    if args.list_interfaces:
        list_interfaces()
        return 0
    
    # 检查必需参数
    if not args.model:
        console.print("[red]错误: 必须指定模型文件 (--model)[/red]")
        console.print("运行 --help 查看使用说明")
        return 1
    
    if not args.interface:
        console.print("[yellow]未指定网卡，尝试自动检测...[/yellow]")
        try:
            import netifaces
            interfaces = netifaces.interfaces()
            # 优先选择常见的网卡
            for candidate in ['en0', 'eth0', 'wlan0', 'Wi-Fi']:
                if candidate in interfaces:
                    args.interface = candidate
                    console.print(f"[green]自动选择网卡: {args.interface}[/green]")
                    break
            
            if not args.interface and interfaces:
                args.interface = interfaces[0]
                console.print(f"[yellow]使用第一个网卡: {args.interface}[/yellow]")
        except:
            pass
        
        if not args.interface:
            console.print("[red]错误: 无法自动检测网卡，请使用 --interface 指定[/red]")
            console.print("运行 --list-interfaces 查看可用网卡")
            return 1
    
    # 检查模型文件
    if not Path(args.model).exists():
        console.print(f"[red]错误: 模型文件不存在: {args.model}[/red]")
        return 1
    
    if not Path(args.preprocessor).exists():
        console.print(f"[red]错误: 预处理器文件不存在: {args.preprocessor}[/red]")
        return 1
    
    # 创建监测器
    try:
        monitor = HostMonitor(
            interface=args.interface,
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            window_size=args.window_size,
            target_ip=args.target_ip,
            save_pcap=args.save_pcap
        )
        
        # 开始监测
        monitor.start(duration=args.duration)
        
        return 0
        
    except PermissionError:
        console.print("\n[red]权限错误: 需要管理员权限捕获数据包[/red]")
        console.print("请使用 sudo 运行:")
        console.print(f"  sudo python3 {' '.join(sys.argv)}")
        return 1
    except Exception as e:
        console.print(f"\n[red]错误: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

