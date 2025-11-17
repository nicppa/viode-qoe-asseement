#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Training Pipeline for Video QoE Assessment
自动化训练流水线 - 在VM中自动生成训练数据并训练模型

This script automates the complete pipeline:
1. Run Mininet experiments to collect training data
2. Train machine learning models
3. Save pretrained models for production use

Usage:
    # 完整流水线（需要约2-4小时）
    sudo python3 scripts/auto_train_pipeline.py --samples 10 --duration 60
    
    # 快速测试（约30分钟）
    sudo python3 scripts/auto_train_pipeline.py --samples 2 --duration 30 --quick
    
    # 仅训练模型（使用已有数据）
    sudo python3 scripts/auto_train_pipeline.py --train-only

Author: Video QoE Assessment System
Date: 2025-11-15
"""

import argparse
import sys
import os
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Fix encoding issues
if sys.stdout.encoding != 'UTF-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Fix asyncio event loop warning
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*coroutine.*')

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_qoe.experiment import ExperimentManager
from video_qoe.monitoring import RealTimePipeline
from video_qoe.utils.logger import get_logger
from video_qoe.capture import PCAPReader, PacketInfo, SlidingWindowBuffer
from video_qoe.features import FeatureCalculator
from video_qoe.output import DataWriter

console = Console()
logger = get_logger('auto_train_pipeline')

# 网络场景配置
SCENARIOS = {
    'low-bandwidth': {
        'bandwidth': 2,    # Mbps
        'latency': 100,    # ms
        'loss': 0.05,      # 5%
        'jitter': 10       # ms
    },
    'mobile-4g': {
        'bandwidth': 5,
        'latency': 50,
        'loss': 0.02,
        'jitter': 5
    },
    'wifi': {
        'bandwidth': 10,
        'latency': 30,
        'loss': 0.01,
        'jitter': 3
    },
    'high-quality': {
        'bandwidth': 20,
        'latency': 10,
        'loss': 0.001,
        'jitter': 1
    },
    'congested': {
        'bandwidth': 3,
        'latency': 150,
        'loss': 0.10,
        'jitter': 20
    },
    'stable': {
        'bandwidth': 15,
        'latency': 20,
        'loss': 0.005,
        'jitter': 2
    }
}

RESOLUTIONS = ['480p', '720p', '1080p']


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='自动化训练流水线 - 自动生成数据并训练模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流水线（推荐）
  sudo python3 scripts/auto_train_pipeline.py --samples 10 --duration 60
  
  # 快速测试
  sudo python3 scripts/auto_train_pipeline.py --samples 2 --duration 30 --quick
  
  # 仅训练（使用已有数据）
  python3 scripts/auto_train_pipeline.py --train-only
  
  # 自定义场景
  sudo python3 scripts/auto_train_pipeline.py --scenarios low-bandwidth mobile-4g --samples 5
"""
    )
    
    # 数据收集参数
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='每个分辨率/场景组合的样本数 (默认: 10)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='每个实验的持续时间（秒）(默认: 60)'
    )
    parser.add_argument(
        '--resolutions',
        nargs='+',
        choices=RESOLUTIONS,
        default=RESOLUTIONS,
        help='要测试的分辨率 (默认: 全部)'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        choices=list(SCENARIOS.keys()),
        default=list(SCENARIOS.keys()),
        help='要测试的网络场景 (默认: 全部)'
    )
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='experiments/',
        help='实验数据输出目录 (默认: experiments/)'
    )
    
    # 训练参数
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models/',
        help='模型输出目录 (默认: models/)'
    )
    parser.add_argument(
        '--model-types',
        nargs='+',
        choices=['xgboost', 'random_forest'],
        default=['xgboost'],
        help='要训练的模型类型 (默认: xgboost)'
    )
    
    # 流程控制
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='仅训练模型（跳过数据收集）'
    )
    parser.add_argument(
        '--collect-only',
        action='store_true',
        help='仅收集数据（跳过训练）'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速模式：使用较少场景和样本'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='模拟运行，不实际执行'
    )
    
    return parser.parse_args()


def apply_quick_mode(args: argparse.Namespace) -> argparse.Namespace:
    """应用快速模式设置"""
    if args.quick:
        console.print("[yellow]快速模式已启用[/yellow]")
        args.samples = min(args.samples, 2)
        args.duration = min(args.duration, 30)
        args.scenarios = ['low-bandwidth', 'mobile-4g', 'wifi']
        console.print(f"  - 样本数: {args.samples}")
        console.print(f"  - 时长: {args.duration}s")
        console.print(f"  - 场景: {', '.join(args.scenarios)}")
    return args


def print_pipeline_plan(args: argparse.Namespace):
    """打印流水线计划"""
    console.print(Panel.fit(
        "[bold cyan]自动化训练流水线[/bold cyan]\n"
        "VM中自动生成训练数据并训练模型",
        border_style="cyan"
    ))
    
    table = Table(title="执行计划", show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("阶段", style="cyan", width=15)
    table.add_column("详情", style="green")
    
    if not args.train_only:
        total_experiments = len(args.resolutions) * len(args.scenarios) * args.samples
        total_time_min = (total_experiments * args.duration) / 60
        
        table.add_row(
            "1️⃣  数据收集",
            f"分辨率: {', '.join(args.resolutions)}\n"
            f"场景: {', '.join(args.scenarios)}\n"
            f"样本/组合: {args.samples}\n"
            f"实验时长: {args.duration}s\n"
            f"[bold]总实验数: {total_experiments}[/bold]\n"
            f"[bold]预计时间: {total_time_min:.1f} 分钟[/bold]"
        )
    
    if not args.collect_only:
        table.add_row(
            "2️⃣  模型训练",
            f"模型类型: {', '.join(args.model_types)}\n"
            f"实验目录: {args.experiments_dir}\n"
            f"模型目录: {args.models_dir}\n"
            f"[bold]预计时间: 5-15 分钟[/bold]"
        )
    
    console.print(table)


def extract_and_save_features(context, resolution: str, logger):
    """从PCAP提取特征并保存为features.csv
    
    Args:
        context: ExperimentContext对象
        resolution: 实际分辨率
        logger: 日志记录器
    """
    pcap_path = context.pcap_path
    exp_dir = context.exp_dir
    
    console.print(f"  [dim]Reading PCAP: {pcap_path.name}...[/dim]")
    
    # 读取PCAP
    reader = PCAPReader(pcap_path)
    packets = []
    
    try:
        for pkt in reader.read_all():
            try:
                packet_info = PacketInfo.from_pyshark_packet(pkt)
                if packet_info:
                    packets.append(packet_info)
            except Exception as e:
                # 跳过无法解析的包
                continue
        
        console.print(f"  [dim]Parsed {len(packets)} packets[/dim]")
        
    except Exception as e:
        logger.warning(f"Error reading PCAP: {e}")
        console.print(f"  [yellow]⚠ PCAP read error: {e}[/yellow]")
    
    if not packets:
        logger.warning("No valid packets found in PCAP")
        console.print(f"  [yellow]⚠ No valid packets, skipping feature extraction[/yellow]")
        return
    
    # 提取特征
    console.print(f"  [dim]Extracting features...[/dim]")
    window = SlidingWindowBuffer(window_size=1.0)
    calculator = FeatureCalculator()
    data_writer = DataWriter(exp_dir, logger=logger)
    
    feature_count = 0
    for pkt in packets:
        window.add_packet(pkt)
        
        if window.should_process():
            window_packets = window.get_window_packets()
            if window_packets:
                try:
                    features = calculator.compute_all_features(window_packets, client_ip=context.client_ip)
                    
                    # 创建虚拟预测（因为这是训练数据）
                    from video_qoe.prediction.predictor import Prediction
                    prediction = Prediction(
                        resolution=resolution,  # 使用实际分辨率
                        confidence=1.0,
                        method='ground_truth'
                    )
                    
                    # 保存特征
                    data_writer.append_data(
                        elapsed=pkt.timestamp - packets[0].timestamp,
                        prediction=prediction,
                        features=features
                    )
                    feature_count += 1
                    
                except Exception as e:
                    logger.debug(f"Error computing features: {e}")
                    continue
    
    # 完成
    data_writer.finalize()
    console.print(f"  [dim]Saved {feature_count} feature samples[/dim]")


def run_single_experiment(resolution: str, scenario: str, scenario_config: Dict,
                         duration: int, experiment_dir: Path, run_id: int) -> bool:
    """运行单个Mininet实验收集数据
    
    Args:
        resolution: 目标分辨率
        scenario: 场景名称
        scenario_config: 网络配置
        duration: 实验时长
        experiment_dir: 实验输出目录
        run_id: 运行ID
        
    Returns:
        True表示成功，False表示失败
    """
    try:
        console.print(f"Starting experiment: {resolution} - {scenario} (run {run_id})")
        
        # 创建实验管理器
        exp_manager = ExperimentManager(logger=logger)
        
        # 设置实验（会自动配置网络条件）
        context = exp_manager.setup_experiment(
            scenario_name=scenario
        )
        
        # 记录目标分辨率到Ground Truth
        if exp_manager.ground_truth:
            exp_manager.ground_truth.video.actual_resolution = resolution
            console.print(f"Ground truth resolution set to: {resolution}")
        
        if not context:
            logger.error("Failed to setup experiment")
            return False
        
        # 获取节点
        h2 = exp_manager.h2
        h1 = exp_manager.h1
        
        # 根据分辨率选择对应的视频文件
        video_dir = '/home/mininet/cn/video'
        video_file = f'video_{resolution}.mp4'
        video_path = f'{video_dir}/{video_file}'
        
        # 检查视频文件是否存在
        check_result = h1.cmd(f'ls {video_path} 2>/dev/null')
        
        if check_result.strip():
            # 使用真实视频文件
            download_file = video_file
            console.print(f"✓ Using video: {video_path} for {resolution}")
        else:
            # 备用：创建测试文件
            logger.warning(f"Video file not found: {video_path}")
            console.print("Creating fallback test file...")
            h1.cmd('mkdir -p /tmp/webserver')
            h1.cmd('dd if=/dev/zero of=/tmp/webserver/test.dat bs=1M count=2 2>/dev/null')
            video_dir = '/tmp/webserver'
            download_file = 'test.dat'
            console.print(f"Using fallback: {video_dir}/{download_file}")
        
        # 创建实时监测流水线
        pipeline = RealTimePipeline(
            interface=context.capture_interface,
            pcap_path=str(context.pcap_path),
            client_ip=context.client_ip,
            node=h2,
            window_size=1.0,
            predictor_type='rule_based',
            output_color=False,
            capture_mode=True
        )
        
        # 运行流水线并生成流量
        with pipeline:
            # 启动持续视频下载（后台）
            console.print(f"Starting video downloads: {download_file}")
            
            # 启动多个并发下载以生成真实流量
            for i in range(3):
                h2.cmd(f'while true; do curl -s http://{context.server_ip}:{context.server_port}/{download_file} > /dev/null 2>&1; sleep 0.3; done &')
            
            # 等待流量稳定
            time.sleep(2)
            console.print("Traffic generation started, monitoring...")
            
            # 运行监测
            stats = pipeline.run(duration=duration)
        
        # 清理实验
        exp_manager.cleanup()
        
        console.print(f"Experiment completed: {stats.predictions_made} predictions")
        
        # NOTE: 跳过实时特征提取以加快数据收集
        # 特征将在训练阶段统一从PCAP提取
        console.print("[dim]PCAP saved, features will be extracted during training[/dim]")
        
        return True
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        try:
            exp_manager.cleanup()
        except:
            pass
        return False


def collect_training_data(args: argparse.Namespace) -> Tuple[bool, int, int]:
    """收集训练数据
    
    Returns:
        (success, total, failed) - 成功标志、总数、失败数
    """
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]阶段 1/2: 收集训练数据[/bold cyan]")
    console.print("=" * 80 + "\n")
    
    experiment_dir = Path(args.experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    total_experiments = len(args.resolutions) * len(args.scenarios) * args.samples
    successful = 0
    failed = 0
    
    console.print(f"将运行 {total_experiments} 个实验...\n")
    
    if args.dry_run:
        console.print("[yellow]模拟运行模式 - 跳过实际执行[/yellow]\n")
        return True, total_experiments, 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("执行实验中", total=total_experiments)
        
        for resolution in args.resolutions:
            for scenario in args.scenarios:
                scenario_config = SCENARIOS[scenario]
                
                console.print(f"\n[cyan]📊 {resolution} - {scenario}[/cyan]")
                console.print(f"   网络: BW={scenario_config['bandwidth']}Mbps, "
                            f"延迟={scenario_config['latency']}ms, "
                            f"丢包={scenario_config['loss']*100:.1f}%")
                
                for run_id in range(1, args.samples + 1):
                    try:
                        success = run_single_experiment(
                            resolution, scenario, scenario_config,
                            args.duration, experiment_dir, run_id
                        )
                        
                        if success:
                            successful += 1
                            console.print(f"   ✓ 运行 {run_id}/{args.samples} 成功")
                        else:
                            failed += 1
                            console.print(f"   ✗ 运行 {run_id}/{args.samples} 失败")
                            
                    except KeyboardInterrupt:
                        console.print("\n[yellow]用户中断[/yellow]")
                        raise
                    except Exception as e:
                        console.print(f"   ✗ 运行 {run_id}/{args.samples} 错误: {e}")
                        failed += 1
                    
                    progress.update(task, advance=1)
                    
                    # 实验间短暂延迟
                    time.sleep(2)
    
    # 打印汇总
    console.print("\n" + "=" * 80)
    console.print("[bold]数据收集汇总[/bold]")
    console.print("=" * 80)
    console.print(f"总实验数: {total_experiments}")
    console.print(f"[green]成功: {successful}[/green]")
    if failed > 0:
        console.print(f"[red]失败: {failed}[/red]")
    console.print(f"输出目录: {experiment_dir}")
    console.print("=" * 80)
    
    return failed == 0, total_experiments, failed


def train_models(args: argparse.Namespace) -> bool:
    """训练机器学习模型
    
    Returns:
        True表示成功，False表示失败
    """
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]阶段 2/2: 训练机器学习模型[/bold cyan]")
    console.print("=" * 80 + "\n")
    
    if args.dry_run:
        console.print("[yellow]模拟运行模式 - 跳过实际执行[/yellow]\n")
        return True
    
    # 首先批量提取特征
    console.print("[cyan]步骤 2.1: 从PCAP批量提取特征[/cyan]\n")
    extract_cmd = [
        sys.executable,
        'scripts/extract_features_from_pcap.py',
        '--experiments-dir', args.experiments_dir
    ]
    
    try:
        result = subprocess.run(extract_cmd, cwd=project_root, check=True)
        console.print("[green]✓ 特征提取完成[/green]\n")
    except subprocess.CalledProcessError:
        console.print("[yellow]⚠ 特征提取失败，将尝试继续训练[/yellow]\n")
    except FileNotFoundError:
        console.print("[yellow]⚠ 特征提取脚本未找到，跳过[/yellow]\n")
    
    console.print("[cyan]步骤 2.2: 训练模型[/cyan]\n")
    
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    all_success = True
    
    for model_type in args.model_types:
        console.print(f"\n[cyan]🤖 训练 {model_type} 模型...[/cyan]\n")
        
        try:
            # 构建训练命令
            cmd = [
                sys.executable,
                'scripts/train_model.py',
                '--experiments-dir', args.experiments_dir,
                '--output-dir', args.models_dir,
                '--model-type', model_type,
                '--class-names', '480p', '720p', '1080p'
            ]
            
            # 执行训练
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                console.print(f"[green]✓ {model_type} 模型训练成功[/green]")
            else:
                console.print(f"[red]✗ {model_type} 模型训练失败[/red]")
                all_success = False
                
        except Exception as e:
            console.print(f"[red]✗ {model_type} 模型训练错误: {e}[/red]")
            all_success = False
    
    return all_success


def save_pipeline_metadata(args: argparse.Namespace, data_success: bool, 
                          train_success: bool, duration: float):
    """保存流水线元数据"""
    metadata = {
        'pipeline_version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': duration,
        'configuration': {
            'samples': args.samples,
            'experiment_duration': args.duration,
            'resolutions': args.resolutions,
            'scenarios': args.scenarios,
            'model_types': args.model_types,
        },
        'results': {
            'data_collection_success': data_success,
            'training_success': train_success,
        },
        'output': {
            'experiments_dir': args.experiments_dir,
            'models_dir': args.models_dir,
        }
    }
    
    metadata_path = Path(args.models_dir) / 'pipeline_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[dim]元数据已保存: {metadata_path}[/dim]")


def main():
    """主函数"""
    args = parse_arguments()
    args = apply_quick_mode(args)
    
    # 打印计划
    print_pipeline_plan(args)
    
    # 确认
    if not args.dry_run:
        console.print("\n[yellow]此流水线将占用大量时间和资源[/yellow]")
        if not args.train_only:
            console.print("[yellow]需要 sudo 权限运行 Mininet 实验[/yellow]")
        response = input("\n继续? (y/n): ")
        if response.lower() != 'y':
            console.print("[yellow]已取消[/yellow]")
            return 0
    
    start_time = time.time()
    
    # 执行流水线
    data_success = True
    train_success = True
    
    try:
        # 阶段1: 收集数据
        if not args.train_only:
            data_success, total, failed = collect_training_data(args)
            
            if not data_success:
                console.print("\n[red]⚠ 数据收集过程中有失败的实验[/red]")
                if failed > total * 0.3:  # 超过30%失败
                    console.print("[red]失败率过高，建议检查后重试[/red]")
                    if not args.collect_only:
                        response = input("是否继续训练模型? (y/n): ")
                        if response.lower() != 'y':
                            return 1
        
        # 阶段2: 训练模型
        if not args.collect_only:
            train_success = train_models(args)
        
        duration = time.time() - start_time
        
        # 保存元数据
        if not args.dry_run:
            save_pipeline_metadata(args, data_success, train_success, duration)
        
        # 最终汇总
        console.print("\n" + "=" * 80)
        console.print("[bold]🎉 流水线执行完成[/bold]")
        console.print("=" * 80)
        console.print(f"总耗时: {duration/60:.1f} 分钟")
        
        if not args.train_only:
            console.print(f"实验数据: {args.experiments_dir}")
        
        if not args.collect_only:
            console.print(f"训练模型: {args.models_dir}")
            console.print("\n[cyan]📦 可用模型:[/cyan]")
            models_dir = Path(args.models_dir)
            for model_file in models_dir.glob('*.pkl'):
                if 'preprocessor' not in model_file.name:
                    console.print(f"  - {model_file.name}")
        
        console.print("\n[bold green]✓ 所有任务完成！[/bold green]")
        console.print("\n[cyan]下一步:[/cyan]")
        console.print("  1. 在宿主机上使用模型:")
        console.print("     python scripts/realtime_capture_host.py --model models/xgboost_model.pkl")
        console.print("\n  2. 测试模型准确性:")
        console.print("     python scripts/test_model.py --model models/xgboost_model.pkl")
        
        return 0 if (data_success and train_success) else 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]流水线被用户中断[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]流水线执行错误: {e}[/red]")
        logger.exception("Pipeline execution error")
        return 1


if __name__ == '__main__':
    sys.exit(main())

