#!/usr/bin/env python3
"""
模型详细评估脚本

功能：
1. 加载训练好的模型和预处理器
2. 在测试集上评估性能
3. 生成详细的评估报告（包括各类别准确率）
4. 可视化混淆矩阵和性能指标
5. 输出分类报告

Usage:
    python3 scripts/evaluate_model.py
    python3 scripts/evaluate_model.py --model-path models/xgboost_model.pkl
    python3 scripts/evaluate_model.py --experiments-dir experiments --detailed
"""

import sys
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_qoe.training.data_loader import ExperimentDataLoader

console = Console()


def load_model_and_preprocessor(model_path: Path, preprocessor_path: Path) -> Tuple:
    """加载模型和预处理器"""
    console.print(f"\n[cyan]加载模型和预处理器...[/cyan]")
    
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    console.print(f"  ✓ 模型: {model_path}")
    
    # 加载预处理器
    preprocessor = joblib.load(preprocessor_path)
    console.print(f"  ✓ 预处理器: {preprocessor_path}")
    
    return model, preprocessor


def evaluate_on_dataset(model, X: np.ndarray, y: np.ndarray, 
                        preprocessor: Dict, dataset_name: str = "Test") -> Dict:
    """在数据集上评估模型"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    
    console.print(f"\n[cyan]在{dataset_name}集上评估...[/cyan]")
    
    # 特征缩放（仅在preprocessor不为None时进行）
    if preprocessor is not None:
        scaler = preprocessor.get('scaler')
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
    else:
        # 如果preprocessor为None，说明数据已经预处理过了
        X_scaled = X
    
    # 预测
    y_pred = model.predict(X_scaled)
    
    # 计算指标
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, average='weighted', zero_division=0
    )
    
    # 各类别指标
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y, y_pred, average=None, zero_division=0)
    
    # 标签映射
    if preprocessor is not None:
        label_encoder = preprocessor.get('label_encoder')
        if label_encoder:
            class_names = label_encoder.classes_
        else:
            class_names = ['1080p', '480p', '720p']
    else:
        class_names = ['1080p', '480p', '720p']
    
    console.print(f"  ✓ {dataset_name}准确率: {accuracy:.4f}")
    console.print(f"  ✓ 加权F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'per_class': {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
            'support': support_per_class
        },
        'class_names': class_names,
        'y_true': y,
        'y_pred': y_pred
    }


def print_dataset_statistics(df: pd.DataFrame):
    """打印数据集统计信息"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]数据集统计信息[/bold cyan]",
        border_style="cyan"
    ))
    
    table = Table(box=box.ROUNDED)
    table.add_column("指标", style="cyan")
    table.add_column("数值", style="green")
    
    table.add_row("总样本数", str(len(df)))
    table.add_row("特征数", str(len(df.columns) - 1))  # 减去标签列
    
    # 各分辨率样本数
    if 'actual_resolution' in df.columns:
        resolution_counts = df['actual_resolution'].value_counts().sort_index()
        for resolution, count in resolution_counts.items():
            percentage = (count / len(df)) * 100
            table.add_row(f"  - {resolution}", f"{count} ({percentage:.1f}%)")
    
    console.print(table)


def print_split_statistics(data_split):
    """打印训练/验证/测试集统计"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]数据集划分[/bold cyan]",
        border_style="cyan"
    ))
    
    total = len(data_split.X_train) + len(data_split.X_val) + len(data_split.X_test)
    
    table = Table(box=box.ROUNDED)
    table.add_column("数据集", style="cyan")
    table.add_column("样本数", style="green")
    table.add_column("占比", style="yellow")
    
    train_pct = (len(data_split.X_train) / total) * 100
    val_pct = (len(data_split.X_val) / total) * 100
    test_pct = (len(data_split.X_test) / total) * 100
    
    table.add_row("训练集", str(len(data_split.X_train)), f"{train_pct:.1f}%")
    table.add_row("验证集", str(len(data_split.X_val)), f"{val_pct:.1f}%")
    table.add_row("测试集", str(len(data_split.X_test)), f"{test_pct:.1f}%")
    table.add_row("[bold]总计[/bold]", f"[bold]{total}[/bold]", "[bold]100.0%[/bold]")
    
    console.print(table)


def print_evaluation_results(results: Dict):
    """打印评估结果"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]模型性能评估[/bold cyan]",
        border_style="cyan"
    ))
    
    # 总体指标
    table = Table(title="总体指标", box=box.ROUNDED)
    table.add_column("指标", style="cyan")
    table.add_column("数值", style="green")
    
    table.add_row("准确率 (Accuracy)", f"{results['accuracy']:.4f}")
    table.add_row("精确率 (Precision)", f"{results['precision']:.4f}")
    table.add_row("召回率 (Recall)", f"{results['recall']:.4f}")
    table.add_row("F1-Score", f"{results['f1']:.4f}")
    
    console.print(table)
    
    # 各类别指标
    console.print()
    table2 = Table(title="各类别性能", box=box.ROUNDED)
    table2.add_column("分辨率", style="cyan")
    table2.add_column("样本数", style="white")
    table2.add_column("精确率", style="green")
    table2.add_column("召回率", style="yellow")
    table2.add_column("F1-Score", style="magenta")
    
    per_class = results['per_class']
    class_names = results['class_names']
    
    for i, class_name in enumerate(class_names):
        table2.add_row(
            class_name,
            str(int(per_class['support'][i])),
            f"{per_class['precision'][i]:.4f}",
            f"{per_class['recall'][i]:.4f}",
            f"{per_class['f1'][i]:.4f}"
        )
    
    console.print(table2)


def print_confusion_matrix(results: Dict):
    """打印混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]混淆矩阵[/bold cyan]",
        border_style="cyan"
    ))
    
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    class_names = results['class_names']
    
    # 创建表格
    table = Table(box=box.ROUNDED)
    table.add_column("真实\\预测", style="cyan")
    for name in class_names:
        table.add_column(name, style="white", justify="right")
    table.add_column("总计", style="yellow", justify="right")
    
    # 添加数据行
    for i, true_class in enumerate(class_names):
        row = [true_class]
        row_sum = 0
        for j in range(len(class_names)):
            count = cm[i, j]
            row_sum += count
            # 高亮对角线（正确预测）
            if i == j:
                row.append(f"[green bold]{count}[/green bold]")
            else:
                row.append(str(count))
        row.append(str(row_sum))
        table.add_row(*row)
    
    # 添加总计行
    totals = ["[bold]总计[/bold]"]
    for j in range(len(class_names)):
        col_sum = cm[:, j].sum()
        totals.append(f"[bold]{col_sum}[/bold]")
    totals.append(f"[bold]{cm.sum()}[/bold]")
    table.add_row(*totals)
    
    console.print(table)


def print_classification_report(results: Dict):
    """打印分类报告"""
    from sklearn.metrics import classification_report
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]详细分类报告[/bold cyan]",
        border_style="cyan"
    ))
    
    report = classification_report(
        results['y_true'],
        results['y_pred'],
        target_names=results['class_names'],
        digits=4
    )
    
    console.print(f"[white]{report}[/white]")


def main():
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/xgboost_model.pkl',
        help='模型文件路径'
    )
    parser.add_argument(
        '--preprocessor-path',
        type=str,
        default='models/preprocessor.pkl',
        help='预处理器文件路径'
    )
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='experiments',
        help='实验数据目录'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='显示详细的分类报告'
    )
    
    args = parser.parse_args()
    
    # 打印标题
    console.print()
    console.print(Panel.fit(
        "[bold cyan]视频QoE模型评估工具[/bold cyan]\n"
        "[white]详细评估模型在测试集上的性能[/white]",
        border_style="cyan",
        padding=(1, 2)
    ))
    
    try:
        # 1. 加载模型和预处理器
        model_path = Path(args.model_path)
        preprocessor_path = Path(args.preprocessor_path)
        
        if not model_path.exists():
            console.print(f"[red]✗ 模型文件不存在: {model_path}[/red]")
            return 1
        
        if not preprocessor_path.exists():
            console.print(f"[red]✗ 预处理器文件不存在: {preprocessor_path}[/red]")
            return 1
        
        model, preprocessor = load_model_and_preprocessor(model_path, preprocessor_path)
        
        # 2. 加载数据
        console.print(f"\n[cyan]加载训练数据...[/cyan]")
        experiments_dir = Path(args.experiments_dir)
        
        # 使用 ExperimentDataLoader 加载数据
        loader = ExperimentDataLoader(experiments_dir)
        df = loader.load_experiments()
        
        if df is None or len(df) == 0:
            console.print(f"[red]✗ 未找到有效的训练数据[/red]")
            console.print(f"[yellow]提示: 请先运行 auto_train_pipeline.py 生成训练数据[/yellow]")
            return 1
        
        console.print(f"  ✓ 加载了 {len(df)} 个样本，来自 {loader.valid_count} 个实验")
        
        # 3. 打印数据集统计
        print_dataset_statistics(df)
        
        # 4. 数据预处理和分割
        console.print(f"\n[cyan]数据预处理和分割...[/cyan]")
        from video_qoe.training.preprocessor import FeaturePreprocessor
        
        temp_preprocessor = FeaturePreprocessor()
        data_split = temp_preprocessor.fit_transform(
            df,
            target_col='actual_resolution',
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
        console.print(f"  ✓ 数据分割完成")
        
        # 5. 打印分割统计
        print_split_statistics(data_split)
        
        # 6. 在测试集上评估
        # 注意：data_split中的数据已经被temp_preprocessor预处理过了，不需要再次预处理
        test_results = evaluate_on_dataset(
            model, data_split.X_test, data_split.y_test,
            None, "测试"  # 传None避免双重预处理
        )
        
        # 7. 在验证集上评估（可选）
        val_results = evaluate_on_dataset(
            model, data_split.X_val, data_split.y_val,
            None, "验证"  # 传None避免双重预处理
        )
        
        # 8. 打印评估结果
        console.print("\n")
        console.print("=" * 80)
        console.print("[bold]测试集性能[/bold]")
        console.print("=" * 80)
        print_evaluation_results(test_results)
        print_confusion_matrix(test_results)
        
        if args.detailed:
            print_classification_report(test_results)
        
        # 9. 验证集性能（如果需要）
        console.print("\n")
        console.print("=" * 80)
        console.print("[bold]验证集性能[/bold]")
        console.print("=" * 80)
        print_evaluation_results(val_results)
        print_confusion_matrix(val_results)
        
        # 10. 总结
        console.print("\n")
        console.print(Panel.fit(
            f"[bold green]✓ 评估完成[/bold green]\n\n"
            f"测试集准确率: {test_results['accuracy']:.4f}\n"
            f"验证集准确率: {val_results['accuracy']:.4f}\n\n"
            f"测试集F1-Score: {test_results['f1']:.4f}\n"
            f"验证集F1-Score: {val_results['f1']:.4f}",
            border_style="green",
            padding=(1, 2)
        ))
        
        return 0
        
    except Exception as e:
        console.print(f"\n[red]✗ 评估失败: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1


if __name__ == '__main__':
    sys.exit(main())

