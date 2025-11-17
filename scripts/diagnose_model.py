#!/usr/bin/env python3
"""
诊断模型和训练数据问题
检查：
1. 模型文件是否正确加载
2. 训练数据的类别分布
3. 模型对不同输入的预测行为
"""

import pickle
import sys
from pathlib import Path
import numpy as np
import joblib
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def load_model_files(model_dir):
    """加载模型和预处理器"""
    model_path = model_dir / 'xgboost_model.pkl'
    preprocessor_path = model_dir / 'preprocessor.pkl'
    
    console.print(f"\n[cyan]检查模型文件...[/cyan]")
    console.print(f"模型路径: {model_path}")
    console.print(f"预处理器路径: {preprocessor_path}")
    
    if not model_path.exists():
        console.print(f"[red]✗ 模型文件不存在: {model_path}[/red]")
        return None, None
    
    if not preprocessor_path.exists():
        console.print(f"[red]✗ 预处理器文件不存在: {preprocessor_path}[/red]")
        return None, None
    
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载预处理器（使用 joblib，因为 FeaturePreprocessor 用 joblib 保存）
    try:
        preprocessor = joblib.load(preprocessor_path)
        console.print("[green]✓ 模型和预处理器加载成功[/green]")
    except Exception as e:
        console.print(f"[yellow]警告: joblib 加载失败，尝试 pickle: {e}[/yellow]")
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        console.print("[green]✓ 模型加载成功，预处理器用 pickle 加载[/green]")
    
    return model, preprocessor


def check_preprocessor(preprocessor):
    """检查预处理器内容"""
    console.print("\n[cyan]预处理器信息：[/cyan]")
    
    # 检查类型
    console.print(f"类型: {type(preprocessor)}")
    
    # 如果是字典
    if isinstance(preprocessor, dict):
        console.print("[yellow]预处理器是字典类型，包含以下键：[/yellow]")
        for key in preprocessor.keys():
            console.print(f"  - {key}: {type(preprocessor[key])}")
        
        # 检查label_encoder
        if 'label_encoder' in preprocessor:
            encoder = preprocessor['label_encoder']
            console.print(f"\n标签编码器类型: {type(encoder)}")
            if hasattr(encoder, 'classes_'):
                console.print(f"已知类别: {encoder.classes_}")
    
    # 如果是对象
    else:
        console.print(f"属性: {dir(preprocessor)}")
        
        # 检查label_encoder
        if hasattr(preprocessor, 'label_encoder'):
            encoder = preprocessor.label_encoder
            console.print(f"\n标签编码器类型: {type(encoder)}")
            if hasattr(encoder, 'classes_'):
                console.print(f"已知类别: {encoder.classes_}")


def check_training_data_distribution(experiments_dir):
    """检查训练数据中的分辨率分布"""
    console.print("\n[cyan]检查训练数据分布...[/cyan]")
    
    import json
    from collections import Counter
    
    resolutions = []
    exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith('exp_')]
    
    if not exp_dirs:
        console.print(f"[yellow]未找到实验数据目录: {experiments_dir}[/yellow]")
        return None
    
    console.print(f"找到 {len(exp_dirs)} 个实验目录")
    
    for exp_dir in exp_dirs:
        gt_file = exp_dir / 'ground_truth.json'
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                data = json.load(f)
                if 'video' in data and 'resolution' in data['video']:
                    resolutions.append(data['video']['resolution'])
    
    if not resolutions:
        console.print("[yellow]未找到任何分辨率数据[/yellow]")
        return None
    
    # 统计分布
    counter = Counter(resolutions)
    
    table = Table(title="训练数据分辨率分布")
    table.add_column("分辨率", style="cyan")
    table.add_column("数量", style="magenta")
    table.add_column("占比", style="green")
    
    total = len(resolutions)
    for resolution, count in counter.most_common():
        percentage = (count / total) * 100
        table.add_row(resolution, str(count), f"{percentage:.1f}%")
    
    console.print(table)
    
    # 警告
    if len(counter) == 1:
        console.print("\n[red]⚠️  警告：训练数据只有一个分辨率类别！[/red]")
        console.print("[yellow]模型无法学习区分不同分辨率。需要收集多种分辨率的训练数据。[/yellow]")
    elif max(counter.values()) / total > 0.8:
        console.print("\n[yellow]⚠️  警告：训练数据严重不平衡！[/yellow]")
        console.print(f"[yellow]主要类别占比超过80%，可能导致模型偏向预测该类别。[/yellow]")
    
    return counter


def test_model_predictions(model, preprocessor):
    """测试模型对不同输入的预测"""
    console.print("\n[cyan]测试模型预测行为...[/cyan]")
    
    # 创建测试样本（35个特征，模拟不同网络条件）
    # 特征顺序: TCP(10) + Traffic(15) + Temporal(10) = 35
    test_cases = [
        {
            "name": "高带宽低延迟",
            "features": [
                # TCP特征 (10个)
                1500, 1500, 1500, 100, 100, 10.0, 1.0, 0.0, 0.0, 0.95,
                # Traffic特征 (15个)
                10.0, 100, 100, 50, 50, 1500, 1500, 1000, 500, 5, 2, 1, 0.5, 0.8, 1.2,
                # Temporal特征 (10个)
                0.01, 0.02, 0.005, 100, 5, 2, 0.05, 0.1, 50, 10
            ]
        },
        {
            "name": "低带宽高延迟",
            "features": [
                # TCP特征 (10个)
                800, 800, 800, 50, 50, 5.0, 0.5, 0.05, 0.1, 0.7,
                # Traffic特征 (15个)
                1.0, 20, 20, 10, 10, 800, 800, 500, 300, 3, 1, 0, 0.3, 0.5, 0.8,
                # Temporal特征 (10个)
                0.05, 0.1, 0.02, 20, 3, 5, 0.2, 0.3, 15, 5
            ]
        },
        {
            "name": "中等网络",
            "features": [
                # TCP特征 (10个)
                1200, 1200, 1200, 70, 70, 7.0, 0.7, 0.02, 0.03, 0.8,
                # Traffic特征 (15个)
                5.0, 50, 50, 25, 25, 1200, 1200, 700, 400, 4, 1, 0, 0.4, 0.6, 1.0,
                # Temporal特征 (10个)
                0.02, 0.05, 0.01, 50, 4, 3, 0.1, 0.15, 30, 7
            ]
        },
        {
            "name": "极差网络",
            "features": [
                # TCP特征 (10个)
                500, 500, 500, 30, 30, 2.0, 0.3, 0.1, 0.2, 0.5,
                # Traffic特征 (15个)
                0.5, 10, 10, 5, 5, 500, 500, 300, 200, 2, 0, 0, 0.2, 0.3, 0.6,
                # Temporal特征 (10个)
                0.1, 0.2, 0.05, 10, 2, 8, 0.3, 0.5, 8, 3
            ]
        },
    ]
    
    table = Table(title="模型预测测试")
    table.add_column("测试场景", style="cyan")
    table.add_column("预测结果", style="magenta")
    table.add_column("预测概率", style="green")
    
    for case in test_cases:
        features = np.array([case['features']])
        
        try:
            # 应用预处理器的scaling（如果需要）
            if isinstance(preprocessor, dict) and 'scaler' in preprocessor:
                # 字典格式的preprocessor
                scaler = preprocessor['scaler']
                if scaler is not None:
                    features_scaled = scaler.transform(features)
                else:
                    features_scaled = features
            else:
                # 假设是FeaturePreprocessor对象或其他格式
                features_scaled = features
            
            # 尝试预测
            prediction = model.predict(features_scaled)[0]
            
            # 尝试获取概率
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                max_proba = f"{max(proba):.2%}"
            else:
                max_proba = "N/A"
            
            # 尝试反转标签
            if isinstance(preprocessor, dict) and 'label_encoder' in preprocessor:
                encoder = preprocessor['label_encoder']
                if encoder is not None and hasattr(encoder, 'inverse_transform'):
                    label = encoder.inverse_transform([int(prediction)])[0]
                else:
                    label = str(prediction)
            elif hasattr(preprocessor, 'inverse_transform_labels'):
                label = preprocessor.inverse_transform_labels([prediction])[0]
            elif hasattr(preprocessor, 'label_encoder'):
                label = preprocessor.label_encoder.inverse_transform([prediction])[0]
            else:
                # Fallback
                label_map = {0: '480p', 1: '720p', 2: '1080p'}
                label = label_map.get(int(prediction), str(prediction))
            
            table.add_row(case['name'], label, max_proba)
            
        except Exception as e:
            import traceback
            table.add_row(case['name'], f"[red]错误: {e}[/red]", "N/A")
            console.print(f"[dim]详细错误: {traceback.format_exc()}[/dim]")
    
    console.print(table)
    
    # 检查是否所有预测都相同
    predictions = []
    for case in test_cases:
        features = np.array([case['features']])
        try:
            prediction = model.predict(features)[0]
            predictions.append(prediction)
        except:
            pass
    
    if predictions and len(set(predictions)) == 1:
        console.print("\n[red]⚠️  所有测试样本预测结果相同！[/red]")
        console.print("[yellow]这表明：[/yellow]")
        console.print("[yellow]1. 训练数据可能只有一个类别[/yellow]")
        console.print("[yellow]2. 特征提取可能有问题[/yellow]")
        console.print("[yellow]3. 模型可能过拟合到单一类别[/yellow]")


def main():
    # 项目根目录
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'models'
    experiments_dir = project_root / 'experiments'
    
    console.print(Panel.fit(
        "[bold cyan]视频QoE模型诊断工具[/bold cyan]\n"
        "检查模型和训练数据的问题",
        border_style="cyan"
    ))
    
    # 1. 加载模型
    model, preprocessor = load_model_files(model_dir)
    if model is None:
        console.print("\n[red]无法继续诊断，请先训练模型或检查模型文件[/red]")
        return 1
    
    # 2. 检查预处理器
    check_preprocessor(preprocessor)
    
    # 3. 检查训练数据分布
    distribution = check_training_data_distribution(experiments_dir)
    
    # 4. 测试模型预测
    test_model_predictions(model, preprocessor)
    
    # 总结
    console.print("\n" + "="*60)
    console.print("[bold cyan]诊断建议：[/bold cyan]")
    
    if distribution and len(distribution) == 1:
        console.print("\n[yellow]主要问题：训练数据只有单一分辨率！[/yellow]")
        console.print("\n[green]解决方案：[/green]")
        console.print("1. 重新运行 auto_train_pipeline.py，确保包含多种分辨率")
        console.print("2. 检查 config/experiments/basic.yaml 中的 target_resolutions 配置")
        console.print("3. 手动运行不同分辨率的实验：")
        console.print("   - 修改视频源分辨率")
        console.print("   - 调整网络参数（带宽、延迟）")
        console.print("   - 确保收集各种场景的数据")
    elif distribution and max(distribution.values()) / sum(distribution.values()) > 0.8:
        console.print("\n[yellow]主要问题：训练数据严重不平衡！[/yellow]")
        console.print("\n[green]解决方案：[/green]")
        console.print("1. 增加少数类别的样本")
        console.print("2. 使用类别权重平衡训练")
        console.print("3. 过采样少数类别或欠采样多数类别")
    else:
        console.print("\n[green]训练数据分布看起来合理。[/green]")
        console.print("[yellow]如果预测仍然不准，可能需要：[/yellow]")
        console.print("1. 检查特征工程是否合理")
        console.print("2. 调整模型超参数")
        console.print("3. 增加训练数据量")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

