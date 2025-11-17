# -*- coding: utf-8 -*-
"""
模型评估器

提供模型性能评估功能：
- 评估指标计算（准确率、精确率、召回率、F1等）
- 混淆矩阵生成
- 特征重要性分析
- 可视化图表生成
- Markdown评估报告生成
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


@dataclass
class EvaluationResult:
    """评估结果
    
    Attributes:
        model_type: 模型类型
        accuracy: 准确率
        precision: 精确率（加权平均）
        recall: 召回率（加权平均）
        f1_score: F1分数（加权平均）
        confusion_matrix: 混淆矩阵
        per_class_metrics: 每个类别的详细指标
        feature_importances: 特征重要性（可选）
        test_samples: 测试样本数
        n_classes: 类别数量
        class_names: 类别名称列表
        timestamp: 评估时间戳
    """
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    per_class_metrics: Dict[str, Dict[str, float]]
    test_samples: int
    n_classes: int
    class_names: List[str]
    feature_importances: Optional[Dict[str, float]] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        # 转换numpy数组为列表以便序列化
        if isinstance(self.confusion_matrix, np.ndarray):
            self._cm_array = self.confusion_matrix
            self.confusion_matrix = self.confusion_matrix.tolist()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于JSON序列化）"""
        result = asdict(self)
        # 确保confusion_matrix是列表
        if isinstance(result['confusion_matrix'], np.ndarray):
            result['confusion_matrix'] = result['confusion_matrix'].tolist()
        return result
    
    def summary(self) -> str:
        """返回评估结果摘要"""
        lines = [
            f"Evaluation Result - {self.model_type}",
            "=" * 60,
            f"Overall Performance:",
            f"  Accuracy:  {self.accuracy:.4f}",
            f"  Precision: {self.precision:.4f}",
            f"  Recall:    {self.recall:.4f}",
            f"  F1-Score:  {self.f1_score:.4f}",
            f"",
            f"Test Samples: {self.test_samples}",
            f"Classes: {self.n_classes} ({', '.join(self.class_names)})",
            f"Timestamp: {self.timestamp}",
        ]
        
        if self.feature_importances:
            lines.extend([
                "",
                f"Feature Importances: {len(self.feature_importances)} features"
            ])
        
        return "\n".join(lines)


class ModelEvaluator:
    """模型评估器
    
    提供完整的模型评估功能：
    1. 计算评估指标（准确率、精确率、召回率、F1）
    2. 生成混淆矩阵
    3. 分析每个类别的性能
    4. 提取特征重要性
    5. 生成可视化图表
    6. 生成Markdown评估报告
    
    Attributes:
        logger: 日志记录器
        
    Example:
        >>> evaluator = ModelEvaluator()
        >>> result = evaluator.evaluate(model, X_test, y_test, class_names=['480p', '720p', '1080p'])
        >>> evaluator.plot_confusion_matrix(result, 'output/cm.png')
        >>> evaluator.plot_feature_importance(result, feature_names, 'output/fi.png')
        >>> evaluator.generate_report(result, 'output/report.md')
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """初始化评估器
        
        Args:
            logger: 日志记录器（可选）
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def evaluate(self,
                 model: Any,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 class_names: Optional[List[str]] = None,
                 model_type: str = "Unknown") -> EvaluationResult:
        """评估模型性能
        
        Args:
            model: 训练好的模型对象
            X_test: 测试特征 (n_samples, n_features)
            y_test: 测试标签 (n_samples,)
            class_names: 类别名称列表（可选）
            model_type: 模型类型名称
            
        Returns:
            EvaluationResult: 评估结果对象
            
        Raises:
            RuntimeError: 如果模型不支持predict方法
        """
        if not hasattr(model, 'predict'):
            raise RuntimeError("Model must have a predict() method")
        
        self.logger.info(f"Evaluating {model_type} model...")
        self.logger.info(f"Test samples: {X_test.shape[0]}, Features: {X_test.shape[1]}")
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Determine class names
        unique_classes = np.unique(y_test)
        n_classes = len(unique_classes)
        if class_names is None:
            class_names = [f"Class_{i}" for i in unique_classes]
        elif len(class_names) != n_classes:
            self.logger.warning(f"class_names length ({len(class_names)}) != n_classes ({n_classes})")
            class_names = [f"Class_{i}" for i in unique_classes]
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(y_test, y_pred, class_names)
        
        # Extract feature importances
        feature_importances = self._extract_feature_importances(model)
        
        # Create evaluation result
        result = EvaluationResult(
            model_type=model_type,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            per_class_metrics=per_class_metrics,
            test_samples=X_test.shape[0],
            n_classes=n_classes,
            class_names=class_names,
            feature_importances=feature_importances
        )
        
        self.logger.info(f"Evaluation complete: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return result
    
    def _calculate_per_class_metrics(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     class_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate detailed metrics for each class
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class name list
            
        Returns:
            Dictionary with class names as keys and metric dictionaries as values
        """
        # Use sklearn's classification_report to get detailed report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        per_class = {}
        unique_classes = np.unique(y_true)
        
        for i, class_label in enumerate(unique_classes):
            class_name = class_names[i] if i < len(class_names) else f"Class_{class_label}"
            class_key = str(class_label)
            
            if class_key in report:
                per_class[class_name] = {
                    'precision': report[class_key]['precision'],
                    'recall': report[class_key]['recall'],
                    'f1-score': report[class_key]['f1-score'],
                    'support': int(report[class_key]['support'])
                }
        
        return per_class
    
    def _extract_feature_importances(self, model: Any) -> Optional[Dict[str, float]]:
        """Extract feature importances (if model supports it)
        
        Args:
            model: Model object
            
        Returns:
            Feature importance dictionary, or None if not supported
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return {
                f"feature_{i}": float(imp)
                for i, imp in enumerate(importances)
            }
        return None
    
    def plot_confusion_matrix(self,
                              result: EvaluationResult,
                              output_path: Union[str, Path],
                              figsize: Tuple[int, int] = (10, 8),
                              cmap: str = 'Blues'):
        """Plot confusion matrix heatmap
        
        Args:
            result: Evaluation result object
            output_path: Output file path
            figsize: Figure size
            cmap: Color map
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get confusion matrix
        cm = result._cm_array if hasattr(result, '_cm_array') else np.array(result.confusion_matrix)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        if HAS_SEABORN:
            # Use seaborn for better appearance
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                       xticklabels=result.class_names,
                       yticklabels=result.class_names,
                       cbar_kws={'label': 'Count'},
                       ax=ax)
        else:
            # Use matplotlib as fallback
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            ax.figure.colorbar(im, ax=ax, label='Count')
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]),
                           ha="center", va="center",
                           color="white" if cm[i, j] > cm.max() / 2 else "black")
            
            ax.set_xticks(np.arange(len(result.class_names)))
            ax.set_yticks(np.arange(len(result.class_names)))
            ax.set_xticklabels(result.class_names)
            ax.set_yticklabels(result.class_names)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {result.model_type}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix plot saved to: {output_path}")
    
    def plot_feature_importance(self,
                                result: EvaluationResult,
                                feature_names: Optional[List[str]] = None,
                                output_path: Union[str, Path] = "feature_importance.png",
                                top_n: int = 10,
                                figsize: Tuple[int, int] = (10, 6)):
        """Plot feature importance bar chart
        
        Args:
            result: Evaluation result object
            feature_names: Feature name list (optional)
            output_path: Output file path
            top_n: Show top N important features
            figsize: Figure size
            
        Raises:
            ValueError: If model does not support feature importance
        """
        if result.feature_importances is None:
            raise ValueError(f"Model {result.model_type} does not support feature importances")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        importances = result.feature_importances
        if feature_names is None:
            feature_names = list(importances.keys())
        
        # Create DataFrame and sort
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': [importances.get(f"feature_{i}", 0) for i in range(len(feature_names))]
        })
        df = df.nlargest(top_n, 'importance')
        
        # Plot horizontal bar chart
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(df['feature'], df['importance'], color='steelblue')
        
        # Add value labels
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['importance'], i, f' {row["importance"]:.4f}',
                   va='center', fontsize=9)
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances - {result.model_type}',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Most important at top
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Feature importance plot saved to: {output_path}")
    
    def generate_report(self,
                       result: EvaluationResult,
                       output_path: Union[str, Path],
                       include_plots: bool = True,
                       cm_plot_path: Optional[str] = None,
                       fi_plot_path: Optional[str] = None):
        """Generate Markdown evaluation report
        
        Args:
            result: Evaluation result object
            output_path: Report output path
            include_plots: Whether to reference plots in the report
            cm_plot_path: Confusion matrix plot relative path
            fi_plot_path: Feature importance plot relative path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build Markdown report
        lines = [
            f"# Model Evaluation Report - {result.model_type}",
            "",
            f"**Evaluation Time:** {result.timestamp}  ",
            f"**Test Samples:** {result.test_samples}  ",
            f"**Number of Classes:** {result.n_classes}  ",
            "",
            "---",
            "",
            "## 📊 Overall Performance",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy | {result.accuracy:.4f} ({result.accuracy*100:.2f}%) |",
            f"| Precision | {result.precision:.4f} |",
            f"| Recall | {result.recall:.4f} |",
            f"| F1-Score | {result.f1_score:.4f} |",
            "",
            "---",
            "",
            "## 📈 Per-Class Performance",
            "",
            "| Class | Precision | Recall | F1-Score | Support |",
            "|-------|-----------|--------|----------|---------|"
        ]
        
        # 添加每个类别的指标
        for class_name, metrics in result.per_class_metrics.items():
            lines.append(
                f"| {class_name} | {metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} | {metrics['f1-score']:.4f} | "
                f"{metrics['support']} |"
            )
        
        lines.extend([
            "",
            "---",
            "",
            "## 🔢 Confusion Matrix",
            ""
        ])
        
        # Add confusion matrix plot reference
        if include_plots and cm_plot_path:
            lines.append(f"![Confusion Matrix]({cm_plot_path})")
            lines.append("")
        
        # Add confusion matrix text representation
        cm = result._cm_array if hasattr(result, '_cm_array') else np.array(result.confusion_matrix)
        lines.append("**Confusion Matrix Values:**")
        lines.append("")
        lines.append("| True \\ Pred | " + " | ".join(result.class_names) + " |")
        lines.append("|" + "---|" * (len(result.class_names) + 1))
        for i, class_name in enumerate(result.class_names):
            row = [class_name] + [str(cm[i, j]) for j in range(len(result.class_names))]
            lines.append("| " + " | ".join(row) + " |")
        
        lines.extend(["", "---", ""])
        
        # Add feature importance
        if result.feature_importances:
            lines.extend([
                "## 🎯 Feature Importance (Top 10)",
                ""
            ])
            
            if include_plots and fi_plot_path:
                lines.append(f"![Feature Importance]({fi_plot_path})")
                lines.append("")
            
            # Sort and display Top 10
            sorted_features = sorted(
                result.feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            lines.append("| Feature | Importance |")
            lines.append("|---------|------------|")
            for feat, imp in sorted_features:
                lines.append(f"| {feat} | {imp:.6f} |")
            
            lines.extend(["", "---", ""])
        
        # Write to file
        report_content = "\n".join(lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Evaluation report saved to: {output_path}")
    
    def save_result(self, result: EvaluationResult, output_path: Union[str, Path]):
        """Save evaluation result as JSON
        
        Args:
            result: Evaluation result object
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Evaluation result saved to: {output_path}")

