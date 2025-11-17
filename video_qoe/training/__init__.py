"""
训练模块

提供模型训练相关的类和函数：
- ExperimentDataLoader: 实验数据加载器
- FeaturePreprocessor: 特征预处理器
- ModelTrainer: 模型训练器（抽象基类）
- XGBoostTrainer: XGBoost训练器
- RandomForestTrainer: Random Forest训练器
- ModelEvaluator: 模型评估器
- Model utilities: 预训练模型加载和管理工具
"""

from .data_loader import ExperimentDataLoader
from .preprocessor import FeaturePreprocessor, DataSplit
from .model_trainer import (
    ModelTrainer,
    XGBoostTrainer,
    RandomForestTrainer,
    TrainingResult
)
from .evaluator import ModelEvaluator, EvaluationResult
from .model_utils import (
    load_pretrained_model,
    list_available_models,
    get_model_info,
    save_model_metadata,
    print_model_summary
)

__all__ = [
    'ExperimentDataLoader',
    'FeaturePreprocessor',
    'DataSplit',
    'ModelTrainer',
    'XGBoostTrainer',
    'RandomForestTrainer',
    'TrainingResult',
    'ModelEvaluator',
    'EvaluationResult',
    'load_pretrained_model',
    'list_available_models',
    'get_model_info',
    'save_model_metadata',
    'print_model_summary',
]

