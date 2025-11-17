"""
预测模块

实现视频质量预测功能。
"""

from video_qoe.prediction.predictor import (
    Prediction,
    RuleBasedPredictor,
    ModelBasedPredictor,
    create_predictor
)

__all__ = [
    'Prediction',
    'RuleBasedPredictor',
    'ModelBasedPredictor',
    'create_predictor',
]

__version__ = '0.1.0'  # Epic 5: Prediction and Real-time Output - Story 5.1

