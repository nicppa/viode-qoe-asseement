"""
实时监测模块

提供端到端的实时视频质量监测流水线。
"""

from video_qoe.monitoring.pipeline import RealTimePipeline, PipelineStats

__all__ = [
    'RealTimePipeline',
    'PipelineStats',
]

__version__ = '0.3.0'  # Epic 5: Prediction and Real-time Output - Story 5.3


