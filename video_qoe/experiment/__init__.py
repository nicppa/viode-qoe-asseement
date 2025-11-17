"""
实验管理模块
Experiment Management Module

负责Mininet实验环境的初始化、配置、运行和清理。

主要组件：
- ExperimentManager: 实验管理器，负责Mininet拓扑管理和网络配置
- NetworkConfig: 网络配置类（带宽、延迟、丢包、抖动）
- MeasuredNetworkConditions: 实测网络条件类
- GroundTruth: Ground Truth数据记录器（待实现）
- ExperimentContext: 实验上下文信息（待实现）

Example:
    >>> from video_qoe.experiment import ExperimentManager, NetworkConfig
    >>> 
    >>> # 方式1: 手动管理
    >>> manager = ExperimentManager()
    >>> h1, h2 = manager.setup_mininet()
    >>> 
    >>> # 配置网络条件
    >>> config = NetworkConfig(bandwidth="2Mbps", delay="50ms", loss="1%")
    >>> manager.configure_network_conditions(config)
    >>> 
    >>> # ... 运行实验 ...
    >>> manager.cleanup()
    >>> 
    >>> # 方式2: 使用with语句（推荐）
    >>> with ExperimentManager() as manager:
    ...     h1, h2 = manager.setup_mininet()
    ...     
    ...     config = NetworkConfig(bandwidth="10Mbps", delay="20ms", loss="0%")
    ...     manager.configure_network_conditions(config)
    ...     
    ...     # ... 运行实验 ...
    ...     # 自动清理
"""

from .manager import ExperimentManager
from .network_config import NetworkConfig, MeasuredNetworkConditions
from .scenario import ScenarioConfig, ScenarioLoader, load_scenario, list_available_scenarios
from .ground_truth import GroundTruth, VideoInfo, NetworkInfo, ExperimentEvent
from .context import ExperimentContext

__all__ = [
    'ExperimentManager',
    'NetworkConfig',
    'MeasuredNetworkConditions',
    'ScenarioConfig',
    'ScenarioLoader',
    'load_scenario',
    'list_available_scenarios',
    'GroundTruth',
    'VideoInfo',
    'NetworkInfo',
    'ExperimentEvent',
    'ExperimentContext',
]

__version__ = '0.5.0'  # Story 2.6: 实验生命周期管理


