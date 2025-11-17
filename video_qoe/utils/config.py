"""
配置管理模块
Configuration Management Module

提供多层配置加载、验证和管理功能。
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """网络配置"""
    bandwidth: str = "10Mbps"          # 带宽
    delay: str = "20ms"                # 延迟
    loss: str = "0%"                   # 丢包率
    jitter: str = "0ms"                # 抖动
    
    def validate(self):
        """验证网络配置"""
        # 验证带宽格式
        if not any(self.bandwidth.endswith(unit) for unit in ['Mbps', 'Kbps', 'Gbps']):
            raise ValueError(f"Invalid bandwidth format: {self.bandwidth}")
        # 验证延迟格式
        if not self.delay.endswith('ms'):
            raise ValueError(f"Invalid delay format: {self.delay}")
        # 验证丢包率格式
        if not self.loss.endswith('%'):
            raise ValueError(f"Invalid loss format: {self.loss}")
        return True


@dataclass
class VideoConfig:
    """视频配置"""
    file: str = ""                      # 视频文件路径
    expected_resolution: str = "720p"   # 期望分辨率
    server_port: int = 8000            # HTTP服务器端口
    segment_duration: int = 4          # 分段时长（秒）
    
    def validate(self):
        """验证视频配置"""
        # 验证分辨率格式
        valid_resolutions = ['144p', '240p', '360p', '480p', '720p', '1080p', '1440p', '4K']
        if self.expected_resolution not in valid_resolutions:
            raise ValueError(f"Invalid resolution: {self.expected_resolution}")
        # 验证端口范围
        if not 1024 <= self.server_port <= 65535:
            raise ValueError(f"Invalid port: {self.server_port}")
        return True


@dataclass
class ModelConfig:
    """模型配置"""
    path: str = "models/xgboost_v1.0.pkl"     # 模型文件路径
    type: str = "xgboost"                      # 模型类型
    confidence_threshold: float = 0.7          # 置信度阈值
    feature_count: int = 35                    # 特征数量
    
    def validate(self):
        """验证模型配置"""
        # 验证置信度范围
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"Invalid confidence threshold: {self.confidence_threshold}")
        # 验证模型类型
        valid_types = ['xgboost', 'lightgbm', 'random_forest', 'lstm']
        if self.type not in valid_types:
            raise ValueError(f"Invalid model type: {self.type}")
        return True


@dataclass
class MonitoringConfig:
    """监测配置"""
    update_interval: float = 1.0        # 更新间隔（秒）
    window_size: int = 10               # 滑动窗口大小（秒）
    enable_color: bool = True           # 启用颜色输出
    enable_events: bool = True          # 启用事件标注
    display_confidence: bool = True     # 显示置信度
    max_delay_tolerance: int = 10       # 最大延迟容忍（秒）
    
    def validate(self):
        """验证监测配置"""
        if self.update_interval <= 0:
            raise ValueError(f"Invalid update interval: {self.update_interval}")
        if self.window_size <= 0:
            raise ValueError(f"Invalid window size: {self.window_size}")
        return True


@dataclass
class OutputConfig:
    """输出配置"""
    base_dir: str = "experiments"       # 基础目录
    save_pcap: bool = True              # 保存PCAP文件
    save_features: bool = True          # 保存特征数据
    save_timeline: bool = True          # 保存时间线
    save_ground_truth: bool = True      # 保存Ground Truth
    save_config: bool = True            # 保存配置
    pcap_filename: str = "capture.pcap"
    features_filename: str = "features.csv"
    timeline_filename: str = "timeline.json"
    log_filename: str = "monitor.log"
    
    def validate(self):
        """验证输出配置"""
        # 验证目录路径
        if not self.base_dir:
            raise ValueError("Output base_dir cannot be empty")
        return True


@dataclass
class ExperimentConfig:
    """实验配置（顶层配置）"""
    # 实验元数据
    name: str = "Default Experiment"
    description: str = "Default configuration for video quality assessment"
    tags: List[str] = field(default_factory=list)
    
    # 子配置
    network: NetworkConfig = field(default_factory=NetworkConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # 其他配置
    log_level: str = "INFO"
    random_seed: int = 42
    
    def validate(self):
        """验证所有配置"""
        self.network.validate()
        self.video.validate()
        self.model.validate()
        self.monitoring.validate()
        self.output.validate()
        
        # 验证日志级别
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.log_level}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """从字典创建配置"""
        # 处理嵌套配置
        if 'network' in data and isinstance(data['network'], dict):
            data['network'] = NetworkConfig(**data['network'])
        if 'video' in data and isinstance(data['video'], dict):
            data['video'] = VideoConfig(**data['video'])
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        if 'monitoring' in data and isinstance(data['monitoring'], dict):
            data['monitoring'] = MonitoringConfig(**data['monitoring'])
        if 'output' in data and isinstance(data['output'], dict):
            data['output'] = OutputConfig(**data['output'])
        
        return cls(**data)


class ConfigManager:
    """配置管理器
    
    支持多层配置加载：
    1. 默认配置（最低优先级）
    2. 场景配置
    3. 用户配置文件
    4. 命令行参数（最高优先级）
    5. 环境变量覆盖
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """初始化配置管理器
        
        Args:
            project_root: 项目根目录，默认自动检测
        """
        if project_root is None:
            # 自动检测项目根目录（从当前文件向上查找）
            current = Path(__file__).resolve()
            for parent in current.parents:
                if (parent / 'setup.py').exists() or (parent / 'video_qoe').exists():
                    project_root = parent
                    break
            else:
                project_root = Path.cwd()
        
        self.project_root = Path(project_root)
        self.config_dir = self.project_root / 'configs'
        self.scenarios_dir = self.config_dir / 'scenarios'
        
        # 加载默认配置
        self.defaults = self._load_default_config()
        
        # 加载所有场景配置
        self.scenarios = self._load_all_scenarios()
        
        logger.info(f"ConfigManager initialized with project root: {self.project_root}")
        logger.info(f"Loaded {len(self.scenarios)} scenario(s)")
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """加载YAML文件"""
        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if data else {}
        except Exception as e:
            logger.error(f"Failed to load YAML file {file_path}: {e}")
            return {}
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        default_file = self.config_dir / 'default_config.yaml'
        config = self._load_yaml(default_file)
        
        if not config:
            logger.warning("Using built-in default configuration")
            # 内置默认配置（fallback）
            config = {
                'experiment': {'name': 'Default', 'description': 'Built-in default'},
                'network': {'bandwidth': '10Mbps', 'delay': '20ms', 'loss': '0%', 'jitter': '0ms'},
                'video': {'file': '', 'expected_resolution': '720p'},
                'model': {'path': 'models/xgboost_v1.0.pkl', 'confidence_threshold': 0.7},
                'monitoring': {'update_interval': 1.0, 'enable_color': True},
                'output': {'save_pcap': True, 'save_features': True},
            }
        
        return config
    
    def _load_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """加载所有场景配置"""
        scenarios = {}
        
        if not self.scenarios_dir.exists():
            logger.warning(f"Scenarios directory not found: {self.scenarios_dir}")
            return scenarios
        
        for scenario_file in self.scenarios_dir.glob('*.yaml'):
            scenario_name = scenario_file.stem
            scenarios[scenario_name] = self._load_yaml(scenario_file)
            logger.debug(f"Loaded scenario: {scenario_name}")
        
        return scenarios
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                result[key] = self._deep_merge(result[key], value)
            else:
                # 覆盖值
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖
        
        环境变量格式：VQE_<SECTION>_<KEY>
        例如：VQE_NETWORK_BANDWIDTH=20Mbps
        """
        for env_key, env_value in os.environ.items():
            if not env_key.startswith('VQE_'):
                continue
            
            # 解析环境变量名称
            parts = env_key[4:].lower().split('_')
            if len(parts) < 2:
                continue
            
            section = parts[0]
            key = '_'.join(parts[1:])
            
            # 应用覆盖
            if section in config and isinstance(config[section], dict):
                # 类型转换
                try:
                    if env_value.lower() in ('true', 'false'):
                        config[section][key] = env_value.lower() == 'true'
                    elif env_value.isdigit():
                        config[section][key] = int(env_value)
                    elif env_value.replace('.', '').isdigit():
                        config[section][key] = float(env_value)
                    else:
                        config[section][key] = env_value
                    
                    logger.debug(f"Applied env override: {env_key} = {env_value}")
                except Exception as e:
                    logger.warning(f"Failed to apply env override {env_key}: {e}")
        
        return config
    
    def load_config(
        self,
        scenario: Optional[str] = None,
        config_file: Optional[str] = None,
        cli_args: Optional[Dict[str, Any]] = None,
        apply_env: bool = True,
        validate: bool = True,
    ) -> ExperimentConfig:
        """加载配置
        
        Args:
            scenario: 场景名称（对应scenarios/目录下的YAML文件）
            config_file: 用户配置文件路径
            cli_args: 命令行参数字典
            apply_env: 是否应用环境变量覆盖
            validate: 是否验证配置
        
        Returns:
            ExperimentConfig实例
        
        优先级（从低到高）：
            默认配置 < 场景配置 < 用户配置文件 < CLI参数 < 环境变量
        """
        # 1. 从默认配置开始
        config = self.defaults.copy()
        logger.debug("Loaded default configuration")
        
        # 2. 合并场景配置
        if scenario:
            if scenario in self.scenarios:
                config = self._deep_merge(config, self.scenarios[scenario])
                logger.info(f"Applied scenario: {scenario}")
            else:
                logger.warning(f"Scenario not found: {scenario}")
                logger.info(f"Available scenarios: {list(self.scenarios.keys())}")
        
        # 3. 合并用户配置文件
        if config_file:
            user_config = self._load_yaml(Path(config_file))
            if user_config:
                config = self._deep_merge(config, user_config)
                logger.info(f"Applied user config: {config_file}")
        
        # 4. 合并CLI参数
        if cli_args:
            config = self._deep_merge(config, cli_args)
            logger.debug(f"Applied CLI arguments: {len(cli_args)} items")
        
        # 5. 应用环境变量覆盖
        if apply_env:
            config = self._apply_env_overrides(config)
        
        # 6. 创建ExperimentConfig对象
        try:
            exp_config = ExperimentConfig.from_dict(config)
        except Exception as e:
            logger.error(f"Failed to create ExperimentConfig: {e}")
            raise ValueError(f"Invalid configuration: {e}")
        
        # 7. 验证配置
        if validate:
            try:
                exp_config.validate()
                logger.info("Configuration validated successfully")
            except Exception as e:
                logger.error(f"Configuration validation failed: {e}")
                raise
        
        return exp_config
    
    def get_scenario_list(self) -> List[str]:
        """获取可用场景列表"""
        return list(self.scenarios.keys())
    
    def save_config(self, config: ExperimentConfig, output_path: Path):
        """保存配置到YAML文件"""
        try:
            config_dict = config.to_dict()
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Configuration saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise


# 便捷函数
def load_default_config() -> ExperimentConfig:
    """加载默认配置（便捷函数）"""
    manager = ConfigManager()
    return manager.load_config()


def load_scenario_config(scenario: str) -> ExperimentConfig:
    """加载场景配置（便捷函数）"""
    manager = ConfigManager()
    return manager.load_config(scenario=scenario)




