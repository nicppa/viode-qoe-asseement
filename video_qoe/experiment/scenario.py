"""
场景加载器模块

提供场景配置的加载、验证和管理功能。
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

from video_qoe.experiment.network_config import NetworkConfig


@dataclass
class ScenarioConfig:
    """场景配置数据类"""
    
    # 基本信息
    name: str
    description: str
    
    # 网络配置
    network: NetworkConfig
    
    # 视频配置
    video_expected_resolution: str = "720p"
    video_expected_bitrate: str = "3Mbps"
    video_duration: int = 180
    
    # 监测配置
    monitoring_update_interval: float = 1.0
    monitoring_confidence_threshold: float = 0.7
    monitoring_window_size: float = 1.0
    
    # HTTP服务器配置
    http_server_port: int = 8000
    http_server_video_dir: str = "/var/html/out"
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 特殊配置
    stall_injection: Optional[Dict[str, Any]] = None
    network_dynamics: Optional[Dict[str, Any]] = None
    
    # 原始YAML数据（用于调试）
    _raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioConfig':
        """从字典创建场景配置
        
        Args:
            data: YAML解析后的字典
            
        Returns:
            ScenarioConfig实例
        """
        # 提取基本信息
        name = data.get('name', 'Unnamed Scenario')
        description = data.get('description', '')
        
        # 提取网络配置
        network_data = data.get('network', {})
        network = NetworkConfig(
            bandwidth=network_data.get('bandwidth', '10Mbps'),
            delay=network_data.get('delay', '0ms'),
            loss=network_data.get('loss', '0%'),
            jitter=network_data.get('jitter')
        )
        
        # 提取视频配置
        video_data = data.get('video', {})
        video_expected_resolution = video_data.get('expected_resolution', '720p')
        video_expected_bitrate = video_data.get('expected_bitrate', '3Mbps')
        video_duration = video_data.get('duration', 180)
        
        # 提取监测配置
        monitoring_data = data.get('monitoring', {})
        monitoring_update_interval = monitoring_data.get('update_interval', 1.0)
        monitoring_confidence_threshold = monitoring_data.get('confidence_threshold', 0.7)
        monitoring_window_size = monitoring_data.get('window_size', 1.0)
        
        # 提取HTTP服务器配置
        http_data = data.get('http_server', {})
        http_server_port = http_data.get('port', 8000)
        http_server_video_dir = http_data.get('video_dir', '/home/mininet/cn/video')
        
        # 提取元数据
        metadata = data.get('metadata', {})
        
        # 提取特殊配置
        stall_injection = data.get('stall_injection')
        network_dynamics = data.get('network_dynamics')
        
        return cls(
            name=name,
            description=description,
            network=network,
            video_expected_resolution=video_expected_resolution,
            video_expected_bitrate=video_expected_bitrate,
            video_duration=video_duration,
            monitoring_update_interval=monitoring_update_interval,
            monitoring_confidence_threshold=monitoring_confidence_threshold,
            monitoring_window_size=monitoring_window_size,
            http_server_port=http_server_port,
            http_server_video_dir=http_server_video_dir,
            metadata=metadata,
            stall_injection=stall_injection,
            network_dynamics=network_dynamics,
            _raw_data=data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'network': self.network.to_dict(),
            'video': {
                'expected_resolution': self.video_expected_resolution,
                'expected_bitrate': self.video_expected_bitrate,
                'duration': self.video_duration,
            },
            'monitoring': {
                'update_interval': self.monitoring_update_interval,
                'confidence_threshold': self.monitoring_confidence_threshold,
                'window_size': self.monitoring_window_size,
            },
            'http_server': {
                'port': self.http_server_port,
                'video_dir': self.http_server_video_dir,
            },
            'metadata': self.metadata,
            'stall_injection': self.stall_injection,
            'network_dynamics': self.network_dynamics,
        }
    
    def is_dynamic_scenario(self) -> bool:
        """是否为动态场景（需要运行时调整网络条件）"""
        return (self.stall_injection and self.stall_injection.get('enabled', False)) or \
               (self.network_dynamics and self.network_dynamics.get('enabled', False))


class ScenarioLoader:
    """场景加载器
    
    负责加载、验证和管理场景配置文件。
    """
    
    def __init__(self, scenarios_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        """初始化场景加载器
        
        Args:
            scenarios_dir: 场景配置文件目录，默认为configs/scenarios
            logger: 日志记录器
        """
        if scenarios_dir is None:
            # 默认使用项目根目录下的configs/scenarios
            project_root = Path(__file__).parent.parent.parent
            scenarios_dir = project_root / 'configs' / 'scenarios'
        
        self.scenarios_dir = Path(scenarios_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # 缓存已加载的场景
        self._scenarios_cache: Dict[str, ScenarioConfig] = {}
    
    def list_scenarios(self) -> List[str]:
        """列出所有可用的场景名称
        
        Returns:
            场景名称列表（不含.yaml扩展名）
        """
        if not self.scenarios_dir.exists():
            self.logger.warning(f"Scenarios directory not found: {self.scenarios_dir}")
            return []
        
        scenario_files = list(self.scenarios_dir.glob('*.yaml'))
        scenario_files.extend(self.scenarios_dir.glob('*.yml'))
        
        # 返回文件名（不含扩展名）
        scenarios = [f.stem for f in scenario_files]
        scenarios.sort()
        
        return scenarios
    
    def load_scenario(self, scenario_name: str, use_cache: bool = True) -> ScenarioConfig:
        """加载场景配置
        
        Args:
            scenario_name: 场景名称（如'low-bandwidth'，不含扩展名）
            use_cache: 是否使用缓存
            
        Returns:
            ScenarioConfig对象
            
        Raises:
            FileNotFoundError: 场景文件不存在
            ValueError: YAML格式错误或配置无效
        """
        # 检查缓存
        if use_cache and scenario_name in self._scenarios_cache:
            self.logger.debug(f"Loading scenario '{scenario_name}' from cache")
            return self._scenarios_cache[scenario_name]
        
        # 查找场景文件
        scenario_file = self.scenarios_dir / f"{scenario_name}.yaml"
        if not scenario_file.exists():
            # 尝试.yml扩展名
            scenario_file = self.scenarios_dir / f"{scenario_name}.yml"
            if not scenario_file.exists():
                raise FileNotFoundError(f"Scenario file not found: {scenario_name}")
        
        self.logger.info(f"Loading scenario from: {scenario_file}")
        
        try:
            # 读取YAML文件
            with open(scenario_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not isinstance(data, dict):
                raise ValueError(f"Invalid YAML format in {scenario_file}")
            
            # 创建场景配置
            scenario = ScenarioConfig.from_dict(data)
            
            # 验证配置
            scenario.network.validate()
            
            # 缓存
            self._scenarios_cache[scenario_name] = scenario
            
            self.logger.info(f"✓ Scenario '{scenario.name}' loaded successfully")
            self.logger.debug(f"  Network: BW={scenario.network.bandwidth}, "
                            f"Delay={scenario.network.delay}, Loss={scenario.network.loss}")
            
            return scenario
            
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error in {scenario_file}: {e}")
            raise ValueError(f"Invalid YAML in {scenario_name}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load scenario {scenario_name}: {e}")
            raise
    
    def get_scenario_summary(self, scenario_name: str) -> Dict[str, Any]:
        """获取场景摘要信息（不完全加载）
        
        Args:
            scenario_name: 场景名称
            
        Returns:
            场景摘要字典
        """
        try:
            scenario = self.load_scenario(scenario_name)
            return {
                'name': scenario.name,
                'description': scenario.description,
                'network': {
                    'bandwidth': scenario.network.bandwidth,
                    'delay': scenario.network.delay,
                    'loss': scenario.network.loss,
                },
                'video_resolution': scenario.video_expected_resolution,
                'is_dynamic': scenario.is_dynamic_scenario(),
                'category': scenario.metadata.get('category', 'unknown'),
                'tags': scenario.metadata.get('tags', []),
            }
        except Exception as e:
            self.logger.error(f"Failed to get summary for {scenario_name}: {e}")
            return {
                'name': scenario_name,
                'error': str(e)
            }
    
    def validate_all_scenarios(self) -> Dict[str, bool]:
        """验证所有场景配置的有效性
        
        Returns:
            {scenario_name: is_valid} 字典
        """
        scenarios = self.list_scenarios()
        results = {}
        
        for scenario_name in scenarios:
            try:
                self.load_scenario(scenario_name, use_cache=False)
                results[scenario_name] = True
                self.logger.info(f"✓ {scenario_name}: Valid")
            except Exception as e:
                results[scenario_name] = False
                self.logger.error(f"✗ {scenario_name}: Invalid - {e}")
        
        return results
    
    def clear_cache(self):
        """清空场景缓存"""
        self._scenarios_cache.clear()
        self.logger.debug("Scenario cache cleared")


# 便捷函数
def load_scenario(scenario_name: str, scenarios_dir: Optional[Path] = None) -> ScenarioConfig:
    """快速加载场景配置
    
    Args:
        scenario_name: 场景名称
        scenarios_dir: 场景目录（可选）
        
    Returns:
        ScenarioConfig对象
    """
    loader = ScenarioLoader(scenarios_dir)
    return loader.load_scenario(scenario_name)


def list_available_scenarios(scenarios_dir: Optional[Path] = None) -> List[str]:
    """列出可用场景
    
    Args:
        scenarios_dir: 场景目录（可选）
        
    Returns:
        场景名称列表
    """
    loader = ScenarioLoader(scenarios_dir)
    return loader.list_scenarios()


if __name__ == '__main__':
    """测试模块"""
    import sys
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('scenario_test')
    
    print("=" * 70)
    print("Testing ScenarioLoader")
    print("=" * 70)
    
    # 测试1: 列出场景
    print("\n[Test 1] List available scenarios")
    loader = ScenarioLoader(logger=logger)
    scenarios = loader.list_scenarios()
    print(f"Found {len(scenarios)} scenarios:")
    for s in scenarios:
        print(f"  - {s}")
    
    # 测试2: 加载场景
    print("\n[Test 2] Load scenarios")
    for scenario_name in scenarios[:3]:  # 只测试前3个
        try:
            scenario = loader.load_scenario(scenario_name)
            print(f"\n✓ Loaded: {scenario.name}")
            print(f"  Description: {scenario.description}")
            print(f"  Network: {scenario.network}")
            print(f"  Dynamic: {scenario.is_dynamic_scenario()}")
        except Exception as e:
            print(f"\n✗ Failed to load {scenario_name}: {e}")
    
    # 测试3: 验证所有场景
    print("\n[Test 3] Validate all scenarios")
    results = loader.validate_all_scenarios()
    valid_count = sum(1 for v in results.values() if v)
    print(f"\nValidation results: {valid_count}/{len(results)} valid")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)



