"""
实验上下文模块

提供实验运行时的上下文信息。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class ExperimentContext:
    """实验上下文
    
    包含实验运行所需的所有关键信息。
    
    Attributes:
        exp_id: 实验ID
        exp_dir: 实验目录
        server_ip: HTTP服务器IP
        server_port: HTTP服务器端口
        capture_interface: 捕获接口名称
        pcap_path: PCAP文件路径
        scenario_name: 场景名称
        client_ip: 客户端IP
        metadata: 额外元数据
    """
    
    # 必需字段
    exp_id: str
    exp_dir: Path
    
    # 网络信息
    server_ip: str = ""
    server_port: int = 0
    client_ip: str = ""
    
    # 捕获信息
    capture_interface: str = "h2-eth0"
    pcap_path: Optional[Path] = None
    
    # 场景信息
    scenario_name: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保exp_dir是Path对象
        if not isinstance(self.exp_dir, Path):
            self.exp_dir = Path(self.exp_dir)
        
        # 如果没有指定pcap_path，使用默认路径
        if self.pcap_path is None:
            self.pcap_path = self.exp_dir / 'capture.pcap'
        elif not isinstance(self.pcap_path, Path):
            self.pcap_path = Path(self.pcap_path)
        
        # 初始化metadata
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            上下文信息字典
        """
        return {
            'exp_id': self.exp_id,
            'exp_dir': str(self.exp_dir),
            'server_ip': self.server_ip,
            'server_port': self.server_port,
            'client_ip': self.client_ip,
            'capture_interface': self.capture_interface,
            'pcap_path': str(self.pcap_path) if self.pcap_path else None,
            'scenario_name': self.scenario_name,
            'metadata': self.metadata,
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"ExperimentContext(exp_id={self.exp_id}, "
                f"scenario={self.scenario_name}, "
                f"server={self.server_ip}:{self.server_port})")


if __name__ == '__main__':
    """测试模块"""
    print("=" * 70)
    print("Testing ExperimentContext")
    print("=" * 70)
    
    # 测试1: 基本创建
    print("\n[Test 1] Create ExperimentContext")
    ctx = ExperimentContext(
        exp_id="exp_test_001",
        exp_dir=Path("/tmp/exp_test_001")
    )
    print(f"Context: {ctx}")
    print(f"  exp_dir type: {type(ctx.exp_dir)}")
    print(f"  pcap_path: {ctx.pcap_path}")
    
    # 测试2: 完整创建
    print("\n[Test 2] Full Context")
    ctx = ExperimentContext(
        exp_id="exp_test_002",
        exp_dir="/tmp/exp_test_002",
        server_ip="10.0.0.1",
        server_port=8000,
        client_ip="10.0.0.2",
        scenario_name="low-bandwidth",
        metadata={'test': True}
    )
    print(f"Context: {ctx}")
    print(f"  Dictionary: {ctx.to_dict()}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)



