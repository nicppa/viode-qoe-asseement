"""
实验管理器模块
Experiment Manager Module

负责Mininet实验环境的初始化、配置和清理。
"""

import time
import signal
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

try:
    from mininet.net import Mininet
    from mininet.node import OVSController
    from mininet.topo import SingleSwitchTopo
    from mininet.link import TCLink
    from mininet.cli import CLI
except ImportError as e:
    raise ImportError(
        "Mininet is not installed. "
        "Please install Mininet first: sudo apt-get install mininet"
    ) from e

from video_qoe.utils.logger import get_logger
from video_qoe.utils.helpers import generate_exp_id, ensure_dir
from video_qoe.experiment.network_config import NetworkConfig, MeasuredNetworkConditions
from video_qoe.experiment.ground_truth import GroundTruth
from video_qoe.experiment.context import ExperimentContext


class ExperimentManager:
    """实验管理器
    
    管理Mininet实验环境的完整生命周期：
    - 初始化Mininet拓扑
    - 配置网络条件
    - 启动HTTP服务器
    - 清理实验环境
    
    拓扑结构：
        h1 (server) --- s1 (switch) --- h2 (client)
    
    Example:
        >>> manager = ExperimentManager()
        >>> manager.setup_mininet()
        >>> # 运行实验...
        >>> manager.cleanup()
    """
    
    def __init__(self, logger=None):
        """初始化实验管理器
        
        Args:
            logger: 日志记录器，如果为None则创建默认logger
        """
        self.logger = logger or get_logger('video_qoe.experiment.manager')
        
        # Mininet相关
        self.net: Optional[Mininet] = None
        self.h1 = None  # Server节点
        self.h2 = None  # Client节点
        
        # HTTP服务器进程
        self.http_process = None
        
        # 实验信息
        self.exp_id = None
        self.exp_dir = None
        
        # 记录是否已初始化
        self.initialized = False
        
        # 网络配置
        self.network_config: Optional[NetworkConfig] = None
        self.measured_conditions: Optional[MeasuredNetworkConditions] = None
        
        # Ground Truth记录器
        self.ground_truth: Optional[GroundTruth] = None
        
        # 注册信号处理（优雅退出）
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.debug("ExperimentManager initialized")
    
    def _signal_handler(self, signum, frame):
        """处理中断信号（Ctrl+C）"""
        self.logger.warning(f"Received signal {signum}, cleaning up...")
        self.cleanup()
        raise KeyboardInterrupt("Experiment interrupted by user")
    
    def setup_mininet(self, num_hosts: int = 2) -> Tuple[Any, Any]:
        """初始化Mininet拓扑
        
        创建简单的单交换机拓扑：
        - h1: 服务器节点（提供视频文件）
        - h2: 客户端节点（下载视频、捕获流量）
        - s1: OpenFlow交换机连接两个节点
        
        Args:
            num_hosts: 主机数量，默认2个
        
        Returns:
            (h1, h2): 服务器和客户端节点句柄
        
        Raises:
            RuntimeError: Mininet初始化失败或连通性测试失败
        
        Example:
            >>> manager = ExperimentManager()
            >>> h1, h2 = manager.setup_mininet()
            >>> print(f"Server IP: {h1.IP()}")
            >>> print(f"Client IP: {h2.IP()}")
        """
        if self.initialized:
            self.logger.warning("Mininet already initialized, skipping...")
            return self.h1, self.h2
        
        self.logger.info("Initializing Mininet topology...")
        
        try:
            # 清理可能存在的旧Mininet实例
            self._cleanup_old_mininet()
            
            # 创建拓扑（单交换机，2个主机）
            self.logger.debug(f"Creating SingleSwitchTopo with {num_hosts} hosts")
            topo = SingleSwitchTopo(n=num_hosts)
            
            # 创建Mininet网络
            # - TCLink: 支持流量控制（带宽、延迟、丢包）
            # - OVSController: 默认OpenFlow控制器
            self.logger.debug("Creating Mininet instance")
            self.net = Mininet(
                topo=topo,
                link=TCLink,
                controller=OVSController,
                autoSetMacs=True,  # 自动设置MAC地址
                autoStaticArp=True,  # 自动设置ARP表
            )
            
            # 启动网络
            self.logger.debug("Starting Mininet network")
            self.net.start()
            
            # 等待网络稳定
            time.sleep(2)
            
            # 获取节点句柄
            self.h1 = self.net.get('h1')
            self.h2 = self.net.get('h2')
            
            if not self.h1 or not self.h2:
                raise RuntimeError("Failed to get host handles from Mininet")
            
            self.logger.info(f"Mininet nodes created:")
            self.logger.info(f"  - h1 (server): {self.h1.IP()}")
            self.logger.info(f"  - h2 (client): {self.h2.IP()}")
            
            # 测试连通性
            self.logger.debug("Testing network connectivity...")
            result = self._test_connectivity()
            
            if not result:
                raise RuntimeError("Mininet connectivity test failed")
            
            self.logger.info("✓ Mininet topology initialized successfully")
            self.initialized = True
            
            return self.h1, self.h2
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Mininet: {e}")
            self.cleanup()  # 清理已创建的资源
            raise
    
    def _test_connectivity(self, timeout: float = 5.0) -> bool:
        """测试网络连通性
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            是否连通
        """
        if not self.net or not self.h1 or not self.h2:
            return False
        
        try:
            # 方法1: 使用Mininet内置pingAll
            self.logger.debug("Running pingAll test")
            loss_percent = self.net.pingAll(timeout=str(timeout))
            
            if loss_percent == 0:
                self.logger.info("✓ Connectivity test passed (pingAll: 0% loss)")
                return True
            else:
                self.logger.warning(f"Connectivity test: {loss_percent}% packet loss")
                
                # 方法2: 手动ping测试
                self.logger.debug("Trying manual ping test")
                result = self.h2.cmd(f'ping -c 3 -W {int(timeout)} {self.h1.IP()}')
                
                if '0% packet loss' in result or '0 errors' in result:
                    self.logger.info("✓ Connectivity test passed (manual ping)")
                    return True
                else:
                    self.logger.error(f"Manual ping failed: {result}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Connectivity test error: {e}")
            return False
    
    def _cleanup_old_mininet(self):
        """清理可能存在的旧Mininet实例
        
        在新实验开始前，确保清理所有旧的Mininet资源
        """
        self.logger.debug("Cleaning up old Mininet instances...")
        
        try:
            # 运行 mn -c 清理命令
            result = subprocess.run(
                ['sudo', 'mn', '-c'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.logger.debug("Old Mininet instances cleaned")
            else:
                self.logger.warning(f"mn -c returned non-zero: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.warning("mn -c command timed out")
        except FileNotFoundError:
            self.logger.debug("mn command not found (maybe Mininet not installed?)")
        except Exception as e:
            self.logger.warning(f"Error cleaning old Mininet: {e}")
    
    def get_node_info(self) -> Dict[str, Any]:
        """获取节点信息
        
        Returns:
            包含节点IP、接口等信息的字典
        """
        if not self.initialized or not self.h1 or not self.h2:
            return {}
        
        info = {
            'h1': {
                'name': 'h1',
                'role': 'server',
                'ip': self.h1.IP(),
                'mac': self.h1.MAC(),
                'interfaces': self._get_interfaces(self.h1),
            },
            'h2': {
                'name': 'h2',
                'role': 'client',
                'ip': self.h2.IP(),
                'mac': self.h2.MAC(),
                'interfaces': self._get_interfaces(self.h2),
            }
        }
        
        return info
    
    def _get_interfaces(self, node) -> list:
        """获取节点的网络接口列表"""
        try:
            output = node.cmd('ip link show')
            interfaces = []
            for line in output.split('\n'):
                if ':' in line and '@' not in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        intf_name = parts[1].strip()
                        if intf_name != 'lo':  # 排除loopback
                            interfaces.append(intf_name)
            return interfaces
        except Exception as e:
            self.logger.warning(f"Failed to get interfaces: {e}")
            return []
    
    def configure_network_conditions(self, config: NetworkConfig) -> bool:
        """配置网络条件
        
        在Mininet拓扑上配置网络条件（带宽、延迟、丢包、抖动）。
        
        Args:
            config: 网络配置对象
            
        Returns:
            是否配置成功
            
        Raises:
            RuntimeError: Mininet未初始化或配置失败
            ValueError: 网络配置参数无效
            
        Example:
            >>> manager = ExperimentManager()
            >>> manager.setup_mininet()
            >>> 
            >>> config = NetworkConfig(
            ...     bandwidth="2Mbps",
            ...     delay="50ms",
            ...     loss="1%"
            ... )
            >>> manager.configure_network_conditions(config)
            True
        """
        if not self.initialized:
            raise RuntimeError("Mininet not initialized. Call setup_mininet() first.")
        
        # 验证配置
        try:
            config.validate()
        except ValueError as e:
            self.logger.error(f"Invalid network configuration: {e}")
            raise
        
        self.logger.info(f"Configuring network conditions: {config}")
        
        try:
            # 获取h2的link（客户端侧的link）
            # linksBetween返回连接s1和h2的links
            links = self.net.linksBetween(self.net.get('s1'), self.h2)
            
            if not links:
                raise RuntimeError("No link found between s1 and h2")
            
            link = links[0]
            
            # 获取h2的接口名称
            # 注意：使用h2的默认接口名称，而不是link.intf2.name（可能返回交换机侧接口）
            h2_intf = 'h2-eth0'
            
            # 验证接口是否存在
            check_intf = self.h2.cmd(f'ip link show {h2_intf} 2>&1')
            if 'does not exist' in check_intf or 'Cannot find' in check_intf:
                # 尝试获取h2的第一个非lo接口
                intfs_output = self.h2.cmd('ip link show')
                for line in intfs_output.split('\n'):
                    if 'eth' in line and ':' in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            h2_intf = parts[1].strip().split('@')[0]
                            break
                self.logger.warning(f"Using detected interface: {h2_intf}")
            
            self.logger.debug(f"Configuring interface: {h2_intf}")
            
            # 清除现有的tc配置
            self.h2.cmd(f'tc qdisc del dev {h2_intf} root 2>/dev/null || true')
            
            # 使用tc命令直接配置
            # 策略改进：根据需求选择合适的层次结构
            
            # 检查是否需要netem
            need_netem = (config._delay_ms > 0 or 
                         config._loss_percent > 0 or 
                         config.jitter is not None)
            need_bandwidth = config._bandwidth_mbps > 0
            
            if need_netem and need_bandwidth:
                # 情况1: 需要延迟/丢包 + 带宽限制
                # 策略：netem作为root，tbf作为其child
                
                # 步骤1: 添加netem作为root
                netem_cmd = f'tc qdisc add dev {h2_intf} root handle 1: netem'
                
                if config._delay_ms > 0:
                    netem_cmd += f' delay {config.delay}'
                
                if config._loss_percent > 0:
                    netem_cmd += f' loss {config._loss_percent}%'
                
                if config.jitter:
                    netem_cmd += f' {config.jitter}'
                
                self.logger.debug(f"Applying netem (root): {netem_cmd}")
                result = self.h2.cmd(netem_cmd)
                if result and 'Error' in result:
                    self.logger.error(f"netem failed: {result}")
                
                # 步骤2: 添加tbf作为child
                burst_kb = max(32, int(config._bandwidth_mbps * 15))
                limit_kb = max(burst_kb * 2, 64)
                
                tbf_cmd = (f'tc qdisc add dev {h2_intf} parent 1:1 handle 10: '
                          f'tbf rate {config._bandwidth_mbps}mbit '
                          f'burst {burst_kb}kbit '
                          f'limit {limit_kb}kbit')
                
                self.logger.debug(f"Applying tbf (child): {tbf_cmd}")
                result = self.h2.cmd(tbf_cmd)
                if result and 'Error' in result:
                    self.logger.error(f"tbf failed: {result}")
                
            elif need_netem:
                # 情况2: 只需要延迟/丢包
                netem_cmd = f'tc qdisc add dev {h2_intf} root handle 1: netem'
                
                if config._delay_ms > 0:
                    netem_cmd += f' delay {config.delay}'
                
                if config._loss_percent > 0:
                    netem_cmd += f' loss {config._loss_percent}%'
                
                if config.jitter:
                    netem_cmd += f' {config.jitter}'
                
                self.logger.debug(f"Applying netem only: {netem_cmd}")
                result = self.h2.cmd(netem_cmd)
                if result and 'Error' in result:
                    self.logger.error(f"netem failed: {result}")
                
            elif need_bandwidth:
                # 情况3: 只需要带宽限制
                burst_kb = max(32, int(config._bandwidth_mbps * 15))
                limit_kb = max(burst_kb * 2, 64)
                
                tbf_cmd = (f'tc qdisc add dev {h2_intf} root handle 1: '
                          f'tbf rate {config._bandwidth_mbps}mbit '
                          f'burst {burst_kb}kbit '
                          f'limit {limit_kb}kbit')
                
                self.logger.debug(f"Applying tbf only: {tbf_cmd}")
                result = self.h2.cmd(tbf_cmd)
                if result and 'Error' in result:
                    self.logger.error(f"tbf failed: {result}")
            
            # 验证h2配置
            tc_check = self.h2.cmd(f'tc qdisc show dev {h2_intf}')
            self.logger.info(f"TC configuration verification (h2):\n{tc_check}")
            
            # 检查配置是否成功
            if need_bandwidth and 'tbf' not in tc_check:
                self.logger.error("TBF configuration failed on h2!")
            if need_netem and 'netem' not in tc_check:
                self.logger.error("NETEM configuration failed on h2!")
            
            # 为了实现双向网络条件，也在h1上配置TC
            # 策略：只配置延迟，不配置丢包（避免双向丢包叠加）
            self.logger.debug("Configuring h1 interface for bidirectional delay...")
            
            h1_intf = 'h1-eth0'
            
            # 验证h1接口是否存在
            check_h1_intf = self.h1.cmd(f'ip link show {h1_intf} 2>&1')
            if 'does not exist' in check_h1_intf or 'Cannot find' in check_h1_intf:
                # 尝试获取h1的第一个非lo接口
                intfs_output = self.h1.cmd('ip link show')
                for line in intfs_output.split('\n'):
                    if 'eth' in line and ':' in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            h1_intf = parts[1].strip().split('@')[0]
                            break
                self.logger.warning(f"Using detected h1 interface: {h1_intf}")
            
            # 清除h1现有的tc配置
            self.h1.cmd(f'tc qdisc del dev {h1_intf} root 2>/dev/null || true')
            
            # 在h1上只应用延迟（不应用丢包，避免双向叠加）
            if config._delay_ms > 0:
                h1_netem_cmd = f'tc qdisc add dev {h1_intf} root handle 1: netem delay {config.delay}'
                
                # 只添加抖动（如果有），不添加丢包
                if config.jitter:
                    h1_netem_cmd += f' {config.jitter}'
                
                self.logger.debug(f"Applying netem on h1 (delay only): {h1_netem_cmd}")
                result = self.h1.cmd(h1_netem_cmd)
                if result and 'Error' in result:
                    self.logger.warning(f"h1 netem configuration failed: {result}")
                else:
                    self.logger.debug("h1 netem configured successfully (delay only)")
                
                # 验证h1配置
                h1_tc_check = self.h1.cmd(f'tc qdisc show dev {h1_intf}')
                self.logger.debug(f"TC configuration verification (h1):\n{h1_tc_check}")
            
            # 保存配置
            self.network_config = config
            
            self.logger.info(f"✓ Network conditions configured successfully")
            self.logger.info(f"  h2 → h1: Bandwidth={config.bandwidth}, Delay={config.delay}, Loss={config.loss}")
            self.logger.info(f"  h1 → h2: Delay={config.delay} only (bidirectional RTT effect)")
            if config.jitter:
                self.logger.info(f"  Jitter: {config.jitter}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to configure network conditions: {e}")
            raise RuntimeError(f"Network configuration failed: {e}")
    
    def measure_network_conditions(self, 
                                   duration: int = 10,
                                   use_iperf: bool = True) -> MeasuredNetworkConditions:
        """测量实际的网络条件
        
        使用iperf和ping测试实际的网络性能，验证配置是否生效。
        
        Args:
            duration: 测试持续时间（秒）
            use_iperf: 是否使用iperf测试带宽
            
        Returns:
            实测网络条件对象
            
        Raises:
            RuntimeError: Mininet未初始化
            
        Example:
            >>> manager.configure_network_conditions(config)
            >>> measured = manager.measure_network_conditions(duration=10)
            >>> print(f"Actual bandwidth: {measured.actual_bandwidth_mbps} Mbps")
        """
        if not self.initialized:
            raise RuntimeError("Mininet not initialized")
        
        self.logger.info(f"Measuring network conditions (duration={duration}s)")
        
        measured = MeasuredNetworkConditions()
        measured.measurement_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 测试1: Ping测试（RTT和丢包）
        self.logger.debug("Running ping test...")
        try:
            # 增加ping包数以获得更准确的丢包统计
            ping_count = min(duration * 10, 200)  # 更多包，最多200个
            result = self.h2.cmd(f'ping -c {ping_count} -i 0.1 {self.h1.IP()}')
            
            # 解析ping结果
            # 示例输出: "10 packets transmitted, 9 received, 10% packet loss"
            # 示例输出: "rtt min/avg/max/mdev = 0.123/0.456/0.789/0.111 ms"
            
            # 解析丢包率
            if 'packet loss' in result:
                for line in result.split('\n'):
                    if 'packet loss' in line:
                        parts = line.split(',')
                        for part in parts:
                            if 'transmitted' in part:
                                measured.total_packets = int(part.split()[0])
                            if 'received' in part:
                                received = int(part.split()[0])
                                measured.lost_packets = measured.total_packets - received
                            if '%' in part and 'packet loss' in part:
                                loss_str = part.split('%')[0].strip().split()[-1]
                                measured.actual_loss_percent = float(loss_str)
            
            # 解析RTT
            if 'rtt min/avg/max' in result or 'round-trip' in result:
                for line in result.split('\n'):
                    if 'rtt' in line.lower() or 'round-trip' in line.lower():
                        # 格式: rtt min/avg/max/mdev = 0.123/0.456/0.789/0.111 ms
                        if '=' in line:
                            values = line.split('=')[1].strip().split('/')
                            if len(values) >= 2:
                                measured.actual_rtt_ms = float(values[1])  # avg RTT
            
            self.logger.info(f"✓ Ping test complete: RTT={measured.actual_rtt_ms}ms, "
                           f"Loss={measured.actual_loss_percent}%")
            
        except Exception as e:
            self.logger.warning(f"Ping test failed: {e}")
        
        # 测试2: iperf测试（带宽）
        if use_iperf:
            self.logger.debug("Running iperf test...")
            try:
                # 在h1上启动iperf服务器
                server_cmd = 'iperf -s -u'
                server_process = self.h1.popen(server_cmd, shell=True)
                
                time.sleep(1)  # 等待服务器启动
                
                # 在h2上运行iperf客户端
                client_cmd = f'iperf -c {self.h1.IP()} -u -b 100M -t {duration}'
                result = self.h2.cmd(client_cmd)
                
                # 解析iperf结果 - 关键：只解析Server Report中的带宽
                # iperf输出两个带宽值：
                # 1. 客户端发送速率（不准确，因为可能被限制）
                # 2. 服务器接收速率（准确的！在"Server Report:"之后）
                lines = result.split('\n')
                in_server_report = False
                for line in lines:
                    # 找到Server Report部分
                    if 'Server Report' in line:
                        in_server_report = True
                        continue
                    
                    # 在Server Report部分解析带宽
                    if in_server_report and ('Mbits/sec' in line or 'Mbps' in line):
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'Mbits/sec' in part or 'Mbps' in part:
                                if i > 0:
                                    try:
                                        measured.actual_bandwidth_mbps = float(parts[i-1])
                                        break  # 找到就退出
                                    except:
                                        pass
                        if measured.actual_bandwidth_mbps:
                            break  # 找到就退出外层循环
                
                # 停止iperf服务器
                server_process.terminate()
                server_process.wait()
                
                self.logger.info(f"✓ iperf test complete: Bandwidth={measured.actual_bandwidth_mbps}Mbps")
                
            except Exception as e:
                self.logger.warning(f"iperf test failed: {e}")
        
        measured.measurement_method = "ping" + ("+iperf" if use_iperf else "")
        self.measured_conditions = measured
        
        return measured
    
    def verify_network_config(self, 
                             config: NetworkConfig,
                             tolerance_bandwidth: float = 10.0,
                             tolerance_delay: float = 5.0,
                             tolerance_loss: float = 1.5) -> Dict[str, Any]:
        """验证网络配置是否生效
        
        配置网络后，测量实际性能并与配置对比。
        
        Args:
            config: 配置的网络条件
            tolerance_bandwidth: 带宽容差（百分比）
            tolerance_delay: 延迟容差（毫秒）
            tolerance_loss: 丢包容差（百分比）
            
        Returns:
            验证结果字典
            
        Example:
            >>> config = NetworkConfig(bandwidth="2Mbps", delay="50ms", loss="1%")
            >>> manager.configure_network_conditions(config)
            >>> result = manager.verify_network_config(config)
            >>> if result['passed']:
            ...     print("Network configuration verified!")
        """
        self.logger.info("Verifying network configuration...")
        
        # 测量实际条件
        measured = self.measure_network_conditions(duration=5, use_iperf=True)
        
        # 对比（传入容差参数）
        comparison = measured.compare_with_config(
            config,
            tolerance_bandwidth=tolerance_bandwidth,
            tolerance_delay=tolerance_delay,
            tolerance_loss=tolerance_loss
        )
        
        # 判断是否通过
        passed = True
        reasons = []
        
        # 检查带宽
        if comparison['bandwidth']:
            bw_data = comparison['bandwidth']
            if not bw_data['within_tolerance']:
                passed = False
                reasons.append(f"Bandwidth mismatch: expected {bw_data['expected']}Mbps, "
                             f"got {bw_data['actual']}Mbps "
                             f"(error: {bw_data['error_percent']:.1f}%)")
        
        # 检查延迟
        if comparison['delay']:
            delay_data = comparison['delay']
            if not delay_data['within_tolerance']:
                passed = False
                reasons.append(f"Delay mismatch: expected RTT {delay_data['expected_rtt']}ms, "
                             f"got {delay_data['actual_rtt']}ms "
                             f"(error: {delay_data['error_ms']:.1f}ms)")
        
        # 检查丢包
        if comparison['loss']:
            loss_data = comparison['loss']
            if not loss_data['within_tolerance']:
                passed = False
                reasons.append(f"Loss mismatch: expected {loss_data['expected']}%, "
                             f"got {loss_data['actual']}% "
                             f"(error: {loss_data['error_percent']:.1f}%)")
        
        result = {
            'passed': passed,
            'configured': config.to_dict(),
            'measured': measured.to_dict(),
            'comparison': comparison,
            'reasons': reasons,
        }
        
        if passed:
            self.logger.info("✓ Network configuration verified successfully")
        else:
            self.logger.warning("✗ Network configuration verification failed:")
            for reason in reasons:
                self.logger.warning(f"  - {reason}")
        
        return result
    
    def start_http_server(self, 
                         video_dir: str = '/home/mininet/cn/video',
                         port: int = 8000) -> Tuple[str, int]:
        """在h1节点启动HTTP服务器
        
        使用Python内置的http.server模块在h1上启动简单的HTTP服务器，
        用于提供视频文件下载。
        
        Args:
            video_dir: 视频文件目录（绝对路径）
            port: HTTP服务器端口号
            
        Returns:
            (server_ip, port): 服务器IP地址和端口号
            
        Raises:
            RuntimeError: Mininet未初始化或服务器启动失败
            
        Example:
            >>> manager.setup_mininet()
            >>> server_ip, port = manager.start_http_server('/home/mininet/cn/video', 8000)
            >>> print(f"Server: http://{server_ip}:{port}/")
        """
        if not self.initialized or not self.h1:
            raise RuntimeError("Mininet not initialized. Call setup_mininet() first.")
        
        if self.http_process:
            self.logger.warning("HTTP server already running, stopping old instance...")
            self.stop_http_server()
        
        self.logger.info(f"Starting HTTP server on h1...")
        self.logger.debug(f"  Directory: {video_dir}")
        self.logger.debug(f"  Port: {port}")
        
        try:
            # 确保视频目录存在
            check_dir = self.h1.cmd(f'test -d {video_dir} && echo "exists" || echo "not found"').strip()
            if 'not found' in check_dir:
                self.logger.warning(f"Directory {video_dir} not found, creating...")
                self.h1.cmd(f'mkdir -p {video_dir}')
            
            # 在h1上启动HTTP服务器（后台运行）
            # 使用nohup确保进程不会因为shell关闭而终止
            cmd = f'cd {video_dir} && nohup python3 -m http.server {port} > /tmp/http_server.log 2>&1 &'
            self.logger.debug(f"Running command: {cmd}")
            result = self.h1.cmd(cmd)
            
            # 等待服务器启动
            time.sleep(2)
            
            # 获取HTTP服务器进程PID
            pid_result = self.h1.cmd(f'pgrep -f "http.server {port}"').strip()
            if pid_result:
                self.logger.debug(f"HTTP server PID: {pid_result}")
                # 保存进程信息（用于后续清理）
                # 注意：这里不保存实际的Popen对象，因为使用了nohup
                self.http_process = {'pid': pid_result, 'port': port}
            else:
                raise RuntimeError("HTTP server process not found after startup")
            
            # 验证服务器可访问性
            server_ip = self.h1.IP()
            test_url = f'http://{server_ip}:{port}/'
            
            self.logger.debug(f"Testing server accessibility: {test_url}")
            
            # 使用h2的curl测试连接（最多尝试3次）
            accessible = False
            for attempt in range(3):
                result = self.h2.cmd(f'curl -s -o /dev/null -w "%{{http_code}}" --connect-timeout 2 {test_url}')
                self.logger.debug(f"Attempt {attempt + 1}: HTTP response code = {result.strip()}")
                
                if '200' in result or '301' in result:
                    accessible = True
                    break
                
                time.sleep(1)
            
            if not accessible:
                # 检查错误日志
                error_log = self.h1.cmd('tail -n 20 /tmp/http_server.log')
                self.logger.error(f"HTTP server log:\n{error_log}")
                raise RuntimeError(f"HTTP server not accessible at {test_url}")
            
            self.logger.info(f"✓ HTTP server started successfully")
            self.logger.info(f"  URL: http://{server_ip}:{port}/")
            self.logger.info(f"  Serving: {video_dir}")
            
            return server_ip, port
            
        except Exception as e:
            self.logger.error(f"Failed to start HTTP server: {e}")
            # 清理失败的进程
            self.h1.cmd(f'pkill -f "http.server {port}"')
            self.http_process = None
            raise
    
    def stop_http_server(self):
        """停止HTTP服务器
        
        优雅地停止在h1上运行的HTTP服务器进程。
        该方法可以安全地多次调用（幂等性）。
        
        Example:
            >>> manager.start_http_server()
            >>> # ... 运行实验 ...
            >>> manager.stop_http_server()
        """
        if not self.http_process:
            self.logger.debug("No HTTP server to stop")
            return
        
        try:
            self.logger.info("Stopping HTTP server...")
            
            # 如果保存了PID信息
            if isinstance(self.http_process, dict) and 'pid' in self.http_process:
                port = self.http_process.get('port')
                # 使用pkill杀掉进程
                if self.h1:
                    self.h1.cmd(f'pkill -f "http.server {port}"')
                    self.logger.info("✓ HTTP server stopped")
            # 如果是Popen对象
            elif hasattr(self.http_process, 'terminate'):
                self.http_process.terminate()
                self.http_process.wait(timeout=5)
                self.logger.info("✓ HTTP server stopped")
            
        except Exception as e:
            self.logger.warning(f"Error stopping HTTP server: {e}")
            # 尝试强制杀掉
            try:
                if isinstance(self.http_process, dict):
                    port = self.http_process.get('port', 8000)
                    if self.h1:
                        self.h1.cmd(f'pkill -9 -f "http.server {port}"')
                elif hasattr(self.http_process, 'kill'):
                    self.http_process.kill()
            except:
                pass
        finally:
            self.http_process = None
    
    def open_cli(self):
        """打开Mininet CLI（调试用）
        
        用于交互式调试和手动测试
        
        Example:
            >>> manager.setup_mininet()
            >>> manager.open_cli()  # 进入Mininet CLI
            mininet> h1 ifconfig
            mininet> h2 ping h1
        """
        if not self.net:
            self.logger.error("Mininet not initialized, cannot open CLI")
            return
        
        self.logger.info("Opening Mininet CLI (type 'exit' to close)")
        CLI(self.net)

    def _save_experiment_config(self,
                               scenario_name: Optional[str] = None,
                               network_config: Optional[NetworkConfig] = None,
                               server_ip: str = None,
                               server_port: int = None,
                               video_dir: str = None):
        """保存实验配置到YAML文件（Story 6.2）
        
        生成完整的实验配置文件，包含所有必要参数以重现实验。
        
        Args:
            scenario_name: 场景名称
            network_config: 网络配置对象
            server_ip: HTTP服务器IP
            server_port: HTTP服务器端口
            video_dir: 视频文件目录
        """
        try:
            import yaml
            from datetime import datetime
            
            # 构建配置数据
            config_data = {
                # 实验基本信息
                'experiment': {
                    'id': self.exp_id,
                    'timestamp': datetime.now().isoformat(),
                    'scenario': scenario_name or 'default',
                },
                
                # 网络配置
                'network': network_config.to_dict() if network_config else {
                    'bandwidth': None,
                    'delay': None,
                    'loss': None,
                    'jitter': None
                },
                
                # 服务器配置
                'server': {
                    'ip': server_ip or (self.h1.IP() if self.h1 else 'unknown'),
                    'port': server_port or 8000,
                    'video_dir': video_dir or '/home/mininet/cn/video'
                },
                
                # 客户端配置
                'client': {
                    'ip': self.h2.IP() if self.h2 else 'unknown',
                    'interface': 'h2-eth0'
                },
                
                # 监测配置（默认值）
                'monitoring': {
                    'window_size': 1.0,
                    'update_interval': 1,
                    'capture_filter': 'tcp'
                },
                
                # 输出配置（默认值）
                'output': {
                    'features_csv': True,
                    'timeline_json': True,
                    'summary_report': True,
                    'enable_color': True
                }
            }
            
            # 保存为YAML文件
            config_file = self.exp_dir / 'config.yaml'
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            self.logger.info(f"  Config saved to: {config_file}")
            
            # 同时保存JSON格式（向后兼容）
            import json
            config_json = self.exp_dir / 'config.json'
            with open(config_json, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save experiment config: {e}")
            return False
    
    def setup_experiment(self, 
                        scenario_name: Optional[str] = None,
                        network_config: Optional[NetworkConfig] = None,
                        video_dir: str = '/home/mininet/cn/video',
                        http_port: int = 8000,
                        base_exp_dir: Path = None) -> ExperimentContext:
        """设置完整实验环境（Story 2.6）
        
        一次性完成所有实验初始化步骤。
        
        Args:
            scenario_name: 场景名称
            network_config: 网络配置
            video_dir: 视频目录
            http_port: HTTP端口
            base_exp_dir: 实验基础目录
            
        Returns:
            ExperimentContext对象
        """
        try:
            self.logger.info("=" * 70)
            self.logger.info("Setting up experiment environment...")
            self.logger.info("=" * 70)
            
            # 步骤1: 生成实验ID
            self.logger.info("[1/7] Generating experiment ID")
            self.exp_id = generate_exp_id()
            
            if base_exp_dir is None:
                base_exp_dir = Path('experiments')
            else:
                base_exp_dir = Path(base_exp_dir)
            
            self.exp_dir = base_exp_dir / self.exp_id
            ensure_dir(self.exp_dir)
            self.logger.info(f"  Experiment ID: {self.exp_id}")
            self.logger.info(f"  Experiment directory: {self.exp_dir}")
            
            # 步骤2: 初始化Ground Truth
            self.logger.info("[2/7] Initializing Ground Truth")
            self.ground_truth = GroundTruth(experiment_id=self.exp_id, logger=self.logger)
            if scenario_name:
                self.ground_truth.set_scenario(scenario_name)
            self.ground_truth.add_event('experiment_setup_start')
            
            # 步骤3: 初始化Mininet
            self.logger.info("[3/7] Initializing Mininet topology")
            self.h1, self.h2 = self.setup_mininet()
            self.ground_truth.add_event('mininet_initialized', {
                'h1_ip': self.h1.IP(),
                'h2_ip': self.h2.IP()
            })
            self.logger.info(f"  Server (h1): {self.h1.IP()}")
            self.logger.info(f"  Client (h2): {self.h2.IP()}")
            
            # 步骤4: 配置网络条件
            if network_config:
                self.logger.info("[4/7] Configuring network conditions")
                self.configure_network_conditions(network_config)
                self.network_config = network_config
                self.ground_truth.record_network_config(network_config)
                self.ground_truth.add_event('network_configured')
                self.logger.info(f"  Network: BW={network_config.bandwidth}, "
                               f"Delay={network_config.delay}, Loss={network_config.loss}")
            else:
                self.logger.info("[4/7] Skipping network configuration")
            
            # 步骤5: 启动HTTP服务器
            self.logger.info("[5/7] Starting HTTP server")
            server_ip, server_port = self.start_http_server(video_dir=video_dir, port=http_port)
            self.ground_truth.add_event('http_server_started', {
                'ip': server_ip,
                'port': server_port,
                'video_dir': video_dir
            })
            self.logger.info(f"  HTTP Server: http://{server_ip}:{server_port}/")
            
            # 步骤6: 保存配置文件（YAML格式 - Story 6.2）
            self.logger.info("[6/7] Saving experiment configuration")
            self._save_experiment_config(
                scenario_name=scenario_name,
                network_config=network_config,
                server_ip=server_ip,
                server_port=server_port,
                video_dir=video_dir
            )
            
            # 步骤7: 创建ExperimentContext
            self.logger.info("[7/7] Creating experiment context")
            context = ExperimentContext(
                exp_id=self.exp_id,
                exp_dir=self.exp_dir,
                server_ip=server_ip,
                server_port=server_port,
                client_ip=self.h2.IP(),
                capture_interface='h2-eth0',
                pcap_path=self.exp_dir / 'capture.pcap',
                scenario_name=scenario_name
            )
            
            self.ground_truth.add_event('experiment_setup_complete')
            
            self.logger.info("=" * 70)
            self.logger.info(f"Experiment setup complete: {self.exp_id}")
            self.logger.info("=" * 70)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Experiment setup failed: {e}", exc_info=True)
            self.cleanup()
            raise RuntimeError(f"Failed to setup experiment: {e}") from e
    
    def finalize_experiment(self, success: bool = True, error_msg: Optional[str] = None) -> Dict[str, Any]:
        """完成实验并生成报告（Story 2.6）
        
        Args:
            success: 实验是否成功完成
            error_msg: 错误信息（如果有）
            
        Returns:
            实验摘要字典
        """
        self.logger.info("=" * 70)
        self.logger.info("Finalizing experiment...")
        self.logger.info("=" * 70)
        
        try:
            # 记录结束事件
            if self.ground_truth:
                if success:
                    self.ground_truth.add_event('experiment_end', {'status': 'success'})
                else:
                    self.ground_truth.add_event('experiment_end', {
                        'status': 'failed',
                        'error': error_msg
                    })
            
            # 生成摘要
            summary = {
                'exp_id': self.exp_id,
                'status': 'success' if success else 'failed',
                'exp_dir': str(self.exp_dir) if self.exp_dir else None,
            }
            
            if self.ground_truth:
                gt_summary = self.ground_truth.get_summary()
                summary.update(gt_summary)
            
            if error_msg:
                summary['error'] = error_msg
            
            # 执行清理（会自动保存Ground Truth）
            self.cleanup()
            
            self.logger.info("=" * 70)
            if success:
                self.logger.info(f"Experiment finalized successfully: {self.exp_id}")
            else:
                self.logger.error(f"Experiment finalized with errors: {self.exp_id}")
            self.logger.info("=" * 70)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error during finalization: {e}", exc_info=True)
            try:
                self.cleanup()
            except:
                pass
            return {
                'exp_id': self.exp_id,
                'status': 'error',
                'error': str(e)
            }

    def cleanup(self):
        """清理实验环境
        
        优雅地清理所有资源：
        - 保存Ground Truth（如果存在）
        - 停止HTTP服务器
        - 停止Mininet网络
        - 释放所有网络资源
        
        该方法可以多次调用（幂等性）
        
        Example:
            >>> manager = ExperimentManager()
            >>> manager.setup_mininet()
            >>> # ... 运行实验 ...
            >>> manager.cleanup()  # 确保资源释放
        """
        self.logger.info("Cleaning up experiment environment...")
        
        # 保存Ground Truth（如果存在且有实验目录）
        if self.ground_truth and self.exp_dir:
            try:
                self.ground_truth.add_event('experiment_cleanup')
                gt_path = Path(self.exp_dir) / 'ground_truth.json'
                self.ground_truth.save(gt_path)
                self.logger.info(f"✓ Ground Truth saved to {gt_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save Ground Truth: {e}")
        
        # 停止HTTP服务器
        self.stop_http_server()
        
        # 停止Mininet
        if self.net:
            try:
                self.logger.debug("Stopping Mininet network")
                self.net.stop()
                self.logger.info("✓ Mininet network stopped")
            except Exception as e:
                self.logger.warning(f"Error stopping Mininet: {e}")
            finally:
                self.net = None
                self.h1 = None
                self.h2 = None
                self.initialized = False
        
        # 最终清理
        try:
            self._cleanup_old_mininet()
        except Exception as e:
            self.logger.debug(f"Final cleanup warning: {e}")
        
        self.logger.info("✓ Cleanup complete")
    
    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持with语句自动清理"""
        self.cleanup()
        return False  # 不抑制异常
    
    def __del__(self):
        """析构时确保清理"""
        try:
            if self.initialized:
                self.cleanup()
        except:
            pass  # 析构函数中不应抛出异常


if __name__ == '__main__':
    """测试模块"""
    from video_qoe.utils.logger import setup_logger
    
    # 设置日志
    logger = setup_logger('video_qoe.test', 'DEBUG')
    
    print("=" * 70)
    print("Testing ExperimentManager - Story 2.1")
    print("=" * 70)
    
    # 测试1: 基本初始化和清理
    print("\n[Test 1] Basic initialization and cleanup")
    try:
        manager = ExperimentManager(logger)
        h1, h2 = manager.setup_mininet()
        
        print(f"[ok] Setup successful")
        print(f"  Server (h1): {h1.IP()}")
        print(f"  Client (h2): {h2.IP()}")
        
        # 获取节点信息
        node_info = manager.get_node_info()
        print(f"\nNode Info:")
        for node_name, info in node_info.items():
            print(f"  {node_name}: IP={info['ip']}, MAC={info['mac']}")
            print(f"    Interfaces: {info['interfaces']}")
        
        # 清理
        manager.cleanup()
        print("[ok] Cleanup successful")
        
    except Exception as e:
        print(f"[error] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 使用with语句
    print("\n[Test 2] Context manager (with statement)")
    try:
        with ExperimentManager(logger) as manager:
            h1, h2 = manager.setup_mininet()
            print(f"[ok] Setup in context manager")
            print(f"  h1: {h1.IP()}, h2: {h2.IP()}")
        
        print("[ok] Auto cleanup on exit")
        
    except Exception as e:
        print(f"[error] Test failed: {e}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


