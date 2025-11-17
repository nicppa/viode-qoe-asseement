"""
工具函数模块
Helper Functions Module

提供通用的工具函数和辅助功能。
"""

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import hashlib


# ==================== 时间处理 ====================

def generate_exp_id(prefix: str = 'exp') -> str:
    """生成唯一实验ID
    
    Args:
        prefix: ID前缀，默认'exp'
    
    Returns:
        格式为 'prefix_YYYYMMDD_HHMMSS' 的唯一ID
    
    Example:
        >>> generate_exp_id()
        'exp_20251108_163450'
        >>> generate_exp_id('test')
        'test_20251108_163450'
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}"


def format_timestamp(dt: Optional[datetime] = None, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    """格式化时间戳
    
    Args:
        dt: datetime对象，默认为当前时间
        fmt: 时间格式，默认'%Y-%m-%d %H:%M:%S'
    
    Returns:
        格式化的时间字符串
    
    Example:
        >>> format_timestamp()
        '2025-11-08 16:34:50'
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(fmt)


def parse_timestamp(ts_str: str, fmt: str = '%Y-%m-%d %H:%M:%S') -> datetime:
    """解析时间字符串
    
    Args:
        ts_str: 时间字符串
        fmt: 时间格式
    
    Returns:
        datetime对象
    
    Example:
        >>> parse_timestamp('2025-11-08 16:34:50')
        datetime(2025, 11, 8, 16, 34, 50)
    """
    return datetime.strptime(ts_str, fmt)


def format_duration(seconds: float, precision: int = 2) -> str:
    """格式化时长为可读字符串
    
    Args:
        seconds: 秒数
        precision: 小数精度
    
    Returns:
        格式化的时长字符串
    
    Example:
        >>> format_duration(65)
        '1m 5s'
        >>> format_duration(3665)
        '1h 1m 5s'
        >>> format_duration(0.5)
        '0.50s'
    """
    if seconds < 1:
        return f"{seconds:.{precision}f}s"
    
    parts = []
    
    # 计算小时、分钟、秒
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return ' '.join(parts)


class Timer:
    """简单的计时器类"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = datetime.now()
        self.end_time = None
        return self
    
    def stop(self):
        """停止计时"""
        self.end_time = datetime.now()
        return self
    
    def elapsed(self) -> float:
        """获取经过的秒数"""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else datetime.now()
        return (end - self.start_time).total_seconds()
    
    def elapsed_str(self) -> str:
        """获取格式化的经过时间"""
        return format_duration(self.elapsed())
    
    def __enter__(self):
        """支持with语句"""
        self.start()
        return self
    
    def __exit__(self, *args):
        """支持with语句"""
        self.stop()


# ==================== 路径处理 ====================

def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在，如不存在则创建
    
    Args:
        path: 目录路径
    
    Returns:
        Path对象
    
    Example:
        >>> ensure_dir('experiments/exp_001')
        Path('experiments/exp_001')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """获取项目根目录
    
    从当前文件向上查找包含setup.py或video_qoe目录的父目录
    
    Returns:
        项目根目录Path对象
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'setup.py').exists() or (parent / 'video_qoe').exists():
            return parent
    return Path.cwd()


def get_relative_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """获取相对于基准目录的相对路径
    
    Args:
        path: 目标路径
        base: 基准目录，默认为项目根目录
    
    Returns:
        相对路径
    """
    path = Path(path).resolve()
    if base is None:
        base = get_project_root()
    else:
        base = Path(base).resolve()
    
    try:
        return path.relative_to(base)
    except ValueError:
        # 如果不在同一路径下，返回绝对路径
        return path


def safe_filename(name: str, max_length: int = 255) -> str:
    """将字符串转换为安全的文件名
    
    Args:
        name: 原始名称
        max_length: 最大长度
    
    Returns:
        安全的文件名
    
    Example:
        >>> safe_filename('Test/Experiment: #1')
        'Test_Experiment_1'
    """
    # 替换非法字符
    safe = re.sub(r'[<>:"/\\|?*]', '_', name)
    # 移除多余的空格和下划线
    safe = re.sub(r'[_\s]+', '_', safe)
    # 去除首尾下划线
    safe = safe.strip('_')
    # 限制长度
    if len(safe) > max_length:
        safe = safe[:max_length]
    return safe


def file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法 ('md5', 'sha1', 'sha256')
    
    Returns:
        哈希值字符串
    """
    file_path = Path(file_path)
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


# ==================== 单位转换 ====================

def parse_bandwidth(bw_str: str) -> float:
    """解析带宽字符串，返回Mbps
    
    Args:
        bw_str: 带宽字符串，如 "10Mbps", "2Gbps", "500Kbps"
    
    Returns:
        带宽值（Mbps）
    
    Example:
        >>> parse_bandwidth('10Mbps')
        10.0
        >>> parse_bandwidth('2Gbps')
        2000.0
        >>> parse_bandwidth('500Kbps')
        0.5
    """
    bw_str = bw_str.strip().replace(' ', '')
    
    if 'Gbps' in bw_str:
        return float(bw_str.replace('Gbps', '')) * 1000
    elif 'Mbps' in bw_str:
        return float(bw_str.replace('Mbps', ''))
    elif 'Kbps' in bw_str:
        return float(bw_str.replace('Kbps', '')) / 1000
    elif 'bps' in bw_str:
        return float(bw_str.replace('bps', '')) / 1_000_000
    else:
        # 假设为Mbps
        return float(bw_str)


def parse_delay(delay_str: str) -> float:
    """解析延迟字符串，返回毫秒
    
    Args:
        delay_str: 延迟字符串，如 "20ms", "0.5s"
    
    Returns:
        延迟值（毫秒）
    
    Example:
        >>> parse_delay('20ms')
        20.0
        >>> parse_delay('0.5s')
        500.0
    """
    delay_str = delay_str.strip().replace(' ', '')
    
    if 's' in delay_str and 'ms' not in delay_str:
        return float(delay_str.replace('s', '')) * 1000
    elif 'ms' in delay_str:
        return float(delay_str.replace('ms', ''))
    else:
        # 假设为毫秒
        return float(delay_str)


def parse_percentage(percent_str: str) -> float:
    """解析百分比字符串
    
    Args:
        percent_str: 百分比字符串，如 "5%", "0.5"
    
    Returns:
        百分比值（0-100）
    
    Example:
        >>> parse_percentage('5%')
        5.0
        >>> parse_percentage('0.05')
        5.0
    """
    percent_str = percent_str.strip().replace(' ', '')
    
    if '%' in percent_str:
        return float(percent_str.replace('%', ''))
    else:
        # 假设为小数形式（0-1），转换为百分比
        value = float(percent_str)
        if value <= 1.0:
            return value * 100
        return value


def format_bandwidth(bw_mbps: float, precision: int = 2) -> str:
    """格式化带宽为可读字符串
    
    Args:
        bw_mbps: 带宽（Mbps）
        precision: 小数精度
    
    Returns:
        格式化的带宽字符串
    
    Example:
        >>> format_bandwidth(10.5)
        '10.50Mbps'
        >>> format_bandwidth(2500)
        '2.50Gbps'
        >>> format_bandwidth(0.5)
        '500.00Kbps'
    """
    if bw_mbps >= 1000:
        return f"{bw_mbps/1000:.{precision}f}Gbps"
    elif bw_mbps >= 1:
        return f"{bw_mbps:.{precision}f}Mbps"
    else:
        return f"{bw_mbps*1000:.{precision}f}Kbps"


def format_size(bytes_size: int, precision: int = 2) -> str:
    """格式化文件大小为可读字符串
    
    Args:
        bytes_size: 字节数
        precision: 小数精度
    
    Returns:
        格式化的大小字符串
    
    Example:
        >>> format_size(1024)
        '1.00KB'
        >>> format_size(1048576)
        '1.00MB'
        >>> format_size(1073741824)
        '1.00GB'
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(bytes_size)
    unit_idx = 0
    
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    
    if unit_idx == 0:
        return f"{int(size)}{units[unit_idx]}"
    else:
        return f"{size:.{precision}f}{units[unit_idx]}"


# ==================== 数据验证 ====================

def validate_port(port: int) -> bool:
    """验证端口号是否有效
    
    Args:
        port: 端口号
    
    Returns:
        是否有效
    """
    return 1 <= port <= 65535


def validate_ip(ip_str: str) -> bool:
    """验证IP地址是否有效（简单验证）
    
    Args:
        ip_str: IP地址字符串
    
    Returns:
        是否有效
    """
    parts = ip_str.split('.')
    if len(parts) != 4:
        return False
    
    try:
        return all(0 <= int(part) <= 255 for part in parts)
    except ValueError:
        return False


def validate_percentage(value: float) -> bool:
    """验证百分比值是否有效
    
    Args:
        value: 百分比值
    
    Returns:
        是否有效
    """
    return 0 <= value <= 100


def clamp(value: float, min_value: float, max_value: float) -> float:
    """将值限制在指定范围内
    
    Args:
        value: 输入值
        min_value: 最小值
        max_value: 最大值
    
    Returns:
        限制后的值
    """
    return max(min_value, min(value, max_value))


# ==================== 数据格式化 ====================

def format_dict_for_display(data: Dict[str, Any], indent: int = 2) -> str:
    """格式化字典为可读字符串
    
    Args:
        data: 字典数据
        indent: 缩进空格数
    
    Returns:
        格式化的字符串
    """
    lines = []
    indent_str = ' ' * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            nested = format_dict_for_display(value, indent + 2)
            lines.append(nested)
        else:
            lines.append(f"{key}: {value}")
    
    return '\n'.join(indent_str + line for line in lines)


def truncate_string(s: str, max_length: int = 50, suffix: str = '...') -> str:
    """截断字符串
    
    Args:
        s: 输入字符串
        max_length: 最大长度
        suffix: 截断后缀
    
    Returns:
        截断后的字符串
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


# ==================== 其他工具 ====================

def retry(func, max_attempts: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """重试装饰器/函数
    
    Args:
        func: 要重试的函数
        max_attempts: 最大尝试次数
        delay: 重试延迟（秒）
        exceptions: 要捕获的异常类型
    
    Returns:
        装饰后的函数或执行结果
    """
    import time
    
    def wrapper(*args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(delay)
        return None
    
    return wrapper


def get_system_info() -> Dict[str, str]:
    """获取系统信息
    
    Returns:
        包含系统信息的字典
    """
    import platform
    
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
    }




