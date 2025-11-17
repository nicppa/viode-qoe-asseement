"""
日志管理模块
Logging Management Module

提供统一的日志记录、格式化和管理功能。
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Union
from datetime import datetime


# 日志级别映射
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

# 默认日志格式
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 彩色输出格式（Rich不可用时使用）
COLOR_FORMAT = {
    'DEBUG': '\033[36m',      # Cyan
    'INFO': '\033[32m',       # Green
    'WARNING': '\033[33m',    # Yellow
    'ERROR': '\033[31m',      # Red
    'CRITICAL': '\033[35m',   # Magenta
    'RESET': '\033[0m',       # Reset
}


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    def __init__(self, fmt=None, datefmt=None, use_color=True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color
    
    def format(self, record):
        if self.use_color and sys.stdout.isatty():
            # 为不同级别添加颜色
            levelname = record.levelname
            if levelname in COLOR_FORMAT:
                record.levelname = (
                    f"{COLOR_FORMAT[levelname]}{levelname}{COLOR_FORMAT['RESET']}"
                )
        
        return super().format(record)


class ExperimentLogger:
    """实验日志管理器
    
    支持：
    - 多级别日志输出
    - 控制台和文件双输出
    - 日志文件轮转
    - 实验目录独立日志
    - 与ConfigManager集成
    """
    
    def __init__(
        self,
        name: str = 'video_qoe',
        level: Union[str, int] = 'INFO',
        log_file: Optional[Path] = None,
        console_output: bool = True,
        file_output: bool = True,
        use_color: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        rotation_type: str = 'size',  # 'size' or 'time'
    ):
        """初始化日志管理器
        
        Args:
            name: Logger名称
            level: 日志级别（'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'）
            log_file: 日志文件路径
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            use_color: 是否使用彩色输出（控制台）
            max_bytes: 日志文件最大字节数（rotation_type='size'时）
            backup_count: 保留的备份文件数量
            rotation_type: 轮转类型（'size'按大小, 'time'按时间）
        """
        self.name = name
        self.log_file = Path(log_file) if log_file else None
        self.level = self._parse_level(level)
        
        # 获取或创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # 清除已存在的handlers（避免重复）
        self.logger.handlers.clear()
        
        # 创建格式化器
        console_formatter = ColoredFormatter(
            fmt=DEFAULT_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
            use_color=use_color
        )
        file_formatter = logging.Formatter(
            fmt=DEFAULT_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT
        )
        
        # 添加控制台handler
        if console_output:
            # 确保stdout使用UTF-8编码
            import io
            if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding != 'utf-8':
                stdout_wrapper = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                console_handler = logging.StreamHandler(stdout_wrapper)
            else:
                console_handler = logging.StreamHandler(sys.stdout)
            
            console_handler.setLevel(self.level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # 添加文件handler
        if file_output and self.log_file:
            # 确保日志目录存在
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            if rotation_type == 'size':
                # 按大小轮转
                file_handler = RotatingFileHandler(
                    filename=str(self.log_file),
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8',
                    errors='replace'  # 替换无法编码的字符
                )
            elif rotation_type == 'time':
                # 按时间轮转（每天）
                file_handler = TimedRotatingFileHandler(
                    filename=str(self.log_file),
                    when='midnight',
                    interval=1,
                    backupCount=backup_count,
                    encoding='utf-8',
                    errors='replace'  # 替换无法编码的字符
                )
            else:
                raise ValueError(f"Invalid rotation_type: {rotation_type}")
            
            file_handler.setLevel(self.level)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # 防止日志向上传播
        self.logger.propagate = False
        
        self.logger.debug(f"Logger '{name}' initialized with level {logging.getLevelName(self.level)}")
    
    def _parse_level(self, level: Union[str, int]) -> int:
        """解析日志级别"""
        if isinstance(level, int):
            return level
        
        level_upper = level.upper()
        if level_upper in LOG_LEVELS:
            return LOG_LEVELS[level_upper]
        else:
            raise ValueError(f"Invalid log level: {level}")
    
    def get_logger(self) -> logging.Logger:
        """获取Logger实例"""
        return self.logger
    
    def set_level(self, level: Union[str, int]):
        """动态设置日志级别"""
        new_level = self._parse_level(level)
        self.logger.setLevel(new_level)
        for handler in self.logger.handlers:
            handler.setLevel(new_level)
        self.logger.info(f"Log level changed to {logging.getLevelName(new_level)}")
    
    def add_file_handler(
        self,
        log_file: Path,
        level: Optional[Union[str, int]] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5
    ):
        """添加额外的文件handler
        
        用于实验运行时添加实验特定的日志文件
        """
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
            errors='replace'  # 替换无法编码的字符
        )
        
        handler_level = self._parse_level(level) if level else self.level
        file_handler.setLevel(handler_level)
        
        formatter = logging.Formatter(
            fmt=DEFAULT_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.info(f"Added file handler: {log_file}")
        
        return file_handler
    
    def remove_file_handlers(self):
        """移除所有文件handlers"""
        for handler in self.logger.handlers[:]:
            if isinstance(handler, (RotatingFileHandler, TimedRotatingFileHandler)):
                handler.close()
                self.logger.removeHandler(handler)
    
    def debug(self, msg, *args, **kwargs):
        """DEBUG级别日志"""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        """INFO级别日志"""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """WARNING级别日志"""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """ERROR级别日志"""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """CRITICAL级别日志"""
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        """记录异常信息（包含堆栈跟踪）"""
        self.logger.exception(msg, *args, **kwargs)


# 全局logger实例
_global_logger: Optional[ExperimentLogger] = None


def setup_logger(
    name: str = 'video_qoe',
    level: Union[str, int] = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    file_output: bool = True,
    use_color: bool = True,
    **kwargs
) -> ExperimentLogger:
    """设置并返回日志管理器（便捷函数）
    
    Args:
        name: Logger名称
        level: 日志级别
        log_file: 日志文件路径
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        use_color: 是否使用彩色输出
        **kwargs: 其他ExperimentLogger参数
    
    Returns:
        ExperimentLogger实例
    
    Example:
        >>> logger = setup_logger('video_qoe', 'INFO', 'logs/app.log')
        >>> logger.info("Application started")
        >>> logger.warning("High latency detected")
    """
    global _global_logger
    
    log_file_path = Path(log_file) if log_file else None
    
    _global_logger = ExperimentLogger(
        name=name,
        level=level,
        log_file=log_file_path,
        console_output=console_output,
        file_output=file_output,
        use_color=use_color,
        **kwargs
    )
    
    return _global_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取Logger实例（便捷函数）
    
    Args:
        name: Logger名称，如果为None则返回全局logger
    
    Returns:
        logging.Logger实例
    
    Example:
        >>> logger = get_logger('video_qoe.features')
        >>> logger.debug("Calculating features...")
    """
    if name is None:
        if _global_logger is None:
            # 如果全局logger未初始化，创建默认logger
            setup_logger()
        return _global_logger.logger
    else:
        return logging.getLogger(name)


def setup_experiment_logger(
    experiment_dir: Path,
    experiment_name: str,
    level: Union[str, int] = 'INFO',
    console_output: bool = True,
    use_color: bool = True,
) -> ExperimentLogger:
    """为特定实验设置日志系统
    
    创建实验目录下的独立日志文件
    
    Args:
        experiment_dir: 实验目录
        experiment_name: 实验名称
        level: 日志级别
        console_output: 是否输出到控制台
        use_color: 是否使用彩色输出
    
    Returns:
        ExperimentLogger实例
    
    Example:
        >>> exp_dir = Path('experiments/exp_001')
        >>> logger = setup_experiment_logger(exp_dir, 'Low Bandwidth Test')
        >>> logger.info("Experiment started")
    """
    # 创建实验目录
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志文件路径
    log_file = experiment_dir / 'monitor.log'
    
    # 创建logger
    logger = setup_logger(
        name=f'video_qoe.experiment.{experiment_name}',
        level=level,
        log_file=log_file,
        console_output=console_output,
        file_output=True,
        use_color=use_color,
        rotation_type='size',
        max_bytes=50 * 1024 * 1024,  # 50MB for experiments
        backup_count=3
    )
    
    # 记录实验开始
    logger.info("=" * 70)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Directory: {experiment_dir}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    return logger


