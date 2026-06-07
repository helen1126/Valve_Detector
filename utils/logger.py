"""日志工具模块

基于 Loguru 的统一日志管理，支持控制台和文件双输出。
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days",
    encoding: str = "utf-8",
) -> None:
    """初始化日志系统

    Args:
        log_dir: 日志文件保存目录
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        rotation: 日志文件轮转大小
        retention: 日志文件保留时间
        encoding: 日志文件编码
    """
    # 移除默认的日志处理器
    logger.remove()

    # 控制台输出（彩色格式）
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # 确保日志目录存在
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 全量日志文件
    logger.add(
        log_path / "valve_detector.log",
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        rotation=rotation,
        retention=retention,
        encoding=encoding,
    )

    # 错误日志文件（仅记录 WARNING 及以上级别）
    logger.add(
        log_path / "error.log",
        level="WARNING",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        rotation=rotation,
        retention=retention,
        encoding=encoding,
    )

    logger.info("日志系统初始化完成，日志目录: {}", log_path.resolve())


def get_logger():
    """获取日志记录器实例

    Returns:
        loguru.Logger: 日志记录器实例
    """
    return logger
