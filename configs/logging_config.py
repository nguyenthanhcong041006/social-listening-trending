import sys
from loguru import logger

def setup_logger(log_level: str = "INFO", log_file: str = "logs/pipeline.log"):
    """
    Configures a unified logging setup across the entire pipeline.
    Supports formatted console output and rotating file logging.
    """
    logger.remove()
    
    # Console output format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File logging with rotation and retention
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="14 days",
            compression="zip",
            encoding="utf-8"
        )
    
    return logger
