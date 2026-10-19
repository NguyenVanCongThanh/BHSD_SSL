"""
Logging Utilities for MedSSL-BHSD.
Provides formatted console logging, TensorBoard, and Weights & Biases helpers.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any


def get_logger(
    name: Optional[str] = None,
    log_file: Optional[str] = None,
    is_main: bool = True
) -> logging.Logger:
    """Initialize and configure a logger with clean formatting.

    Args:
        name: Name of the logger. If None or empty, configures the root logger.
        log_file: Optional path to a file where logs should be written.
        is_main: True if current process is main process (rank 0).
                 Non-main processes only log WARNING and higher to avoid console clutter.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if is_main else logging.WARNING)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file and is_main:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
