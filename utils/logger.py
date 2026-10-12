"""
Logging Utilities for MedSSL-BHSD.
Provides formatted console logging, TensorBoard, and Weights & Biases helpers.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any


def get_logger(name: str = "MedSSL-BHSD", log_file: Optional[str] = None) -> logging.Logger:
    """Initialize and configure a logger with clean formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
