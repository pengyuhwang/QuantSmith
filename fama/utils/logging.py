"""FAMA 框架的日志工具。"""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str) -> "logging.Logger":
    """按照 README 中“Instrumentation”说明返回配置好的 logger。

    Args:
        name: 调用模块希望使用的日志名称。

    Returns:
        配置了基础格式化器的 :class:`logging.Logger` 实例。
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler: Optional[logging.Handler] = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
