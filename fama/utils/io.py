"""与 README “单次流程”对应的 IO 辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def read_yaml(path: str) -> dict:
    """按 README 描述从 ``path`` 读取 YAML 配置。

    Args:
        path: YAML 文件路径，例如 ``fama/config/defaults.yaml``。

    Returns:
        解析后的字典；若文件不存在则返回空字典。
    """

    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: str, data: dict) -> None:
    """按照 README 的“Artifacts & IO”小节将数据写入 YAML。

    Args:
        path: 输出文件路径。
        data: 待序列化的字典数据。
    """

    yaml_path = Path(path)
    ensure_dir(str(yaml_path.parent))
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True, allow_unicode=True)


def ensure_dir(path: str) -> None:
    """在写入产物前确保目录存在（README “单次流程”）。

    Args:
        path: 需要创建的目录路径。
    """

    Path(path).mkdir(parents=True, exist_ok=True)
