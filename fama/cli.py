"""FAMA 项目的命令行接口模块。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from fama.mining.orchestrator import PromptOrchestrator
from fama.utils.io import read_yaml, write_yaml


def _build_parser() -> "argparse.ArgumentParser":
    """构建带有 CSS/CoE 开关的单次挖掘 CLI 解析器。"""

    parser = argparse.ArgumentParser(description="FAMA single-run factor miner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine_parser = subparsers.add_parser("mine", help="Generate new factors via CSS/CoE + LLM prompt")
    mine_parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config" / "defaults.yaml"),
        help="Path to the YAML configuration file.",
    )
    mine_parser.add_argument("--skip-css", action="store_true", help="Disable CSS exemplar selection.")
    mine_parser.add_argument("--skip-coe", action="store_true", help="Disable CoE context building.")
    mine_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write generated expressions as YAML.",
    )
    mine_parser.set_defaults(func=handle_mine)

    return parser


def handle_mine(args: "argparse.Namespace") -> None:
    """处理 ``mine`` 子命令，并传递跳过标志给 PromptOrchestrator。"""

    config = _load_config(args.config)
    orchestrator = PromptOrchestrator(config)
    expressions = orchestrator.run(use_css=not args.skip_css, use_coe=not args.skip_coe)
    if args.output:
        write_yaml(args.output, {"expressions": expressions})
        print(f"Wrote {len(expressions)} expressions to {args.output}")
        return

    if expressions:
        print("Generated expressions:")
        for expr in expressions:
            print(f"- {expr}")
    else:
        print(
            "No expressions were generated. Ensure CSS/CoE toggles are enabled "
            "and provide a valid LLM API key if you expect live suggestions."
        )


def main() -> None:
    """CLI 入口函数，负责拼接解析器与各个处理函数。"""

    parser = _build_parser()
    args = parser.parse_args()
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return
    handler(args)


def _load_config(path: str) -> Dict[str, Any]:
    """加载默认配置，并合并来自 ``path`` 的可选覆盖配置。"""

    default_path = Path(__file__).resolve().parent / "config" / "defaults.yaml"
    project_root = default_path.parents[2]
    config = read_yaml(str(default_path))
    if not path:
        _normalize_paths(config, project_root)
        config["_config_path"] = str(default_path.resolve())
        return config

    resolved = _resolve_config_path(path)
    if resolved is None:
        raise FileNotFoundError(
            f"Config file '{path}' not found relative to current directory or package root."
        )
    if resolved.resolve() == default_path.resolve():
        _normalize_paths(config, project_root)
        config["_config_path"] = str(default_path.resolve())
        return config

    user_cfg = read_yaml(str(resolved))
    config = _merge_dicts(config, user_cfg)
    _normalize_paths(config, project_root)
    config["_config_path"] = str(resolved.resolve())
    return config


def _resolve_config_path(path: str) -> Path | None:
    """解析配置文件路径，以兼容多种启动位置。"""

    candidate = Path(path)
    if candidate.exists():
        return candidate
    pkg_root = Path(__file__).resolve().parent.parent
    alt = pkg_root / path
    if alt.exists():
        return alt
    cwd_alt = Path.cwd() / path
    if cwd_alt.exists():
        return cwd_alt
    return None


def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """递归地合并两份配置字典。"""

    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_paths(config: Dict[str, Any], project_root: Path) -> None:
    """将配置中的 paths 统一为绝对路径（相对路径按项目根目录解释）。"""

    paths = config.get("paths")
    if not isinstance(paths, dict):
        return
    for key, value in list(paths.items()):
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        paths[key] = str(candidate)


if __name__ == "__main__":
    for i in range(5):
        print(f"第{i}次迭代：")
        main()
