from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from fama.data.factor_space import FactorSet, deserialize_factor_set, serialize_factor_set

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path_like: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _build_base_source_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    paths = cfg.get("paths", {}) if isinstance(cfg, Mapping) else {}
    return {
        "alpha101": _resolve_project_path(paths.get("base_alpha101_cache", "./data/factor_cache_new/alpha101.yaml")),
        "alpha158": _resolve_project_path(paths.get("base_alpha158_cache", "./data/factor_cache_new/alpha158.yaml")),
    }


def resolve_base_factor_cache(
    cfg: Mapping[str, Any] | None,
    *,
    selected_sources: Iterable[str] | None = None,
    include_llm_library_in_base: bool | None = None,
) -> tuple[Path, list[str]]:
    """Resolve and persist the runtime base factor cache from configured sources.

    Returns:
        resolved_path: merged base factor cache path (written every call).
        selected_sources: effective source names used for this merge.
    """

    cfg = cfg or {}
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping.")

    paths = cfg.get("paths", {}) if isinstance(cfg, Mapping) else {}
    base_cfg = cfg.get("base_catalog", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(base_cfg, Mapping):
        base_cfg = {}

    source_paths = _build_base_source_paths(cfg)
    llm_library_path = _resolve_project_path(
        paths.get("llm_factor_library", "./data/factor_cache_new/LLM_library.yaml")
    )
    resolved_path = _resolve_project_path(
        paths.get("base_factor_cache_resolved", "./tmp/base_factor_cache_resolved.yaml")
    )

    selected = list(selected_sources) if selected_sources is not None else list(
        base_cfg.get("selected_sources", ["alpha101", "alpha158"])
    )
    selected = [str(item).strip() for item in selected if str(item).strip()]
    if not selected:
        raise ValueError("base_catalog.selected_sources 不能为空，至少指定一个基础因子源。")

    unknown = [name for name in selected if name not in source_paths]
    if unknown:
        raise ValueError(f"未知 base 源: {unknown}。可选: {sorted(source_paths.keys())}")

    include_llm = (
        bool(include_llm_library_in_base)
        if include_llm_library_in_base is not None
        else bool(base_cfg.get("include_llm_library_in_base", False))
    )

    source_items: list[tuple[str, Path]] = [(name, source_paths[name]) for name in selected]
    if include_llm:
        source_items.append(("llm_library", llm_library_path))

    merged: list[Any] = []
    by_name: dict[str, tuple[str, str]] = {}
    for source_name, source_path in source_items:
        if not source_path.exists():
            raise FileNotFoundError(f"base 源不存在: {source_name} -> {source_path}")
        fs = deserialize_factor_set(str(source_path))
        if source_name == "alpha101":
            overlap = sum(1 for factor in fs.factors if factor.name.startswith("alpha158_"))
            if overlap:
                print(
                    f"[factor_catalog] Warning: alpha101 源中检测到 {overlap} 个 alpha158_ 因子，"
                    "将按源文件原样参与合并。"
                )
        for factor in fs.factors:
            existing = by_name.get(factor.name)
            if existing is None:
                by_name[factor.name] = (factor.expression, source_name)
                merged.append(factor)
                continue
            existing_expr, existing_source = existing
            if existing_expr != factor.expression:
                raise ValueError(
                    f"因子重名但表达式不一致: {factor.name} "
                    f"(来源 {existing_source} vs {source_name})"
                )

    if not merged:
        source_label = ", ".join(name for name, _ in source_items)
        raise ValueError(f"合并后 base 因子为空，请检查源文件内容: {source_label}")

    serialize_factor_set(FactorSet(merged), str(resolved_path))
    return resolved_path, [name for name, _ in source_items]


def load_factor_name_set(path: str | Path) -> set[str]:
    fs = deserialize_factor_set(str(path))
    return {factor.name for factor in fs.factors}
