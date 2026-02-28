from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .models import SelectionConfig


def load_selection_config(cfg: Mapping[str, Any] | None) -> SelectionConfig:
    cfg = cfg or {}
    selection_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection"), Mapping) else {}
    workflow_cfg = cfg.get("workflow", {}) if isinstance(cfg.get("workflow"), Mapping) else {}

    def _pick(key: str, legacy_key: str | None, default: Any) -> Any:
        if key in selection_cfg:
            return selection_cfg.get(key)
        if legacy_key and legacy_key in workflow_cfg:
            return workflow_cfg.get(legacy_key)
        return default

    scope_cfg = selection_cfg.get("scope", {}) if isinstance(selection_cfg.get("scope"), Mapping) else {}

    config = SelectionConfig(
        ric_threshold=float(_pick("ric_threshold", "ric_threshold", 0.08)),
        corr_threshold=float(_pick("corr_threshold", "corr_threshold", 0.65)),
        llm_self_corr_threshold=float(_pick("llm_self_corr_threshold", "llm_self_corr_threshold", 0.7)),
        min_corr_obs=int(_pick("min_corr_obs", "min_corr_obs", 1000)),
        require_full_asset_coverage=bool(_pick("require_full_asset_coverage", None, True)),
        enable_new_vs_old_llm=bool(_pick("enable_new_vs_old_llm", None, True)),
        enable_new_vs_new_llm=bool(_pick("enable_new_vs_new_llm", None, True)),
        dedup_strategy=str(_pick("dedup_strategy", None, "greedy_by_min_abs_ric")),
        log_topk=max(1, int(_pick("log_topk", None, 3))),
        scope_use_ric_assets=bool(scope_cfg.get("use_ric_assets", True)),
        scope_use_ric_window=bool(scope_cfg.get("use_ric_window", True)),
    )

    if config.min_corr_obs <= 0:
        raise ValueError(f"selection.min_corr_obs must be positive, got {config.min_corr_obs}.")
    if config.dedup_strategy != "greedy_by_min_abs_ric":
        raise ValueError(
            f"Unsupported selection.dedup_strategy={config.dedup_strategy}; "
            "only 'greedy_by_min_abs_ric' is currently supported."
        )
    return config


def apply_selection_overrides(
    config: SelectionConfig,
    *,
    ric_threshold: float | None = None,
    corr_threshold: float | None = None,
    llm_self_corr_threshold: float | None = None,
    min_corr_obs: int | None = None,
) -> SelectionConfig:
    patched = config
    if ric_threshold is not None:
        patched = replace(patched, ric_threshold=float(ric_threshold))
    if corr_threshold is not None:
        patched = replace(patched, corr_threshold=float(corr_threshold))
    if llm_self_corr_threshold is not None:
        patched = replace(patched, llm_self_corr_threshold=float(llm_self_corr_threshold))
    if min_corr_obs is not None:
        patched = replace(patched, min_corr_obs=int(min_corr_obs))
    if patched.min_corr_obs <= 0:
        raise ValueError(f"min_corr_obs must be positive, got {patched.min_corr_obs}.")
    return patched

