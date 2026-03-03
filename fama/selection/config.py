from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .models import SelectionConfig


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _pick_from_mappings(selection_cfg: Mapping[str, Any], workflow_cfg: Mapping[str, Any], key: str, legacy_key: str | None, default: Any) -> Any:
    if key in selection_cfg:
        return selection_cfg.get(key)
    if legacy_key and legacy_key in workflow_cfg:
        return workflow_cfg.get(legacy_key)
    return default


def _read_criterion(
    criteria_cfg: Mapping[str, Any],
    name: str,
    *,
    enabled_default: bool,
    threshold_default: float,
    legacy_threshold: Any = None,
    legacy_enabled: Any = None,
) -> tuple[bool, float]:
    raw = criteria_cfg.get(name)
    node = raw if isinstance(raw, Mapping) else {}
    enabled = bool(node.get("enabled", legacy_enabled if legacy_enabled is not None else enabled_default))
    threshold = float(node.get("threshold", legacy_threshold if legacy_threshold is not None else threshold_default))
    return enabled, threshold


def _clean_optional_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_assets(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        token = value.strip()
        return (token,) if token else ()
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return tuple(items)
    text = str(value).strip()
    return (text,) if text else ()


def _resolve_asset_mode(scope_cfg: Mapping[str, Any]) -> str:
    raw_mode = scope_cfg.get("asset_mode")
    if raw_mode is None:
        legacy_flag = scope_cfg.get("use_ric_assets", True)
        mode = "global" if bool(legacy_flag) else "custom"
    else:
        mode = str(raw_mode).strip().lower()
    allowed = {"global", "custom"}
    if mode not in allowed:
        raise ValueError(
            f"selection.scope.asset_mode 必须是 {sorted(allowed)} 之一，当前={mode!r}。"
        )
    return mode


def _resolve_window_mode(
    scope_cfg: Mapping[str, Any],
    *,
    key: str,
    default_mode: str,
    legacy_key: str,
) -> str:
    raw_mode = scope_cfg.get(key)
    if raw_mode is None:
        legacy_flag = scope_cfg.get(legacy_key, scope_cfg.get("use_ric_window", True))
        mode = default_mode if bool(legacy_flag) else "custom"
    else:
        mode = str(raw_mode).strip().lower()
    allowed = {"train", "valid", "full", "custom"}
    if mode not in allowed:
        raise ValueError(
            f"selection.scope.{key} 必须是 {sorted(allowed)} 之一，当前={mode!r}。"
        )
    return mode


def load_selection_config(cfg: Mapping[str, Any] | None) -> SelectionConfig:
    cfg = cfg or {}
    selection_cfg = _as_mapping(cfg.get("selection"))
    workflow_cfg = _as_mapping(cfg.get("workflow"))
    scope_cfg = _as_mapping(selection_cfg.get("scope"))
    criteria_cfg = _as_mapping(selection_cfg.get("criteria"))
    complexity_cfg = _as_mapping(selection_cfg.get("complexity"))

    legacy_ric_threshold = _pick_from_mappings(selection_cfg, workflow_cfg, "ric_threshold", "ric_threshold", 0.08)
    legacy_base_corr_threshold = _pick_from_mappings(selection_cfg, workflow_cfg, "corr_threshold", "corr_threshold", 0.65)
    legacy_self_corr_threshold = _pick_from_mappings(
        selection_cfg,
        workflow_cfg,
        "llm_self_corr_threshold",
        "llm_self_corr_threshold",
        0.7,
    )

    train_ric_enabled, train_ric_threshold = _read_criterion(
        criteria_cfg,
        "train_min_abs_ric",
        enabled_default=True,
        threshold_default=0.08,
        legacy_threshold=legacy_ric_threshold,
    )
    train_base_corr_enabled, train_base_corr_threshold = _read_criterion(
        criteria_cfg,
        "train_max_abs_corr_base",
        enabled_default=True,
        threshold_default=0.65,
        legacy_threshold=legacy_base_corr_threshold,
    )
    train_old_corr_enabled, train_old_corr_threshold = _read_criterion(
        criteria_cfg,
        "train_max_abs_corr_old_llm",
        enabled_default=True,
        threshold_default=0.7,
        legacy_threshold=legacy_self_corr_threshold,
        legacy_enabled=selection_cfg.get("enable_new_vs_old_llm", True),
    )
    train_new_corr_enabled, train_new_corr_threshold = _read_criterion(
        criteria_cfg,
        "train_max_abs_corr_new_llm",
        enabled_default=True,
        threshold_default=0.7,
        legacy_threshold=legacy_self_corr_threshold,
        legacy_enabled=selection_cfg.get("enable_new_vs_new_llm", True),
    )

    valid_ric_enabled, valid_ric_threshold = _read_criterion(
        criteria_cfg,
        "valid_min_abs_ric",
        enabled_default=False,
        threshold_default=0.0,
    )
    valid_base_corr_enabled, valid_base_corr_threshold = _read_criterion(
        criteria_cfg,
        "valid_max_abs_corr_base",
        enabled_default=False,
        threshold_default=0.65,
    )
    valid_old_corr_enabled, valid_old_corr_threshold = _read_criterion(
        criteria_cfg,
        "valid_max_abs_corr_old_llm",
        enabled_default=False,
        threshold_default=0.7,
    )
    valid_new_corr_enabled, valid_new_corr_threshold = _read_criterion(
        criteria_cfg,
        "valid_max_abs_corr_new_llm",
        enabled_default=False,
        threshold_default=0.7,
    )

    max_ops_enabled, max_ops_threshold = _read_criterion(
        criteria_cfg,
        "max_operator_count",
        enabled_default=bool(complexity_cfg.get("enabled", True)),
        threshold_default=float(complexity_cfg.get("max_ops", 10)),
    )
    max_depth_enabled, max_depth_threshold = _read_criterion(
        criteria_cfg,
        "max_nesting_depth",
        enabled_default=bool(complexity_cfg.get("enabled", True)),
        threshold_default=float(complexity_cfg.get("max_depth", 3)),
    )
    asset_mode = _resolve_asset_mode(scope_cfg)
    scope_assets = _clean_assets(scope_cfg.get("assets"))
    train_window_mode = _resolve_window_mode(
        scope_cfg,
        key="train_window_mode",
        default_mode="train",
        legacy_key="use_train_window",
    )
    valid_window_mode = _resolve_window_mode(
        scope_cfg,
        key="valid_window_mode",
        default_mode="valid",
        legacy_key="use_valid_window",
    )

    config = SelectionConfig(
        min_corr_obs=int(_pick_from_mappings(selection_cfg, workflow_cfg, "min_corr_obs", "min_corr_obs", 1000)),
        log_topk=max(1, int(_pick_from_mappings(selection_cfg, workflow_cfg, "log_topk", None, 3))),
        require_full_asset_coverage=bool(
            _pick_from_mappings(selection_cfg, workflow_cfg, "require_full_asset_coverage", None, True)
        ),
        scope_asset_mode=asset_mode,
        scope_assets=scope_assets,
        scope_train_window_mode=train_window_mode,
        scope_valid_window_mode=valid_window_mode,
        scope_train_start_date=_clean_optional_date(scope_cfg.get("train_start_date")),
        scope_train_end_date=_clean_optional_date(scope_cfg.get("train_end_date")),
        scope_valid_start_date=_clean_optional_date(scope_cfg.get("valid_start_date")),
        scope_valid_end_date=_clean_optional_date(scope_cfg.get("valid_end_date")),
        train_min_abs_ric_enabled=train_ric_enabled,
        train_min_abs_ric_threshold=float(train_ric_threshold),
        train_max_abs_corr_base_enabled=train_base_corr_enabled,
        train_max_abs_corr_base_threshold=float(train_base_corr_threshold),
        train_max_abs_corr_old_llm_enabled=train_old_corr_enabled,
        train_max_abs_corr_old_llm_threshold=float(train_old_corr_threshold),
        train_max_abs_corr_new_llm_enabled=train_new_corr_enabled,
        train_max_abs_corr_new_llm_threshold=float(train_new_corr_threshold),
        valid_min_abs_ric_enabled=valid_ric_enabled,
        valid_min_abs_ric_threshold=float(valid_ric_threshold),
        valid_max_abs_corr_base_enabled=valid_base_corr_enabled,
        valid_max_abs_corr_base_threshold=float(valid_base_corr_threshold),
        valid_max_abs_corr_old_llm_enabled=valid_old_corr_enabled,
        valid_max_abs_corr_old_llm_threshold=float(valid_old_corr_threshold),
        valid_max_abs_corr_new_llm_enabled=valid_new_corr_enabled,
        valid_max_abs_corr_new_llm_threshold=float(valid_new_corr_threshold),
        max_operator_count_enabled=max_ops_enabled,
        max_operator_count_threshold=int(max_ops_threshold),
        max_nesting_depth_enabled=max_depth_enabled,
        max_nesting_depth_threshold=int(max_depth_threshold),
    )

    if config.min_corr_obs <= 0:
        raise ValueError(f"selection.min_corr_obs must be positive, got {config.min_corr_obs}.")
    if config.max_operator_count_threshold <= 0:
        raise ValueError(
            f"selection.criteria.max_operator_count.threshold must be positive, got {config.max_operator_count_threshold}."
        )
    if config.max_nesting_depth_threshold <= 0:
        raise ValueError(
            f"selection.criteria.max_nesting_depth.threshold must be positive, got {config.max_nesting_depth_threshold}."
        )
    if config.scope_asset_mode == "custom" and not config.scope_assets:
        raise ValueError(
            "selection.scope.asset_mode=custom 时，必须设置 selection.scope.assets。"
        )
    if config.scope_train_window_mode == "custom":
        if not config.scope_train_start_date or not config.scope_train_end_date:
            raise ValueError(
                "selection.scope.train_window_mode=custom 时，必须同时设置 "
                "selection.scope.train_start_date 和 selection.scope.train_end_date。"
            )
    if config.scope_valid_window_mode == "custom":
        if not config.scope_valid_start_date or not config.scope_valid_end_date:
            raise ValueError(
                "selection.scope.valid_window_mode=custom 时，必须同时设置 "
                "selection.scope.valid_start_date 和 selection.scope.valid_end_date。"
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
        patched = replace(patched, train_min_abs_ric_threshold=float(ric_threshold))
    if corr_threshold is not None:
        patched = replace(patched, train_max_abs_corr_base_threshold=float(corr_threshold))
    if llm_self_corr_threshold is not None:
        patched = replace(
            patched,
            train_max_abs_corr_old_llm_threshold=float(llm_self_corr_threshold),
            train_max_abs_corr_new_llm_threshold=float(llm_self_corr_threshold),
        )
    if min_corr_obs is not None:
        patched = replace(patched, min_corr_obs=int(min_corr_obs))
    if patched.min_corr_obs <= 0:
        raise ValueError(f"min_corr_obs must be positive, got {patched.min_corr_obs}.")
    return patched
