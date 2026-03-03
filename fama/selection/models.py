from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

CORR_COLUMNS = ["llm_factor", "base_factor", "weighted_corr", "abs_corr", "total_obs", "asset_pairs"]


def empty_corr_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CORR_COLUMNS)


@dataclass(frozen=True)
class SelectionConfig:
    min_corr_obs: int
    log_topk: int = 3
    require_full_asset_coverage: bool = True
    scope_asset_mode: str = "global"
    scope_assets: tuple[str, ...] = ()
    scope_train_window_mode: str = "train"
    scope_valid_window_mode: str = "valid"
    scope_train_start_date: str | None = None
    scope_train_end_date: str | None = None
    scope_valid_start_date: str | None = None
    scope_valid_end_date: str | None = None

    train_min_abs_ric_enabled: bool = True
    train_min_abs_ric_threshold: float = 0.08
    train_max_abs_corr_base_enabled: bool = True
    train_max_abs_corr_base_threshold: float = 0.65
    train_max_abs_corr_old_llm_enabled: bool = True
    train_max_abs_corr_old_llm_threshold: float = 0.7
    train_max_abs_corr_new_llm_enabled: bool = True
    train_max_abs_corr_new_llm_threshold: float = 0.7

    valid_min_abs_ric_enabled: bool = False
    valid_min_abs_ric_threshold: float = 0.0
    valid_max_abs_corr_base_enabled: bool = False
    valid_max_abs_corr_base_threshold: float = 0.65
    valid_max_abs_corr_old_llm_enabled: bool = False
    valid_max_abs_corr_old_llm_threshold: float = 0.7
    valid_max_abs_corr_new_llm_enabled: bool = False
    valid_max_abs_corr_new_llm_threshold: float = 0.7

    max_operator_count_enabled: bool = True
    max_operator_count_threshold: int = 10
    max_nesting_depth_enabled: bool = True
    max_nesting_depth_threshold: int = 3


@dataclass
class SelectionInput:
    train_ric_df: pd.DataFrame
    valid_ric_df: pd.DataFrame
    new_llm_df: pd.DataFrame
    base_df: pd.DataFrame
    expr_map: dict[str, str]
    assets: list[str]
    train_start_date: str | None = None
    train_end_date: str | None = None
    valid_start_date: str | None = None
    valid_end_date: str | None = None
    old_llm_df: pd.DataFrame | None = None
    old_llm_df_loader: Callable[[], pd.DataFrame] | None = None


@dataclass
class SelectionResult:
    metrics_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    passed_factors: list[str] = field(default_factory=list)
    failed_factors: list[str] = field(default_factory=list)

    corr_train_new_vs_base: pd.DataFrame = field(default_factory=empty_corr_df)
    corr_train_new_vs_old_llm: pd.DataFrame = field(default_factory=empty_corr_df)
    corr_train_new_vs_new_llm: pd.DataFrame = field(default_factory=empty_corr_df)
    corr_valid_new_vs_base: pd.DataFrame = field(default_factory=empty_corr_df)
    corr_valid_new_vs_old_llm: pd.DataFrame = field(default_factory=empty_corr_df)
    corr_valid_new_vs_new_llm: pd.DataFrame = field(default_factory=empty_corr_df)

    skipped_criteria: dict[str, str] = field(default_factory=dict)

    elapsed_train_corr_base: float = 0.0
    elapsed_train_corr_old_llm: float = 0.0
    elapsed_train_corr_new_llm: float = 0.0
    elapsed_valid_corr_base: float = 0.0
    elapsed_valid_corr_old_llm: float = 0.0
    elapsed_valid_corr_new_llm: float = 0.0
    elapsed_total: float = 0.0
