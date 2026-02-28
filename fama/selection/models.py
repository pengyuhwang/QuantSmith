from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

CORR_COLUMNS = ["llm_factor", "base_factor", "weighted_corr", "abs_corr", "total_obs", "asset_pairs"]


def empty_corr_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CORR_COLUMNS)


@dataclass(frozen=True)
class SelectionConfig:
    ric_threshold: float
    corr_threshold: float
    llm_self_corr_threshold: float
    min_corr_obs: int
    require_full_asset_coverage: bool = True
    enable_new_vs_old_llm: bool = True
    enable_new_vs_new_llm: bool = True
    dedup_strategy: str = "greedy_by_min_abs_ric"
    log_topk: int = 3
    scope_use_ric_assets: bool = True
    scope_use_ric_window: bool = True


@dataclass
class SelectionInput:
    ric_df: pd.DataFrame
    new_llm_df: pd.DataFrame
    base_df: pd.DataFrame
    assets: list[str]
    start_date: str | None = None
    end_date: str | None = None
    old_llm_df: pd.DataFrame | None = None
    old_llm_df_loader: Callable[[], pd.DataFrame] | None = None


@dataclass
class SelectionResult:
    ric_passed_factors: list[str] = field(default_factory=list)
    passed_after_base_corr: list[str] = field(default_factory=list)
    passed_factors: list[str] = field(default_factory=list)

    dropped_by_ric: list[str] = field(default_factory=list)
    dropped_by_base_corr: list[str] = field(default_factory=list)
    dropped_by_old_llm_corr: list[str] = field(default_factory=list)
    dropped_by_new_llm_corr: list[str] = field(default_factory=list)

    corr_new_vs_base: pd.DataFrame = field(default_factory=empty_corr_df)
    corr_new_vs_old_llm: pd.DataFrame = field(default_factory=empty_corr_df)
    corr_new_vs_new_llm: pd.DataFrame = field(default_factory=empty_corr_df)

    old_llm_gate_enabled: bool = False
    old_llm_gate_applied: bool = False
    old_llm_gate_reason: str | None = None

    new_llm_gate_enabled: bool = False
    new_llm_gate_applied: bool = False
    new_llm_gate_reason: str | None = None

    elapsed_new_vs_base: float = 0.0
    elapsed_new_vs_old_llm: float = 0.0
    elapsed_new_vs_new_llm: float = 0.0
    elapsed_total: float = 0.0

