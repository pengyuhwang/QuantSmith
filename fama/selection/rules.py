from __future__ import annotations

from typing import Iterable

import pandas as pd


def apply_ric_gate(
    ric_df: pd.DataFrame,
    *,
    assets: Iterable[str],
    ric_threshold: float,
    require_full_asset_coverage: bool = True,
) -> list[str]:
    assets = [str(asset) for asset in assets]
    if not assets or ric_df is None or ric_df.empty:
        return []

    asset_col = "asset" if "asset" in ric_df.columns else "unique_id"
    if asset_col not in ric_df.columns:
        raise ValueError("RIC 表缺少资产列（asset/unique_id）。")

    ric_subset = ric_df[ric_df[asset_col].astype(str).isin(assets)].copy()
    if ric_subset.empty:
        return []

    ric_subset["abs_ric"] = pd.to_numeric(ric_subset["ric"], errors="coerce").abs()
    ric_grouped = ric_subset.groupby("factor_tag").agg(
        asset_coverage=(asset_col, "nunique"),
        min_abs_ric=("abs_ric", "min"),
    )
    if require_full_asset_coverage:
        coverage_cond = ric_grouped["asset_coverage"] == len(assets)
    else:
        coverage_cond = ric_grouped["asset_coverage"] > 0

    ric_pass = ric_grouped[coverage_cond & (ric_grouped["min_abs_ric"] >= float(ric_threshold))]
    return sorted(ric_pass.index.astype(str).tolist())


def apply_base_corr_gate(
    candidates: Iterable[str],
    corr_df: pd.DataFrame,
    *,
    corr_threshold: float,
) -> list[str]:
    candidate_set = {str(item) for item in candidates}
    if not candidate_set:
        return []
    if corr_df is None or corr_df.empty:
        return sorted(candidate_set)

    corr_max = corr_df.groupby("llm_factor")["abs_corr"].max()
    passed = {
        factor
        for factor in candidate_set
        if (factor in corr_max.index) and (float(corr_max.loc[factor]) <= float(corr_threshold))
    }
    return sorted(passed)


def dedup_new_factors_by_self_corr(
    candidates: list[str],
    self_corr_df: pd.DataFrame,
    ric_df: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], list[str]]:
    """Greedy dedup for new factors using self-correlation, keep stronger RIC first."""

    if not candidates:
        return [], []
    if self_corr_df is None or self_corr_df.empty:
        return sorted(candidates), []

    pair_df = self_corr_df[self_corr_df["llm_factor"] != self_corr_df["base_factor"]].copy()
    pair_df = pair_df[
        pair_df["llm_factor"].isin(candidates)
        & pair_df["base_factor"].isin(candidates)
    ]
    pair_df = pair_df[pair_df["abs_corr"] > threshold]
    if pair_df.empty:
        return sorted(candidates), []

    ric_strength = (
        ric_df[ric_df["factor_tag"].isin(candidates)]
        .assign(abs_ric=lambda df: pd.to_numeric(df["ric"], errors="coerce").abs())
        .groupby("factor_tag")["abs_ric"]
        .min()
    )

    ordered = sorted(
        candidates,
        key=lambda name: (-float(ric_strength.get(name, 0.0)), name),
    )
    adjacency: dict[str, set[str]] = {name: set() for name in candidates}
    for row in pair_df.itertuples():
        lhs = str(row.llm_factor)
        rhs = str(row.base_factor)
        if lhs == rhs:
            continue
        adjacency.setdefault(lhs, set()).add(rhs)
        adjacency.setdefault(rhs, set()).add(lhs)

    kept: list[str] = []
    dropped: set[str] = set()
    for factor in ordered:
        if factor in dropped:
            continue
        kept.append(factor)
        for peer in adjacency.get(factor, set()):
            if peer not in kept:
                dropped.add(peer)

    return sorted(kept), sorted(dropped)

