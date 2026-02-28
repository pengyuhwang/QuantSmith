from __future__ import annotations

from typing import Iterable

import pandas as pd


def apply_corr_scope(
    df: pd.DataFrame,
    assets: Iterable[str] | None,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """Apply the same scope filters as compute_pairwise_corr for preview logging."""

    if df is None or df.empty:
        return df
    scoped = df
    if assets:
        assets_set = {str(asset) for asset in assets}
        if "unique_id" in scoped.columns:
            scoped = scoped[scoped["unique_id"].astype(str).isin(assets_set)]
    if "time" in scoped.columns:
        time_series = pd.to_datetime(scoped["time"], errors="coerce")
        if start_date:
            scoped = scoped[time_series >= pd.to_datetime(start_date)]
            time_series = pd.to_datetime(scoped["time"], errors="coerce")
        if end_date:
            scoped = scoped[time_series <= pd.to_datetime(end_date)]
    return scoped


def _factor_sample(df: pd.DataFrame, limit: int = 5) -> str:
    if df is None or df.empty or "factor_tag" not in df.columns:
        return "none"
    factors = sorted({str(item) for item in df["factor_tag"].dropna().tolist()})
    if not factors:
        return "none"
    if len(factors) <= limit:
        return ", ".join(factors)
    return f"{', '.join(factors[:limit])} ..."


def corr_input_summary(df: pd.DataFrame, *, label: str) -> str:
    """Build a concise, readable summary for correlation input tables."""

    if df is None or df.empty:
        return f"{label}: factors=0 [none] | rows=0 | assets=0 | window=N/A->N/A"
    factor_cnt = int(df["factor_tag"].nunique()) if "factor_tag" in df.columns else 0
    asset_cnt = int(df["unique_id"].nunique()) if "unique_id" in df.columns else 0
    if "time" in df.columns:
        ts = pd.to_datetime(df["time"], errors="coerce").dropna()
        if ts.empty:
            window = "N/A->N/A"
        else:
            window = f"{ts.min().date()}->{ts.max().date()}"
    else:
        window = "N/A->N/A"
    return (
        f"{label}: factors={factor_cnt} [{_factor_sample(df)}] | "
        f"rows={len(df)} | assets={asset_cnt} | window={window}"
    )


def top_self_corr(corr_df: pd.DataFrame) -> pd.DataFrame:
    if corr_df is None or corr_df.empty:
        return pd.DataFrame(columns=["llm_factor", "base_factor", "weighted_corr", "abs_corr", "total_obs"])
    return (
        corr_df.sort_values("abs_corr", ascending=False)
        .groupby("llm_factor", as_index=False)
        .first()
    )


def format_ric_passed_details(
    ric_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    factors: Iterable[str],
) -> list[str]:
    lines: list[str] = []
    if ric_df is None or ric_df.empty:
        return lines
    asset_col = "asset" if "asset" in ric_df.columns else "unique_id"
    for factor in sorted({str(item) for item in factors}):
        ric_rows = ric_df[ric_df["factor_tag"] == factor]
        if ric_rows.empty:
            continue
        ric_str = "; ".join(f"{getattr(row, asset_col)}:{row.ric:.4f}" for row in ric_rows.itertuples())
        top_corr_str = "无"
        if corr_df is not None and not corr_df.empty:
            top_corr = (
                corr_df[corr_df["llm_factor"] == factor]
                .sort_values("abs_corr", ascending=False)
                .head(1)
            )
            if not top_corr.empty:
                row = top_corr.iloc[0]
                top_corr_str = f"{row['base_factor']}:{row['weighted_corr']:.4f}(|{row['abs_corr']:.4f}|)"
        lines.append(f"  - {factor} | RIC[{ric_str}] | TopCorr[{top_corr_str}]")
    return lines

