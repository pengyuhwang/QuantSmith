#!/usr/bin/env python
"""Compute correlations between LLM factors and base factor library."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LLM factors with base factors.")
    parser.add_argument(
        "--llm-path",
        default="/Users/hpy/PycharmProjects/FAMA/scripts/LLM_factors/dsl_LLM_factors_new.parquet",
        help="Path to LLM factor parquet (long format).",
    )
    parser.add_argument(
        "--base-dir",
        default="/Users/hpy/PycharmProjects/FAMA/data/base_factors",
        help="Directory containing base factor parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/hpy/PycharmProjects/FAMA/factor_correlation/output",
        help="Directory to store correlation csv/plots.",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=60,
        help="Minimum overlapping observations required for a correlation.",
    )
    parser.add_argument(
        "--topk-base",
        type=int,
        default=10,
        help="Number of base factors (by |corr| max) to keep in the heatmap.",
    )
    parser.add_argument(
        "--per-factor-top",
        type=int,
        default=3,
        help="Top-N base factors (per LLM factor) to export in summary CSV.",
    )
    return parser.parse_args()


def _ensure_factor_tag(df: pd.DataFrame) -> pd.DataFrame:
    if "factor_tag" in df.columns:
        return df
    if "factor" in df.columns:
        return df.rename(columns={"factor": "factor_tag"})
    raise ValueError("Input DataFrame missing factor/factor_tag column")


def load_llm_factors(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"LLM factor parquet not found: {path}")
    df = pd.read_parquet(path)
    df = _ensure_factor_tag(df)
    df = df[["time", "unique_id", "factor_tag", "value"]].dropna(subset=["time", "unique_id", "factor_tag"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def load_base_factors(base_dir: Path) -> pd.DataFrame:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base factor directory not found: {base_dir}")
    frames: list[pd.DataFrame] = []
    for file in sorted(base_dir.glob("*.parquet")):
        df = pd.read_parquet(file)
        df = _ensure_factor_tag(df)
        df = df[["time", "unique_id", "factor_tag", "value"]]
        df["time"] = pd.to_datetime(df["time"])
        df["factor_tag"] = df["factor_tag"].astype(str)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No parquet files found under {base_dir}")
    return pd.concat(frames, ignore_index=True)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    x_std = np.nanstd(x_rank)
    y_std = np.nanstd(y_rank)
    if x_std < 1e-12 or y_std < 1e-12:
        return np.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def compute_pairwise_corr(
    llm_df: pd.DataFrame,
    base_df: pd.DataFrame,
    min_obs: int,
    assets: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    corr_sum: Dict[Tuple[str, str], float] = defaultdict(float)
    weight_sum: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    # 可选裁剪资产与时间窗口
    if assets:
        assets_set = set(assets)
        llm_df = llm_df[llm_df["unique_id"].isin(assets_set)]
        base_df = base_df[base_df["unique_id"].isin(assets_set)]
    if start_date:
        llm_df = llm_df[llm_df["time"] >= pd.to_datetime(start_date)]
        base_df = base_df[base_df["time"] >= pd.to_datetime(start_date)]
    if end_date:
        llm_df = llm_df[llm_df["time"] <= pd.to_datetime(end_date)]
        base_df = base_df[base_df["time"] <= pd.to_datetime(end_date)]

    for uid, llm_group in llm_df.groupby("unique_id"):
        base_group = base_df[base_df["unique_id"] == uid]
        if base_group.empty:
            continue

        llm_wide = llm_group.pivot_table(index="time", columns="factor_tag", values="value", aggfunc="mean")
        base_wide = base_group.pivot_table(index="time", columns="factor_tag", values="value", aggfunc="mean")
        common_idx = llm_wide.index.intersection(base_wide.index)
        if len(common_idx) < min_obs:
            continue
        llm_wide = llm_wide.loc[common_idx]
        base_wide = base_wide.loc[common_idx]

        for llm_factor, llm_series in llm_wide.items():
            llm_array = llm_series.to_numpy()
            if np.nanstd(llm_array) < 1e-12:
                continue
            for base_factor, base_series in base_wide.items():
                mask = (~llm_series.isna()) & (~base_series.isna())
                obs = int(mask.sum())
                if obs < min_obs:
                    continue
                x_vals = llm_series[mask].to_numpy()
                y_vals = base_series[mask].to_numpy()
                corr = _spearman_corr(x_vals, y_vals)
                if np.isnan(corr):
                    continue
                key = (llm_factor, base_factor)
                corr_sum[key] += corr * obs
                weight_sum[key] += obs
                pair_counts[key] += 1

    records = []
    for key, weight in weight_sum.items():
        llm_factor, base_factor = key
        corr = corr_sum[key] / float(weight)
        records.append(
            {
                "llm_factor": llm_factor,
                "base_factor": base_factor,
                "weighted_corr": corr,
                "abs_corr": abs(corr),
                "total_obs": weight,
                "asset_pairs": pair_counts[key],
            }
        )
    return pd.DataFrame(records)


def plot_heatmap(
    corr_df: pd.DataFrame,
    output_path: Path,
    topk_base: int,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if corr_df.empty:
        print("No correlation pairs to plot.")
        return
    matrix = corr_df.pivot(index="llm_factor", columns="base_factor", values="weighted_corr")
    if matrix.empty:
        print("Correlation matrix is empty after pivot; skip heatmap.")
        return

    if topk_base > 0 and matrix.shape[1] > topk_base:
        top_base = (
            corr_df.groupby("base_factor")["abs_corr"]
            .max()
            .sort_values(ascending=False)
            .head(topk_base)
            .index
        )
        matrix = matrix.loc[:, [col for col in matrix.columns if col in top_base]]

    matrix = matrix.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if matrix.empty:
        print("No finite correlations to plot.")
        return

    plt.figure(
        figsize=(
            max(8, 0.4 * matrix.shape[1]),
            max(6, 0.4 * matrix.shape[0]),
        )
    )
    sns.heatmap(
        matrix,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="lightgrey",
        cbar_kws={"label": "Weighted Spearman Corr"},
    )
    plt.title("LLM vs Base Factor Correlation Heatmap")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_df = load_llm_factors(Path(args.llm_path))
    base_df = load_base_factors(Path(args.base_dir))
    corr_df = compute_pairwise_corr(llm_df, base_df, min_obs=args.min_obs)

    corr_sorted = corr_df.sort_values("abs_corr", ascending=False)
    corr_csv = output_dir / "llm_vs_base_correlation.csv"
    corr_sorted.to_csv(corr_csv, index=False)
    print(f"[correlation] Saved pairwise statistics to {corr_csv}")

    if not corr_df.empty and args.per_factor_top > 0:
        per_factor_sorted = corr_df.sort_values(
            ["llm_factor", "abs_corr"], ascending=[True, False]
        )
        top_each = per_factor_sorted.groupby("llm_factor", as_index=False).head(args.per_factor_top)
        top_csv = output_dir / f"llm_vs_base_top{args.per_factor_top}.csv"
        top_each.to_csv(top_csv, index=False)
        print(f"[correlation] Saved per-LLM top {args.per_factor_top} pairs to {top_csv}")

    # heatmap_path = output_dir / "llm_vs_base_heatmap.png"
    # plot_heatmap(corr_df, heatmap_path, args.topk_base)
    # if heatmap_path.exists():
    #     print(f"[correlation] Heatmap saved to {heatmap_path}")


if __name__ == "__main__":
    main()
