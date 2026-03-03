from __future__ import annotations

from typing import Iterable

import pandas as pd

from utils.complexity import compute_expression_complexity


def _asset_col(ric_df: pd.DataFrame) -> str:
    if "asset" in ric_df.columns:
        return "asset"
    if "unique_id" in ric_df.columns:
        return "unique_id"
    raise ValueError("RIC 表缺少资产列（asset/unique_id）。")


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

    asset_col = _asset_col(ric_df)
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


def summarize_ric_metrics(ric_df: pd.DataFrame, factors: Iterable[str], *, prefix: str) -> pd.DataFrame:
    factor_ids = sorted({str(item) for item in factors})
    out = pd.DataFrame({"factor_id": factor_ids})
    out[f"{prefix}_ric"] = pd.NA
    out[f"{prefix}_ic"] = pd.NA
    out[f"{prefix}_icir"] = pd.NA
    out[f"{prefix}_min_abs_ric"] = pd.NA
    out[f"{prefix}_asset_coverage"] = 0

    if ric_df is None or ric_df.empty or not factor_ids:
        return out

    df = ric_df.copy()
    if "factor_tag" not in df.columns:
        return out
    df = df[df["factor_tag"].astype(str).isin(set(factor_ids))]
    if df.empty:
        return out

    asset_col = _asset_col(df)
    df["factor_tag"] = df["factor_tag"].astype(str)
    df["ric"] = pd.to_numeric(df["ric"], errors="coerce")
    df["abs_ric"] = df["ric"].abs()
    if "ic" in df.columns:
        df["ic"] = pd.to_numeric(df["ic"], errors="coerce")
    if "icir" in df.columns:
        df["icir"] = pd.to_numeric(df["icir"], errors="coerce")

    grouped = df.groupby("factor_tag", as_index=False).agg(
        **{
            f"{prefix}_ric": ("ric", "mean"),
            f"{prefix}_min_abs_ric": ("abs_ric", "min"),
            f"{prefix}_asset_coverage": (asset_col, "nunique"),
        }
    )
    if "ic" in df.columns:
        ic_series = df.groupby("factor_tag")["ic"].mean()
        grouped[f"{prefix}_ic"] = grouped["factor_tag"].map(ic_series)
    if "icir" in df.columns:
        icir_series = df.groupby("factor_tag")["icir"].mean()
        grouped[f"{prefix}_icir"] = grouped["factor_tag"].map(icir_series)

    grouped = grouped.rename(columns={"factor_tag": "factor_id"})
    merged = out.drop(columns=[f"{prefix}_ric", f"{prefix}_ic", f"{prefix}_icir", f"{prefix}_min_abs_ric", f"{prefix}_asset_coverage"]).merge(
        grouped,
        on="factor_id",
        how="left",
    )

    for col in [f"{prefix}_ric", f"{prefix}_ic", f"{prefix}_icir", f"{prefix}_min_abs_ric"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    if f"{prefix}_asset_coverage" in merged.columns:
        merged[f"{prefix}_asset_coverage"] = (
            pd.to_numeric(merged[f"{prefix}_asset_coverage"], errors="coerce").fillna(0).astype(int)
        )
    return merged


def summarize_corr_metrics(corr_df: pd.DataFrame, factors: Iterable[str], *, prefix: str) -> pd.DataFrame:
    factor_ids = sorted({str(item) for item in factors})
    out = pd.DataFrame({"factor_id": factor_ids})
    out[f"{prefix}_max_corr"] = pd.NA
    out[f"{prefix}_max_corr_factor_id"] = pd.NA
    out[f"{prefix}_max_corr_signed"] = pd.NA
    out[f"{prefix}_max_corr_obs"] = pd.NA
    if corr_df is None or corr_df.empty or not factor_ids:
        return out

    df = corr_df.copy()
    if "llm_factor" not in df.columns:
        return out
    df["llm_factor"] = df["llm_factor"].astype(str)
    df = df[df["llm_factor"].isin(set(factor_ids))]
    if df.empty:
        return out

    df["abs_corr"] = pd.to_numeric(df["abs_corr"], errors="coerce")
    df["weighted_corr"] = pd.to_numeric(df["weighted_corr"], errors="coerce")
    df = df.sort_values("abs_corr", ascending=False).groupby("llm_factor", as_index=False).first()
    df = df.rename(
        columns={
            "llm_factor": "factor_id",
            "abs_corr": f"{prefix}_max_corr",
            "base_factor": f"{prefix}_max_corr_factor_id",
            "weighted_corr": f"{prefix}_max_corr_signed",
            "total_obs": f"{prefix}_max_corr_obs",
        }
    )

    merged = out.drop(
        columns=[f"{prefix}_max_corr", f"{prefix}_max_corr_factor_id", f"{prefix}_max_corr_signed", f"{prefix}_max_corr_obs"]
    ).merge(df[["factor_id", f"{prefix}_max_corr", f"{prefix}_max_corr_factor_id", f"{prefix}_max_corr_signed", f"{prefix}_max_corr_obs"]], on="factor_id", how="left")
    merged[f"{prefix}_max_corr"] = pd.to_numeric(merged[f"{prefix}_max_corr"], errors="coerce")
    merged[f"{prefix}_max_corr_signed"] = pd.to_numeric(merged[f"{prefix}_max_corr_signed"], errors="coerce")
    merged[f"{prefix}_max_corr_obs"] = pd.to_numeric(merged[f"{prefix}_max_corr_obs"], errors="coerce")
    return merged


def merge_combined_max_corr(
    metrics_df: pd.DataFrame,
    *,
    prefix: str,
    lhs_metric_col: str,
    lhs_factor_col: str,
    rhs_metric_col: str,
    rhs_factor_col: str,
) -> pd.DataFrame:
    out = metrics_df.copy()
    lhs_metric = pd.to_numeric(out[lhs_metric_col], errors="coerce")
    rhs_metric = pd.to_numeric(out[rhs_metric_col], errors="coerce")

    lhs_score = lhs_metric.fillna(-1.0)
    rhs_score = rhs_metric.fillna(-1.0)
    choose_lhs = lhs_score >= rhs_score

    out[f"{prefix}_max_corr"] = lhs_metric.where(choose_lhs, rhs_metric)
    out[f"{prefix}_max_corr_factor_id"] = out[lhs_factor_col].where(choose_lhs, out[rhs_factor_col])
    both_missing = lhs_metric.isna() & rhs_metric.isna()
    out.loc[both_missing, f"{prefix}_max_corr"] = pd.NA
    out.loc[both_missing, f"{prefix}_max_corr_factor_id"] = pd.NA
    return out


def build_complexity_metrics(factors: Iterable[str], expr_map: dict[str, str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for factor_id in sorted({str(item) for item in factors}):
        expr = expr_map.get(factor_id)
        if not isinstance(expr, str) or not expr.strip():
            records.append(
                {
                    "factor_id": factor_id,
                    "operator_count": pd.NA,
                    "nesting_depth": pd.NA,
                    "expression_size": pd.NA,
                }
            )
            continue

        try:
            ops, depth = compute_expression_complexity(expr)
            records.append(
                {
                    "factor_id": factor_id,
                    "operator_count": int(ops),
                    "nesting_depth": int(depth),
                    "expression_size": int(len(expr)),
                }
            )
        except Exception:
            records.append(
                {
                    "factor_id": factor_id,
                    "operator_count": pd.NA,
                    "nesting_depth": pd.NA,
                    "expression_size": int(len(expr)),
                }
            )
    return pd.DataFrame(records)


def apply_selection_criteria(
    metrics_df: pd.DataFrame,
    *,
    config,
    assets_count: int,
    has_old_llm_reference: bool,
    has_new_llm_pair: bool,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    out = metrics_df.copy()
    skipped_criteria: dict[str, str] = {}
    ordered_criteria: list[str] = []
    criterion_masks: list[pd.Series] = []

    def _register(name: str, mask: pd.Series) -> None:
        ordered_criteria.append(name)
        out[f"pass_{name}"] = mask.astype(bool)
        criterion_masks.append(mask.astype(bool))

    train_cov = pd.to_numeric(out.get("train_asset_coverage"), errors="coerce").fillna(0)
    valid_cov = pd.to_numeric(out.get("valid_asset_coverage"), errors="coerce").fillna(0)

    if config.train_min_abs_ric_enabled:
        train_ric = pd.to_numeric(out.get("train_min_abs_ric"), errors="coerce")
        cov_mask = train_cov == assets_count if config.require_full_asset_coverage else train_cov > 0
        _register(
            "train_min_abs_ric",
            cov_mask & train_ric.notna() & (train_ric >= float(config.train_min_abs_ric_threshold)),
        )

    if config.train_max_abs_corr_base_enabled:
        train_base_corr = pd.to_numeric(out.get("train_base_max_corr"), errors="coerce")
        _register(
            "train_max_abs_corr_base",
            train_base_corr.notna() & (train_base_corr <= float(config.train_max_abs_corr_base_threshold)),
        )

    if config.train_max_abs_corr_old_llm_enabled:
        if has_old_llm_reference:
            train_old_corr = pd.to_numeric(out.get("train_old_llm_max_corr"), errors="coerce")
            _register(
                "train_max_abs_corr_old_llm",
                train_old_corr.isna() | (train_old_corr <= float(config.train_max_abs_corr_old_llm_threshold)),
            )
        else:
            skipped_criteria["train_max_abs_corr_old_llm"] = "no_historical_llm_reference"

    if config.train_max_abs_corr_new_llm_enabled:
        if has_new_llm_pair:
            train_new_corr = pd.to_numeric(out.get("train_new_llm_max_corr"), errors="coerce")
            _register(
                "train_max_abs_corr_new_llm",
                train_new_corr.isna() | (train_new_corr <= float(config.train_max_abs_corr_new_llm_threshold)),
            )
        else:
            skipped_criteria["train_max_abs_corr_new_llm"] = "less_than_2_new_factors"

    if config.valid_min_abs_ric_enabled:
        valid_ric = pd.to_numeric(out.get("valid_min_abs_ric"), errors="coerce")
        cov_mask = valid_cov == assets_count if config.require_full_asset_coverage else valid_cov > 0
        _register(
            "valid_min_abs_ric",
            cov_mask & valid_ric.notna() & (valid_ric >= float(config.valid_min_abs_ric_threshold)),
        )

    if config.valid_max_abs_corr_base_enabled:
        valid_base_corr = pd.to_numeric(out.get("valid_base_max_corr"), errors="coerce")
        _register(
            "valid_max_abs_corr_base",
            valid_base_corr.notna() & (valid_base_corr <= float(config.valid_max_abs_corr_base_threshold)),
        )

    if config.valid_max_abs_corr_old_llm_enabled:
        if has_old_llm_reference:
            valid_old_corr = pd.to_numeric(out.get("valid_old_llm_max_corr"), errors="coerce")
            _register(
                "valid_max_abs_corr_old_llm",
                valid_old_corr.isna() | (valid_old_corr <= float(config.valid_max_abs_corr_old_llm_threshold)),
            )
        else:
            skipped_criteria["valid_max_abs_corr_old_llm"] = "no_historical_llm_reference"

    if config.valid_max_abs_corr_new_llm_enabled:
        if has_new_llm_pair:
            valid_new_corr = pd.to_numeric(out.get("valid_new_llm_max_corr"), errors="coerce")
            _register(
                "valid_max_abs_corr_new_llm",
                valid_new_corr.isna() | (valid_new_corr <= float(config.valid_max_abs_corr_new_llm_threshold)),
            )
        else:
            skipped_criteria["valid_max_abs_corr_new_llm"] = "less_than_2_new_factors"

    if config.max_operator_count_enabled:
        ops = pd.to_numeric(out.get("operator_count"), errors="coerce")
        _register(
            "max_operator_count",
            ops.notna() & (ops <= int(config.max_operator_count_threshold)),
        )

    if config.max_nesting_depth_enabled:
        depth = pd.to_numeric(out.get("nesting_depth"), errors="coerce")
        _register(
            "max_nesting_depth",
            depth.notna() & (depth <= int(config.max_nesting_depth_threshold)),
        )

    if criterion_masks:
        pass_matrix = pd.concat(criterion_masks, axis=1)
        final_pass = pass_matrix.all(axis=1)
    else:
        final_pass = pd.Series(True, index=out.index)

    out["final_status"] = final_pass.map({True: "success", False: "failure"})
    failure_stage: list[str] = []
    failure_reason: list[str] = []
    for idx in out.index:
        failed = [name for name in ordered_criteria if not bool(out.at[idx, f"pass_{name}"])]
        failure_stage.append(failed[0] if failed else "")
        failure_reason.append(";".join(failed) if failed else "")
    out["failure_stage"] = failure_stage
    out["failure_reason"] = failure_reason
    return out, skipped_criteria, ordered_criteria
