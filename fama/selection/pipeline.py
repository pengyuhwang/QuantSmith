from __future__ import annotations

import time

import pandas as pd

from utils.compute_correlation_new import compute_pairwise_corr

from .models import SelectionConfig, SelectionInput, SelectionResult, empty_corr_df
from .rules import (
    apply_selection_criteria,
    build_complexity_metrics,
    merge_combined_max_corr,
    summarize_corr_metrics,
    summarize_ric_metrics,
)


def _resolve_scope(
    selection_input: SelectionInput,
    config: SelectionConfig,
    *,
    window_kind: str,
) -> tuple[list[str] | None, str | None, str | None]:
    if config.scope_asset_mode == "global":
        assets = selection_input.assets
    elif config.scope_asset_mode == "custom":
        assets = [str(item) for item in config.scope_assets]
    else:
        raise ValueError(f"Unsupported selection asset mode: {config.scope_asset_mode}")
    if window_kind == "train":
        mode = config.scope_train_window_mode
        custom_start = config.scope_train_start_date
        custom_end = config.scope_train_end_date
    elif window_kind == "valid":
        mode = config.scope_valid_window_mode
        custom_start = config.scope_valid_start_date
        custom_end = config.scope_valid_end_date
    else:
        raise ValueError(f"Unsupported window_kind={window_kind}")

    if mode == "train":
        return assets, selection_input.train_start_date, selection_input.train_end_date
    if mode == "valid":
        return assets, selection_input.valid_start_date, selection_input.valid_end_date
    if mode == "full":
        return assets, None, None
    if mode == "custom":
        return assets, custom_start, custom_end
    raise ValueError(f"Unsupported {window_kind} window mode: {mode}")


def _merge_metrics(base, addon):
    return base.merge(addon, on="factor_id", how="left")


def run_selection_pipeline(
    selection_input: SelectionInput,
    config: SelectionConfig,
) -> SelectionResult:
    started = time.perf_counter()
    result = SelectionResult()

    candidates = sorted({str(item) for item in selection_input.new_llm_df.get("factor_tag", [])})
    if not candidates:
        result.metrics_df = pd.DataFrame(columns=["factor_id"])
        result.elapsed_total = time.perf_counter() - started
        return result

    metrics = build_complexity_metrics(candidates, selection_input.expr_map)
    metrics = _merge_metrics(metrics, summarize_ric_metrics(selection_input.train_ric_df, candidates, prefix="train"))
    metrics = _merge_metrics(metrics, summarize_ric_metrics(selection_input.valid_ric_df, candidates, prefix="valid"))

    train_assets, train_start, train_end = _resolve_scope(selection_input, config, window_kind="train")
    valid_assets, valid_start, valid_end = _resolve_scope(selection_input, config, window_kind="valid")

    t0 = time.perf_counter()
    result.corr_train_new_vs_base = compute_pairwise_corr(
        selection_input.new_llm_df,
        selection_input.base_df,
        min_obs=config.min_corr_obs,
        assets=train_assets,
        start_date=train_start,
        end_date=train_end,
    )
    if result.corr_train_new_vs_base is None:
        result.corr_train_new_vs_base = empty_corr_df()
    result.elapsed_train_corr_base = time.perf_counter() - t0
    metrics = _merge_metrics(metrics, summarize_corr_metrics(result.corr_train_new_vs_base, candidates, prefix="train_base"))

    t0 = time.perf_counter()
    result.corr_valid_new_vs_base = compute_pairwise_corr(
        selection_input.new_llm_df,
        selection_input.base_df,
        min_obs=config.min_corr_obs,
        assets=valid_assets,
        start_date=valid_start,
        end_date=valid_end,
    )
    if result.corr_valid_new_vs_base is None:
        result.corr_valid_new_vs_base = empty_corr_df()
    result.elapsed_valid_corr_base = time.perf_counter() - t0
    metrics = _merge_metrics(metrics, summarize_corr_metrics(result.corr_valid_new_vs_base, candidates, prefix="valid_base"))

    old_llm_df = selection_input.old_llm_df
    if old_llm_df is None and selection_input.old_llm_df_loader is not None:
        old_llm_df = selection_input.old_llm_df_loader()
    has_old_llm_reference = old_llm_df is not None and not old_llm_df.empty

    if has_old_llm_reference:
        t0 = time.perf_counter()
        result.corr_train_new_vs_old_llm = compute_pairwise_corr(
            selection_input.new_llm_df,
            old_llm_df,
            min_obs=config.min_corr_obs,
            assets=train_assets,
            start_date=train_start,
            end_date=train_end,
        )
        if result.corr_train_new_vs_old_llm is None:
            result.corr_train_new_vs_old_llm = empty_corr_df()
        result.elapsed_train_corr_old_llm = time.perf_counter() - t0
        metrics = _merge_metrics(
            metrics,
            summarize_corr_metrics(result.corr_train_new_vs_old_llm, candidates, prefix="train_old_llm"),
        )

        t0 = time.perf_counter()
        result.corr_valid_new_vs_old_llm = compute_pairwise_corr(
            selection_input.new_llm_df,
            old_llm_df,
            min_obs=config.min_corr_obs,
            assets=valid_assets,
            start_date=valid_start,
            end_date=valid_end,
        )
        if result.corr_valid_new_vs_old_llm is None:
            result.corr_valid_new_vs_old_llm = empty_corr_df()
        result.elapsed_valid_corr_old_llm = time.perf_counter() - t0
        metrics = _merge_metrics(
            metrics,
            summarize_corr_metrics(result.corr_valid_new_vs_old_llm, candidates, prefix="valid_old_llm"),
        )
    else:
        result.corr_train_new_vs_old_llm = empty_corr_df()
        result.corr_valid_new_vs_old_llm = empty_corr_df()
        metrics = _merge_metrics(
            metrics,
            summarize_corr_metrics(result.corr_train_new_vs_old_llm, candidates, prefix="train_old_llm"),
        )
        metrics = _merge_metrics(
            metrics,
            summarize_corr_metrics(result.corr_valid_new_vs_old_llm, candidates, prefix="valid_old_llm"),
        )

    has_new_llm_pair = len(candidates) >= 2
    if has_new_llm_pair:
        t0 = time.perf_counter()
        corr_train_new_vs_new = compute_pairwise_corr(
            selection_input.new_llm_df,
            selection_input.new_llm_df,
            min_obs=config.min_corr_obs,
            assets=train_assets,
            start_date=train_start,
            end_date=train_end,
        )
        if corr_train_new_vs_new is None:
            corr_train_new_vs_new = empty_corr_df()
        if not corr_train_new_vs_new.empty:
            corr_train_new_vs_new = corr_train_new_vs_new[
                corr_train_new_vs_new["llm_factor"] != corr_train_new_vs_new["base_factor"]
            ]
        result.corr_train_new_vs_new_llm = corr_train_new_vs_new
        result.elapsed_train_corr_new_llm = time.perf_counter() - t0
        metrics = _merge_metrics(
            metrics,
            summarize_corr_metrics(result.corr_train_new_vs_new_llm, candidates, prefix="train_new_llm"),
        )

        t0 = time.perf_counter()
        corr_valid_new_vs_new = compute_pairwise_corr(
            selection_input.new_llm_df,
            selection_input.new_llm_df,
            min_obs=config.min_corr_obs,
            assets=valid_assets,
            start_date=valid_start,
            end_date=valid_end,
        )
        if corr_valid_new_vs_new is None:
            corr_valid_new_vs_new = empty_corr_df()
        if not corr_valid_new_vs_new.empty:
            corr_valid_new_vs_new = corr_valid_new_vs_new[
                corr_valid_new_vs_new["llm_factor"] != corr_valid_new_vs_new["base_factor"]
            ]
        result.corr_valid_new_vs_new_llm = corr_valid_new_vs_new
        result.elapsed_valid_corr_new_llm = time.perf_counter() - t0
        metrics = _merge_metrics(
            metrics,
            summarize_corr_metrics(result.corr_valid_new_vs_new_llm, candidates, prefix="valid_new_llm"),
        )
    else:
        result.corr_train_new_vs_new_llm = empty_corr_df()
        result.corr_valid_new_vs_new_llm = empty_corr_df()
        metrics = _merge_metrics(
            metrics,
            summarize_corr_metrics(result.corr_train_new_vs_new_llm, candidates, prefix="train_new_llm"),
        )
        metrics = _merge_metrics(
            metrics,
            summarize_corr_metrics(result.corr_valid_new_vs_new_llm, candidates, prefix="valid_new_llm"),
        )

    metrics = merge_combined_max_corr(
        metrics,
        prefix="train",
        lhs_metric_col="train_base_max_corr",
        lhs_factor_col="train_base_max_corr_factor_id",
        rhs_metric_col="train_old_llm_max_corr",
        rhs_factor_col="train_old_llm_max_corr_factor_id",
    )
    metrics = merge_combined_max_corr(
        metrics,
        prefix="valid",
        lhs_metric_col="valid_base_max_corr",
        lhs_factor_col="valid_base_max_corr_factor_id",
        rhs_metric_col="valid_old_llm_max_corr",
        rhs_factor_col="valid_old_llm_max_corr_factor_id",
    )

    metrics, skipped_criteria, _ = apply_selection_criteria(
        metrics,
        config=config,
        assets_count=len(selection_input.assets),
        has_old_llm_reference=has_old_llm_reference,
        has_new_llm_pair=has_new_llm_pair,
    )

    metrics = metrics.sort_values("factor_id").reset_index(drop=True)
    result.metrics_df = metrics
    result.skipped_criteria = skipped_criteria
    result.passed_factors = sorted(metrics.loc[metrics["final_status"] == "success", "factor_id"].astype(str).tolist())
    result.failed_factors = sorted(metrics.loc[metrics["final_status"] != "success", "factor_id"].astype(str).tolist())
    result.elapsed_total = time.perf_counter() - started
    return result
