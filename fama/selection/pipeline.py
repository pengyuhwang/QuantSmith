from __future__ import annotations

import time

from utils.compute_correlation_new import compute_pairwise_corr

from .models import SelectionConfig, SelectionInput, SelectionResult, empty_corr_df
from .rules import apply_base_corr_gate, apply_ric_gate, dedup_new_factors_by_self_corr


def _resolve_scope(selection_input: SelectionInput, config: SelectionConfig) -> tuple[list[str] | None, str | None, str | None]:
    assets = selection_input.assets if config.scope_use_ric_assets else None
    start_date = selection_input.start_date if config.scope_use_ric_window else None
    end_date = selection_input.end_date if config.scope_use_ric_window else None
    return assets, start_date, end_date


def run_selection_pipeline(
    selection_input: SelectionInput,
    config: SelectionConfig,
) -> SelectionResult:
    started = time.perf_counter()
    result = SelectionResult(
        old_llm_gate_enabled=config.enable_new_vs_old_llm,
        new_llm_gate_enabled=config.enable_new_vs_new_llm,
    )

    all_candidates = sorted(
        {
            str(item)
            for item in selection_input.new_llm_df.get("factor_tag", [])
        }
    )

    ric_passed = apply_ric_gate(
        selection_input.ric_df,
        assets=selection_input.assets,
        ric_threshold=config.ric_threshold,
        require_full_asset_coverage=config.require_full_asset_coverage,
    )
    result.ric_passed_factors = ric_passed
    result.dropped_by_ric = sorted(set(all_candidates) - set(ric_passed))
    if not ric_passed:
        result.old_llm_gate_reason = "no_ric_passed_factor"
        result.new_llm_gate_reason = "no_ric_passed_factor"
        result.elapsed_total = time.perf_counter() - started
        return result

    llm_ric_df = selection_input.new_llm_df[
        selection_input.new_llm_df["factor_tag"].isin(set(ric_passed))
    ]
    if llm_ric_df.empty:
        result.old_llm_gate_reason = "no_llm_rows_after_ric_filter"
        result.new_llm_gate_reason = "no_llm_rows_after_ric_filter"
        result.elapsed_total = time.perf_counter() - started
        return result

    scoped_assets, scoped_start, scoped_end = _resolve_scope(selection_input, config)

    t0 = time.perf_counter()
    corr_new_vs_base = compute_pairwise_corr(
        llm_ric_df,
        selection_input.base_df,
        min_obs=config.min_corr_obs,
        assets=scoped_assets,
        start_date=scoped_start,
        end_date=scoped_end,
    )
    result.elapsed_new_vs_base = time.perf_counter() - t0
    result.corr_new_vs_base = corr_new_vs_base if corr_new_vs_base is not None else empty_corr_df()

    passed = apply_base_corr_gate(
        ric_passed,
        result.corr_new_vs_base,
        corr_threshold=config.corr_threshold,
    )
    result.passed_after_base_corr = passed
    result.dropped_by_base_corr = sorted(set(ric_passed) - set(passed))
    if not passed:
        result.old_llm_gate_reason = "no_factor_passed_new_vs_base"
        result.new_llm_gate_reason = "no_factor_passed_new_vs_base"
        result.elapsed_total = time.perf_counter() - started
        return result

    if not config.enable_new_vs_old_llm:
        result.old_llm_gate_reason = "disabled_by_selection_config"
    else:
        old_llm_df = selection_input.old_llm_df
        if old_llm_df is None and selection_input.old_llm_df_loader is not None:
            old_llm_df = selection_input.old_llm_df_loader()
        if old_llm_df is None or old_llm_df.empty:
            result.old_llm_gate_reason = "no_historical_llm_factor_values"
        else:
            result.old_llm_gate_applied = True
            llm_passed_df = selection_input.new_llm_df[
                selection_input.new_llm_df["factor_tag"].isin(set(passed))
            ]
            t0 = time.perf_counter()
            corr_new_vs_old_llm = compute_pairwise_corr(
                llm_passed_df,
                old_llm_df,
                min_obs=config.min_corr_obs,
                assets=scoped_assets,
                start_date=scoped_start,
                end_date=scoped_end,
            )
            result.elapsed_new_vs_old_llm = time.perf_counter() - t0
            result.corr_new_vs_old_llm = (
                corr_new_vs_old_llm if corr_new_vs_old_llm is not None else empty_corr_df()
            )
            if result.corr_new_vs_old_llm.empty:
                result.old_llm_gate_reason = "no_overlap_for_new_vs_old_llm"
            else:
                self_corr_max = result.corr_new_vs_old_llm.groupby("llm_factor")["abs_corr"].max()
                drop_old = sorted(
                    {factor for factor, val in self_corr_max.items() if val > config.llm_self_corr_threshold}
                )
                result.dropped_by_old_llm_corr = drop_old
                if drop_old:
                    drop_set = set(drop_old)
                    passed = [factor for factor in passed if factor not in drop_set]

    if not config.enable_new_vs_new_llm:
        result.new_llm_gate_reason = "disabled_by_selection_config"
    elif len(passed) < 2:
        result.new_llm_gate_reason = "less_than_2_candidates"
    else:
        result.new_llm_gate_applied = True
        llm_passed_df = selection_input.new_llm_df[
            selection_input.new_llm_df["factor_tag"].isin(set(passed))
        ]
        t0 = time.perf_counter()
        corr_new_vs_new = compute_pairwise_corr(
            llm_passed_df,
            llm_passed_df,
            min_obs=config.min_corr_obs,
            assets=scoped_assets,
            start_date=scoped_start,
            end_date=scoped_end,
        )
        result.elapsed_new_vs_new_llm = time.perf_counter() - t0
        corr_new_vs_new = corr_new_vs_new if corr_new_vs_new is not None else empty_corr_df()
        if not corr_new_vs_new.empty:
            corr_new_vs_new = corr_new_vs_new[corr_new_vs_new["llm_factor"] != corr_new_vs_new["base_factor"]]
        result.corr_new_vs_new_llm = corr_new_vs_new

        if result.corr_new_vs_new_llm.empty:
            result.new_llm_gate_reason = "no_overlap_for_new_vs_new_llm"
        else:
            deduped, drop_new = dedup_new_factors_by_self_corr(
                candidates=list(passed),
                self_corr_df=result.corr_new_vs_new_llm,
                ric_df=selection_input.ric_df,
                threshold=config.llm_self_corr_threshold,
            )
            passed = deduped
            result.dropped_by_new_llm_corr = drop_new

    result.passed_factors = sorted(passed)
    result.elapsed_total = time.perf_counter() - started
    return result

