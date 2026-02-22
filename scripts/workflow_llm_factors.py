#!/usr/bin/env python
"""End-to-end workflow: LLM → factor values → RIC → correlation → append to factor cache."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from fama.cli import _load_config
from fama.mining.orchestrator import PromptOrchestrator
from utils.factor_collection_dsl import FactorCollectionDSLNew
from utils.compute_correlation_new import (
    compute_pairwise_corr,
    load_base_factors,
    load_llm_factors,
)
from fama.data.factor_space import deserialize_factor_set, serialize_factor_set, Factor, FactorSet
from fama.utils.io import ensure_dir
from fama.data.dataloader import available_factor_inputs, load_market_data
from fama.factors.alpha_lib import validate_alpha_syntax_strict
from utils.factor_catalog import load_factor_name_set, resolve_base_factor_cache
from utils.ric_engine import compute_rankic_from_files, resolve_ric_params

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DSL_COLLECTION_KWARGS = {
    "mode": "llm",
    "profile": "llm",
    "cache_kind": "llm_factor_cache",
    "default_output_name": "dsl_LLM_factors_new.parquet",
}


def _pick_path(arg_value: str | Path | None, cfg_value: str | None, fallback: Path | None = None) -> Path | None:
    raw = arg_value if arg_value is not None else cfg_value
    if raw is None:
        return fallback
    return Path(raw).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full LLM factor mining workflow.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "fama" / "config" / "defaults.yaml"),
        help="Path to defaults.yaml (overrides supported).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of mining iterations. Omit to follow cfg.workflow.iterations.",
    )
    parser.add_argument(
        "--ric-threshold",
        type=float,
        default=None,
        help="RIC threshold per asset. Omit to follow cfg.workflow.ric_threshold.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=None,
        help="Max absolute correlation vs base factors. Omit to follow cfg.workflow.corr_threshold.",
    )
    parser.add_argument(
        "--llm-self-corr-threshold",
        type=float,
        default=None,
        help="Max absolute correlation for new-vs-old and new-vs-new LLM filtering. Omit to follow cfg.workflow.llm_self_corr_threshold.",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=None,
        help="Asset ids to enforce RIC threshold on. Omit to follow cfg.ric.assets (or cfg.coe.benchmark_assets).",
    )
    parser.add_argument(
        "--min-ric-obs",
        type=int,
        default=None,
        help="Minimum observations for RIC. Omit to follow cfg.ric.min_obs.",
    )
    parser.add_argument(
        "--min-corr-obs",
        type=int,
        default=None,
        help="Minimum overlapping obs for correlation. Omit to follow cfg.workflow.min_corr_obs.",
    )
    parser.add_argument(
        "--llm-output",
        default=None,
        help="Override cfg.paths.llm_output (optional).",
    )
    parser.add_argument(
        "--llm-factor-parquet",
        default=None,
        help="Override cfg.paths.llm_factor_parquet (optional).",
    )
    parser.add_argument(
        "--base-factor-dir",
        default=None,
        help="Override cfg.paths.base_factor_dir (optional).",
    )
    parser.add_argument(
        "--ric-output",
        default=None,
        help="Override cfg.paths.factor_ric_llm (optional).",
    )
    parser.add_argument(
        "--corr-output",
        default=None,
        help="Override cfg.paths.corr_output_new_vs_base (or legacy cfg.paths.corr_output).",
    )
    parser.add_argument(
        "--corr-output-new-vs-old-llm",
        default=None,
        help="Override cfg.paths.corr_output_new_vs_old_llm (optional).",
    )
    parser.add_argument(
        "--corr-output-new-vs-new-llm",
        default=None,
        help="Override cfg.paths.corr_output_new_vs_new_llm (optional).",
    )
    parser.add_argument(
        "--price-path",
        default=None,
        help="Override cfg.paths.market_data (optional).",
    )
    parser.add_argument(
        "--ric-start",
        default=None,
        help="Optional start date for RIC calculation (YYYY-MM-DD). Omit to follow cfg.ric.start_date/cfg.coe.ric_start_date.",
    )
    parser.add_argument(
        "--ric-end",
        default=None,
        help="Optional end date for RIC calculation (YYYY-MM-DD). Omit to follow cfg.ric.end_date/cfg.coe.ric_end_date.",
    )
    return parser.parse_args()


def filter_by_thresholds(
    ric_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    ric_threshold: float,
    corr_threshold: float,
    assets: Iterable[str],
) -> list[str]:
    """Return factor names that pass all thresholds."""

    assets = list(assets)
    if ric_df is None or ric_df.empty:
        return []

    asset_col = "asset" if "asset" in ric_df.columns else "unique_id"
    if asset_col not in ric_df.columns:
        raise ValueError("RIC 表缺少资产列（asset/unique_id）。")

    # RIC filter: factor must exist for all assets and clear the |RIC| threshold
    ric_subset = ric_df[ric_df[asset_col].isin(assets)].copy()
    ric_subset["abs_ric"] = ric_subset["ric"].abs()
    ric_grouped = ric_subset.groupby("factor_tag").agg(
        asset_coverage=(asset_col, "nunique"),
        min_abs_ric=("abs_ric", "min"),
    )
    ric_pass = set(
        ric_grouped[
            (ric_grouped["asset_coverage"] == len(assets))
            & (ric_grouped["min_abs_ric"] >= ric_threshold)
        ].index
    )

    if corr_df is None or corr_df.empty:
        # 没有相关性数据时，只根据 RIC 返回
        return sorted(ric_pass)

    # Correlation filter: max abs corr vs base factors below threshold
    corr_max = corr_df.groupby("llm_factor")["abs_corr"].max()
    corr_pass = {
        factor
        for factor in ric_pass
        # 缺少相关性结果视为不通过，确保有观测才入库
        if (factor in corr_max.index) and (corr_max.loc[factor] <= corr_threshold)
    }

    return sorted(ric_pass & corr_pass)


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
        .assign(abs_ric=lambda df: df["ric"].abs())
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


def _apply_corr_scope(
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


def _corr_input_summary(df: pd.DataFrame, *, label: str) -> str:
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


def append_to_factor_cache(
    factors_path: Path,
    expressions: dict[str, str],
    explanations: dict[str, str | None],
    references: dict[str, list[str] | None],
) -> None:
    fs = deserialize_factor_set(str(factors_path))
    existing = {f.name for f in fs.factors}
    for name, expr in expressions.items():
        if name in existing:
            continue
        fs.factors.append(
            Factor(
                name=name,
                expression=expr,
                explanation=explanations.get(name),
                references=references.get(name),
            )
        )
    serialize_factor_set(fs, str(factors_path))


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    cfg["_config_path"] = str(Path(args.config).expanduser().resolve())
    paths = cfg.setdefault("paths", {})
    workflow_cfg = cfg.get("workflow", {}) if isinstance(cfg.get("workflow"), dict) else {}

    args.iterations = int(args.iterations) if args.iterations is not None else int(workflow_cfg.get("iterations", 1000))
    args.ric_threshold = (
        float(args.ric_threshold)
        if args.ric_threshold is not None
        else float(workflow_cfg.get("ric_threshold", 0.08))
    )
    args.corr_threshold = (
        float(args.corr_threshold)
        if args.corr_threshold is not None
        else float(workflow_cfg.get("corr_threshold", 0.65))
    )
    args.llm_self_corr_threshold = (
        float(args.llm_self_corr_threshold)
        if args.llm_self_corr_threshold is not None
        else float(workflow_cfg.get("llm_self_corr_threshold", 0.7))
    )
    args.min_corr_obs = (
        int(args.min_corr_obs)
        if args.min_corr_obs is not None
        else int(workflow_cfg.get("min_corr_obs", 1000))
    )

    if args.iterations <= 0:
        raise ValueError(f"iterations 必须为正整数，当前={args.iterations}")
    if args.min_corr_obs <= 0:
        raise ValueError(f"min_corr_obs 必须为正整数，当前={args.min_corr_obs}")

    price_path = _pick_path(args.price_path, paths.get("market_data"))
    llm_output_path = _pick_path(args.llm_output, paths.get("llm_output"), PROJECT_ROOT / "tmp" / "llm-factor.yaml")
    llm_factor_parquet_path = _pick_path(
        args.llm_factor_parquet,
        paths.get("llm_factor_parquet"),
        PROJECT_ROOT / "scripts" / "LLM_factors" / "dsl_LLM_factors_new.parquet",
    )
    base_factor_dir = _pick_path(
        args.base_factor_dir,
        paths.get("base_factor_dir"),
        PROJECT_ROOT / "data" / "base_factors",
    )
    ric_output_path = _pick_path(
        args.ric_output,
        paths.get("factor_ric_llm"),
        PROJECT_ROOT / "tmp" / "llm_factor_ric.csv",
    )
    corr_output_new_vs_base_path = _pick_path(
        args.corr_output,
        paths.get("corr_output_new_vs_base") or paths.get("corr_output"),
        PROJECT_ROOT / "tmp" / "new_llm_vs_base_corr.csv",
    )
    corr_output_new_vs_old_llm_path = _pick_path(
        args.corr_output_new_vs_old_llm,
        paths.get("corr_output_new_vs_old_llm"),
        PROJECT_ROOT / "tmp" / "new_llm_vs_old_llm_corr.csv",
    )
    corr_output_new_vs_new_llm_path = _pick_path(
        args.corr_output_new_vs_new_llm,
        paths.get("corr_output_new_vs_new_llm"),
        PROJECT_ROOT / "tmp" / "new_llm_vs_new_llm_corr.csv",
    )
    llm_factor_library_path = _pick_path(
        None,
        paths.get("llm_factor_library"),
        PROJECT_ROOT / "data" / "factor_cache_new" / "LLM_library.yaml",
    )
    if price_path is None:
        raise ValueError("无法确定 market_data 路径，请设置 cfg.paths.market_data 或传入 --price-path。")
    if llm_factor_parquet_path is None:
        raise ValueError("无法确定 llm_factor_parquet 路径，请设置 cfg.paths.llm_factor_parquet 或传入 --llm-factor-parquet。")
    if base_factor_dir is None:
        raise ValueError("无法确定 base_factor_dir 路径，请设置 cfg.paths.base_factor_dir 或传入 --base-factor-dir。")
    if ric_output_path is None:
        raise ValueError("无法确定 LLM RIC 路径，请设置 cfg.paths.factor_ric_llm 或传入 --ric-output。")
    if corr_output_new_vs_base_path is None:
        raise ValueError("无法确定 new-vs-base 相关性路径，请设置 cfg.paths.corr_output_new_vs_base 或传入 --corr-output。")
    if corr_output_new_vs_old_llm_path is None:
        raise ValueError(
            "无法确定 new-vs-old-llm 相关性路径，请设置 cfg.paths.corr_output_new_vs_old_llm 或传入 --corr-output-new-vs-old-llm。"
        )
    if corr_output_new_vs_new_llm_path is None:
        raise ValueError(
            "无法确定 new-vs-new-llm 相关性路径，请设置 cfg.paths.corr_output_new_vs_new_llm 或传入 --corr-output-new-vs-new-llm。"
        )
    if llm_factor_library_path is None:
        raise ValueError("无法确定 LLM 正式入库路径，请设置 cfg.paths.llm_factor_library。")

    price_path = str(price_path.resolve())
    llm_factor_parquet_path = llm_factor_parquet_path.resolve()
    base_factor_dir = base_factor_dir.resolve()
    ric_output_path = ric_output_path.resolve()
    corr_output_new_vs_base_path = corr_output_new_vs_base_path.resolve()
    corr_output_new_vs_old_llm_path = corr_output_new_vs_old_llm_path.resolve()
    corr_output_new_vs_new_llm_path = corr_output_new_vs_new_llm_path.resolve()
    llm_factor_library_path = llm_factor_library_path.resolve()
    llm_output_path = llm_output_path.resolve() if llm_output_path is not None else None
    cfg["paths"]["market_data"] = price_path
    cfg["paths"]["llm_factor_library"] = str(llm_factor_library_path)

    base_resolved_path, base_sources = resolve_base_factor_cache(cfg)
    base_factor_name_set = load_factor_name_set(base_resolved_path)

    # Resolve market data and assert amount exists
    md_path = price_path
    md = load_market_data(md_path)
    avail = available_factor_inputs(md)
    print(f"[workflow] Using market data: {md_path}")
    print(f"[workflow] Available fields: {avail}")
    if "AMOUNT" not in avail:
        raise RuntimeError("Market data missing AMOUNT; cannot evaluate AMOUNT-dependent factors.")
    ric_assets, min_ric_obs, ric_start, ric_end = resolve_ric_params(
        cfg,
        assets=args.assets,
        min_obs=args.min_ric_obs,
        start_date=args.ric_start,
        end_date=args.ric_end,
    )
    if not ric_assets:
        raise ValueError("无法确定 RIC 资产列表，请设置 --assets 或 cfg.ric.assets/cfg.coe.benchmark_assets。")
    print(f"[workflow] RIC window: {ric_start or 'unbounded'} → {ric_end or 'unbounded'} (fallback to ric/coe config)")
    print(
        "[workflow] Runtime params | "
        f"ric_assets={','.join(ric_assets)} | min_ric_obs={min_ric_obs} | "
        f"ric_threshold={args.ric_threshold} | corr_threshold={args.corr_threshold} | "
        f"llm_self_corr_threshold={args.llm_self_corr_threshold}"
    )
    print(
        "[workflow] Base catalog | "
        f"sources={','.join(base_sources)} | resolved_cache={base_resolved_path} | "
        f"base_factor_count={len(base_factor_name_set)}"
    )

    for it in range(args.iterations):
        # if it != 0 and it % 20 == 0:
        #     time.sleep(180)
        iter_started = time.perf_counter()
        print(f"[workflow] Iteration {it + 1}/{args.iterations}")
        # Track existing LLM factors before this iteration
        llm_cache_path = Path(cfg["paths"].get("llm_factor_cache"))
        try:
            llm_fs = deserialize_factor_set(str(llm_cache_path))
        except Exception:
            llm_fs = FactorSet([])
        existing_names = {f.name for f in llm_fs.factors}

        # 1) Generate expressions via CLI orchestrator
        print("[workflow] Call PromptOrchestrator.run(use_css=True, use_coe=True)")
        stage_started = time.perf_counter()
        orchestrator = PromptOrchestrator(cfg)
        expressions = orchestrator.run(use_css=True, use_coe=True)
        print(
            f"[workflow] PromptOrchestrator finished | generated={len(expressions)} | "
            f"elapsed={time.perf_counter() - stage_started:.2f}s"
        )
        if llm_output_path:
            ensure_dir(str(llm_output_path.parent))
            from fama.utils.io import write_yaml

            write_yaml(llm_output_path, {"expressions": expressions})
            print(f"[workflow] Saved llm_output: {llm_output_path}")

        if not expressions:
            print("[workflow] No expressions generated; continue to next iteration.")
            print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")
            continue

        # 2) Compute factor values for LLM cache
        # Sanitize LLM factor cache to avoid legacy junk
        llm_cache_path = Path(cfg["paths"].get("llm_factor_cache"))
        try:
            llm_fs = deserialize_factor_set(str(llm_cache_path))
        except Exception:
            llm_fs = FactorSet([])
        items = []
        dirty_payload = None
        # Fallback raw load to recover from malformed cache
        if not llm_fs.factors:
            import yaml

            dirty_payload = yaml.safe_load(llm_cache_path.read_text()) or []
            if isinstance(dirty_payload, dict) and "expressions" in dirty_payload:
                dirty_payload = dirty_payload.get("expressions") or []
            for idx, item in enumerate(dirty_payload):
                if isinstance(item, dict):
                    name = item.get("name") or f"LLM_Factor_recovered_{idx+1}"
                    expr = item.get("expression")
                    expl = item.get("explanation")
                    items.append(Factor(name=name, expression=expr, explanation=expl))
        else:
            items = llm_fs.factors
        cleaned = []
        for f in items:
            expr = getattr(f, "expression", None)
            if not isinstance(expr, str) or not expr.strip():
                print(f"[workflow] Dropping invalid cached factor (empty expression): {f}")
                continue
            ok, reason = validate_alpha_syntax_strict(
                expr,
                allowed_variables=avail,
                allowed_ops=set(op.upper() for op in cfg.get("llm", {}).get("operator_whitelist", [])) or None,
            )
            if not ok:
                print(f"[workflow] Dropping invalid cached factor {f.name}: {reason}")
                continue
            cleaned.append(f)
        if len(cleaned) != len(items):
            serialize_factor_set(FactorSet(cleaned), str(llm_cache_path))

        collector = FactorCollectionDSLNew(config_path=args.config, **LLM_DSL_COLLECTION_KWARGS)
        # Identify newly added factors after orchestrator run
        llm_fs_after = deserialize_factor_set(str(llm_cache_path))
        new_factors = [f for f in llm_fs_after.factors if f.name not in existing_names]
        if not new_factors:
            print("[workflow] No new valid factors added this iteration; continue to next.")
            print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")
            continue

        # Compute only for new factors
        tmp_cache = FactorSet(new_factors)
        temp_cache_path = PROJECT_ROOT / "tmp" / "llm_factor_cache_tmp.yaml"
        ensure_dir(str(temp_cache_path.parent))
        serialize_factor_set(tmp_cache, str(temp_cache_path))

        from fama.utils.io import ensure_dir as _ensure
        _ensure(str(llm_factor_parquet_path.parent))
        collector_tmp = FactorCollectionDSLNew(
            config_path=args.config,
            factor_cache_path=temp_cache_path,
            **LLM_DSL_COLLECTION_KWARGS,
        )
        print(
            f"[workflow] Call FactorCollectionDSLNew.update_dsl_factors | "
            f"new_factors={len(new_factors)} | cache={temp_cache_path} | output={llm_factor_parquet_path}"
        )
        stage_started = time.perf_counter()
        llm_parquet = collector_tmp.update_dsl_factors(output_path=llm_factor_parquet_path, batch_size=1)
        print(f"[workflow] update_dsl_factors finished | elapsed={time.perf_counter() - stage_started:.2f}s")
        # 如果 KunQuant 全失败/无输出，跳过并不入库
        try:
            llm_df_check = pd.read_parquet(llm_parquet)
            if llm_df_check.empty:
                print("[workflow] 新因子计算结果为空，本轮跳过。")
                print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")
                continue
        except Exception:
            print("[workflow] 新因子计算结果读取失败，本轮跳过。")
            print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")
            continue
        calc_start = pd.to_datetime(llm_df_check["time"], errors="coerce").min()
        calc_end = pd.to_datetime(llm_df_check["time"], errors="coerce").max()
        asset_count = int(llm_df_check["unique_id"].nunique())
        factor_count = int(llm_df_check["factor_tag"].nunique())
        print(
            f"[workflow] New factor values ready | rows={len(llm_df_check)} | factors={factor_count} | "
            f"assets={asset_count} | window={calc_start.date() if pd.notna(calc_start) else 'N/A'}->"
            f"{calc_end.date() if pd.notna(calc_end) else 'N/A'}"
        )

        # 3) RankIC
        print(
            f"[workflow] Call compute_rankic_from_files | factor={llm_parquet} | output={ric_output_path} | "
            f"assets={','.join(ric_assets)} | min_obs={min_ric_obs} | "
            f"window={ric_start or 'unbounded'}->{ric_end or 'unbounded'}"
        )
        stage_started = time.perf_counter()
        ric_df = compute_rankic_from_files(
            factor_path=Path(llm_parquet),
            price_path=price_path,
            output_path=ric_output_path,
            assets=ric_assets,
            min_obs=min_ric_obs,
            start_date=ric_start,
            end_date=ric_end,
            include_ic=True,
            include_icir=True,
            config_path=cfg.get("_config_path"),
            calendar_anchor_symbol=cfg.get("backtest", {}).get("calendar_anchor_symbol"),
        )
        print(f"[workflow] compute_rankic_from_files finished | elapsed={time.perf_counter() - stage_started:.2f}s")

        # 4) Correlation vs base factors (only for RIC-passed factors)
        # Filter RIC to new factors only
        ric_df = ric_df[ric_df["factor_tag"].isin({f.name for f in new_factors})]
        ric_passed = set(filter_by_thresholds(ric_df, None, args.ric_threshold, 1.0, ric_assets))
        if not ric_passed:
            print("[workflow] No factors passed RIC thresholds; skip correlation and continue.")
            print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")
            continue

        llm_df = load_llm_factors(Path(llm_parquet))
        llm_df = llm_df[llm_df["factor_tag"].isin(ric_passed)]
        if llm_df.empty:
            print("[workflow] No LLM factors remain after RIC filter; continue to next iteration.")
            print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")
            continue
        base_df = load_base_factors(base_factor_dir)
        # base_df = base_df[base_df["factor_tag"].isin(base_factor_name_set)]
        if base_df.empty:
            raise RuntimeError(
                f"base_factor_dir 中未找到选定 base 因子，请检查 base 源与目录是否一致: {base_factor_dir}"
            )
        llm_df_scope = _apply_corr_scope(llm_df, ric_assets, ric_start, ric_end)
        base_df_scope = _apply_corr_scope(base_df, ric_assets, ric_start, ric_end)
        print(
            f"[workflow] Corr[new_vs_base] scope | assets={','.join(ric_assets)} | "
            f"window={ric_start or 'unbounded'}->{ric_end or 'unbounded'} | min_obs={args.min_corr_obs}"
        )
        print(f"[workflow] Corr[new_vs_base] {_corr_input_summary(llm_df_scope, label='new(passed_ric)')}")
        print(f"[workflow] Corr[new_vs_base] {_corr_input_summary(base_df_scope, label='base(reference_dir)')}")
        print("[workflow] Corr[new_vs_base] computing pairwise correlations...")
        stage_started = time.perf_counter()
        corr_df = compute_pairwise_corr(
            llm_df,
            base_df,
            min_obs=args.min_corr_obs,
            assets=ric_assets,
            start_date=ric_start,
            end_date=ric_end,
        )
        corr_pairs = len(corr_df) if corr_df is not None else 0
        corr_llm_factors = int(corr_df["llm_factor"].nunique()) if corr_df is not None and not corr_df.empty else 0
        print(
            f"[workflow] Corr[new_vs_base] done | pairs={corr_pairs} | llm_factors={corr_llm_factors} | "
            f"elapsed={time.perf_counter() - stage_started:.2f}s"
        )
        ensure_dir(str(corr_output_new_vs_base_path.parent))
        corr_df.to_csv(corr_output_new_vs_base_path, index=False)

        # RIC 通过因子的对照输出（含 Top1 相关性）
        print("[workflow] RIC 通过的因子明细：")
        for factor in sorted(ric_passed):
            ric_rows = ric_df[ric_df["factor_tag"] == factor]
            ric_str = "; ".join(f"{row.asset}:{row.ric:.4f}" for row in ric_rows.itertuples())
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
            print(f"  - {factor} | RIC[{ric_str}] | TopCorr[{top_corr_str}]")

        # 5) Filter and append to factor cache
        passed = filter_by_thresholds(
            ric_df,
            corr_df,
            ric_threshold=args.ric_threshold,
            corr_threshold=args.corr_threshold,
            assets=ric_assets,
        )
        if not passed:
            print("[workflow] No factors passed thresholds; continue to next iteration.")
            print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")
            continue

        # 4.1) Correlation: new-llm vs old-llm
        factor_cache_path = llm_factor_library_path
        try:
            factor_cache_fs = deserialize_factor_set(str(factor_cache_path))
        except Exception:
            factor_cache_fs = FactorSet([])
        historical_llm = [f for f in factor_cache_fs.factors if "LLM" in f.name.upper()]
        corr_cols = ["llm_factor", "base_factor", "weighted_corr", "abs_corr", "total_obs", "asset_pairs"]
        self_corr_old_df = pd.DataFrame(columns=corr_cols)
        if historical_llm and passed:
            existing_cache_path = PROJECT_ROOT / "tmp" / "llm_factor_cache_existing.yaml"
            ensure_dir(str(existing_cache_path.parent))
            serialize_factor_set(FactorSet(historical_llm), str(existing_cache_path))
            collector_existing = FactorCollectionDSLNew(
                config_path=args.config,
                factor_cache_path=existing_cache_path,
                **LLM_DSL_COLLECTION_KWARGS,
            )
            existing_parquet = collector_existing.update_dsl_factors(
                output_path=llm_factor_parquet_path.parent / "existing_llm_factors.parquet",
                batch_size=100,
            )
            existing_df = load_llm_factors(Path(existing_parquet))
            llm_df_passed = llm_df[llm_df["factor_tag"].isin(passed)]
            llm_df_passed_scope = _apply_corr_scope(llm_df_passed, ric_assets, ric_start, ric_end)
            existing_df_scope = _apply_corr_scope(existing_df, ric_assets, ric_start, ric_end)
            print(
                f"[workflow] Corr[new_vs_old_llm] scope | assets={','.join(ric_assets)} | "
                f"window={ric_start or 'unbounded'}->{ric_end or 'unbounded'} | min_obs={args.min_corr_obs}"
            )
            print(
                f"[workflow] Corr[new_vs_old_llm] "
                f"{_corr_input_summary(llm_df_passed_scope, label='new(passed_thresholds)')}"
            )
            print(
                f"[workflow] Corr[new_vs_old_llm] "
                f"{_corr_input_summary(existing_df_scope, label='old(llm_library)')}"
            )
            print("[workflow] Corr[new_vs_old_llm] computing pairwise correlations...")
            stage_started = time.perf_counter()
            self_corr_old_df = compute_pairwise_corr(
                llm_df_passed,
                existing_df,
                min_obs=args.min_corr_obs,
                assets=ric_assets,
                start_date=ric_start,
                end_date=ric_end,
            )
            old_pairs = len(self_corr_old_df) if self_corr_old_df is not None else 0
            print(
                f"[workflow] Corr[new_vs_old_llm] done | pairs={old_pairs} | "
                f"elapsed={time.perf_counter() - stage_started:.2f}s"
            )
            if self_corr_old_df.empty:
                self_corr_old_df = pd.DataFrame(columns=corr_cols)
                print("[workflow] No overlap to compute new-vs-old-llm correlation.")
            else:
                top_self_corr_old = (
                    self_corr_old_df.sort_values("abs_corr", ascending=False)
                    .groupby("llm_factor", as_index=False)
                    .first()
                )
                print("[workflow] Top self-corr: new-llm vs old-llm:")
                for row in top_self_corr_old.itertuples():
                    print(
                        f"  - {row.llm_factor} vs {row.base_factor} | corr={row.weighted_corr:.4f} "
                        f"|abs|={row.abs_corr:.4f} | obs={row.total_obs}"
                    )
                self_corr_max = self_corr_old_df.groupby("llm_factor")["abs_corr"].max()
                drop = {factor for factor, val in self_corr_max.items() if val > args.llm_self_corr_threshold}
                if drop:
                    print(
                        f"[workflow] Dropped {len(drop)} factors due to new-vs-old-llm corr > {args.llm_self_corr_threshold}: "
                        f"{', '.join(sorted(drop))}"
                    )
                    passed = [p for p in passed if p not in drop]
        else:
            print("[workflow] No historical LLM factors for new-vs-old-llm correlation.")
        ensure_dir(str(corr_output_new_vs_old_llm_path.parent))
        self_corr_old_df.to_csv(corr_output_new_vs_old_llm_path, index=False)

        # 4.2) Correlation: new-llm vs new-llm (same iteration), then dedup
        self_corr_new_df = pd.DataFrame(columns=corr_cols)
        if len(passed) >= 2:
            llm_df_passed = llm_df[llm_df["factor_tag"].isin(passed)]
            llm_df_passed_scope = _apply_corr_scope(llm_df_passed, ric_assets, ric_start, ric_end)
            print(
                f"[workflow] Corr[new_vs_new_llm] scope | assets={','.join(ric_assets)} | "
                f"window={ric_start or 'unbounded'}->{ric_end or 'unbounded'} | min_obs={args.min_corr_obs}"
            )
            print(
                f"[workflow] Corr[new_vs_new_llm] "
                f"{_corr_input_summary(llm_df_passed_scope, label='new(candidates)')}"
            )
            print("[workflow] Corr[new_vs_new_llm] computing pairwise correlations...")
            stage_started = time.perf_counter()
            self_corr_new_df = compute_pairwise_corr(
                llm_df_passed,
                llm_df_passed,
                min_obs=args.min_corr_obs,
                assets=ric_assets,
                start_date=ric_start,
                end_date=ric_end,
            )
            new_pairs = len(self_corr_new_df) if self_corr_new_df is not None else 0
            print(
                f"[workflow] Corr[new_vs_new_llm] done | pairs={new_pairs} | "
                f"elapsed={time.perf_counter() - stage_started:.2f}s"
            )
            if not self_corr_new_df.empty:
                self_corr_new_df = self_corr_new_df[self_corr_new_df["llm_factor"] != self_corr_new_df["base_factor"]]
            if self_corr_new_df.empty:
                self_corr_new_df = pd.DataFrame(columns=corr_cols)
                print("[workflow] No overlap to compute new-vs-new-llm correlation.")
            else:
                top_self_corr_new = (
                    self_corr_new_df.sort_values("abs_corr", ascending=False)
                    .groupby("llm_factor", as_index=False)
                    .first()
                )
                print("[workflow] Top self-corr: new-llm vs new-llm:")
                for row in top_self_corr_new.itertuples():
                    print(
                        f"  - {row.llm_factor} vs {row.base_factor} | corr={row.weighted_corr:.4f} "
                        f"|abs|={row.abs_corr:.4f} | obs={row.total_obs}"
                    )
                passed, drop_new = dedup_new_factors_by_self_corr(
                    candidates=list(passed),
                    self_corr_df=self_corr_new_df,
                    ric_df=ric_df,
                    threshold=args.llm_self_corr_threshold,
                )
                if drop_new:
                    print(
                        f"[workflow] Dropped {len(drop_new)} factors due to new-vs-new-llm corr > {args.llm_self_corr_threshold}: "
                        f"{', '.join(drop_new)}"
                    )
        else:
            print("[workflow] Less than 2 passed factors; skip new-vs-new-llm correlation.")
        ensure_dir(str(corr_output_new_vs_new_llm_path.parent))
        self_corr_new_df.to_csv(corr_output_new_vs_new_llm_path, index=False)
        if not passed:
            print("[workflow] No factors passed thresholds; continue to next iteration.")
            print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")
            continue

        # Map expressions/explanations from LLM cache
        llm_cache_path = Path(cfg["paths"].get("llm_factor_cache"))
        llm_fs = deserialize_factor_set(str(llm_cache_path))
        expr_map = {f.name: f.expression for f in llm_fs.factors}
        expl_map = {f.name: f.explanation for f in llm_fs.factors}
        ref_map = {f.name: getattr(f, "references", None) for f in llm_fs.factors}
        selected_exprs = {name: expr_map[name] for name in passed if name in expr_map}
        selected_expl = {name: expl_map.get(name) for name in passed if name in expl_map}
        selected_refs = {name: ref_map.get(name) for name in passed}

        factors_path = llm_factor_library_path
        append_to_factor_cache(factors_path, selected_exprs, selected_expl, selected_refs)
        print(f"[workflow] Appended {len(selected_exprs)} factors to {factors_path}")
        print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")


if __name__ == "__main__":
    main()
