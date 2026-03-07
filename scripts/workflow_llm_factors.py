#!/usr/bin/env python
"""End-to-end workflow: LLM → factor values → RIC → correlation → append to factor cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from fama.cli import _load_config
from fama.mining.orchestrator import PromptOrchestrator
from utils.factor_collection_dsl import FactorCollectionDSLNew
from utils.compute_correlation_new import load_base_factors, load_llm_factors
from fama.data.factor_space import deserialize_factor_set, serialize_factor_set, Factor, FactorSet
from fama.utils.io import ensure_dir
from fama.data.dataloader import available_factor_inputs, load_market_data
from fama.factors.alpha_lib import validate_alpha_syntax_strict
from utils.factor_catalog import load_factor_name_set, resolve_base_factor_cache
from utils.ric_engine import compute_rankic_from_files, resolve_ric_params
from fama.selection.config import apply_selection_overrides, load_selection_config
from fama.memory import append_memory_csv, build_memory_records
from fama.memory.llm_agents import run_retrieval_planner, run_round_analyst
from fama.memory.round_memory import (
    FactorLibraryIndex,
    append_round_memory_csv,
    build_retrieval_packet,
    build_round_memory_row,
    build_round_packet,
    extract_reference_names_from_plan,
    load_recent_round_context,
    render_retrieval_guidance,
)
from fama.selection.models import SelectionInput
from fama.selection.pipeline import run_selection_pipeline
from fama.selection.reporting import (
    apply_corr_scope,
    corr_input_summary,
)

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


def _resolve_train_valid_windows(
    cfg: dict,
    *,
    train_start_override: str | None = None,
    train_end_override: str | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    windows_cfg = cfg.get("windows", {}) if isinstance(cfg.get("windows"), dict) else {}
    train_cfg = windows_cfg.get("train", {}) if isinstance(windows_cfg.get("train"), dict) else {}
    valid_cfg = windows_cfg.get("valid", {}) if isinstance(windows_cfg.get("valid"), dict) else {}

    ric_cfg = cfg.get("ric", {}) if isinstance(cfg.get("ric"), dict) else {}
    coe_cfg = cfg.get("coe", {}) if isinstance(cfg.get("coe"), dict) else {}

    train_start = (
        train_start_override
        if train_start_override is not None
        else train_cfg.get("start_date") or ric_cfg.get("start_date") or coe_cfg.get("ric_start_date")
    )
    train_end = (
        train_end_override
        if train_end_override is not None
        else train_cfg.get("end_date") or ric_cfg.get("end_date") or coe_cfg.get("ric_end_date")
    )
    valid_start = valid_cfg.get("start_date")
    valid_end = valid_cfg.get("end_date")
    return train_start, train_end, valid_start, valid_end


def _resolve_selection_scope_window(
    *,
    mode: str,
    train_start: str | None,
    train_end: str | None,
    valid_start: str | None,
    valid_end: str | None,
    custom_start: str | None,
    custom_end: str | None,
    scope_name: str,
) -> tuple[str | None, str | None, str]:
    mode_text = str(mode).strip().lower()
    if mode_text == "train":
        return train_start, train_end, "windows.train"
    if mode_text == "valid":
        return valid_start, valid_end, "windows.valid"
    if mode_text == "full":
        return None, None, "full"
    if mode_text == "custom":
        return custom_start, custom_end, f"{scope_name}.start/end"
    raise ValueError(f"Unsupported {scope_name} mode: {mode_text}")


def _factor_to_meta(item: Factor) -> dict[str, object]:
    return {
        "expression": item.expression,
        "explanation": item.explanation,
        "references": item.references or [],
    }


def _load_factor_meta(path: Path) -> dict[str, dict[str, object]]:
    try:
        fs = deserialize_factor_set(str(path))
    except Exception:
        fs = FactorSet([])
    return {f.name: _factor_to_meta(f) for f in fs.factors if isinstance(getattr(f, "name", None), str)}


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
        help="Train min-abs-RIC threshold. Omit to follow cfg.selection.criteria.train_min_abs_ric.threshold.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=None,
        help="Train max-abs-corr threshold vs base. Omit to follow cfg.selection.criteria.train_max_abs_corr_base.threshold.",
    )
    parser.add_argument(
        "--llm-self-corr-threshold",
        type=float,
        default=None,
        help="Train max-abs-corr threshold for new-vs-old/new-vs-new LLM. Omit to follow cfg.selection.criteria.*.",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=None,
        help="Asset ids to enforce RIC threshold on. Omit to follow cfg.assets.",
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
        help="Minimum overlapping obs for correlation. Omit to follow cfg.selection.min_corr_obs.",
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
        help="Optional train-window start date (YYYY-MM-DD). Omit to follow cfg.windows.train.start_date.",
    )
    parser.add_argument(
        "--ric-end",
        default=None,
        help="Optional train-window end date (YYYY-MM-DD). Omit to follow cfg.windows.train.end_date.",
    )
    return parser.parse_args()


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
    selection_cfg = load_selection_config(cfg)
    memory_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}
    retrieval_cfg = memory_cfg.get("retrieval_planner", {}) if isinstance(memory_cfg.get("retrieval_planner"), dict) else {}
    round_analyst_cfg = memory_cfg.get("round_analyst", {}) if isinstance(memory_cfg.get("round_analyst"), dict) else {}

    args.iterations = int(args.iterations) if args.iterations is not None else int(workflow_cfg.get("iterations", 1000))
    selection_cfg = apply_selection_overrides(
        selection_cfg,
        ric_threshold=args.ric_threshold,
        corr_threshold=args.corr_threshold,
        llm_self_corr_threshold=args.llm_self_corr_threshold,
        min_corr_obs=args.min_corr_obs,
    )
    args.ric_threshold = float(selection_cfg.train_min_abs_ric_threshold)
    args.corr_threshold = float(selection_cfg.train_max_abs_corr_base_threshold)
    args.llm_self_corr_threshold = float(
        max(
            selection_cfg.train_max_abs_corr_old_llm_threshold,
            selection_cfg.train_max_abs_corr_new_llm_threshold,
        )
    )
    args.min_corr_obs = int(selection_cfg.min_corr_obs)

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
    ric_valid_output_path = _pick_path(
        None,
        paths.get("factor_ric_llm_valid"),
        PROJECT_ROOT / "tmp" / "llm_factor_ric_valid.csv",
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
    corr_output_valid_new_vs_base_path = _pick_path(
        None,
        paths.get("corr_output_valid_new_vs_base"),
        PROJECT_ROOT / "tmp" / "new_llm_vs_base_corr_valid.csv",
    )
    corr_output_valid_new_vs_old_llm_path = _pick_path(
        None,
        paths.get("corr_output_valid_new_vs_old_llm"),
        PROJECT_ROOT / "tmp" / "new_llm_vs_old_llm_corr_valid.csv",
    )
    corr_output_valid_new_vs_new_llm_path = _pick_path(
        None,
        paths.get("corr_output_valid_new_vs_new_llm"),
        PROJECT_ROOT / "tmp" / "new_llm_vs_new_llm_corr_valid.csv",
    )
    llm_factor_library_path = _pick_path(
        None,
        paths.get("llm_factor_library"),
        PROJECT_ROOT / "data" / "factor_cache_new" / "LLM_library.yaml",
    )
    success_cases_path = _pick_path(
        None,
        paths.get("factor_success_cases"),
        PROJECT_ROOT / "tmp" / "factor_success_cases.csv",
    )
    failure_cases_path = _pick_path(
        None,
        paths.get("factor_failure_cases"),
        PROJECT_ROOT / "tmp" / "factor_failure_cases.csv",
    )
    round_memory_path = _pick_path(
        None,
        paths.get("round_memory_csv"),
        PROJECT_ROOT / "data" / "memory" / "round_memory.csv",
    )
    retrieval_prompt_dump_path = (PROJECT_ROOT / "tmp" / "retrieval_planner_prompt.json").resolve()
    round_analyst_prompt_dump_path = (PROJECT_ROOT / "tmp" / "round_analyst_prompt.json").resolve()
    if price_path is None:
        raise ValueError("无法确定 market_data 路径，请设置 cfg.paths.market_data 或传入 --price-path。")
    if llm_factor_parquet_path is None:
        raise ValueError("无法确定 llm_factor_parquet 路径，请设置 cfg.paths.llm_factor_parquet 或传入 --llm-factor-parquet。")
    if base_factor_dir is None:
        raise ValueError("无法确定 base_factor_dir 路径，请设置 cfg.paths.base_factor_dir 或传入 --base-factor-dir。")
    if ric_output_path is None:
        raise ValueError("无法确定 LLM RIC 路径，请设置 cfg.paths.factor_ric_llm 或传入 --ric-output。")
    if ric_valid_output_path is None:
        raise ValueError("无法确定 LLM 验证集 RIC 路径，请设置 cfg.paths.factor_ric_llm_valid。")
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
    if success_cases_path is None:
        raise ValueError("无法确定 success cases 路径，请设置 cfg.paths.factor_success_cases。")
    if failure_cases_path is None:
        raise ValueError("无法确定 failure cases 路径，请设置 cfg.paths.factor_failure_cases。")
    if round_memory_path is None:
        raise ValueError("无法确定 round memory 路径，请设置 cfg.paths.round_memory_csv。")

    price_path = str(price_path.resolve())
    llm_factor_parquet_path = llm_factor_parquet_path.resolve()
    base_factor_dir = base_factor_dir.resolve()
    ric_output_path = ric_output_path.resolve()
    ric_valid_output_path = ric_valid_output_path.resolve()
    corr_output_new_vs_base_path = corr_output_new_vs_base_path.resolve()
    corr_output_new_vs_old_llm_path = corr_output_new_vs_old_llm_path.resolve()
    corr_output_new_vs_new_llm_path = corr_output_new_vs_new_llm_path.resolve()
    corr_output_valid_new_vs_base_path = corr_output_valid_new_vs_base_path.resolve()
    corr_output_valid_new_vs_old_llm_path = corr_output_valid_new_vs_old_llm_path.resolve()
    corr_output_valid_new_vs_new_llm_path = corr_output_valid_new_vs_new_llm_path.resolve()
    llm_factor_library_path = llm_factor_library_path.resolve()
    success_cases_path = success_cases_path.resolve()
    failure_cases_path = failure_cases_path.resolve()
    round_memory_path = round_memory_path.resolve()
    llm_output_path = llm_output_path.resolve() if llm_output_path is not None else None
    cfg["paths"]["market_data"] = price_path
    cfg["paths"]["llm_factor_library"] = str(llm_factor_library_path)

    base_resolved_path, base_sources = resolve_base_factor_cache(cfg)
    base_factor_name_set = load_factor_name_set(base_resolved_path)
    base_meta_map = _load_factor_meta(Path(base_resolved_path))

    # Resolve market data and assert amount exists
    md_path = price_path
    md = load_market_data(md_path)
    avail = available_factor_inputs(md)
    print(f"[workflow] Using market data: {md_path}")
    print(f"[workflow] Available fields: {avail}")
    if "AMOUNT" not in avail:
        raise RuntimeError("Market data missing AMOUNT; cannot evaluate AMOUNT-dependent factors.")
    global_assets, min_ric_obs, _, _ = resolve_ric_params(
        cfg,
        assets=args.assets,
        min_obs=args.min_ric_obs,
        start_date=None,
        end_date=None,
    )
    if selection_cfg.scope_asset_mode == "global":
        selection_metric_assets = [str(asset) for asset in (global_assets or [])]
    elif selection_cfg.scope_asset_mode == "custom":
        selection_metric_assets = [str(asset) for asset in selection_cfg.scope_assets]
    else:
        raise ValueError(f"Unsupported selection asset mode: {selection_cfg.scope_asset_mode}")

    train_start, train_end, valid_start, valid_end = _resolve_train_valid_windows(
        cfg,
        train_start_override=args.ric_start,
        train_end_override=args.ric_end,
    )
    corr_train_start, corr_train_end, train_scope_source = _resolve_selection_scope_window(
        mode=selection_cfg.scope_train_window_mode,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        custom_start=selection_cfg.scope_train_start_date,
        custom_end=selection_cfg.scope_train_end_date,
        scope_name="selection.scope.train",
    )
    corr_valid_start, corr_valid_end, valid_scope_source = _resolve_selection_scope_window(
        mode=selection_cfg.scope_valid_window_mode,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        custom_start=selection_cfg.scope_valid_start_date,
        custom_end=selection_cfg.scope_valid_end_date,
        scope_name="selection.scope.valid",
    )
    if not global_assets:
        raise ValueError("无法确定全局资产列表，请设置 --assets 或 cfg.assets。")
    if not selection_metric_assets:
        raise ValueError("selection 指标资产为空，请检查 selection.scope.asset_mode / selection.scope.assets。")
    print(
        f"[workflow] Train window: {train_start or 'unbounded'} → {train_end or 'unbounded'} | "
        f"Valid window: {valid_start or 'unbounded'} → {valid_end or 'unbounded'}"
    )
    print(
        "[workflow] Runtime params | "
        f"global_assets={','.join(global_assets)} | "
        f"selection_assets(mode={selection_cfg.scope_asset_mode})={','.join(selection_metric_assets)} | "
        f"min_ric_obs={min_ric_obs} | "
        f"train_min_abs_ric={selection_cfg.train_min_abs_ric_threshold} | "
        f"train_max_corr_base={selection_cfg.train_max_abs_corr_base_threshold} | "
        f"min_corr_obs={args.min_corr_obs}"
    )
    print(
        "[workflow] Criteria enabled | "
        f"train_ric={selection_cfg.train_min_abs_ric_enabled} | "
        f"train_corr_base={selection_cfg.train_max_abs_corr_base_enabled} | "
        f"train_corr_old={selection_cfg.train_max_abs_corr_old_llm_enabled} | "
        f"train_corr_new={selection_cfg.train_max_abs_corr_new_llm_enabled} | "
        f"valid_ric={selection_cfg.valid_min_abs_ric_enabled} | "
        f"valid_corr_base={selection_cfg.valid_max_abs_corr_base_enabled} | "
        f"valid_corr_old={selection_cfg.valid_max_abs_corr_old_llm_enabled} | "
        f"valid_corr_new={selection_cfg.valid_max_abs_corr_new_llm_enabled} | "
        f"max_ops={selection_cfg.max_operator_count_enabled}:{selection_cfg.max_operator_count_threshold} | "
        f"max_depth={selection_cfg.max_nesting_depth_enabled}:{selection_cfg.max_nesting_depth_threshold}"
    )
    print(
        "[workflow] Corr scope source | "
        f"train(mode={selection_cfg.scope_train_window_mode})={train_scope_source}:"
        f"{corr_train_start or 'unbounded'}->{corr_train_end or 'unbounded'} | "
        f"valid(mode={selection_cfg.scope_valid_window_mode})={valid_scope_source}:"
        f"{corr_valid_start or 'unbounded'}->{corr_valid_end or 'unbounded'}"
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
        ensure_dir(str(retrieval_prompt_dump_path.parent))
        retrieval_prompt_dump_path.write_text(
            json.dumps(
                {
                    "agent": "retrieval_planner",
                    "round_id": it + 1,
                    "status": "not_invoked_yet",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        round_analyst_prompt_dump_path.write_text(
            json.dumps(
                {
                    "agent": "round_analyst",
                    "round_id": it + 1,
                    "status": "not_invoked_yet",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        memory_guidance = ""
        memory_reference_names: list[str] = []
        if bool(memory_cfg.get("enabled", True)) and round_memory_path.exists():
            factor_library = FactorLibraryIndex.from_paths(
                base_factor_cache_path=base_resolved_path,
                llm_factor_library_path=llm_factor_library_path,
            )
            retrieval_packet = build_retrieval_packet(
                round_memory_path,
                recent_rounds=int(retrieval_cfg.get("recent_rounds", 10) or 10),
                top_pass_rounds=int(retrieval_cfg.get("top_pass_rounds", 3) or 3),
                top_duplicate_rounds=int(retrieval_cfg.get("top_duplicate_rounds", 3) or 3),
            )
            retrieval_plan = run_retrieval_planner(
                cfg,
                retrieval_packet,
                factor_library=factor_library,
                dump_path=retrieval_prompt_dump_path,
            )
            memory_guidance = render_retrieval_guidance(retrieval_plan)
            memory_reference_names = extract_reference_names_from_plan(retrieval_plan)
            if memory_guidance:
                print(
                    f"[workflow] Memory retrieval ready | refs={len(memory_reference_names)} | "
                    f"guidance_lines={len(memory_guidance.splitlines())} | "
                    f"prompt_dump={retrieval_prompt_dump_path}"
                )

        # 1) Generate expressions via CLI orchestrator
        print("[workflow] Call PromptOrchestrator.run(use_css=True, use_coe=True)")
        stage_started = time.perf_counter()
        orchestrator = PromptOrchestrator(cfg)
        expressions = orchestrator.run(
            use_css=True,
            use_coe=True,
            memory_guidance=memory_guidance,
            memory_reference_names=memory_reference_names,
        )
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

        new_factor_names = {f.name for f in new_factors}
        computed_factor_names = {
            str(name)
            for name in llm_df_check["factor_tag"].dropna().astype(str).unique().tolist()
        }
        dropped_not_computable = sorted(new_factor_names - computed_factor_names)
        if dropped_not_computable:
            print(
                f"[workflow] Dropped {len(dropped_not_computable)} factors: no computable values | "
                f"{', '.join(dropped_not_computable)}"
            )
        new_factors = [f for f in new_factors if f.name in computed_factor_names]
        if not new_factors:
            print("[workflow] 本轮新因子均不可计算，跳过。")
            print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")
            continue
        new_factor_names = {f.name for f in new_factors}

        llm_df = load_llm_factors(Path(llm_parquet))
        llm_df = llm_df[llm_df["factor_tag"].isin(new_factor_names)]
        if llm_df.empty:
            print("[workflow] 新因子值表中不存在可筛选因子，跳过本轮。")
            print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")
            continue

        # 3) RankIC (train + valid)
        print(
            f"[workflow] Call compute_rankic_from_files(train) | factor={llm_parquet} | output={ric_output_path} | "
            f"assets={','.join(selection_metric_assets)} | min_obs={min_ric_obs} | "
            f"window={train_start or 'unbounded'}->{train_end or 'unbounded'}"
        )
        stage_started = time.perf_counter()
        train_ric_df = compute_rankic_from_files(
            factor_path=Path(llm_parquet),
            price_path=price_path,
            output_path=ric_output_path,
            assets=selection_metric_assets,
            min_obs=min_ric_obs,
            start_date=train_start,
            end_date=train_end,
            include_ic=True,
            include_icir=True,
            config_path=cfg.get("_config_path"),
        )
        print(f"[workflow] compute_rankic_from_files(train) finished | elapsed={time.perf_counter() - stage_started:.2f}s")
        train_ric_df = train_ric_df[train_ric_df["factor_tag"].isin(new_factor_names)]

        print(
            f"[workflow] Call compute_rankic_from_files(valid) | factor={llm_parquet} | output={ric_valid_output_path} | "
            f"assets={','.join(selection_metric_assets)} | min_obs={min_ric_obs} | "
            f"window={valid_start or 'unbounded'}->{valid_end or 'unbounded'}"
        )
        stage_started = time.perf_counter()
        valid_ric_df = compute_rankic_from_files(
            factor_path=Path(llm_parquet),
            price_path=price_path,
            output_path=ric_valid_output_path,
            assets=selection_metric_assets,
            min_obs=min_ric_obs,
            start_date=valid_start,
            end_date=valid_end,
            include_ic=True,
            include_icir=True,
            config_path=cfg.get("_config_path"),
        )
        print(f"[workflow] compute_rankic_from_files(valid) finished | elapsed={time.perf_counter() - stage_started:.2f}s")
        valid_ric_df = valid_ric_df[valid_ric_df["factor_tag"].isin(new_factor_names)]

        # 4) Prepare reference factor values
        base_df = load_base_factors(base_factor_dir)
        if base_df.empty:
            raise RuntimeError(
                f"base_factor_dir 中未找到 base 因子，请检查目录内容: {base_factor_dir}"
            )

        train_llm_scope = apply_corr_scope(llm_df, selection_metric_assets, corr_train_start, corr_train_end)
        train_base_scope = apply_corr_scope(base_df, selection_metric_assets, corr_train_start, corr_train_end)
        valid_llm_scope = apply_corr_scope(llm_df, selection_metric_assets, corr_valid_start, corr_valid_end)
        valid_base_scope = apply_corr_scope(base_df, selection_metric_assets, corr_valid_start, corr_valid_end)
        print(
            f"[workflow] Corr scope(train) | assets={','.join(selection_metric_assets)} | "
            f"window={corr_train_start or 'unbounded'}->{corr_train_end or 'unbounded'} | min_obs={args.min_corr_obs}"
        )
        print(f"[workflow] Corr input(train.new): {corr_input_summary(train_llm_scope, label='new')}")
        print(f"[workflow] Corr input(train.base): {corr_input_summary(train_base_scope, label='base')}")
        print(
            f"[workflow] Corr scope(valid) | assets={','.join(selection_metric_assets)} | "
            f"window={corr_valid_start or 'unbounded'}->{corr_valid_end or 'unbounded'} | min_obs={args.min_corr_obs}"
        )
        print(f"[workflow] Corr input(valid.new): {corr_input_summary(valid_llm_scope, label='new')}")
        print(f"[workflow] Corr input(valid.base): {corr_input_summary(valid_base_scope, label='base')}")

        factor_cache_path = llm_factor_library_path
        try:
            factor_cache_fs = deserialize_factor_set(str(factor_cache_path))
        except Exception:
            factor_cache_fs = FactorSet([])
        historical_llm = [f for f in factor_cache_fs.factors if "LLM" in f.name.upper() and f.name not in new_factor_names]

        old_llm_loader = None
        if historical_llm:
            def _load_existing_llm_df() -> pd.DataFrame:
                existing_cache_path = PROJECT_ROOT / "tmp" / "llm_factor_cache_existing.yaml"
                ensure_dir(str(existing_cache_path.parent))
                serialize_factor_set(FactorSet(historical_llm), str(existing_cache_path))
                collector_existing = FactorCollectionDSLNew(
                    config_path=args.config,
                    factor_cache_path=existing_cache_path,
                    **LLM_DSL_COLLECTION_KWARGS,
                )
                print("[workflow] Prepare historical LLM factor values for correlation...")
                existing_parquet = collector_existing.update_dsl_factors(
                    output_path=llm_factor_parquet_path.parent / "existing_llm_factors.parquet",
                    batch_size=100,
                )
                existing_df = load_llm_factors(Path(existing_parquet))
                train_old_scope = apply_corr_scope(existing_df, selection_metric_assets, corr_train_start, corr_train_end)
                valid_old_scope = apply_corr_scope(existing_df, selection_metric_assets, corr_valid_start, corr_valid_end)
                print(f"[workflow] Corr input(train.old_llm): {corr_input_summary(train_old_scope, label='old_llm')}")
                print(f"[workflow] Corr input(valid.old_llm): {corr_input_summary(valid_old_scope, label='old_llm')}")
                return existing_df

            old_llm_loader = _load_existing_llm_df
        else:
            print("[workflow] No historical LLM factors for old-LLM correlation.")

        expr_map = {f.name: f.expression for f in new_factors}
        print("[workflow] Call selection.run_selection_pipeline (full metrics + intersection filter)")
        stage_started = time.perf_counter()
        selection_result = run_selection_pipeline(
            SelectionInput(
                train_ric_df=train_ric_df,
                valid_ric_df=valid_ric_df,
                new_llm_df=llm_df,
                base_df=base_df,
                expr_map=expr_map,
                assets=selection_metric_assets,
                train_start_date=train_start,
                train_end_date=train_end,
                valid_start_date=valid_start,
                valid_end_date=valid_end,
                old_llm_df_loader=old_llm_loader,
            ),
            selection_cfg,
        )
        print(
            f"[workflow] selection.run_selection_pipeline finished | "
            f"elapsed={time.perf_counter() - stage_started:.2f}s"
        )

        ensure_dir(str(corr_output_new_vs_base_path.parent))
        selection_result.corr_train_new_vs_base.to_csv(corr_output_new_vs_base_path, index=False)
        ensure_dir(str(corr_output_new_vs_old_llm_path.parent))
        selection_result.corr_train_new_vs_old_llm.to_csv(corr_output_new_vs_old_llm_path, index=False)
        ensure_dir(str(corr_output_new_vs_new_llm_path.parent))
        selection_result.corr_train_new_vs_new_llm.to_csv(corr_output_new_vs_new_llm_path, index=False)
        ensure_dir(str(corr_output_valid_new_vs_base_path.parent))
        selection_result.corr_valid_new_vs_base.to_csv(corr_output_valid_new_vs_base_path, index=False)
        ensure_dir(str(corr_output_valid_new_vs_old_llm_path.parent))
        selection_result.corr_valid_new_vs_old_llm.to_csv(corr_output_valid_new_vs_old_llm_path, index=False)
        ensure_dir(str(corr_output_valid_new_vs_new_llm_path.parent))
        selection_result.corr_valid_new_vs_new_llm.to_csv(corr_output_valid_new_vs_new_llm_path, index=False)

        metrics_df = selection_result.metrics_df
        passed = selection_result.passed_factors
        failed = selection_result.failed_factors
        print(
            f"[workflow] Metrics evaluated | candidates={len(metrics_df)} | "
            f"passed={len(passed)} | failed={len(failed)}"
        )
        if selection_result.skipped_criteria:
            skipped_text = ", ".join(f"{k}:{v}" for k, v in sorted(selection_result.skipped_criteria.items()))
            print(f"[workflow] Skipped criteria: {skipped_text}")

        def _print_top_corr(label: str, corr_df: pd.DataFrame) -> None:
            if corr_df is None or corr_df.empty:
                print(f"[workflow] {label}: no overlap")
                return
            print(f"[workflow] {label}:")
            topk = max(1, int(selection_cfg.log_topk))
            top_rows = corr_df.sort_values("abs_corr", ascending=False).head(topk)
            for row in top_rows.itertuples():
                print(
                    f"  - {row.llm_factor} vs {row.base_factor} | "
                    f"corr={row.weighted_corr:.4f} |abs|={row.abs_corr:.4f} | obs={row.total_obs}"
                )

        _print_top_corr("Top corr(train, new_vs_base)", selection_result.corr_train_new_vs_base)
        _print_top_corr("Top corr(train, new_vs_old_llm)", selection_result.corr_train_new_vs_old_llm)
        _print_top_corr("Top corr(train, new_vs_new_llm)", selection_result.corr_train_new_vs_new_llm)

        if failed:
            failed_view = metrics_df[metrics_df["final_status"] != "success"][["factor_id", "failure_reason"]].head(5)
            print("[workflow] Failed factors preview:")
            for row in failed_view.itertuples(index=False):
                print(f"  - {row.factor_id} | reason={row.failure_reason}")

        candidate_meta_map = {f.name: _factor_to_meta(f) for f in new_factors}
        llm_meta_map = _load_factor_meta(llm_factor_library_path)
        reference_meta_map = dict(base_meta_map)
        reference_meta_map.update(llm_meta_map)
        memory_df = build_memory_records(
            metrics_df,
            round_id=str(it + 1),
            batch_id=f"iter_{it + 1}",
            factor_meta_map=candidate_meta_map,
            reference_meta_map=reference_meta_map,
        )
        success_df = memory_df[memory_df["final_status"] == "success"]
        failure_df = memory_df[memory_df["final_status"] != "success"]
        append_memory_csv(success_cases_path, success_df)
        append_memory_csv(failure_cases_path, failure_df)
        print(
            f"[workflow] Memory updated | success_rows={len(success_df)} -> {success_cases_path} | "
            f"failure_rows={len(failure_df)} -> {failure_cases_path}"
        )

        recent_context: list[dict[str, object]] = []
        if bool(memory_cfg.get("enabled", True)) and round_memory_path.exists():
            recent_context = load_recent_round_context(
                round_memory_path,
                limit=int(round_analyst_cfg.get("recent_context_rounds", 2) or 2),
            )
        round_packet = build_round_packet(
            metrics_df,
            round_id=str(it + 1),
            batch_id=f"iter_{it + 1}",
            factor_meta_map=candidate_meta_map,
            reference_meta_map=reference_meta_map,
            recent_context=recent_context,
        )
        round_reflection = {}
        if bool(memory_cfg.get("enabled", True)):
            round_reflection = run_round_analyst(
                cfg,
                round_packet,
                dump_path=round_analyst_prompt_dump_path,
            )
        round_row = build_round_memory_row(round_packet, round_reflection)
        append_round_memory_csv(round_memory_path, round_row)
        if bool(memory_cfg.get("enabled", True)):
            print(
                f"[workflow] Round memory updated | round={it + 1} -> {round_memory_path} | "
                f"prompt_dump={round_analyst_prompt_dump_path}"
            )
        else:
            print(
                f"[workflow] Round memory updated without reflection | round={it + 1} -> {round_memory_path}"
            )

        if not passed:
            print("[workflow] No factors passed intersection criteria; continue to next iteration.")
            print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")
            continue

        expl_map = {f.name: f.explanation for f in new_factors}
        ref_map = {f.name: getattr(f, "references", None) for f in new_factors}
        selected_exprs = {name: expr_map[name] for name in passed if name in expr_map}
        selected_expl = {name: expl_map.get(name) for name in passed}
        selected_refs = {name: ref_map.get(name) for name in passed}

        factors_path = llm_factor_library_path
        append_to_factor_cache(factors_path, selected_exprs, selected_expl, selected_refs)
        print(f"[workflow] Appended {len(selected_exprs)} factors to {factors_path}")
        print(f"[workflow] Iteration elapsed={time.perf_counter() - iter_started:.2f}s")


if __name__ == "__main__":
    main()
