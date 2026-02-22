#!/usr/bin/env python
"""DSL factor-value computation collection (unified KunQuant backend)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import pandas as pd
import yaml

from fama.data.factor_space import deserialize_factor_set
from utils.kun_backend import compute_factor_values_kunquant_new
from utils.factor_collection import FactorCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "fama" / "config" / "defaults.yaml"

_CACHE_KEY_DEFAULT_PATH = {
    "base_factor_cache": "./tmp/base_factor_cache_resolved.yaml",
    "llm_factor_cache": "./data/factor_cache_new/LLM_factors.yaml",
}


class FactorCollectionDSLNew(FactorCollection):
    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        factor_cache_path: str | Path | None = None,
        *,
        cache_kind: str = "base_factor_cache",
        default_output_name: str = "dsl_factors_new.parquet",
        mode: str = "base",
        profile: str | None = None,
        factor_dir: str | Path | None = None,
        benchmark_symbol: str | None = None,
        factor_config_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            mode=mode,
            profile=profile,
            factor_dir=factor_dir,
            benchmark_symbol=benchmark_symbol,
            factor_config_path=factor_config_path,
            config_path=config_path,
        )
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (PROJECT_ROOT / self.config_path).resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        with self.config_path.open("r", encoding="utf-8") as fp:
            self.cfg: dict[str, Any] = yaml.safe_load(fp) or {}

        self.cache_kind = cache_kind
        if self.cache_kind not in _CACHE_KEY_DEFAULT_PATH:
            raise ValueError(
                f"Unsupported cache_kind: {self.cache_kind}. "
                f"Expect one of: {', '.join(sorted(_CACHE_KEY_DEFAULT_PATH))}."
            )

        self.default_output_name = str(default_output_name)
        if not self.default_output_name.strip():
            raise ValueError("default_output_name 不能为空。")

        paths = self.cfg.get("paths", {}) if isinstance(self.cfg, dict) else {}
        raw_cache_path = (
            Path(factor_cache_path)
            if factor_cache_path is not None
            else Path(paths.get(self.cache_kind, _CACHE_KEY_DEFAULT_PATH[self.cache_kind]))
        )
        self.factor_cache_path = (
            raw_cache_path
            if raw_cache_path.is_absolute()
            else (PROJECT_ROOT / raw_cache_path).resolve()
        )
        if not self.factor_cache_path.exists():
            if self.cache_kind == "base_factor_cache":
                from utils.factor_catalog import resolve_base_factor_cache

                resolved_path, _ = resolve_base_factor_cache(self.cfg)
                self.factor_cache_path = Path(resolved_path).resolve()
            if not self.factor_cache_path.exists():
                raise FileNotFoundError(
                    f"{self.cache_kind} 文件不存在：{self.factor_cache_path}。"
                )

    def update_dsl_factors(
        self,
        *,
        output_path: str | Path | None = None,
        threads: int | None = None,
        batch_size: int,
    ) -> Path:
        if batch_size is None:
            raise ValueError("batch_size 必须显式传入。")
        if int(batch_size) <= 0:
            raise ValueError("batch_size 必须是正整数。")

        factor_set = deserialize_factor_set(str(self.factor_cache_path))
        if not factor_set.factors:
            raise ValueError("factor cache 为空，无法计算 DSL 因子。")
        expressions = [factor.expression for factor in factor_set.factors]

        market_df = self._build_market_frame()
        symbols = market_df.index.get_level_values("symbol").unique().astype(str).tolist()
        dt_index = pd.to_datetime(market_df.index.get_level_values("date"))
        span_start = dt_index.min().date().isoformat() if len(dt_index) else "N/A"
        span_end = dt_index.max().date().isoformat() if len(dt_index) else "N/A"
        sample_symbols = ", ".join(symbols[:8]) + (" ..." if len(symbols) > 8 else "")
        compute_cfg = self.cfg.get("compute", {}) if isinstance(self.cfg, dict) else {}
        kun_threads = threads if threads is not None else int(compute_cfg.get("threads", 4))
        layout = str(compute_cfg.get("layout", "TS"))
        print(
            f"[DSL-New] 调用 update_dsl_factors | factors={len(expressions)} | "
            f"threads={kun_threads} | layout={layout} | batch_size={int(batch_size)}"
        )
        print(
            f"[DSL-New] 行情窗口: {span_start} -> {span_end} | 资产数: {len(symbols)} | "
            f"资产样例: {sample_symbols}"
        )
        started = time.perf_counter()
        kun_df, fallback = compute_factor_values_kunquant_new(
            market_df,
            expressions,
            threads=kun_threads,
            layout=layout,
            batch_size=int(batch_size),
        )
        elapsed = time.perf_counter() - started
        print(f"[DSL-New] 因子值计算耗时: {elapsed:.2f}s")
        if fallback:
            print(f"[DSL-New] KunQuant 无法解析 {len(fallback)} 个表达式，已跳过（不回退 Python）。")
            if kun_df is not None and not kun_df.empty:
                success_cols = [col for col in kun_df.columns if col not in fallback]
                kun_df = kun_df[success_cols]

        if kun_df is None or kun_df.empty:
            print("[DSL-New] KunQuant 全部解析失败，未生成因子。")
            return Path(output_path) if output_path else (self.factor_dir / self.default_output_name)

        stacked = kun_df.stack(future_stack=True).rename("value").reset_index()
        stacked = stacked.rename(columns={"date": "time", "symbol": "unique_id", "level_2": "factor_tag"})
        expr_to_name = {factor.expression: factor.name for factor in factor_set.factors}
        stacked["factor_tag"] = stacked["factor_tag"].map(expr_to_name).fillna(stacked["factor_tag"])
        stacked = stacked[["time", "unique_id", "factor_tag", "value"]]
        stacked = stacked.drop_duplicates(subset=["time", "unique_id", "factor_tag"])

        save_path = Path(output_path) if output_path else (self.factor_dir / self.default_output_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix == ".csv":
            stacked.to_csv(save_path, index=False)
        else:
            stacked.to_parquet(save_path, index=False)
        print(f"[DSL-New] 因子计算完成，写入 {save_path}（{len(stacked)} 行）。")
        return save_path

    def _build_market_frame(self) -> pd.DataFrame:
        required = {"time", "unique_id", "open", "high", "low", "close", "volume", "amount"}
        missing = required - set(self.native_price.columns)
        if missing:
            raise ValueError(f"行情数据缺少列: {missing}")

        native = deepcopy(self.native_price)
        frames = []
        cols = ["open", "high", "low", "close", "volume", "amount"]
        for uid, frame in native.groupby("unique_id", sort=False):
            df = frame.set_index("time")[cols].reindex(self.working_days).ffill().bfill()
            df["unique_id"] = uid
            frames.append(df.reset_index().rename(columns={"index": "time"}))
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.set_index(["time", "unique_id"]).sort_index()
        merged.index.names = ["date", "symbol"]
        return merged

if __name__ == "__main__":
    collector = FactorCollectionDSLNew()
    collector.update_dsl_factors(batch_size=500)
