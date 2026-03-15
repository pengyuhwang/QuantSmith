"""Chain-of-Experience manager that follows the two-phase construction described in FAMA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from fama.data.factor_space import FactorSet
from utils.efficientCalculation import EfficientCalculator


@dataclass
class CoEChain:
    chain_id: int
    cluster_id: int
    factor_names: list[str] = field(default_factory=list)
    expressions: list[str] = field(default_factory=list)
    representative_score: float = 0.0

    def to_list(self) -> list[str]:
        return self.expressions.copy()


class CoEManager:
    """Manage initial chain generation and enhanced updates."""

    def __init__(self) -> None:
        self._chains: list[CoEChain] = []
        self._expr_to_chain: dict[str, int] = {}
        self._gamma_scores: dict[str, float] = {}
        self._gamma_assets: dict[str, Optional[str]] = {}
        self._ric_by_asset: dict[str, dict[str, float]] = {}
        self._ic_by_asset: dict[str, dict[str, float]] = {}
        self._icir_by_asset: dict[str, dict[str, float]] = {}
        self._precomputed_ric: Optional[pd.DataFrame] = None
        self._logger = None
        self._forward_returns: Optional[pd.Series] = None
        self._ric_calc = EfficientCalculator()
        self._min_samples = 10
        self.max_depth = 5
        self.min_rankic = 0.05
        self.prompt_chains = 3
        self.prompt_expr_chars: Optional[int] = None
        self.benchmark_assets: list[str] = []
        self.ric_start: Optional[pd.Timestamp] = None
        self.ric_end: Optional[pd.Timestamp] = None

    @property
    def chains(self) -> list[CoEChain]:
        return self._chains

    def set_forward_returns(self, series: Optional[pd.Series]) -> None:
        self._forward_returns = series

    def set_precomputed_ric(self, table: Optional[pd.DataFrame]) -> None:
        if table is None:
            self._precomputed_ric = None
            return
        required = {"factor_tag", "unique_id", "ric"}
        missing = required - set(table.columns)
        if missing:
            if self._logger:
                self._logger.warning("预计算 RIC 缺少必要列 %s，将忽略该文件。", missing)
            self._precomputed_ric = None
            return
        df = table.copy()
        df["ric"] = pd.to_numeric(df["ric"], errors="coerce")
        for col in ["ic", "icir"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["ric"])
        if "abs_ric" not in df.columns:
            df["abs_ric"] = df["ric"].abs()
        else:
            df["abs_ric"] = pd.to_numeric(df["abs_ric"], errors="coerce").abs()
        if "ic" in df.columns and "abs_ic" not in df.columns:
            df["abs_ic"] = df["ic"].abs()
        if "icir" in df.columns and "abs_icir" not in df.columns:
            df["abs_icir"] = df["icir"].abs()
        self._precomputed_ric = df

    def rebuild_from_clusters(
        self,
        factor_set: FactorSet,
        factor_frame: pd.DataFrame,
        clusters: list[list[int]],
    ) -> None:
        """Rebuild CoE chains from scratch based on current clusters."""

        self._chains = []
        self._expr_to_chain = {}
        if not clusters:
            return
        self._gamma_scores = {}
        self._gamma_assets = {}
        self._ric_by_asset = {}
        self._ic_by_asset = {}
        self._icir_by_asset = {}
        expr_to_name = {factor.expression: factor.name for factor in factor_set.factors}
        self._gamma_scores = self._compute_rankic_scores(factor_frame, expr_to_name)
        cluster_infos: list[tuple[int, float, list[tuple[float, str, str]]]] = []
        for cluster_id, members in enumerate(clusters):
            member_data: list[tuple[float, str, str]] = []
            for idx in members:
                if idx >= len(factor_set.factors):
                    continue
                factor = factor_set.factors[idx]
                gamma = self._gamma_scores.get(factor.expression)
                if gamma is None:
                    continue
                member_data.append((gamma, factor.name, factor.expression))
            if not member_data:
                continue
            rep_gamma = max(member_data, key=lambda item: abs(item[0]))[0]
            member_data.sort(key=lambda item: abs(item[0]))
            if self.max_depth and len(member_data) > self.max_depth:
                member_data = member_data[-self.max_depth :]
            cluster_infos.append((cluster_id, rep_gamma, member_data))
        cluster_infos.sort(key=lambda item: abs(item[1]), reverse=True)
        limit = self.prompt_chains or len(cluster_infos)
        selected_infos = cluster_infos[:limit]
        for cluster_id, rep_gamma, member_data in selected_infos:
            chain_id = len(self._chains)
            names = [name for _, name, _ in member_data]
            exprs = [expr for _, _, expr in member_data]
            chain = CoEChain(
                chain_id=chain_id,
                cluster_id=cluster_id,
                factor_names=names,
                expressions=exprs,
                representative_score=rep_gamma,
            )
            self._chains.append(chain)
            for expr in exprs:
                self._expr_to_chain[expr] = chain_id
        if self._logger:
            self._logger.info("CoE rebuilt %d chains.", len(self._chains))
            self.log_chains(self._logger)

    def _sort_members_by_gamma(self, members: list[int], factor_set: FactorSet) -> list[int]:
        def score(idx: int) -> float:
            expr = factor_set.factors[idx].expression
            return self._gamma_scores.get(expr, 0.0)

        return sorted(members, key=score)

    def match_chain(self, expression: str) -> Optional[CoEChain]:
        chain_id = self._expr_to_chain.get(expression)
        if chain_id is None:
            return self._chains[0] if self._chains else None
        return self._chains[chain_id]

    def get_chain_expressions(self, chain: Optional[CoEChain]) -> list[str]:
        if chain is None:
            return []
        return chain.to_list()

    def format_top_chains(self, max_expr_chars: Optional[int] = None) -> list[str]:
        limit = max_expr_chars if max_expr_chars is not None else self.prompt_expr_chars
        selected = self._select_prompt_chains()
        formatted: list[str] = []
        allowed_metrics = set(self.prompt_metrics) if self.prompt_metrics is not None else {"ric", "ic", "icir"}
        for chain in selected:
            expressions = []
            for idx, expr in enumerate(chain.expressions):
                name = chain.factor_names[idx] if idx < len(chain.factor_names) else None
                label = self._truncate_expr(expr, limit)
                score = self._gamma_scores.get(expr)
                best_asset = self._gamma_assets.get(expr)
                bench_label = best_asset or "/".join(self.benchmark_assets) or "ALL"
                expr_label = f"{name}: {label}" if name else label
                if score is None:
                    expressions.append(expr_label)
                    continue
                metrics: list[str] = []
                if "ric" in allowed_metrics:
                    metrics.append(f"ric={score:.3f}")
                ic_val = self._ic_by_asset.get(expr, {}).get(best_asset) if best_asset else None
                icir_val = self._icir_by_asset.get(expr, {}).get(best_asset) if best_asset else None
                if "ic" in allowed_metrics and ic_val is not None and not pd.isna(ic_val):
                    metrics.append(f"ic={ic_val:.3f}")
                if "icir" in allowed_metrics and icir_val is not None and not pd.isna(icir_val):
                    metrics.append(f"icir={icir_val:.3f}")
                metrics.append(f"benchmark_assets={bench_label}")
                expressions.append(f"{expr_label} ({', '.join(metrics)})")
            if expressions:
                formatted.append(" -> ".join(expressions))
        return formatted

    def top_chain_factor_names(self) -> list[str]:
        """Return unique factor names from the chains selected for prompting."""

        names: set[str] = set()
        for chain in self._select_prompt_chains():
            names.update(chain.factor_names)
        return sorted(names)

    def register_existing_expression(self, expr: str, chain_id: int) -> None:
        self._expr_to_chain[expr] = chain_id

    def integrate_factor(
        self,
        chain: Optional[CoEChain],
        factor_name: str,
        expression: str,
        factor_series: pd.Series,
        factor_frame: pd.DataFrame,
    ) -> Optional[CoEChain]:
        if chain is None or factor_series.empty:
            return None
        result = self._compute_single_rankic(factor_series, expression)
        if result is None:
            return None
        gamma_new, best_asset, per_asset = result
        self._ric_by_asset[expression] = per_asset
        self._gamma_scores[expression] = gamma_new
        self._gamma_assets[expression] = best_asset
        chain_scores = [self._gamma_scores.get(expr, 0.0) for expr in chain.expressions]
        if chain_scores and gamma_new <= max(chain_scores):
            return None

        match_expr = self._find_highest_corr_expression(chain, factor_series, factor_frame)
        if match_expr is None:
            self._append_to_chain(chain, factor_name, expression)
            return chain

        match_idx = chain.expressions.index(match_expr)
        if match_idx == len(chain.expressions) - 1:
            self._append_to_chain(chain, factor_name, expression)
            return chain

        return self._split_chain(chain, match_idx, factor_name, expression)

    def _find_highest_corr_expression(
        self,
        chain: CoEChain,
        new_series: pd.Series,
        factor_frame: pd.DataFrame,
    ) -> Optional[str]:
        best_expr = None
        best_corr = -np.inf
        for expr in chain.expressions:
            if expr not in factor_frame.columns:
                continue
            base_series = factor_frame[expr]
            corr = base_series.corr(new_series)
            corr = 0.0 if pd.isna(corr) else abs(float(corr))
            if corr > best_corr:
                best_corr = corr
                best_expr = expr
        return best_expr

    def _append_to_chain(self, chain: CoEChain, factor_name: str, expression: str) -> None:
        chain.factor_names.append(factor_name)
        chain.expressions.append(expression)
        self._expr_to_chain[expression] = chain.chain_id
        self._enforce_chain_depth(chain)

    def _split_chain(self, chain: CoEChain, split_idx: int, factor_name: str, expression: str) -> CoEChain:
        new_chain_id = len(self._chains)
        new_names = chain.factor_names[: split_idx + 1] + [factor_name]
        new_exprs = chain.expressions[: split_idx + 1] + [expression]
        new_chain = CoEChain(chain_id=new_chain_id, cluster_id=chain.cluster_id, factor_names=new_names, expressions=new_exprs)
        self._enforce_chain_depth(new_chain)
        self._chains.append(new_chain)
        for expr in new_chain.expressions:
            self._expr_to_chain[expr] = new_chain_id
        return new_chain

    def _enforce_chain_depth(self, chain: CoEChain) -> None:
        if not self.max_depth:
            return
        while len(chain.expressions) > self.max_depth:
            removed_expr = chain.expressions.pop(0)
            chain.factor_names.pop(0)
            self._expr_to_chain.pop(removed_expr, None)

    def attach_logger(self, logger) -> None:
        self._logger = logger

    def log_chains(self, logger, max_preview: int = 5) -> None:
        if not self._chains:
            logger.info("CoE chains are empty.")
            return
        for chain in self._chains:
            preview = " -> ".join(chain.expressions[:max_preview])
            if len(chain.expressions) > max_preview:
                preview += " -> ..."
            logger.info(
                "CoE chain #%d (cluster=%d, length=%d): %s",
                chain.chain_id,
                chain.cluster_id,
                len(chain.expressions),
                preview,
            )

    def _compute_rankic_scores(self, factor_frame: pd.DataFrame, expr_to_name: dict[str, str]) -> dict[str, float]:
        if self._precomputed_ric is not None:
            return self._scores_from_precomputed(expr_to_name)
        scores: dict[str, float] = {}
        if self._forward_returns is None or factor_frame.empty:
            return scores
        for expr in factor_frame.columns:
            series = factor_frame[expr]
            per_asset = self._compute_asset_scores(series, expr)
            self._ric_by_asset[expr] = per_asset
            if not per_asset:
                continue
            best_asset, best_score = max(per_asset.items(), key=lambda item: abs(item[1]))
            if abs(best_score) < self.min_rankic:
                continue
            scores[expr] = best_score
            self._gamma_assets[expr] = best_asset
        return scores

    def _scores_from_precomputed(self, expr_to_name: dict[str, str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        table = self._precomputed_ric
        if table is None:
            return scores
        # reset caches for ic/icir when loading fresh table
        self._ric_by_asset = {}
        self._ic_by_asset = {}
        self._icir_by_asset = {}
        tag_set = set(table["factor_tag"])
        for expr, name in expr_to_name.items():
            subset = table[table["factor_tag"] == name]
            if subset.empty:
                continue
            subset = subset.sort_values("abs_ric", ascending=False)
            best_row = subset.iloc[0]
            score = float(best_row["ric"])
            if abs(score) < self.min_rankic:
                continue
            scores[expr] = score
            self._gamma_assets[expr] = best_row["unique_id"]
            self._ric_by_asset[expr] = {row["unique_id"]: float(row["ric"]) for _, row in subset.iterrows()}
            if "ic" in subset.columns:
                self._ic_by_asset[expr] = {
                    row["unique_id"]: float(row["ic"]) for _, row in subset.dropna(subset=["ic"]).iterrows()
                }
            if "icir" in subset.columns:
                self._icir_by_asset[expr] = {
                    row["unique_id"]: float(row["icir"]) for _, row in subset.dropna(subset=["icir"]).iterrows()
                }
        missing = [name for name in expr_to_name.values() if name not in tag_set]
        if missing and self._logger:
            preview = ", ".join(missing[:5])
            self._logger.info("部分因子未在 factor_ric.csv 中找到: %s%s", preview, "..." if len(missing) > 5 else "")
        return scores

    def _compute_asset_scores(self, series: pd.Series, expr: Optional[str] = None) -> dict[str, float]:
        scores: dict[str, float] = {}
        if self._forward_returns is None:
            return scores
        returns = self._forward_returns
        candidates = self.benchmark_assets or series.index.get_level_values(1).unique().tolist()
        for asset in candidates:
            mask = series.index.get_level_values(1) == asset
            if not mask.any():
                continue
            series_asset = series[mask]
            returns_asset = returns[returns.index.get_level_values(1) == asset]
            if isinstance(series_asset.index, pd.MultiIndex):
                series_asset = series_asset.droplevel(1)
            if isinstance(returns_asset.index, pd.MultiIndex):
                returns_asset = returns_asset.droplevel(1)
            joined = pd.concat([series_asset, returns_asset], axis=1, join="inner").dropna()
            if self.ric_start is not None:
                joined = joined[joined.index >= self.ric_start]
            if self.ric_end is not None:
                joined = joined[joined.index <= self.ric_end]
            if len(joined) < self._min_samples:
                continue
            series_clean = joined.iloc[:, 0]
            returns_clean = joined.iloc[:, 1]
            if series_clean.nunique() <= 1 or returns_clean.nunique() <= 1:
                continue
            try:
                values = joined.iloc[:, 0].to_numpy()
                rets = joined.iloc[:, 1].to_numpy()
                ric = self._ric_calc.efficent_cal_ric(values, rets)
                ic = self._ric_calc.efficient_cal_ic(values, rets)
                icir = self._ric_calc.efficient_cal_icir(values, rets)
            except Exception:
                continue
            if pd.isna(ric):
                continue
            scores[asset] = float(ric)
            if expr is not None:
                self._ic_by_asset.setdefault(expr, {})[asset] = float(ic) if not pd.isna(ic) else None
                if icir is not None and not pd.isna(icir):
                    self._icir_by_asset.setdefault(expr, {})[asset] = float(icir)
        return scores

    def _compute_single_rankic(self, series: pd.Series, expr: Optional[str] = None) -> Optional[tuple[float, Optional[str], dict[str, float]]]:
        per_asset = self._compute_asset_scores(series, expr)
        if not per_asset:
            return None
        best_asset, best_score = max(per_asset.items(), key=lambda item: abs(item[1]))
        if abs(best_score) < self.min_rankic:
            return None
        return best_score, best_asset, per_asset

    @staticmethod
    def _truncate_expr(expr: str, limit: Optional[int]) -> str:
        clean = " ".join(expr.split())
        if not limit or limit <= 0 or len(clean) <= limit:
            return clean
        return clean[: limit - 3] + "..."

    def _select_prompt_chains(self) -> list[CoEChain]:
        ranked = sorted(
            self._chains,
            key=lambda chain: abs(chain.representative_score),
            reverse=True,
        )
        return ranked[: self.prompt_chains] if self.prompt_chains else ranked
