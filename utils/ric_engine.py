from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import pandas as pd
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from fama.utils.io import ensure_dir
from utils.backtest_utils import prepare_price_data
from utils.efficientCalculation import EfficientCalculator

DEFAULT_MIN_OBS = 1000


def resolve_ric_params(
    cfg: Mapping[str, Any] | None,
    *,
    assets: Iterable[str] | None = None,
    min_obs: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[str] | None, int, str | None, str | None]:
    """Resolve RIC runtime parameters from cfg + overrides."""

    cfg = cfg or {}
    ric_cfg = cfg.get("ric", {}) if isinstance(cfg, Mapping) else {}
    coe_cfg = cfg.get("coe", {}) if isinstance(cfg, Mapping) else {}

    resolved_assets: list[str] | None
    if assets is not None:
        resolved_assets = [str(item) for item in assets]
    else:
        cfg_assets = ric_cfg.get("assets") if isinstance(ric_cfg, Mapping) else None
        if not cfg_assets:
            cfg_assets = coe_cfg.get("benchmark_assets") if isinstance(coe_cfg, Mapping) else None
        if isinstance(cfg_assets, str):
            cfg_assets = [cfg_assets]
        resolved_assets = [str(item) for item in cfg_assets] if cfg_assets else None

    resolved_min_obs = int(min_obs if min_obs is not None else ric_cfg.get("min_obs", DEFAULT_MIN_OBS))
    if resolved_min_obs <= 0:
        raise ValueError(f"min_obs must be positive, got {resolved_min_obs}.")

    resolved_start = start_date if start_date is not None else ric_cfg.get("start_date") or coe_cfg.get("ric_start_date")
    resolved_end = end_date if end_date is not None else ric_cfg.get("end_date") or coe_cfg.get("ric_end_date")
    return resolved_assets, resolved_min_obs, resolved_start, resolved_end


def load_factor_long_table(path: str | Path) -> pd.DataFrame:
    """Load factor long table from parquet/csv and normalize schema."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Factor file not found: {file_path}")

    if file_path.suffix == ".parquet":
        frame = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        frame = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported factor file type: {file_path.suffix}")

    return normalize_factor_long_table(frame)


def normalize_factor_long_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize factor long table to required columns: time, unique_id, factor_tag, value."""

    rename_map: dict[str, str] = {}
    cols = set(frame.columns)

    if "time" not in cols and "date" in cols:
        rename_map["date"] = "time"
    if "unique_id" not in cols:
        if "asset" in cols:
            rename_map["asset"] = "unique_id"
        elif "symbol" in cols:
            rename_map["symbol"] = "unique_id"
    if "factor_tag" not in cols and "factor" in cols:
        rename_map["factor"] = "factor_tag"
    if "value" not in cols and "factor_value" in cols:
        rename_map["factor_value"] = "value"

    normalized = frame.rename(columns=rename_map).copy()
    required = {"time", "unique_id", "factor_tag", "value"}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(f"Factor table missing required columns: {sorted(missing)}")

    normalized["time"] = pd.to_datetime(normalized["time"])
    normalized["unique_id"] = normalized["unique_id"].astype(str)
    normalized["factor_tag"] = normalized["factor_tag"].astype(str)
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    return normalized[["time", "unique_id", "factor_tag", "value"]]


def compute_rankic_table(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    assets: Iterable[str] | None = None,
    min_obs: int = DEFAULT_MIN_OBS,
    start_date: str | None = None,
    end_date: str | None = None,
    include_ic: bool = False,
    include_icir: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute per-asset RankIC for a factor long table."""

    if min_obs <= 0:
        raise ValueError(f"min_obs must be positive, got {min_obs}.")

    factor_data = normalize_factor_long_table(factor_df)
    if start_date:
        factor_data = factor_data[factor_data["time"] >= pd.to_datetime(start_date)]
    if end_date:
        factor_data = factor_data[factor_data["time"] <= pd.to_datetime(end_date)]

    if assets is None:
        candidate_assets = sorted(set(factor_data["unique_id"]) & set(price_df.columns.astype(str)))
    else:
        candidate_assets = [str(item) for item in assets]

    if not candidate_assets:
        raise ValueError("No assets available for RIC calculation.")

    missing_assets = [asset for asset in candidate_assets if asset not in price_df.columns]
    if missing_assets:
        raise ValueError(f"Assets missing from price data: {missing_assets}")

    returns_df = price_df[candidate_assets].pct_change(1).shift(-1).dropna()
    calculator = EfficientCalculator()
    need_ic = bool(include_ic or include_icir)

    records: list[dict[str, Any]] = []
    grouped = factor_data.groupby(["unique_id", "factor_tag"], sort=False)
    grouped_iter = grouped
    if show_progress and tqdm is not None:
        grouped_iter = tqdm(
            grouped,
            total=grouped.ngroups,
            desc="计算RIC、IC、ICIR",
            dynamic_ncols=True,
        )
    for (asset_id, factor_tag), group in grouped_iter:
        if asset_id not in returns_df.columns:
            continue
        factor_series = group.set_index("time")["value"].sort_index()
        returns_series = returns_df[asset_id]

        common_dates = factor_series.index.intersection(returns_series.index)
        if len(common_dates) < min_obs:
            continue

        factor_aligned = factor_series.loc[common_dates]
        returns_aligned = returns_series.loc[common_dates]

        valid_mask = ~(factor_aligned.isna() | returns_aligned.isna())
        factor_clean = factor_aligned[valid_mask]
        returns_clean = returns_aligned[valid_mask]
        if len(factor_clean) < min_obs:
            continue
        if factor_clean.nunique() <= 1 or returns_clean.nunique() <= 1:
            continue

        ric = calculator.efficent_cal_ric(factor_clean.values, returns_clean.values)
        if pd.isna(ric):
            continue

        record: dict[str, Any] = {
            "unique_id": asset_id,
            "factor_tag": factor_tag,
            "ric": float(ric),
            "sample_count": int(len(factor_clean)),
            "start_date": factor_clean.index.min(),
            "end_date": factor_clean.index.max(),
        }

        if need_ic:
            ic = calculator.efficient_cal_ic(factor_clean.values, returns_clean.values)
            record["ic"] = ic
            if include_icir:
                ic_std = calculator.efficient_cal_ic_std(factor_clean.values, returns_clean.values)
                record["icir"] = ic / ic_std if ic_std and not pd.isna(ic_std) else pd.NA

        records.append(record)

    base_cols = ["unique_id", "asset", "factor_tag", "ric", "sample_count", "start_date", "end_date", "abs_ric"]
    extra_cols: list[str] = []
    if need_ic:
        extra_cols.extend(["ic", "abs_ic"])
    if include_icir:
        extra_cols.extend(["icir", "abs_icir"])

    if not records:
        return pd.DataFrame(columns=base_cols + extra_cols)

    ric_df = pd.DataFrame(records)
    ric_df["asset"] = ric_df["unique_id"]
    ric_df["abs_ric"] = ric_df["ric"].abs()
    if need_ic and "ic" in ric_df.columns:
        ric_df["abs_ic"] = pd.to_numeric(ric_df["ic"], errors="coerce").abs()
    if include_icir and "icir" in ric_df.columns:
        ric_df["abs_icir"] = pd.to_numeric(ric_df["icir"], errors="coerce").abs()

    order = [
        "unique_id",
        "asset",
        "factor_tag",
        "ric",
        "sample_count",
        "start_date",
        "end_date",
        "abs_ric",
    ]
    if need_ic:
        order.extend(["ic", "abs_ic"])
    if include_icir:
        order.extend(["icir", "abs_icir"])

    ric_df = ric_df[order].sort_values("abs_ric", ascending=False).reset_index(drop=True)
    return ric_df


def compute_rankic_from_files(
    factor_path: str | Path,
    price_path: str | Path,
    *,
    output_path: str | Path | None = None,
    assets: Iterable[str] | None = None,
    min_obs: int = DEFAULT_MIN_OBS,
    start_date: str | None = None,
    end_date: str | None = None,
    include_ic: bool = False,
    include_icir: bool = False,
    config_path: str | Path | None = None,
    calendar_anchor_symbol: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Load factor/price files and compute per-asset RankIC table."""

    started = time.perf_counter()
    factor_df = load_factor_long_table(factor_path)
    factor_count = int(factor_df["factor_tag"].nunique()) if not factor_df.empty else 0
    factor_assets = int(factor_df["unique_id"].nunique()) if not factor_df.empty else 0
    if factor_df.empty:
        factor_start = "N/A"
        factor_end = "N/A"
    else:
        factor_time = pd.to_datetime(factor_df["time"], errors="coerce")
        factor_time = factor_time.dropna()
        if factor_time.empty:
            factor_start = "N/A"
            factor_end = "N/A"
        else:
            factor_start = factor_time.min().date().isoformat()
            factor_end = factor_time.max().date().isoformat()
    requested_assets = [str(a) for a in assets] if assets is not None else []
    requested_assets_str = ", ".join(requested_assets) if requested_assets else "AUTO"
    print(
        f"[RIC] 输入因子: {factor_path} | factors={factor_count} | factor_assets={factor_assets} | "
        f"time={factor_start}->{factor_end}"
    )
    print(
        f"[RIC] 参数: assets={requested_assets_str} | min_obs={min_obs} | "
        f"window={start_date or 'unbounded'}->{end_date or 'unbounded'} | "
        f"include_ic={bool(include_ic)} | include_icir={bool(include_icir)}"
    )
    _, price_df, _, _ = prepare_price_data(
        data_path=str(price_path),
        calendar_anchor_symbol=calendar_anchor_symbol,
        config_path=config_path,
    )
    price_df = price_df.ffill().bfill()

    ric_df = compute_rankic_table(
        factor_df,
        price_df,
        assets=assets,
        min_obs=min_obs,
        start_date=start_date,
        end_date=end_date,
        include_ic=include_ic,
        include_icir=include_icir,
        show_progress=show_progress,
    )

    if output_path is not None:
        out = Path(output_path)
        ensure_dir(str(out.parent))
        ric_df.to_csv(out, index=False)
    elapsed = time.perf_counter() - started
    passed_rows = len(ric_df)
    passed_factors = int(ric_df["factor_tag"].nunique()) if not ric_df.empty else 0
    out_label = str(output_path) if output_path is not None else "<memory>"
    print(
        f"[RIC] 完成: rows={passed_rows} | factors={passed_factors} | output={out_label} | elapsed={elapsed:.2f}s"
    )

    return ric_df


__all__ = [
    "DEFAULT_MIN_OBS",
    "resolve_ric_params",
    "load_factor_long_table",
    "normalize_factor_long_table",
    "compute_rankic_table",
    "compute_rankic_from_files",
]
