from __future__ import annotations

import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from utils.backtest_utils import prepare_price_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = PROJECT_ROOT / "fama" / "config" / "defaults.yaml"


def _load_defaults(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path).expanduser() if config_path else DEFAULTS_PATH
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _resolve_project_path(path_like: str | Path, project_root: Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _resolve_default_factor_dir(mode: str, defaults: dict[str, Any], project_root: Path) -> Path:
    paths = defaults.get("paths", {}) if isinstance(defaults, dict) else {}
    if mode == "llm":
        raw = paths.get("llm_factor_parquet", "./scripts/LLM_factors/dsl_LLM_factors_new.parquet")
    elif mode == "base":
        raw = paths.get("factor_base_parquet", "./factor_value_prepared/data/factors/dsl_factors_new.parquet")
    else:
        raise ValueError(f"Unsupported mode/profile: {mode}. Expect one of: base, llm.")
    return _resolve_project_path(raw, project_root).parent


def _resolve_benchmark_symbol(price_df: pd.DataFrame, benchmark_symbol: str) -> str:
    if benchmark_symbol in price_df.columns:
        return benchmark_symbol

    for fallback in ("000852.SH", "000300.SH"):
        if fallback in price_df.columns:
            warnings.warn(
                f"benchmark_symbol={benchmark_symbol} 不在 price_df 中，自动回退到 {fallback}。",
                RuntimeWarning,
                stacklevel=2,
            )
            return fallback

    candidate_cols = [col for col in price_df.columns if col != "empty"]
    if not candidate_cols:
        raise ValueError("price_df 无可用列，无法计算 benchmark return。")
    fallback = candidate_cols[0]
    warnings.warn(
        f"benchmark_symbol={benchmark_symbol} 不在 price_df 中，自动回退到 {fallback}。",
        RuntimeWarning,
        stacklevel=2,
    )
    return fallback


class FactorCollection:
    def __init__(
        self,
        *,
        mode: str = "base",
        profile: str | None = None,
        factor_dir: str | Path | None = None,
        benchmark_symbol: str | None = None,
        factor_config_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ):
        self.base_dir = PROJECT_ROOT
        self.config_path = _resolve_project_path(config_path, self.base_dir) if config_path else DEFAULTS_PATH
        self.defaults = _load_defaults(self.config_path)

        resolved_mode = (profile or mode or "base").lower()
        if resolved_mode not in {"base", "llm"}:
            raise ValueError(f"Unsupported mode/profile: {resolved_mode}. Expect one of: base, llm.")
        self.mode = resolved_mode

        paths_cfg = self.defaults.get("paths", {}) if isinstance(self.defaults, dict) else {}
        market_data_raw = paths_cfg.get("market_data", "./data/fof_price_updating.parquet")
        market_data_path = _resolve_project_path(market_data_raw, self.base_dir)
        self.native_price, self.price_df, self.open_price_df, self.working_days = prepare_price_data(
            data_path=str(market_data_path),
            config_path=self.config_path,
        )

        self.max_date = self.native_price["time"].max()
        self.min_date = self.native_price["time"].min()

        default_factor_dir = _resolve_default_factor_dir(self.mode, self.defaults, self.base_dir)
        self.factor_dir = _resolve_project_path(factor_dir, self.base_dir) if factor_dir is not None else default_factor_dir
        self.factor_dir.mkdir(parents=True, exist_ok=True)

        default_factor_cfg = self.base_dir / "utils" / "FactorConfig.yaml"
        self.factor_config_path = (
            _resolve_project_path(factor_config_path, self.base_dir) if factor_config_path is not None else default_factor_cfg
        )
        with open(self.factor_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.available_unique_ids = self.config.get("TargetAssetsIds", self.native_price["unique_id"].unique().tolist())
        self.benchmark_symbol = benchmark_symbol or ("000852.SH" if self.mode == "llm" else "000300.SH")

    def update_all(self):
        """
        更新所有因子
        """
        self.update_talib()
        self.update_alphas()

    def check_newest(self, file_path):
        """
        检查存储的因子是否是最新的
        :param file_path: 存储因子的文件路径
        :return: bool, 是否是最新的因子
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False  # 如果文件不存在，则表明不是最新的因子
        # 读取存储的因子数据
        factor_df = self.read_factor_file(file_path)
        # 检查因子数据是否为空
        if factor_df.empty:
            return False
        # 检查因子是否更新到最新
        if factor_df["time"].max() < self.max_date:
            return False
        # 检查因子中包含的标的是否是最新的, available_unique_ids 中的每个都需要在factor_df中存在
        factor_df_unique_ids = factor_df["unique_id"].unique()
        if not all([unique_id in factor_df_unique_ids for unique_id in self.available_unique_ids]):
            return False
        return True

    def update_alphas(self):
        """
        这里会每次都将所有因子重新计算一遍，如果后期标的数量增多，应该迁移为增量更新，包括时间增量和标的增量两方面
        """
        calculating_modules = ["alpha101"]
        # 先读取已经存储的因子，确定是否需要更新。如果不存在文件或者因子最大日期小于当前最大日期，则需要更新
        newest_flat = [
            self.check_newest(self.factor_dir / f"{module_name}.parquet") for module_name in calculating_modules
        ]
        if all(newest_flat):
            return
        print("开始更新因子...")
        from alphatest.ExtFunction import get_run_lib, run_lib_module, parse_out_to_factor_df, collect_input_dict

        native_price, working_days = deepcopy(self.native_price), deepcopy(self.working_days)
        # if self.available_unique_ids:
        #     native_price = native_price[native_price["unique_id"].isin(self.available_unique_ids)]
        native_price = native_price.rename(columns={"time": "date"})

        lib = get_run_lib(calculating_modules, libname="fof_nmw_factor")

        print("开始加载原始数据...")
        input_dict, other_info = collect_input_dict(native_price, working_days)
        num_time, num_stocks = other_info["num_time"], other_info["num_stocks"]

        # for k in ["open", "high", "low", "close", "volume", "amount"]:
        #     if k in input_dict:
        #         # 转为 float64 且保证是 C 连续内存
        #         input_dict[k] = np.ascontiguousarray(input_dict[k], dtype=np.float64)
        for k in ["open", "high", "low", "close", "volume", "amount"]:
            if k in input_dict:
                # 保持 float32，但用 ascontiguousarray 保证内存布局符合 C 扩展的要求
                input_dict[k] = np.ascontiguousarray(input_dict[k], dtype=np.float32)
        print("数据加载完毕, 开始因子计算...")
        print("-----------------------------------")

        for module_name in tqdm(calculating_modules):
            out = run_lib_module(input_dict, lib, module_name, num_time, num_stocks, multi_thread_num=4)
            factor_df = parse_out_to_factor_df(out, other_info, working_days=working_days).dropna()
            factor_df.to_parquet(os.path.join(self.factor_dir, f"{module_name}.parquet"))
            print(f"finish {module_name}")

    def read_factor_file(self, file_path):
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        # 读取存储的因子数据
        if file_path.suffix == ".parquet":
            factor_df = pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            factor_df = pd.read_csv(file_path, parse_dates=["time"])
        else:
            raise ValueError(f"不支持的文件格式{file_path}")
        return factor_df

    def update_talib(self):
        """
        talib 的因子更新会增量更新，包括时间增量更新与标的增量更新。如果要完全重新计算，则需要删除原有的因子文件
        """
        save_file_path = self.factor_dir / "talib.parquet"
        if self.check_newest(save_file_path):
            return
        print("开始更新talib因子...")
        # 假设如果存储到文件中的因子，日期截至日期必然是相同的
        # 故对于非最新日期的标的，只需要计算后期的因子即可
        # 对缺失标的，则需要对所有日期都进行计算

        default_cal_start = None

        to_update_ids = []
        to_calculate_new_ids = []

        # 读取存储的因子数据
        factor_df = self.read_factor_file(save_file_path)

        if factor_df is None:
            to_calculate_new_ids = self.available_unique_ids
        else:
            to_update_ids = factor_df["unique_id"].unique().tolist()
            to_calculate_new_ids = list(set(self.available_unique_ids) - set(to_update_ids))

            default_cal_start = factor_df["time"].min()  # 如果已经有存储的因子，默认计算的开始时间为存储的因子最小日期

        all_results = []
        # 计算新标的因子
        if to_calculate_new_ids:
            print(
                f"检测到新标的, 数量{len(to_calculate_new_ids)}，开始计算: {to_calculate_new_ids}, 从{'标的起始日' if default_cal_start is None else default_cal_start}开始"
            )
        all_results += calculate_talib(
            default_cal_start,
            to_calculate_new_ids,
            self.native_price,
            self.price_df,
            benchmark_symbol=self.benchmark_symbol,
        )
        # 计算已有标的因子
        cal_start = factor_df["time"].max() if factor_df is not None else default_cal_start
        if to_update_ids:
            print(f"检测到已有标的, 数量{len(to_update_ids)}，开始计算: {to_update_ids}, 从{'标的起始日' if cal_start is None else cal_start}开始")
        all_results += calculate_talib(
            cal_start,
            to_update_ids,
            self.native_price,
            self.price_df,
            benchmark_symbol=self.benchmark_symbol,
        )

        # 过滤计算结果中的空值以及空表
        all_results = [result.dropna() for result in all_results if result is not None]
        all_results = [result for result in all_results if not result.empty]

        # 拼接数据
        if factor_df is not None:
            all_results = [factor_df] + all_results
        if all_results:
            factor_df = pd.concat(all_results, ignore_index=True)
            factor_df = factor_df.sort_values(["unique_id", "time"])
        # 检查factor_df有效
        assert factor_df is not None and not factor_df.empty, "因子计算失败，请检查数据或计算逻辑"
        factor_df = factor_df[["time", "factor", "unique_id", "value"]].drop_duplicates()
        # 检查因子数据是否重复, 这里的重复是指同一时间同一标的同一因子有多个值
        assert len(factor_df) == len(
            factor_df.drop_duplicates(subset=["time", "factor", "unique_id"])
        ), "因子数据重复，请检查数据"
        # 保存因子数据
        factor_df = factor_df.reset_index(drop=True)
        factor_df.to_parquet(save_file_path, index=False)
        print("talib因子更新完成")

    def load_factor_df(self, factor_set_list=None):
        """
        加载因子数据
        :param factor_set_list: 因子集合列表，默认加载所有因子集合
        :return: 因子数据
        """
        if factor_set_list is None:
            factor_set_list = ["talib", "alpha158", "alpha101", "alpha360"]
        all_factor_df = []
        for factor_set in factor_set_list:
            factor_df = self.read_factor_file(self.factor_dir / f"{factor_set}.parquet")
            if factor_df is not None:
                factor_df['factor_set'] = factor_set  # 添加因子集合标识
                all_factor_df.append(factor_df)
            else:
                print(f"因子集合{factor_set}未加载成功，请先运行因子更新")
        if all_factor_df:
            return pd.concat(all_factor_df, ignore_index=True)
        else:
            print("没有找到因子数据，请先运行因子更新")
            return pd.DataFrame()


def calculate_talib(cal_start, to_calculate_ids, native_price, price_df, benchmark_symbol="000300.SH"):
    from talib import abstract
    import talib
    import numpy as np
    from arch import arch_model
    cal_start = pd.Timestamp(cal_start) if cal_start is not None else None

    exclude = {"MAVP"}
    work_period = 20
    all_results = []
    for code in to_calculate_ids:
        part_df = native_price[native_price["unique_id"] == code].set_index("time")
        indicator_results = {}
        for indicator in talib.get_functions():
            if indicator in exclude:
                continue
            try:
                result = abstract.Function(indicator, part_df, timeperiod=work_period)
                if len(result.output_names) == 1:
                    indicator_results[indicator] = result.outputs
                else:
                    for out_name in result.output_names:
                        indicator_results[f"{indicator}_{out_name}"] = result.outputs[out_name]
            except Exception as e:
                print(f"Error calculating {indicator}: {e}")

        indicator_df = (
            pd.DataFrame(indicator_results, index=part_df.index)
            .reset_index()
            .melt(id_vars="time", var_name="factor", value_name="value")
            .dropna()
            .sort_values("time")
        )
        indicator_df["unique_id"] = code

        # 过滤时间
        if cal_start is not None:
            indicator_df = indicator_df[indicator_df["time"] > cal_start]

        all_results.append(indicator_df)

        yx_factors = {}
        yx_factors["cum_return_20d"] = part_df["close"].pct_change(20)
        yx_factors["cum_return_6m"] = part_df["close"].pct_change(180)

        bench_col = _resolve_benchmark_symbol(price_df, benchmark_symbol)
        _bench_return = price_df[bench_col].pct_change()
        _excess_return = part_df["close"].pct_change().sub(_bench_return)
        rolling_excess_mean_ = _excess_return.rolling(window=180, min_periods=2).mean()
        rolling_excess_std_ = _excess_return.rolling(window=180, min_periods=2).std()
        yx_factors["mid_info_ratio"] = rolling_excess_mean_ / rolling_excess_std_
        risk_free_ret = part_df["close"].pct_change()
        rolling_rf_mean_ = risk_free_ret.rolling(window=180, min_periods=2).mean()
        rolling_rf_std_ = risk_free_ret.rolling(window=180, min_periods=2).std()
        yx_factors["mid_sharpe"] = rolling_rf_mean_ / rolling_rf_std_ * np.sqrt(252)

        data = part_df["close"].pct_change().dropna() * 100
        garch_means = {}
        garch_stds = {}
        for arch_start in part_df.index[10:]:
            try:
                model = arch_model(
                    data[data.index <= arch_start], vol="GARCH", p=1, q=1, lags=1, mean="AR", rescale=False
                )
                model_fitted = model.fit(disp="off", show_warning=False, last_obs=arch_start)
                res = model_fitted.forecast(start=arch_start, method="analytic", horizon=20)
                garch_std = np.sqrt(res.variance.mean(axis=1))
                garch_mean = res.mean.mean(axis=1)
                garch_means[arch_start] = garch_mean.item()
                garch_stds[arch_start] = garch_std.item()
            except Exception as e:
                print(f"Error calculating GARCH for {code} at {arch_start}: {e}")
                garch_means[arch_start] = np.nan
                garch_stds[arch_start] = np.nan
        yx_factors["garch_mean"] = pd.Series(garch_means)
        yx_factors["garch_std"] = pd.Series(garch_stds)

        yx_df = pd.DataFrame(yx_factors)
        if cal_start is not None:
            yx_df = yx_df[yx_df.index > cal_start]
        if not yx_df.empty:
            yx_df = (
                yx_df.reset_index()
                .melt(id_vars="index", var_name="factor", value_name="value")
                .dropna()
                .sort_values("index")
                .rename(columns={"index": "time"})
            )
            yx_df["unique_id"] = code
            all_results.append(yx_df)
    return all_results
