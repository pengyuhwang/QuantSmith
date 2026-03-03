import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from copy import deepcopy
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "fama" / "config" / "defaults.yaml"


def _load_calendar_anchor_symbol_from_config(config_path=None):
    """Load calendar anchor symbol from config.

    Priority:
    1) top-level assets first symbol (project-wide unified asset setting)
    2) backtest.calendar_anchor_symbol (legacy fallback)
    3) ric.assets first symbol (legacy fallback for old configs)
    """
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    assets = payload.get("assets") if isinstance(payload, dict) else None
    if assets is None and isinstance(payload, dict):
        assets = payload.get("asset")
    if isinstance(assets, (list, tuple)):
        for item in assets:
            text = str(item).strip()
            if text:
                return text
    elif isinstance(assets, str) and assets.strip():
        return assets.strip()

    backtest_cfg = payload.get("backtest") if isinstance(payload, dict) else None
    if isinstance(backtest_cfg, dict):
        symbol = backtest_cfg.get("calendar_anchor_symbol")
        if isinstance(symbol, str) and symbol.strip():
            return symbol.strip()
    # Legacy fallback: old config layout where assets lived under ric.assets
    ric_cfg = payload.get("ric") if isinstance(payload, dict) else None
    if isinstance(ric_cfg, dict):
        assets = ric_cfg.get("assets")
        if isinstance(assets, (list, tuple)):
            for item in assets:
                text = str(item).strip()
                if text:
                    return text
        elif isinstance(assets, str) and assets.strip():
            return assets.strip()
    return None


def _resolve_calendar_anchor_symbol(native_price, calendar_anchor_symbol=None, config_path=None):
    """Resolve a valid anchor symbol for trading calendar extraction."""
    symbol = calendar_anchor_symbol or _load_calendar_anchor_symbol_from_config(config_path=config_path) or "000852.SH"
    symbols = native_price["unique_id"].dropna().astype(str)
    available = set(symbols.unique().tolist())
    if symbol in available:
        return symbol

    for fallback in ("000852.SH", "000300.SH"):
        if fallback in available:
            warnings.warn(
                f"calendar_anchor_symbol={symbol} 不在行情数据中，自动回退到 {fallback}。",
                RuntimeWarning,
                stacklevel=2,
            )
            return fallback

    if not available:
        raise ValueError("行情数据缺少 unique_id，无法构建 working_days。")

    # Pick the most complete symbol as fallback calendar anchor.
    fallback = (
        native_price.groupby("unique_id")["time"]
        .nunique()
        .sort_values(ascending=False)
        .index[0]
    )
    warnings.warn(
        f"calendar_anchor_symbol={symbol} 不在行情数据中，自动回退到 {fallback}。",
        RuntimeWarning,
        stacklevel=2,
    )
    return fallback


class NMWBacktester:
    def __init__(self, price_df, benchmark=None, open_price=None, close_price=None, fee=0.0):
        self.price_df = price_df.copy()
        if "empty" not in self.price_df.columns:
            self.price_df.insert(0, "empty", 1.0)

        self.open_price = None
        if open_price is not None:
            self.open_price = open_price.copy()
            if "empty" not in self.open_price.columns:
                self.open_price.insert(0, "empty", 1.0)

        self.close_price = None
        if close_price is not None:
            self.close_price = close_price.copy()
            if "empty" not in self.close_price.columns:
                self.close_price.insert(0, "empty", 1.0)

        self.benchmark = benchmark

        self.asset_kind = {
            "empty": "货基",
            "159934.SZ": "黄金",
            "159532.SZ": "中证2000",
            "563080.SH": "中证A50",
            "930050.CSI": "中证A50",
            "932000.CSI": "中证2000",
            "Au9999.SGE": "黄金",
            "五年国债": "五年国债",
        }
        self.kind_color = {
            "货基": 3,
            "黄金": 1,
            "中证2000": 9,
            "中证A50": 4,
            "五年国债": 0,
        }

        self.fee = fee

    def backtest(
            self,
            target,
            start_date=None,
            end_date=None,
            D=10,
            fillna_asset="empty",
            async_mode=False,
            adjust_by_open_price=False,
            open_price=None,
            close_price=None,
            skip_empty_warehouse=False,
    ):
        """
        async_mode: bool, 是否使用异步模式,只有D>0的时候这个参数才生效。会将资金分为D+1份,每天调仓一份. D=0时只有一份资金, 所以不支持异步模式。

        adjust_by_open_price: bool, 是否使用开盘价调仓。如果设置的话要么调用的时候传入 open_price 和 close_price, 要么在初始化的时候传入 open_price 和 close_price。
        """
        if async_mode and D < 1:
            warnings.warn("D<1时, async_mode参数无效, 已经自动设置为同步调仓")
            async_mode = False

        target = deepcopy(target)

        open_price = self.open_price if open_price is None else open_price
        close_price = self.close_price if close_price is None else close_price

        if adjust_by_open_price and (open_price is None or close_price is None):
            raise ValueError(
                "adjust_by_open_price为True时, open_price和close_price必须提供, 可以在初始化或者调用时传入"
            )

        if async_mode:
            # 先采样信号，异步调仓相当于信号的执行有惯性锁仓，但执行的还是D=0的信号。
            to_merge_pos = []
            target = clip_time_range(target, start_date, end_date)
            for i in range(D + 1):  # 每天调仓一份(merge_position时会自动平均合并)
                resample_target = target.iloc[i:: D + 1].reindex(target.index).ffill().fillna(fillna_asset)
                _pos = backtest_nmw(
                    resample_target,
                    self.price_df,
                    start_date,
                    end_date,
                    D=0,
                    fillna_asset=fillna_asset,
                    skip_empty_warehouse=skip_empty_warehouse,
                    fee=self.fee,
                )
                if adjust_by_open_price:  # 如果使用开盘价调仓
                    _pos = adjust_position_by_open_price(_pos, open_price, close_price)
                to_merge_pos.append(_pos)
            pos = merge_position(to_merge_pos)
            pos["target"] = target.reindex(pos.index)
        else:
            pos = backtest_nmw(
                target,
                self.price_df,
                start_date,
                end_date,
                D,
                fillna_asset=fillna_asset,
                skip_empty_warehouse=skip_empty_warehouse,
                fee=self.fee,
            )
            if adjust_by_open_price:  # 如果使用开盘价调仓
                pos = adjust_position_by_open_price(pos, open_price, close_price)

        return pos

    def multi_weight_backtest(
            self,
            target,
            start_date=None,
            end_date=None,
            D=0,  # not implemented yet
            fillna_asset="empty",
            async_mode=False,
            adjust_by_open_price=False,
            open_price=None,
            close_price=None,
    ):
        """
        target: index为日期, 列为资产, 值为权重的DataFrame (输入时应该保证和为1, 不足1的部分会用empty来补充)
        """
        if D != 0:
            raise NotImplementedError("多权重调仓暂时不支持D>0的情况")
        if adjust_by_open_price:
            raise NotImplementedError("多权重调仓暂时不支持开盘价调仓")

        target = deepcopy(target)

        assert type(target) == pd.DataFrame, "target必须是DataFrame类型"

        target = target.fillna(0)  # 填充空值为0
        if "empty" not in target.columns:
            target.insert(0, "empty", 1.0 - target.sum(axis=1))

        # 保证权重和为1
        target = target.div(target.sum(axis=1), axis=0)  # 按行归一化

        open_price = self.open_price if open_price is None else open_price
        close_price = self.close_price if close_price is None else close_price

        if adjust_by_open_price and (open_price is None or close_price is None):
            raise ValueError(
                "adjust_by_open_price为True时, open_price和close_price必须提供, 可以在初始化或者调用时传入"
            )

        price_df = self.price_df.copy()

        # 确保索引类型为 datetime
        target.index = pd.to_datetime(target.index)
        price_df.index = pd.to_datetime(price_df.index)

        # 确保初始日期有信号
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
            start_date = max(start_date, target.index[0])
        else:
            start_date = target.index[0]

        if end_date is not None:
            end_date = pd.Timestamp(end_date)
            end_date = min(end_date, target.index[-1])
        else:
            end_date = target.index[-1]

        # 准备交易日列表
        trading_days = price_df.index.tolist()

        # 准备日收益率
        daily_return_df = price_df.pct_change()

        # 获取处理日期列表
        processing_days = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)].index

        # 对齐调仓信号
        target = target.shift(D + 1).reindex(processing_days).ffill().fillna(0)
        # 对和为0的行，将对应的empty的权重设置为1
        target.loc[target.sum(axis=1) == 0, "empty"] = 1.0

        daily_return_df = daily_return_df.reindex(processing_days).ffill().fillna(0)

        weighted_daily_return_df = daily_return_df.mul(target, axis=0)  # 按行乘以权重

        combined_daily_return = weighted_daily_return_df.sum(axis=1)  # 按行求和

        # 获取持仓资产, 权重大于0的资产拼成一个元组， 如果只有一个资产，就直接显示这个字符串
        holding_asset = target.apply(
            lambda x: tuple(x[x > 0].index) if len(x[x > 0]) > 1 else x[x > 0].index[0], axis=1
        )

        position_adjust = pd.DataFrame(
            {
                "holding_asset": holding_asset,
                "daily_return": combined_daily_return,
                "nv": (1 + combined_daily_return).cumprod(),
            },
            index=processing_days,
        )

        return position_adjust

    def plot_pnl(
            self,
            position_adjust,
            title="净值曲线",
            start_date=None,
            end_date=None,
            draw_benchmark=False,
    ):
        if draw_benchmark and self.benchmark is None:
            raise ValueError("benchmark is not provided.")
        position_adjust = position_adjust.copy()
        if start_date is not None:
            position_adjust = position_adjust[position_adjust.index >= start_date]
        if end_date is not None:
            position_adjust = position_adjust[position_adjust.index <= end_date]
        plot_pnl(position_adjust, title, self.benchmark if draw_benchmark else None)

    def plot_detail_pnl(
            self,
            position_adjust,
            title="调仓详情",
            start_date=None,
            end_date=None,
            draw_benchmark=False,
            show=True,
            figsize=(12, 6),
    ):
        if draw_benchmark and self.benchmark is None:
            raise ValueError("benchmark is not provided.")
        position_adjust = position_adjust.copy()
        if start_date is not None:
            position_adjust = position_adjust[position_adjust.index >= start_date]
        if end_date is not None:
            position_adjust = position_adjust[position_adjust.index <= end_date]
        # 获取颜色映射
        cmap = plt.get_cmap("tab10")  # 'tab10' 是一个有10种不同颜色的colormap

        # 创建一个新的图形
        fig, ax = plt.subplots(figsize=figsize)

        def fill_prev(series, source):
            series = series.copy()
            assert len(series) == len(source)
            for i in range(1, len(series)):
                if np.isnan(series.iloc[i - 1]) and not np.isnan(series.iloc[i]):
                    series.iloc[i - 1] = source.iloc[i - 1]
            return series

        # 绘制 base_df 中的净值数据
        available_assets = position_adjust["holding_asset"].unique()
        available_asset_kind = {asset: self.asset_kind[asset] for asset in available_assets}
        for code, kind in available_asset_kind.items():
            _ = position_adjust.apply(lambda x: x.nv if x.holding_asset == code else np.nan, axis=1)
            if code != "empty":
                _ = fill_prev(_, position_adjust["nv"])
            ax.plot(_, label=kind, color=cmap(self.kind_color[kind]), linewidth=2)

        if draw_benchmark:
            benchmark_nv = self.benchmark.reindex(position_adjust.index).ffill()
            benchmark_nv = benchmark_nv.div(benchmark_nv.iloc[0])
            ax.plot(benchmark_nv, label="基准净值", color=cmap(5), linewidth=2)

        # 添加标题和标签
        ax.set_title(title)
        ax.set_ylabel("净值")
        ax.set_xlabel("日期")

        # 显示图例
        ax.legend(loc="upper left")

        # 显示图形
        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    @staticmethod
    def calculate_metrics(
            position_adjust,
            annual_trading_days=252,
            return_as_df=False,
            start_date=None,
            end_date=None,
    ):
        position_adjust = position_adjust.copy()
        if start_date is not None:
            position_adjust = position_adjust[position_adjust.index >= start_date]
        if end_date is not None:
            position_adjust = position_adjust[position_adjust.index <= end_date]
        return calculate_metrics(position_adjust, annual_trading_days, return_as_df)

    def generate_nmw_sig(self, assets, S=20):
        process_df = self.price_df[assets].copy()
        momentum_sig = process_df.pct_change(S).dropna()
        result_sig = momentum_sig.apply(lambda x: x.argmax(), axis=1)
        result_sig = result_sig.map({i: name for i, name in enumerate(assets)})
        return result_sig


def get_trading_days_diff(date1, date2, trading_days):
    if date1 >= date2:
        return 0
    return len([d for d in trading_days if date1 < d <= date2])


# 起始日期默认与牛魔王保持一致
def backtest_nmw(
        target, price_df, start_date=None, end_date=None, D=10, fillna_asset="empty", skip_empty_warehouse=False,
        fee=0.0
):
    # 确保索引类型为 datetime
    target.index = pd.to_datetime(target.index)
    price_df.index = pd.to_datetime(price_df.index)

    # 确保初始日期有信号
    if start_date is not None:
        start_date = max(start_date, target.index[0])
    else:
        start_date = target.index[0]

    if end_date is not None:
        end_date = min(end_date, target.index[-1])
    else:
        end_date = target.index[-1]

    # 准备交易日列表
    trading_days = price_df.index.tolist()

    # 准备日收益率
    daily_return_df = price_df.pct_change()

    # 对齐调仓信号
    target = target.reindex(daily_return_df.index).ffill().fillna(fillna_asset)

    # 获取处理日期列表
    processing_days = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)].index

    # 生成调仓单
    position_adjust = pd.DataFrame(
        np.nan,
        index=processing_days,
        columns=["target", "holding_asset", "daily_return", "nv"],
    )
    position_adjust["target"] = position_adjust["target"].astype(str)
    position_adjust["holding_asset"] = position_adjust["holding_asset"].astype(str)
    # 初始化首次调仓
    last_adjust_date = start_date
    position_adjust.loc[start_date, "target"] = target[
        start_date
    ]  # target 是收盘时发出的信号，holding asset 是真正产生收益的持仓资产，相差两天
    # 遍历数据行
    for i, current_date in enumerate(processing_days):
        # 跳过回测开始日期前的数据
        if current_date <= start_date:
            continue

        # 如果距离上次调仓日期小于等于 D 天，保持前一天的基金
        trading_days_diff = get_trading_days_diff(last_adjust_date, current_date, trading_days)

        if (skip_empty_warehouse and target.loc[processing_days[i - 1]] == fillna_asset) or trading_days_diff > D:
            # 调仓逻辑，根据目标基金计算
            target_asset = target[processing_days[i]]  # i - 2 天前生成了信号，i - 1 天提交订单, 第 i 天正式持仓

            # 如果目标基金发生变化，更新调仓日期
            if target_asset != position_adjust.loc[processing_days[i - 1], "target"]:  # 如果目标基金和当前持有的不同
                last_adjust_date = current_date  # 记录改变持仓日
        else:
            target_asset = position_adjust.loc[processing_days[i - 1], "target"]

        # if trading_days_diff <= D:
        #     target_asset = position_adjust.loc[processing_days[i - 1], "target"]
        # else:
        #     # 调仓逻辑，根据目标基金计算
        #     target_asset = target[processing_days[i]]  # i - 2 天前生成了信号，i - 1 天提交订单, 第 i 天正式持仓

        #     # 如果目标基金发生变化，更新调仓日期
        #     if target_asset != position_adjust.loc[processing_days[i - 1], "target"]:  # 如果目标基金和当前持有的不同
        #         last_adjust_date = current_date  # 记录改变持仓日

        position_adjust.loc[current_date, "target"] = target_asset
    position_adjust["holding_asset"] = (
        position_adjust["target"].shift(2).fillna(fillna_asset)
    )  # target的后一天操作，再后一天产生收益
    for i, asset in position_adjust["holding_asset"].items():
        if pd.isna(asset):
            continue
        position_adjust.loc[i, "daily_return"] = daily_return_df.loc[i, asset]

    position_adjust["nv"] = (1 + position_adjust["daily_return"]).cumprod()

    if fee > 0:
        position_adjust = minus_position_fee(position_adjust, fee)

    return position_adjust


def minus_position_fee(position_adjust, fee):
    if "detail" not in position_adjust.columns:
        holding_asset_position = position_adjust.copy().reset_index()
        holding_asset_position["ratio"] = 1
        holding_asset_position = holding_asset_position.pivot(
            index="time", columns="holding_asset", values="ratio"
        ).fillna(0)
    else:
        holding_asset_position = position_adjust["detail"].apply(lambda x: pd.Series(x.to_dict())).fillna(0.0)
    hsr = (holding_asset_position - holding_asset_position.shift(1)).abs().fillna(0)
    cost_daily_return = (hsr * fee).sum(1)
    position_adjust = position_adjust.copy()
    position_adjust["daily_return"] = position_adjust["daily_return"] - cost_daily_return
    position_adjust["nv"] = (1 + position_adjust["daily_return"]).cumprod()
    return position_adjust


def calculate_empty_ratio(holding_assets, empty_tag="empty"):
    return {"empty_ratio": sum(holding_assets == empty_tag) / len(holding_assets)}


def clip_time_range(position_adjust, start_date=None, end_date=None):
    position_adjust = position_adjust.copy()
    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        position_adjust = position_adjust[position_adjust.index >= start_date]
    if end_date is not None:
        end_date = pd.Timestamp(end_date)
        position_adjust = position_adjust[position_adjust.index <= end_date]
    return position_adjust


def calculate_turnover_rate(pos):
    # 根据持仓表计算平均年换手率和年化平均月换手率
    # pos中的列有 holding_asset, daily_return, nv
    # holding_asset 与下一期持仓不同的日期标注出来，那些日期发生了换手，其净值*2就是发生的买卖金额
    # 某期的换手率是 发生的 买卖总金额/2 / 平均总资产
    pos = pos.copy()
    if 'hrs' in pos.columns:
        # 如果日换手率列存在于pos中，则直接使用该列计算
        daily_turnover = pos['hrs'].fillna(0)
        monthly_turnover_rate = daily_turnover.resample("ME").sum().mean()
        annual_monthly_turnover_rate = monthly_turnover_rate * 12
        yearly_turnover_rate = daily_turnover.resample("YE").sum().mean()
        return {"年化平均月换手率": annual_monthly_turnover_rate, "平均年换手率": yearly_turnover_rate}

    pos["turnover_day"] = pos["holding_asset"].shift(-1) != pos["holding_asset"]
    # 按月统计每个月turnover_day的净值之和*2
    pos["turnover_money"] = pos["turnover_day"] * pos["nv"]  # 这里不乘2就是单边的，因为这里买卖一定一样
    monthly_turnover = pos.resample("ME").agg({"turnover_money": "sum", "nv": "mean"})
    monthly_turnover["turnover_rate"] = monthly_turnover["turnover_money"] / monthly_turnover["nv"]
    yearly_turnover = pos.resample("YE").agg({"turnover_money": "sum", "nv": "mean"})
    yearly_turnover["turnover_rate"] = yearly_turnover["turnover_money"] / yearly_turnover["nv"]
    # 计算年化平均月换手率
    monthly_turnover_rate = monthly_turnover["turnover_rate"].mean()
    annual_monthly_turnover_rate = monthly_turnover_rate * 12
    # 计算平均年换手率
    yearly_turnover_rate = yearly_turnover["turnover_rate"].mean()
    return {"年化平均月换手率": annual_monthly_turnover_rate, "平均年换手率": yearly_turnover_rate}


def measure_position(
        position_adjust,
        tester,
        start_date=None,
        end_date=None,
        benchmark=None,
        holding_metric=False,
        rebalance_metric=False,
        empty_metric=False,
        empty_tag="empty",
        win_rate_metric=False,
        win_rate_consider_assets=None,
        adverse_excess_consider_assets=None,
        long_empty_mode=False,
        # ic_metric=False,
):
    """
    封装了计算策略表现的函数，包括计算净值、年化收益、年化波动率、夏普比率、最大回撤、Calmar比率等。
    """
    position_adjust = clip_time_range(position_adjust, start_date, end_date)

    metrics = tester.calculate_metrics(position_adjust)
    if benchmark is not None:
        metrics.update(calculate_performance_with_benchmark(position_adjust.nv, benchmark))
    metrics.update(calculate_performance(position_adjust.nv, 3))
    metrics.update(calculate_performance(position_adjust.nv, 5))
    metrics.update(calculate_performance(position_adjust.nv, 10))
    if holding_metric:
        metrics.update(calculate_empty_ratio(position_adjust.holding_asset, empty_tag=empty_tag))
        metrics.update(adjust_metrics(position_adjust))
    if rebalance_metric:
        try:
            _res, _, _ = calculate_rebalance_win_rate(position_adjust, empty_tag=empty_tag, benchmark=benchmark)
        except Exception as e:
            _res, _, _ = calculate_rebalance_win_rate(position_adjust, empty_tag=empty_tag)
        metrics.update(_res)
    if empty_metric:
        metrics.update(measure_empty(position_adjust, benchmark=benchmark, empty_tag=empty_tag))

    try:
        # 新增盈亏比
        metrics.update(calculate_pl_ratio(position_adjust))
    except Exception as e:
        print(f"计算盈亏比时发生错误: {e}")

    try:
        # 新增换手率
        metrics.update(calculate_turnover_rate(position_adjust))
    except Exception as e:
        print(f"计算换手率时发生错误: {e}")

    # 新增资产的资产胜率以及对抗调仓胜率
    if win_rate_metric:
        metrics.update(
            calculate_win_rate_metrics(
                position_adjust,
                tester.price_df,
                win_rate_consider_assets,
                adverse_excess_consider_assets,
                long_empty_mode=long_empty_mode,
            )
        )

    return metrics


def measure_position_excess(
        position_adjust,
        tester=None,
        start_date=None,
        end_date=None,
        benchmark_daily_return=None,
        annual_trading_days=252,
):
    """
    封装了超额收益的评价函数。
    tester 并不需要，仅前向兼容。
    """
    position_adjust = clip_time_range(position_adjust, start_date, end_date)

    daily_returns = position_adjust["nv"].pct_change().dropna()
    excess_returns = daily_returns - benchmark_daily_return.reindex(daily_returns.index).fillna(0)
    excess_return_nv = excess_returns.cumsum().add(1)

    # Calculate cumulative return
    cumulative_return = excess_returns.sum()

    compound_cumulative_return = excess_returns.add(1).prod() - 1

    # Calculate annualized return
    annualized_return = excess_returns.mean() * annual_trading_days

    # Calculate annualized volatility
    annualized_volatility = excess_returns.std() * np.sqrt(annual_trading_days)

    # Calculate annualized Sharpe ratio
    annualized_sharpe = safe_divide(annualized_return, annualized_volatility)

    # Calculate maximum drawdown
    rolling_max = excess_return_nv.cummax()
    drawdown_pct = excess_return_nv / rolling_max - 1
    drawdown_num = excess_return_nv - rolling_max
    max_drawdown_pct = drawdown_pct.min()
    max_drawdown_num = drawdown_num.min()

    metrics = {
        "累积单利超额": cumulative_return,
        "年化单利超额": annualized_return,
        "年化超额波动率": annualized_volatility,
        "年化超额夏普": annualized_sharpe,
        "超额最大回撤占比": max_drawdown_pct,
        "超额最大回撤数值": max_drawdown_num,
        # "累积复利超额": compound_cumulative_return,
    }

    def calculate_simple_excess_win_performance(daily_returns, years=3):
        # 计算起始日期
        end_date = daily_returns.index[-1]  # 最后一个日期
        start_date = end_date - pd.DateOffset(years=years)  # 计算起始日期
        # 筛选出在指定时间范围内的数据
        data = daily_returns[start_date:end_date].infer_objects(copy=False)
        # 定义不同的持有期（1月、3月、6月、1年）
        holding_periods = [1, 3, 6, 12]  # 单位：月, 一个月22个交易日
        performance = {}
        for period in holding_periods:
            returns = data.rolling(period * 22).sum().dropna()
            avg_return = sum(returns) / len(returns) if not returns.empty else 0
            profit_probability = sum(1 for r in returns if r > 0) / len(returns) if not returns.empty else 0
            performance[f"{years}年持{period}月平均超额"] = avg_return
            performance[f"{years}年持{period}月超额胜率"] = profit_probability
        return performance

    metrics.update(calculate_simple_excess_win_performance(excess_returns, 3))
    metrics.update(calculate_simple_excess_win_performance(excess_returns, 5))
    metrics.update(calculate_simple_excess_win_performance(excess_returns, 10))
    return metrics


def measure_position_geo_excess(
        position_adjust,
        tester=None,
        start_date=None,
        end_date=None,
        benchmark_daily_nv=None,
        annual_trading_days=252,
):
    """
    封装了超额收益的评价函数。
    tester 并不需要，仅前向兼容。
    """
    position_adjust = clip_time_range(position_adjust, start_date, end_date)

    if benchmark_daily_nv is None:
        raise ValueError("benchmark_daily_nv 必须提供以计算超额收益。")

    strategy_nv = position_adjust["nv"]
    benchmark_nv = benchmark_daily_nv.reindex(strategy_nv.index).ffill()

    # 归一化净值
    strategy_nv_norm = strategy_nv / strategy_nv.iloc[0]
    benchmark_nv_norm = benchmark_nv / benchmark_nv.iloc[0]

    # 计算几何超额净值
    excess_return_nv = strategy_nv_norm / benchmark_nv_norm
    excess_returns = excess_return_nv.pct_change().dropna()

    # Calculate cumulative return
    cumulative_return = excess_return_nv.iloc[-1] - 1

    # Calculate annualized return
    total_days = len(excess_returns)
    annualized_return = (1 + cumulative_return) ** (annual_trading_days / total_days) - 1 if total_days > 0 else 0

    # Calculate annualized volatility
    annualized_volatility = excess_returns.std() * np.sqrt(annual_trading_days)

    # Calculate annualized Sharpe ratio
    annualized_sharpe = safe_divide(annualized_return, annualized_volatility)

    # Calculate maximum drawdown
    rolling_max = excess_return_nv.cummax()
    drawdown_pct = excess_return_nv / rolling_max - 1
    max_drawdown_pct = drawdown_pct.min()

    metrics = {
        "累积超额(几何)": cumulative_return,
        "年化超额(几何)": annualized_return,
        "年化超额波动率(几何)": annualized_volatility,
        "年化超额夏普(几何)": annualized_sharpe,
        "超额最大回撤(几何)": max_drawdown_pct,
    }

    def calculate_geometric_excess_win_performance(excess_nv, years=3):
        # 计算起始日期
        end_date = excess_nv.index[-1]  # 最后一个日期
        start_date = end_date - pd.DateOffset(years=years)  # 计算起始日期
        # 筛选出在指定时间范围内的数据
        data = excess_nv[start_date:end_date].infer_objects(copy=False)
        # 定义不同的持有期（1月、3月、6月、1年）
        holding_periods = [1, 3, 6, 12]  # 单位：月, 一个月22个交易日
        performance = {}
        for period in holding_periods:
            returns = data.pct_change(period * 22).dropna()
            avg_return = returns.mean() if not returns.empty else 0
            profit_probability = (returns > 0).sum() / len(returns) if not returns.empty else 0
            performance[f"{years}年持{period}月平均超额(几何)"] = avg_return
            performance[f"{years}年持{period}月超额胜率(几何)"] = profit_probability
        return performance

    metrics.update(calculate_geometric_excess_win_performance(excess_return_nv, 3))
    metrics.update(calculate_geometric_excess_win_performance(excess_return_nv, 5))
    metrics.update(calculate_geometric_excess_win_performance(excess_return_nv, 10))
    return metrics


def calculate_win_rate_metrics(
        pos, price_df, win_rate_consider_assets=None, adverse_detail_consider_assets=None, long_empty_mode=False
):
    if long_empty_mode:
        win_rate_consider_assets = ["empty", "long"]
        adverse_detail_consider_assets = ["empty-long", "long-empty"]
    else:
        if win_rate_consider_assets is None:
            # 默认考虑A50和2000
            win_rate_consider_assets = ["930050.CSI", "932000.CSI"]
        if adverse_detail_consider_assets is None:
            adverse_detail_consider_assets = [
                "930050.CSI-932000.CSI",
                "930050.CSI-empty",
                "932000.CSI-930050.CSI",
                "932000.CSI-empty",
                "empty-930050.CSI",
                "empty-932000.CSI",
            ]
    # 获得的持仓日胜率
    holding_asset_win_rate = holding_asset_win_rate_analysis(pos, long_empty_mode=long_empty_mode)
    # 获得对抗调仓胜率
    adverse_rebalance_metrics = calculate_adverse_rebalance_metrics(
        pos, price_df, return_as_dict=True, long_empty_mode=long_empty_mode
    )

    # 写入目标资产结果
    res = {}
    for asset in win_rate_consider_assets:
        if asset in holding_asset_win_rate:
            res[f"持仓日胜率_{asset}"] = holding_asset_win_rate[asset]["holding_win_rate"]
        else:
            res[f"持仓日胜率_{asset}"] = np.nan
    for asset in win_rate_consider_assets:
        if asset in holding_asset_win_rate:
            res[f"持仓日胜率(除空仓)_{asset}"] = holding_asset_win_rate[asset]["holding_win_rate_wo_emp"]
        else:
            res[f"持仓日胜率(除空仓)_{asset}"] = np.nan
    # 写入总对抗调仓胜率
    adverse_rebalance_win_rate = adverse_rebalance_metrics["adverse_rebalance_win_rate"]
    for asset in win_rate_consider_assets:
        if asset in adverse_rebalance_win_rate:
            res[f"对抗调仓胜率_{asset}"] = adverse_rebalance_win_rate[asset]
        else:
            res[f"对抗调仓胜率_{asset}"] = np.nan
    # 写入详情对抗胜率
    detail_adverse_rebalance_win_rate = adverse_rebalance_metrics["detail_adverse_rebalance_win_rate"]
    for asset in adverse_detail_consider_assets:
        if asset in detail_adverse_rebalance_win_rate:
            res[f"对抗调仓胜率_{asset}"] = detail_adverse_rebalance_win_rate[asset]
        else:
            res[f"对抗调仓胜率_{asset}"] = np.nan
    # 写入平均对抗超额收益
    mean_adverse_excess_return = adverse_rebalance_metrics["mean_adverse_excess_return"]
    for asset in adverse_detail_consider_assets:
        if asset in mean_adverse_excess_return:
            res[f"平均对抗超额_{asset}"] = mean_adverse_excess_return[asset]
        else:
            res[f"平均对抗超额_{asset}"] = np.nan
    # 总对抗超额
    sum_adverse_excess_return = adverse_rebalance_metrics["sum_adverse_excess_return"]
    for asset in adverse_detail_consider_assets:
        if asset in sum_adverse_excess_return:
            res[f"总对抗超额_{asset}"] = sum_adverse_excess_return[asset]
        else:
            res[f"总对抗超额_{asset}"] = np.nan
    return res


def calculate_performance(net_value, years=3):
    """
    计算近x年持有1月、3月、6月、1年的平均收益和盈利概率。

    :param net_value: pandas.Series，日期为索引，净值为值的时间序列。
    :param years: int, 计算的时间范围，默认为3年。
    :return: dict, 各持有期的平均收益和盈利概率。
    """
    # 计算起始日期
    end_date = net_value.index[-1]  # 最后一个日期
    start_date = end_date - pd.DateOffset(years=years)  # 计算起始日期

    # 筛选出在指定时间范围内的数据
    data = net_value[start_date:end_date].infer_objects(copy=False)

    # 定义不同的持有期（1月、3月、6月、1年）
    holding_periods = [1, 3, 6, 12]  # 单位：月, 一个月22个交易日

    performance = {}

    for period in holding_periods:
        returns = data.pct_change(period * 22).dropna()
        # 计算平均收益
        avg_return = sum(returns) / len(returns) if not returns.empty else 0

        # 计算盈利概率（盈利的概率，即回报率 > 0）
        profit_probability = sum(1 for r in returns if r > 0) / len(returns) if not returns.empty else 0

        # 将结果存储在字典中
        performance[f"{years}年持{period}月平均收益"] = avg_return
        performance[f"{years}年持{period}月盈利概率"] = profit_probability

    return performance


def calculate_performance_with_benchmark(strategy_nv, benchmark_nv):
    # 计算上行捕获与下行捕获
    strategy_returns = strategy_nv.pct_change().dropna()
    benchmark_returns = benchmark_nv.pct_change().reindex(strategy_returns.index).dropna()
    strategy_returns = strategy_returns.reindex(benchmark_returns.index)

    assert len(strategy_returns) == len(benchmark_returns)

    # benchmark_up = benchmark_returns[benchmark_returns > 0].add(1).prod() - 1
    # strategy_up = strategy_returns[benchmark_returns > 0].add(1).prod() - 1
    benchmark_up = benchmark_returns[benchmark_returns > 0].mean()
    strategy_up = strategy_returns[benchmark_returns > 0].mean()
    up_capture = safe_divide(strategy_up, benchmark_up)

    benchmark_down = benchmark_returns[benchmark_returns < 0].mean()
    strategy_down = strategy_returns[benchmark_returns < 0].mean()
    down_capture = safe_divide(strategy_down, benchmark_down)

    # 补充alpha 和 beta 的计算
    X = sm.add_constant(benchmark_returns)
    model = sm.OLS(strategy_returns, X).fit()
    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]

    simple_anual_alpha = alpha * 252
    compound_anual_alpha = (1 + alpha) ** 252 - 1

    return {
        "上行捕获": up_capture,
        "下行捕获": down_capture,
        "单利年化alpha": simple_anual_alpha,
        "复利年化alpha": compound_anual_alpha,
        "beta": beta,
    }


def calculate_pl_ratio(position_adjust):
    """
    计算盈亏比，平均盈利与平均亏损之比 $$\frac{总盈利金额/盈利天数}{总亏损金额/亏损天数}$$
    """
    daily_return = position_adjust["daily_return"].dropna().copy()
    mean_profit = daily_return[daily_return > 0].mean()
    mean_loss = abs(daily_return[daily_return < 0].mean())
    mean_pl_ratio = safe_divide(mean_profit, mean_loss)

    sum_profit = daily_return[daily_return > 0].sum()
    sum_loss = abs(daily_return[daily_return < 0].sum())
    profit_days = len(daily_return[daily_return > 0])
    loss_days = len(daily_return[daily_return < 0])
    sum_pl_ratio = safe_divide(sum_profit, sum_loss)

    return {
        "单利总盈利": sum_profit,
        "单利总亏损": sum_loss,
        "盈利天数": profit_days,
        "亏损天数": loss_days,
        "平均日度盈利": mean_profit,
        "平均日度亏损": mean_loss,
        "日度盈亏比": sum_pl_ratio,
        "平均化日度盈亏比": mean_pl_ratio,
    }


def plot_pnl(position_adjust, title="净值曲线", benchmark=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    if benchmark is None:
        position_adjust["nv"].plot(ax=ax)
    else:
        position_adjust["nv"].plot(ax=ax, label="策略净值")
        benchmark = benchmark.copy().reindex(position_adjust.index)
        benchmark = benchmark.div(benchmark.iloc[0])
        benchmark.plot(ax=ax, label="基准净值")
        ax.legend()
    ax.set_title(title)
    ax.set_ylabel("净值")
    ax.set_xlabel("日期")
    plt.show()


def adjust_metrics(_pos):
    res = {}
    asset_ids = _pos["holding_asset"].astype("category").cat.codes
    adjust_ids = (asset_ids.diff().fillna(0) != 0).cumsum()
    adjust_count = adjust_ids.value_counts()
    res["平均持仓交易日数"] = adjust_count.mean()
    res["持仓交易日数中位数"] = adjust_count.median()
    res["持仓交易日数最大值"] = adjust_count.max()
    res["持仓交易日数最小值"] = adjust_count.min()
    res["持仓交易日<5天"] = (adjust_count < 5).sum()
    return res


def calculate_metrics(position_adjust, annual_trading_days=252, return_as_df=False):
    # Calculate cumulative return
    cumulative_return = safe_divide(position_adjust["nv"].iloc[-1], position_adjust["nv"].iloc[0]) - 1

    # Calculate annualized return
    total_days = len(position_adjust)
    annualized_return = (1 + cumulative_return) ** (annual_trading_days / total_days) - 1

    # Calculate annualized volatility
    daily_returns = position_adjust["daily_return"]
    annualized_volatility = daily_returns.std() * np.sqrt(annual_trading_days)

    # Calculate annualized Sharpe ratio
    annualized_sharpe = safe_divide(annualized_return, annualized_volatility)

    # Calculate maximum drawdown
    rolling_max = position_adjust["nv"].cummax()
    drawdown = position_adjust["nv"] / rolling_max - 1
    max_drawdown = drawdown.min()

    # Calculate Calmar ratio
    calmar_ratio = safe_divide(annualized_return, abs(max_drawdown))

    res = {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "annualized_sharpe": annualized_sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }
    if return_as_df:
        return pd.DataFrame(res, index=[0])
    return res


def sim_position_from_nv(nv):
    position = nv.rename("nv").to_frame()
    position["daily_return"] = position.nv.pct_change().fillna(0)
    return position


def test_sim_by_nv(_nv, benchmark=None):
    _ = {}

    _nv = _nv.copy()
    _nv = sim_position_from_nv(_nv)
    _["all"] = calculate_metrics(_nv)
    if benchmark is not None:
        _["all"].update(calculate_performance_with_benchmark(_nv.nv, benchmark))
    _["all"].update(calculate_performance(_nv.nv, 3))
    _["all"].update(calculate_performance(_nv.nv, 5))
    _["all"].update(calculate_performance(_nv.nv, 10))

    _nv_from2016 = _nv[_nv.index >= pd.Timestamp("2016-01-04")].copy()
    _["from2016"] = calculate_metrics(_nv_from2016)
    if benchmark is not None:
        _["from2016"].update(calculate_performance_with_benchmark(_nv_from2016.nv, benchmark))
    _["from2016"].update(calculate_performance(_nv_from2016.nv, 3))
    _["from2016"].update(calculate_performance(_nv_from2016.nv, 5))
    _["from2016"].update(calculate_performance(_nv_from2016.nv, 10))

    _start = _nv.index[0]
    _end = _nv.index[-1]

    intervals = pd.interval_range(_start, _end, freq="YS").tolist()
    if len(intervals) > 0:
        if _start < intervals[0].left:
            intervals.insert(0, pd.Interval(left=_start, right=intervals[0].left))
        if _end > intervals[-1].right:
            intervals.append(pd.Interval(left=intervals[-1].right, right=_end))

    for interval in intervals:
        _position = _nv[interval.left: interval.right]
        _[interval.left.year] = calculate_metrics(_position)

    return _


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_trans(size, _up, _down):
    if _up == _down:
        return np.ones(size)
    weights = np.arange(-size // 2, size // 2, 1)
    weights = sigmoid(weights)
    weights = (_up - _down) * weights + _down
    return weights


def yearly_bull_bear_test(tester, position_adjust, is_bull, _start=None, _end=None):
    if _start is None:
        _start = position_adjust.index[0]
    if _end is None:
        _end = position_adjust.index[-1]
    intervals = pd.interval_range(_start, _end, freq="YS").tolist()
    if len(intervals) > 0:
        intervals.insert(0, pd.Interval(left=_start, right=intervals[0].left))
        intervals.append(pd.Interval(left=intervals[-1].right, right=_end))

    res = []
    days_index = position_adjust.index
    is_bull = is_bull.copy().reindex(days_index, fill_value="ffill").fillna(False)
    for interval in intervals:
        _position = position_adjust[interval.left: interval.right]

        # # 处理 bull 与 bear
        for bull_flag in [True, False]:
            bull_position = _position[is_bull[interval.left: interval.right] == bull_flag].copy()
            if len(bull_position) > 2:  # 超过2天才有统计的必要
                _ = {
                    "year": interval.left.year,
                    "is_bull": bull_flag,
                    "days_cnt": len(bull_position),
                }
                bull_position["nv"] = bull_position["daily_return"].add(1).cumprod()
                _.update(tester.calculate_metrics(bull_position, start_date=interval.left, end_date=interval.right))
                res.append(_)
    return res


def year_test(tester, position_adjust, _start=None, _end=None):
    if _start is None:
        _start = position_adjust.index[0]
    if _end is None:
        _end = position_adjust.index[-1]
    intervals = pd.interval_range(_start, _end, freq="YS").tolist()
    if len(intervals) > 0:
        intervals.insert(0, pd.Interval(left=_start, right=intervals[0].left))
        intervals.append(pd.Interval(left=intervals[-1].right, right=_end))

    res = {}
    for interval in intervals:
        _position = position_adjust[interval.left: interval.right]
        res[interval.left.year] = tester.calculate_metrics(_position, start_date=interval.left, end_date=interval.right)
        res[interval.left.year].update(calculate_empty_ratio(_position.holding_asset))
    return res


def calculate_entropy(probabilities, soft=True):
    # 使用numpy计算熵
    probabilities = np.array(probabilities)
    probabilities -= np.min(probabilities)
    probabilities = softmax(probabilities) if soft else probabilities
    return -np.sum(probabilities * np.log2(probabilities))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sigmoid_trans(size, _up, _down):
    if _up == _down:
        return np.ones(size)
    weights = np.arange(-size // 2, size // 2, 1)
    weights = sigmoid(weights)
    weights = (_up - _down) * weights + _down
    return weights


def yearly_bull_bear_test(tester, position_adjust, is_bull, _start=None, _end=None):
    if _start is None:
        _start = position_adjust.index[0]
    if _end is None:
        _end = position_adjust.index[-1]
    intervals = pd.interval_range(_start, _end, freq="YS").tolist()
    if len(intervals) > 0:
        intervals.insert(0, pd.Interval(left=_start, right=intervals[0].left))
        intervals.append(pd.Interval(left=intervals[-1].right, right=_end))

    res = []
    days_index = position_adjust.index
    is_bull = is_bull.copy().reindex(days_index, fill_value="ffill").fillna(False)
    for interval in intervals:
        _position = position_adjust[interval.left: interval.right]

        # 处理 bull 与 bear
        for bull_flag in [True, False]:
            bull_position = _position[is_bull[interval.left: interval.right] == bull_flag].copy()
            if len(bull_position) > 2:  # 超过2天才有统计的必要
                _ = {
                    "year": interval.left.year,
                    "is_bull": bull_flag,
                    "days_cnt": len(bull_position),
                }
                bull_position["nv"] = bull_position["daily_return"].add(1).cumprod()
                _.update(tester.calculate_metrics(bull_position, start_date=interval.left, end_date=interval.right))
                res.append(_)
    return res


def year_test(tester, position_adjust, _start=None, _end=None):
    if _start is None:
        _start = position_adjust.index[0]
    if _end is None:
        _end = position_adjust.index[-1]
    intervals = pd.interval_range(_start, _end, freq="YS").tolist()
    if len(intervals) > 0:
        intervals.insert(0, pd.Interval(left=_start, right=intervals[0].left))
        intervals.append(pd.Interval(left=intervals[-1].right, right=_end))

    res = {}
    for interval in intervals:
        _position = position_adjust[interval.left: interval.right]
        res[interval.left.year] = tester.calculate_metrics(_position, start_date=interval.left, end_date=interval.right)
        res[interval.left.year].update(calculate_empty_ratio(_position.holding_asset))
    return res


def adjust_position_by_open_price(position_adjust, open_price, close_price):
    if "target" not in position_adjust.columns:
        raise ValueError(
            "你要修改为open价调仓的数据中没有 target 列，请检查是不是已经是开盘价调仓了。暂不支持直接对合成持仓调整开盘价。"
        )

    # 计算昨收到今开的收益
    daily_return_last_close_open = open_price.div(close_price.shift(1)).sub(1).fillna(0)
    if "empty" not in daily_return_last_close_open.columns:
        daily_return_last_close_open["empty"] = 0.0
    # 计算今开到今收的收益
    daily_return_open_close = close_price.div(open_price).sub(1).fillna(0)
    if "empty" not in daily_return_open_close.columns:
        daily_return_open_close["empty"] = 0.0
    daily_return_close_close = close_price.pct_change().fillna(0)
    if "empty" not in daily_return_close_close.columns:
        daily_return_close_close["empty"] = 0.0
    # 信号发生后，下一天能及时调仓
    position_adjust["holding_asset"] = position_adjust["target"].shift(1).fillna("empty")

    # 针对每次调仓，调整收益，当发生资产切换时，收益包括两部分：
    # 1. 上一个持仓资产昨收到今开的收益
    # 2. 今天的持仓资产从今开到今收的收益
    _holding = position_adjust["holding_asset"]
    for i in range(1, len(_holding)):
        _date = _holding.index[i]

        if _holding.iloc[i] != _holding.iloc[i - 1]:
            adjust_return = (1 + daily_return_last_close_open.loc[_date, _holding.iloc[i]]) * (
                    1 + daily_return_open_close.loc[_date, _holding.iloc[i]]
            ) - 1
            position_adjust.loc[_date, "daily_return"] = adjust_return
        else:
            position_adjust.loc[_date, "daily_return"] = daily_return_close_close.loc[_date, _holding.iloc[i]]
    # 重新计算净值
    position_adjust["nv"] = position_adjust["daily_return"].add(1).cumprod()
    return position_adjust


def run_test(
        model_name,
        threshold,
        sigmoid_weight_down,
        D,
        daily_return,
        tester,
        targets,
        all_scores_df,
        S=20,
        start_date=pd.Timestamp("2016-01-04"),
        using_model_score=None,
        benchmark=None,
        test_with_holding_metric=True,
        is_bull=None,
        open_price=None,
        close_price=None,
        change_use_open_price=False,
        test_with_rebalance_metric=False,
        test_with_empty_metric=False,
        async_mode=False,
        test_with_win_rate_metric=True,
        win_rate_consider_assets=None,
        end_date=None,
        skip_empty_warehouse=False,
):
    """
    运行测试，并返回结果。

    如果设置 change_use_open_price=True，则持仓发生切换的时候，会以 open_price 作为价格为成交价。
    成交日的收益由两部分组成：一部分是上一个持仓资产昨收到今开的收益，另一部分是今天的持仓资产从今开到今收的收益。
    否则如果设置 change_use_open_price=False，则持仓发生切换的时候，都是以 close_price 作为价格为成交价。每日收益都是作收到今收的收益。

    D: int, 锁仓天数
    S: int, 动量信号窗口大小
    """

    if benchmark is None:
        warnings.warn("run_test param `benchmark` is None, will not calculate benchmark metrics")
    if is_bull is not None:
        warnings.warn("`is_bull` is deprecated, please do not use it")
    if change_use_open_price:
        assert (
                open_price is not None and close_price is not None
        ), "`open_price` and `close_price` must be provided when `change_use_open_price` is True"

    _weights = sigmoid_trans(S, 1, sigmoid_weight_down)[::-1]
    _sig = daily_return.rolling(S).apply(lambda x: (x * _weights + 1).prod() - 1, raw=True).dropna().copy()
    _sig_pred = (_sig < 0).astype(int)[targets]
    if all_scores_df is not None:
        _sig_pred = _sig_pred.reindex(all_scores_df.index)
    _sig.insert(0, "empty", 0.0)

    sig_belief = {}
    sig_belief["entropy"] = _sig.apply(calculate_entropy, axis=1)
    sig_belief["std"] = _sig.std(axis=1)

    def top2_diff(arr):
        arr.sort()
        return arr[-1] - arr[-2]

    sig_belief["top_diff"] = _sig.apply(top2_diff, axis=1, raw=True)
    sig_belief["max"] = _sig.max(axis=1)

    sig_belief = pd.DataFrame(sig_belief)

    _y_pred_3cls = _sig.apply(lambda x: x.argmax(), axis=1).rename("y")

    if model_name != "no" and using_model_score is not None:
        for tag in using_model_score:
            tag_parts = tag.split("_")
            assert len(tag_parts) >= 3, r"using_model_score must be in the format of `{model_name}_{y_i}_{kind}`"
            y_i = int(tag_parts[1])
            kind = tag_parts[2]
            _pred = all_scores_df[tag]
            if isinstance(_pred.iloc[0], np.ndarray):
                _pred = _pred.apply(lambda x: x[1])
            _pred = (_pred > threshold).astype(int)
            if kind == "up":
                _y_pred_3cls[(_pred == 1) & (_y_pred_3cls == 0)] = y_i + 1
            else:
                _y_pred_3cls[(_pred == 1) & (_y_pred_3cls == y_i + 1)] = 0
    elif model_name != "no" and using_model_score is None:
        all_enhanced_preds = {}
        to_enhance = _sig_pred.reindex(all_scores_df.index).copy()
        for y_i in range(len(targets[:2])):
            _pred = all_scores_df[f"{model_name}_{y_i}"]
            if isinstance(_pred.iloc[0], np.ndarray):
                _pred = _pred.apply(lambda x: x[1])
            _pred = (_pred > threshold).astype(int)
            _enhance = to_enhance.iloc[:, y_i] | _pred  # 牛魔王的空仓信号或者模型的空仓信号生效（为1则空仓）
            all_enhanced_preds[y_i] = _enhance
        for i in range(len(targets[:2])):
            _index = (_y_pred_3cls == i + 1) & (all_enhanced_preds[i] == 1)
            _y_pred_3cls[_index] = 0

    real_sigs = _y_pred_3cls.map(lambda x: targets[x - 1] if x > 0 else "empty")
    position_adjust = tester.backtest(
        real_sigs,
        start_date=start_date,
        end_date=end_date,
        D=D,
        async_mode=async_mode,
        adjust_by_open_price=change_use_open_price,
        open_price=open_price,
        close_price=close_price,
        skip_empty_warehouse=skip_empty_warehouse,
    )

    final_metrics = {
        "all": measure_position(
            position_adjust,
            tester,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            holding_metric=test_with_holding_metric,
            rebalance_metric=test_with_rebalance_metric,
            empty_metric=test_with_empty_metric,
            win_rate_metric=test_with_win_rate_metric,
            win_rate_consider_assets=win_rate_consider_assets,
        ),
        "from2016": measure_position(
            position_adjust,
            tester,
            start_date=pd.Timestamp("2016-01-04"),
            end_date=end_date,
            benchmark=benchmark,
            holding_metric=test_with_holding_metric,
            rebalance_metric=test_with_rebalance_metric,
            empty_metric=test_with_empty_metric,
            win_rate_metric=test_with_win_rate_metric,
            win_rate_consider_assets=win_rate_consider_assets,
        ),
    }

    final_metrics.update(year_test(tester, position_adjust, _start=start_date))

    return position_adjust, final_metrics, sig_belief


def set_chinese_font():
    """
    设置中文字体
    """
    import platform

    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "SimHei"
    elif system == "Darwin":  # macOS
        plt.rcParams["font.family"] = "Heiti TC"
    else:  # Assume Linux
        plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False


def calculate_rebalance_win_rate(position_df, empty_tag="empty", benchmark=None):
    """
    计算持仓胜率（计入空仓与不计入空仓），以及如果设置了benchmark，相对于benchmark的空仓胜率
    """
    df = position_df.copy()
    df.index.name = "time"
    df = df.reset_index()
    # 识别调仓点：holding_asset发生变化的位置
    df["is_rebalance"] = df["holding_asset"] != df["holding_asset"].shift(1)

    # 生成持仓周期分组标识
    df["period"] = df["is_rebalance"].cumsum()

    def compound_return(returns):
        return (returns + 1).prod() - 1

    if benchmark is not None:
        df["benchmark_return"] = benchmark.pct_change().reindex(df["time"]).reset_index(drop=True)
        period_returns = df.groupby("period", as_index=False).agg(
            start_time=("time", "first"),
            holding_asset=("holding_asset", "first"),
            total_return=("daily_return", compound_return),
            benchmark_return=("benchmark_return", compound_return),
        )
    else:
        # 计算每个持仓周期的累计收益
        period_returns = df.groupby("period", as_index=False).agg(
            start_time=("time", "first"),
            holding_asset=("holding_asset", "first"),
            total_return=("daily_return", compound_return),
        )

    # 筛选有效调仓周期（排除初始空持仓）
    valid_periods = period_returns[period_returns["holding_asset"] != empty_tag]
    # 计算胜率
    win_count_wo_emp = (valid_periods["total_return"] > 0).sum()
    total_trades_wo_emp = len(valid_periods)
    rate_wo_emp = win_count_wo_emp / total_trades_wo_emp if total_trades_wo_emp > 0 else 0.0

    win_count = (period_returns["total_return"] > 0).sum()
    total_trades = len(period_returns)
    rate = win_count / total_trades if total_trades > 0 else 0.0

    average_win_return = period_returns["total_return"].mean()

    # 计算持仓盈亏比
    valid_holding_profit = valid_periods[valid_periods["total_return"] > 0]["total_return"].sum()
    valid_holding_loss = -valid_periods[valid_periods["total_return"] < 0]["total_return"].sum()
    holding_pl_ratio = safe_divide(valid_holding_profit, valid_holding_loss)

    valid_mean_holding_profit = valid_periods[valid_periods["total_return"] > 0]["total_return"].mean()
    valid_mean_holding_loss = -valid_periods[valid_periods["total_return"] < 0]["total_return"].mean()
    mean_holding_pl_ratio = safe_divide(valid_mean_holding_profit, valid_mean_holding_loss)

    _res = {
        "平均调仓收益": average_win_return,
        "调仓胜率": rate,
        "持仓盈利次数": win_count,
        "总调仓次数": total_trades,
        "多仓方向准确率": rate_wo_emp,
        "持仓盈利次数(除空仓)": win_count_wo_emp,
        "总调仓次数(除空仓)": total_trades_wo_emp,
        "持仓盈亏比": holding_pl_ratio,
        "平均化持仓盈亏比": mean_holding_pl_ratio,
    }

    if benchmark is not None:
        # 计算相对benchmark的空仓胜率
        valid_periods = period_returns[period_returns["holding_asset"] == empty_tag]
        empty_win_count = (valid_periods["total_return"] > valid_periods["benchmark_return"]).sum()
        empty_total_trades = len(valid_periods)
        empty_rate = empty_win_count / empty_total_trades if empty_total_trades > 0 else 0.0

        weighted_long_empty_rate = safe_divide(
            rate_wo_emp * total_trades_wo_emp + empty_rate * empty_total_trades,
            total_trades_wo_emp + empty_total_trades,
            default=np.nan,
        )
        _res.update(
            {
                "空仓方向准确率": empty_rate,
                "相对benchmark空仓盈利次数": empty_win_count,
                "相对benchmark总空仓次数": empty_total_trades,
                "总方向准确率": weighted_long_empty_rate,
            }
        )
    return _res, df, period_returns


def safe_divide(a, b, default=np.nan):
    return a / b if not np.isclose(b, 0) else default


def measure_empty(position_df, empty_tag="empty", benchmark=None, annual_days=252):
    _res = {}
    if benchmark is None:
        warnings.warn("benchmark is not provided, empty metric is not calculated.")
        return _res
    df = position_df.copy()
    df["benchmark_return"] = benchmark.pct_change().reindex(df.index)
    # 避险天数
    empty_days = len(df[df["holding_asset"] == empty_tag])

    empty_parts = df[df["holding_asset"] == empty_tag].copy()
    holding_parts = df[df["holding_asset"] != empty_tag].copy()

    # 避险增益计算：避险期内(空仓收益-基准收益)复利累积；若为正值代表规避的市场风险，若为负数代表错过的市场收益。正数代表是一个整体上规避的风险比损失的收益多的有效避险策略。
    empty_gain = (empty_parts["daily_return"] - empty_parts["benchmark_return"]).add(1).prod() - 1
    annual_empty_gain = (1 + empty_gain) ** (annual_days / empty_days) - 1 if empty_days > 0 else 0
    _res["累积避险增益"] = empty_gain
    _res["年化避险增益"] = annual_empty_gain

    # 平均规避风险、平均错过收益；最大规避风险、最大错过收益；总规避风险、总错过收益：所有成功规避风险时基准损失（收益率的相反数）的平均值、最大值和单利求和，所有错过收益的平均值、最大值和单利求和
    empty_parts["avoid_risk"] = empty_parts["daily_return"] - empty_parts["benchmark_return"]
    empty_parts["avoid_risk"] = empty_parts["avoid_risk"].clip(lower=0)
    empty_parts["miss_return"] = empty_parts["benchmark_return"] - empty_parts["daily_return"]
    empty_parts["miss_return"] = empty_parts["miss_return"].clip(lower=0)
    _res["平均日规避风险"] = empty_parts["avoid_risk"].mean()
    _res["平均日错过收益"] = empty_parts["miss_return"].mean()
    _res["最大日规避风险"] = empty_parts["avoid_risk"].max()
    _res["最大日错过收益"] = empty_parts["miss_return"].max()
    _res["总规避风险(单利)"] = empty_parts["avoid_risk"].sum()
    _res["总错过收益(单利)"] = empty_parts["miss_return"].sum()

    _res["规避风险比错过收益"] = safe_divide(_res["总规避风险(单利)"], _res["总错过收益(单利)"])

    # 避险机会成本比率：$$\frac{总错过收益}{策略收益单利总和} \times 100\%$$，总错过收益一定为正数，策略收益可能为负数。无论空仓机会成本比是正数还是负数，都是越接近0越好，代表机会成本越小。若比值为200%，代表错过的市场收益是现在收益的2倍。若比值为-100%，说明若全都不错，能够让策略不赚不亏。

    _res["避险机会成本比率"] = safe_divide(_res["总错过收益(单利)"], holding_parts["daily_return"].sum())
    return _res


def calculate_down_vol_compression_ratio(source_nv, model_nv):
    """
    计算下行波动率压缩比：
    $$\frac{\sigma^-_{原始组合}-\sigma^-_{避险组合}}{\sigma^-_{原始组合}}$$
    衡量避险模型对下行风险的抑制能力，其中 $$\sigma^-$$ 为下行标准差
    """

    def downside_std(returns):
        negative_returns = returns[returns < 0]
        # 没有负收益，波动率为 0
        return np.sqrt((negative_returns ** 2).sum() / len(negative_returns)) if len(negative_returns) > 0 else 0

    source_returns = source_nv.pct_change().dropna()
    model_returns = model_nv.pct_change().dropna()

    source_down_vol = downside_std(source_returns)
    model_down_vol = downside_std(model_returns)

    return safe_divide(source_down_vol - model_down_vol, source_down_vol)


class PositionDetail:
    """持仓详情类，用于管理资产占比"""

    EMPTY = "empty"

    def __init__(self, holdings=None):
        """
        初始化持仓详情

        参数:
        holdings (dict): 资产及其权重的字典，如果为None则默认为空仓(100%现金)
        """
        self.holdings = {}

        if holdings is None:
            self.holdings[self.EMPTY] = 0.0  # 空类，用于加法计算
        elif isinstance(holdings, str):
            self.holdings[holdings] = 1.0
        elif isinstance(holdings, (list, tuple)):
            for asset in holdings:
                self.holdings[asset] = 1.0 / len(holdings)
        elif isinstance(holdings, PositionDetail):
            self.holdings = holdings.holdings.copy()
        elif isinstance(holdings, dict):
            self.holdings = holdings.copy()
        else:
            raise TypeError(f"不支持的初始化类型{type(holdings)}")

    def __mul__(self, scalar):
        """持仓权重与标量相乘"""
        if not isinstance(scalar, (int, float)):
            return NotImplemented

        new_holdings = {asset: weight * scalar for asset, weight in self.holdings.items()}
        return PositionDetail(new_holdings)

    def __rmul__(self, scalar):
        """支持右乘"""
        return self.__mul__(scalar)

    def __add__(self, other):
        """合并两个持仓"""
        if not isinstance(other, PositionDetail):
            return NotImplemented

        new_holdings = self.holdings.copy()

        for asset, weight in other.holdings.items():
            if asset in new_holdings:
                new_holdings[asset] += weight
            else:
                new_holdings[asset] = weight

        result = PositionDetail(new_holdings)
        return result

    def __str__(self):
        """返回持仓的字符串表示"""
        return str(self.holdings)

    def to_dict(self):
        """返回持仓字典的副本"""
        return self.holdings.copy()

    def get_holding_asset(self):
        """获取持仓资产表示（empty/str/tuple）"""
        assets = [asset for asset, weight in self.holdings.items() if weight > 0 and asset != self.EMPTY]

        if not assets:
            return self.EMPTY
        elif len(assets) == 1:
            return assets[0]
        else:
            return tuple(assets)

    @classmethod
    def check_weight(cls, details):
        if isinstance(details, PositionDetail):
            # 检查权重和是否为1
            total_weight = sum(details.holdings.values())
            return np.isclose(total_weight, 1.0)
        elif isinstance(details, (list, tuple)):
            return [cls.check_weight(detail) for detail in details]
        elif isinstance(details, pd.Series):
            return details.map(cls.check_weight)
        else:
            raise TypeError(f"不支持的类型{type(details)}")


def merge_position(to_merge_positions, weights=None, merge_target=False):
    """
    如果 weights 为 None, 则默认每天按照等权合并。如果weights为一维数组, 则数组长度需要等于to_merge_positions的长度,
    每天都按照这个权重合并。如果weights为二维的DataFrame, 则DataFrame的index应该为权重日期, 权重日期的长度不应小于各个
    to_merge_positions的长度, DataFrame的columns应该等于to_merge_positions的columns, DataFrame的values应该是权重。
    权重每行和为1。

    如果to_merge_positions里的每个position的index不一样, 会合并出一个公有的日期,然后再合并。 注意如果某天0号position是有交易信息的，
    而1号没有交易信息，则加权的时候仍然会给0号weights里的权重。因为模拟的是初始就将各个资金分成了多份。
    """
    merge_cnt = len(to_merge_positions)
    if weights is None:
        weights = np.array([1 / merge_cnt] * merge_cnt)
    elif isinstance(weights, pd.Series):
        weights = weights.values
        assert len(weights) == merge_cnt, "weights长度应该等于to_merge_positions的长度"
    elif isinstance(weights, (list, tuple)):
        assert len(weights) == merge_cnt, "weights长度应该等于to_merge_positions的长度"
        weights = np.array(weights)
    elif isinstance(weights, np.ndarray):
        if len(weights.shape) == 1:
            assert len(weights) == merge_cnt, "weights长度应该等于to_merge_positions的长度"
        elif len(weights.shape) > 1:
            raise NotImplementedError("暂时不支持二维数组，指定每天的权重应使用DataFrame")
    elif isinstance(weights, pd.DataFrame):
        assert len(weights.columns) == merge_cnt, "weights的列数应该等于to_merge_positions的长度"
        assert len(weights) >= len(to_merge_positions[0]), "weights的长度应该不小于to_merge_positions的长度"
        assert np.allclose(weights.sum(axis=1), 1), "权重每行和应该为1"
    else:
        raise ValueError("weights只支持None, list, tuple, np.ndarray, pd.Series, pd.DataFrame")

    # 合并出公有日期
    common_daily_return = pd.concat([position["daily_return"] for position in to_merge_positions], axis=1).fillna(0)

    # 扩展生成每日权重
    if isinstance(weights, np.ndarray):
        weights = np.tile(weights, (len(common_daily_return), 1))
        weights = pd.DataFrame(
            weights,
            index=common_daily_return.index,
            columns=[i for i in range(merge_cnt)],
        )
    else:
        weights = weights.reindex(common_daily_return.index, method="ffill").fillna(0)

    # 计算每日收益
    merged_daily_return = (common_daily_return.values * weights.values).sum(axis=1)

    merged_position = deepcopy(to_merge_positions[0].reindex(common_daily_return.index))
    merged_position["daily_return"] = merged_daily_return

    # 为每个to_merge_positions添加detail列辅助计算
    to_merge_positions = deepcopy(to_merge_positions)
    for i in range(merge_cnt):
        to_merge_positions[i] = to_merge_positions[i].reindex(common_daily_return.index)
        to_merge_positions[i]["holding_asset"] = to_merge_positions[i]["holding_asset"].fillna(PositionDetail.EMPTY)
        if "detail" not in to_merge_positions[i]:
            to_merge_positions[i]["detail"] = to_merge_positions[i]["holding_asset"].apply(PositionDetail)

    # 对每天进行处理
    merged_position["detail"] = None  # 新增详细信息列

    for date in common_daily_return.index:
        combined_position = sum(
            [to_merge_positions[i].loc[date, "detail"] * weights.loc[date, i] for i in range(merge_cnt)],
            PositionDetail(),
        )

        assert PositionDetail.check_weight(combined_position), f"日期{date}权重和不为1"
        merged_position.at[date, "detail"] = combined_position
        merged_position.at[date, "holding_asset"] = combined_position.get_holding_asset()

        if merge_target and "target" in to_merge_positions[i].columns:
            key = "target_detail" if "target_detail" in to_merge_positions[i].columns else "target"
            combined_target = sum(
                [PositionDetail(to_merge_positions[i].loc[date, key]) * weights.loc[date, i] for i in range(merge_cnt)],
                PositionDetail(),
            )
            merged_position.at[date, "target"] = combined_target.get_holding_asset()
            merged_position.at[date, "target_detail"] = combined_target

    if not merge_target and "target" in merged_position:  # 删除target列, 以免干扰
        merged_position.pop("target")

    # 更新净值列
    merged_position["nv"] = (1 + merged_position["daily_return"]).cumprod()

    return merged_position


def prepare_price_data(
        common_price_path=None,
        au9999_price_path=None,
        data_path="data/fof_price_updating.parquet",
        calendar_anchor_symbol=None,
        config_path=None,
):
    if common_price_path is not None or au9999_price_path is not None:
        warnings.warn(
            "注意， common_price_path 和 au9999_price_path 的参数未来将被弃用，为方便api自动更新所有数据转为使用parquet存储。"
            "请使用 data_path 指定目标数据文件替代，该数据文件中已经包括了黄金数据，不过国外指数如纳指和标普等指数api获取上存在问题，暂不放入其中。"
            "如果你需要加载的是所有宽基数据，请先注意目标parquet文件是否已经更新了所有新宽基。详情可以咨询姚垿(yaoxu@163.sufe.edu.cn)。",
            DeprecationWarning,
            stacklevel=2,
        )

        native_price = (
            pd.read_excel(common_price_path, parse_dates=["时间"])
            .rename(
                columns=lambda x: {
                    "代码": "unique_id",
                    "开盘价(元)": "open",
                    "最高价(元)": "high",
                    "最低价(元)": "low",
                    "收盘价(元)": "close",
                    "成交量(万股)": "volume",
                    "成交金额(万元)": "amount",
                    "时间": "time",
                }.get(
                    x, x
                )  # 保留未映射的列
            )
            .dropna()
        )

    else:
        native_price = pd.read_parquet(data_path)

    # 准备当日收盘价数据：交易日序列由可配置锚点资产决定
    anchor_symbol = _resolve_calendar_anchor_symbol(
        native_price,
        calendar_anchor_symbol=calendar_anchor_symbol,
        config_path=config_path,
    )
    working_days = native_price[native_price.unique_id == anchor_symbol].set_index("time").sort_index().index
    price_df = (
        native_price.pivot(index="time", columns="unique_id", values="close").ffill().bfill().reindex(working_days)
    )

    if au9999_price_path is not None:
        try:
            au9999 = pd.read_excel(au9999_price_path, parse_dates=["Date"], index_col=0, skiprows=1)["Au9999.SGE"]
            # assert len(au9999) == price_df.shape[0]
            au9999 = au9999.reindex(working_days)
            price_df["Au9999.SGE"] = au9999
            price_df["000140.SH"] = price_df["000140.CSI"]
        except:
            pass

    # 准备开盘价数据
    open_price_df = (
        native_price.pivot(index="time", columns="unique_id", values="open").ffill().bfill().reindex(working_days)
    )

    if au9999_price_path is not None:
        try:
            open_price_df["Au9999.SGE"] = (
                au9999  # 黄金开盘价数据暂时拿不到，用当日收盘价代替，一般黄金单日涨跌幅不会太大
            )
            open_price_df["000140.SH"] = open_price_df["000140.CSI"]
        except:
            pass

    return native_price, price_df, open_price_df, working_days


def part_win_measure(pos):
    # 持仓胜率
    win_days = pos[pos["daily_return"] > 0].shape[0]
    total_days = pos.shape[0]
    holding_win_rate = win_days / total_days

    # 分母去除空仓的持仓胜率
    holding_days = pos[pos["holding_asset"] != "empty"].shape[0]
    holding_win_rate_wo_emp = win_days / holding_days if holding_days > 0 else 0.0

    # 累积收益
    cum_return = pos["daily_return"].add(1).cumprod().iloc[-1] - 1
    # 单利累积
    simple_cum = pos["daily_return"].sum()
    # 年化收益
    annual_return = (cum_return + 1) ** (252 / total_days) - 1
    return {
        "cumulative_return": cum_return,
        "annual_return": annual_return,
        "holding_win_rate": holding_win_rate,
        "win_days": win_days,
        "total_days": total_days,
        "simple_cum": simple_cum,
        "holding_win_rate_wo_emp": holding_win_rate_wo_emp,
    }


def holding_asset_win_rate_analysis(pos, long_empty_mode=False):
    if long_empty_mode:
        pos = pos.copy()
        pos["holding_asset"] = pos["holding_asset"].map(lambda x: "empty" if x == "empty" else "long")

    assets = pos.holding_asset.unique().tolist()
    res = {}
    for asset in assets:
        res[asset] = {}
        _pos = pos[pos.holding_asset == asset].copy()
        res[asset] = part_win_measure(_pos)
    return res


def plot_holding_asset_win_analysis_all(res_pos, show=True, figsize=(12, 14)):
    res = {}
    for k, v in res_pos.items():
        res[k] = holding_asset_win_rate_analysis(v)
    # 绘制 index为 不同资产，column为不同策略的
    # holding_win_rate 热力图、total_days 热力图、simple_cum 热力图
    holding_win_rate_df = pd.DataFrame({k: pd.DataFrame(v).T["holding_win_rate"] for k, v in res.items()})
    holding_win_rate_df.columns = res_pos.keys()

    total_days_df = pd.DataFrame({k: pd.DataFrame(v).T["total_days"] for k, v in res.items()})
    total_days_df.columns = res_pos.keys()

    win_days_df = pd.DataFrame({k: pd.DataFrame(v).T["win_days"] for k, v in res.items()})
    win_days_df.columns = res_pos.keys()

    simple_cum_df = pd.DataFrame({k: pd.DataFrame(v).T["simple_cum"] for k, v in res.items()})
    simple_cum_df.columns = res_pos.keys()

    # 将三个df绘制热力图，放在从上到下三张子图上
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    to_draw = [
        (holding_win_rate_df, "持仓胜率"),
        (total_days_df, "持仓天数"),
        (win_days_df, "持仓盈利天数"),
        (simple_cum_df, "单利累积收益"),
    ]
    for ax, (df, title) in zip(axes, to_draw):
        # 如果title为持仓天数，则将格式设置为整数，否则为.2%
        fmt = ".0f" if title in ["持仓天数", "持仓盈利天数"] else ".2%"
        sns.heatmap(
            df,
            annot=True,
            fmt=fmt,
            cmap="Reds",
            cbar=True,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(title)
        ax.set_ylabel("资产")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment="right")
    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes, to_draw


def plot_adverse_win_headmap_all(res_pos, price_df, show=True, figsize=(12, 12)):
    res = {}
    for k, v in res_pos.items():
        res[k] = calculate_adverse_rebalance_metrics(v, price_df, return_as_dict=True)
    # 绘制 index为 不同资产，column为不同策略的 adverse_rebalance_win_rate heat map， 还有 adverse_rebalance_total_cnt  adverse_rebalance_win_cnt
    adverse_rebalance_win_rate_df = pd.DataFrame({k: v["adverse_rebalance_win_rate"] for k, v in res.items()})
    adverse_rebalance_total_cnt_df = pd.DataFrame({k: v["adverse_rebalance_total_cnt"] for k, v in res.items()})
    adverse_rebalance_win_cnt_df = pd.DataFrame({k: v["adverse_rebalance_win_cnt"] for k, v in res.items()})
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    to_draw = [
        (adverse_rebalance_win_rate_df, "对抗调仓胜率"),
        (adverse_rebalance_total_cnt_df, "对抗调仓总次数"),
        (adverse_rebalance_win_cnt_df, "对抗调仓胜利次数"),
    ]
    for ax, (df, title) in zip(axes, to_draw):
        # 如果title为持仓天数，则将格式设置为整数，否则为.2%
        fmt = ".0f" if title in ["对抗调仓总次数", "对抗调仓胜利次数"] else ".2%"
        sns.heatmap(
            df,
            annot=True,
            fmt=fmt,
            cmap="Reds",
            cbar=True,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(title)
        ax.set_ylabel("资产")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment="right")
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes, to_draw


# 通用绘图函数
def plot_monthly_attribution(pos, title=None, figsize=(14, 7), show=True):
    # 将索引转换为日期格式
    pos.index = pd.to_datetime(pos.index)

    # 计算月度资产收益汇总
    pos["year_month"] = pos.index.to_period("M")
    monthly_returns = pos.groupby(["year_month", "holding_asset"]).daily_return.sum().unstack().fillna(0)

    # 资产颜色映射（使用更美观的颜色）
    asset_colors = {}
    predefined_colors = {"Au9999.SGE": "#F9D949", "930050.CSI": "#E63946", "932000.CSI": "#457B9D"}
    for asset in monthly_returns.columns:
        asset_colors[asset] = predefined_colors.get(asset, None)

    # 随机分配未指定颜色的资产
    for asset, color in asset_colors.items():
        if color is None:
            asset_colors[asset] = plt.cm.tab20(hash(asset) % 20)

    # 区分正负收益
    pos_returns = monthly_returns.clip(lower=0)
    neg_returns = monthly_returns.clip(upper=0)

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制正收益堆积图
    pos_returns.plot.bar(stacked=True, color=[asset_colors[a] for a in monthly_returns.columns], ax=ax, legend=False)

    # 绘制负收益堆积图
    neg_returns.plot.bar(stacked=True, color=[asset_colors[a] for a in monthly_returns.columns], ax=ax, legend=False)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("月度收益归因分析")
    plt.xlabel("月份")
    plt.ylabel("收益")

    # 显示图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="资产", bbox_to_anchor=(0, 1), loc="upper left")

    # 调整x轴刻度显示密度
    ax.set_xticks(range(0, len(monthly_returns.index), max(len(monthly_returns) // 12, 1)))
    ax.set_xticklabels(
        [str(date) for date in monthly_returns.index[:: max(len(monthly_returns) // 12, 1)]], rotation=45
    )

    if title is not None:
        plt.title(title)

    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def calculate_adverse_rebalance_metrics(position, price_df, return_as_dict=True, long_empty_mode=False):
    _pos = position.copy()

    if "detail" not in _pos.columns:
        holding_asset_position = _pos.copy().reset_index()
        holding_asset_position["ratio"] = 1
        holding_asset_position = holding_asset_position.pivot(
            index="time", columns="holding_asset", values="ratio"
        ).fillna(0)
    else:
        holding_asset_position = _pos["detail"].apply(lambda x: pd.Series(x.to_dict())).fillna(0.0)

    # 获得调仓阶段 # period_returns['start_time'].apply(lambda x: holding_asset_position.loc[x])
    _, rebalance_df, period_returns = calculate_rebalance_win_rate(_pos)

    real_period_position = period_returns.set_index("period")["start_time"].apply(
        lambda x: holding_asset_position.loc[x]
    )
    adverse_period_position = real_period_position.shift(1).fillna(0)
    assets_daily_returns = price_df[adverse_period_position.columns].pct_change().fillna(0)
    # 将上一期的asset替换成下一期的asset，也就是period+1然后替换
    replace_period_action = period_returns[["period", "holding_asset"]].copy()
    replace_period_action["period"] = replace_period_action["period"] + 1
    replace_dict = replace_period_action.set_index("period").to_dict()["holding_asset"]
    # 生成adverse_df
    adverse_df = rebalance_df[["time", "holding_asset", "period"]].copy()
    adverse_df["holding_asset"] = adverse_df["period"].map(replace_dict).fillna("empty")

    adverse_df_holding_percent = adverse_df.set_index("time")["period"].apply(lambda x: adverse_period_position.loc[x])
    adverse_df["daily_return"] = (
        adverse_df_holding_percent.mul(assets_daily_returns.reindex(adverse_df_holding_percent.index))
        .sum(axis=1)
        .reset_index(drop=True)
    )

    # daily_return_df = price_df.copy().pct_change().dropna()
    # if "empty" not in daily_return_df.columns:
    #     daily_return_df.insert(0, "empty", 0.0)
    # adverse_df["daily_return"] = adverse_df.apply(
    #     lambda x: (daily_return_df.loc[x["time"], x["holding_asset"]] if pd.notnull(x.holding_asset) else np.nan),
    #     axis=1,
    # )

    if long_empty_mode:
        adverse_df["holding_asset"] = adverse_df["holding_asset"].map(lambda x: "empty" if x == "empty" else "long")
        period_returns["holding_asset"] = period_returns["holding_asset"].map(
            lambda x: "empty" if x == "empty" else "long"
        )

    def compound_return(returns):
        return (returns + 1).prod() - 1

    adverse_period_returns = adverse_df.groupby("period", as_index=False).agg(
        start_time=("time", "first"),
        holding_asset=("holding_asset", "first"),
        total_return=("daily_return", compound_return),
    )

    # adverse_period_returns 是如果延续上期决策，当期的收益； period_returns 是如果当期决策，当期的收益
    # 增加计算对抗超额收益的相关代码
    adverse_excess_df = pd.merge(
        adverse_period_returns, period_returns, on=["period", "start_time"], suffixes=("_adverse", "_current")
    )
    adverse_excess_df["tag"] = (
            adverse_excess_df["holding_asset_adverse"] + "-" + adverse_excess_df["holding_asset_current"]
    )
    adverse_excess_df["excess_return"] = (
            adverse_excess_df["total_return_current"] - adverse_excess_df["total_return_adverse"]
    )
    adverse_excess_df = adverse_excess_df[["period", "tag", "excess_return"]].dropna()
    adverse_excess_df = adverse_excess_df.groupby("tag").agg(
        mean_adverse_excess_return=("excess_return", "mean"),
        sum_adverse_excess_return=("excess_return", "sum"),
    )

    # 计算胜率
    working_period_df = period_returns.copy()
    working_period_df["tag"] = adverse_period_returns["holding_asset"] + "-" + period_returns["holding_asset"]
    working_period_df["if_win"] = period_returns.total_return > adverse_period_returns.total_return
    working_period_df = working_period_df.iloc[1:]  # 删除第一个period，因为没有对比
    working_period_df = working_period_df.set_index("period", drop=False)
    result = {}
    result["adverse_rebalance_win_cnt"] = working_period_df.groupby("holding_asset")["if_win"].sum()
    result["adverse_rebalance_total_cnt"] = working_period_df.groupby("holding_asset")["if_win"].size()
    result["adverse_rebalance_win_rate"] = result["adverse_rebalance_win_cnt"] / result["adverse_rebalance_total_cnt"]

    # 补充根据tag的详细对抗调仓胜率
    result["detail_adverse_rebalance_win_cnt"] = working_period_df.groupby("tag")["if_win"].sum()
    result["detail_adverse_rebalance_total_cnt"] = working_period_df.groupby("tag")["if_win"].size()
    result["detail_adverse_rebalance_win_rate"] = (
            result["detail_adverse_rebalance_win_cnt"] / result["detail_adverse_rebalance_total_cnt"]
    )

    result = pd.DataFrame(result)

    if not return_as_dict:
        result = pd.concat([result, adverse_excess_df], axis=1)
        return result

    result = result.to_dict()
    result.update(adverse_excess_df.to_dict())
    return result


def t_test_multi_ic(ic_list, alternative="two-sided"):
    """
    检验多期IC均值的显著性
    :param ic_list: 各期的IC/RIC列表（例如过去12个月的值）
    :param alternative: 检验方向
    :return: t值, p值
    """
    t_stat, p_value = stats.ttest_1samp(ic_list, popmean=0, alternative=alternative)
    return t_stat, p_value


def map_position_to_signal(position):
    # 解析position表，得到资产的信号强度，如果有detail列，则会按照detail中的持仓权重转换为信号, 否则就是one hot权重
    assets = position.holding_asset.unique()
    signal = pd.DataFrame(0.0, index=position.index, columns=assets)
    for i, row in position.iterrows():
        if "detail" in position.columns:
            for asset, weight in row["detail"].items():
                signal.loc[i, asset] = weight
        else:
            signal.loc[i, row["holding_asset"]] = 1
    return signal


def vectorize_ic_calculation(signal_df, returns_df, ic_type="ic", return_as_list=False):
    """
    向量化计算 IC 和 RIC, ic_type 可选 ic 或 ric
    """
    if ic_type == "ric":
        signal = signal_df.rank(axis=1, method="average").values
        returns = returns_df.rank(axis=1, method="average").values
    else:
        signal = signal_df.values
        returns = returns_df.values

    signal_mean = signal.mean(axis=1, keepdims=True)
    returns_mean = returns.mean(axis=1, keepdims=True)
    signal_centered = signal - signal_mean
    return_centered = returns - returns_mean

    n = signal_centered.shape[1]  # 每行的样本数
    covariance = np.sum(signal_centered * return_centered, axis=1) / (n - 1)
    std_signal = np.sqrt(np.sum(signal_centered ** 2, axis=1) / (n - 1))
    std_return = np.sqrt(np.sum(return_centered ** 2, axis=1) / (n - 1))

    ic = covariance / (std_signal * std_return)
    if return_as_list:
        ic = ic.tolist()
    return ic


def calculate_ic_metrics(price_df, position=None, signals=None, shift=0, alternative="greater"):
    """
    计算IC-mean、RIC-mean、IC-std、RIC-std 以及 对应的T检验t值和P值

    可以输入position或者signals，不能同时为None，在已经输入了position的情况下，signals参数会被忽略

    shift是预测的滞后，默认使用调仓表里的holding asset，是没有滞后的。
    """
    assert position is not None or signals is not None, "position和signals至少有一个不为空"
    if position is not None and signals is not None:
        warnings.warn("signals参数将被忽略")

    if signals is None:
        signals = map_position_to_signal(position)

    signals = signals.shift(shift).dropna()
    daily_returns = price_df[signals.columns.to_list()].pct_change().reindex(signals.index).dropna()
    signals = signals.reindex(daily_returns.index)  # 保证去除Na后index一致

    # signal_array = signals.values
    # daily_return_array = daily_returns.values

    # ic_list = []
    # ric_list = []
    # for i in range(1, len(signals)):
    #     ic = np.corrcoef(signal_array[i], daily_return_array[i])[0, 1]
    #     ic_list.append(ic)
    #     ric = stats.spearmanr(signal_array[i], daily_return_array[i])[0]
    #     ric_list.append(ric)
    ic_list = vectorize_ic_calculation(signals, daily_returns, ic_type="ic")
    ric_list = vectorize_ic_calculation(signals, daily_returns, ic_type="ric")

    ic_mean = np.mean(ic_list)
    ic_std = np.std(ic_list)
    ric_mean = np.mean(ric_list)
    ric_std = np.std(ric_list)
    t_ic, p_ic = t_test_multi_ic(ic_list, alternative)
    t_ric, p_ric = t_test_multi_ic(ric_list, alternative)
    return {
        "IC-mean": ic_mean,
        "IC-std": ic_std,
        "IC-t": t_ic,
        "IC-p": p_ic,
        "RIC-mean": ric_mean,
        "RIC-std": ric_std,
        "RIC-t": t_ric,
        "RIC-p": p_ric,
    }


if __name__ == "__main__":
    # 测试代码
    # 准备因子数据
    from nmw.FactorCollection import FactorCollection

    # 准备原始价格数据与tester
    from nmw.backtest_utilts_new import *
    import pandas as pd
    from config import BASEDIR
    from pathlib import Path
    import json
    import pandas as pd

    set_chinese_font()

    # factor_collection = FactorCollection()
    # factor_collection.update_all()

    # factor_df = factor_collection.load_factor_df(["talib", "alpha101", "alpha158", "alpha360"])
    # factor_df["factor_tag"] = factor_df["factor_set"] + "." + factor_df["factor"]  # 因子标签

    native_price, price_df, open_price_df, working_days = prepare_price_data(
        data_path="data/fof_price_updating.parquet"
    )

    # native_price = native_price.query('time <= @end_date')
    # price_df = price_df[:end_date]

    zz800 = price_df["000906.SH"]
    tester = NMWBacktester(price_df)

    import joblib

    ob = joblib.load("nmw/wjl/outputs/滚动res/1000_20日.pkl")
    res_pos = {}
    res_pos["000852.SH_avg_10_20"] = ob["000852.SH_average_10"]

    res_pos["000852.SH_combo_10_20"] = ob["000852.SH_combo_10"]
    ob = joblib.load("nmw/wjl/outputs/滚动res/500_20日.pkl")
    res_pos["000905.SH_avg_10_20"] = ob["000905.SH_average_10"]
    res_pos["000905.SH_combo_10_20"] = ob["000905.SH_combo_10"]
    # res = calculate_adverse_rebalance_metrics(res_pos["000852.SH_avg_10_20"], price_df, return_as_dict=True, long_empty_mode=True)
    for key in res_pos:
        res_pos[key] = res_pos[key].reset_index().drop_duplicates(subset=["time"], keep="first").set_index("time")
        res_pos[key] = clip_time_range(res_pos[key], end_date="2025-08-08")
        res_pos[key]["nv"] = res_pos[key]["daily_return"].add(1).cumprod()

    # tester.backtest(res_pos['000852.SH_avg_10_20']['holding_asset'].shift(-2)['2015-05-05':], D=5, skip_empty_warehouse=True, start_date=pd.Timestamp('2015-05-05'))
    pos = merge_position(
        [
            res_pos["000852.SH_avg_10_20"]["2021-01-01":"2025-08-08"],
            res_pos["000905.SH_avg_10_20"]["2021-01-01":"2025-08-08"],
        ]
    )
    res = measure_position(pos, tester, win_rate_metric=True, long_empty_mode=True, rebalance_metric=True)
    pass
