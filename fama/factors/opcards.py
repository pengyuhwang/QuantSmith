"""LLM 提示中可用的算子语义卡片。"""

from __future__ import annotations

OP_CARDS: dict[str, str] = {
    # "RANK": "RANK(x): 按日期对所有标的做截面排名，输出范围 [0,1]，并列取平均。",
    "DELTA": "DELTA(x, n=1): 每个标的的时间差分 x_t - x_{t-n}；n 默认 1，建议显式写整数常量。",
    "DELAY": "DELAY(x, n=1): 将序列向后平移 n 个 bar；n 默认 1，建议显式写整数常量。",
    "TS_MEAN": "TS_MEAN(x, n): 每个标的最近 n 个 bar (含当期) 的时间序列均值，历史不足返回 NA。",
    "TS_SUM": "TS_SUM(x, n): 每个标的最近 n 个 bar 的滚动求和。",
    "TS_STDDEV": "TS_STDDEV(x, n): 每个标的最近 n 个 bar (含当期) 的时间序列标准差 (总体)，历史不足返回 NA。",
    "TS_MIN": "TS_MIN(x, n): 最近 n 个 bar 的滚动最小值。",
    "TS_MAX": "TS_MAX(x, n): 最近 n 个 bar 的滚动最大值。",
    "TS_PRODUCT": "TS_PRODUCT(x, n): 最近 n 个 bar 的连乘结果（适合累积收益）。",
    "TS_ARGMAX": "TS_ARGMAX(x, n): 最近 n 个 bar 内最大值所在的相对位置（0 为窗口起点）。",
    "TS_ARGMIN": "TS_ARGMIN(x, n): 最近 n 个 bar 内最小值所在的相对位置。",
    "TS_RANK": "TS_RANK(x, n): 当前值在最近 n 个 bar 中的排序百分位。",
    "CORREL": "CORREL(x, y, n): 每个标的最近 n 个 bar 的皮尔逊相关系数，范围 [-1, 1]，历史不足返回 NA。",
    "COVAR": "COVAR(x, y, n): 最近 n 个 bar 的协方差，用于衡量联合波动。",
    "SIGN": "SIGN(x): 元素级符号函数，输出 {-1, 0, 1}。",
    "ABS": "ABS(x): 元素级绝对值。",
    "DECAY_LINEAR": "DECAY_LINEAR(x, n): 线性加权的滚动平均，越近的 bar 权重越大。",
    # "SCALE": "SCALE(x): 当日截面标准化，使 ∑|x| = 1，用于消除量纲。",
    "IF": "IF(cond, a, b): 条件选择，cond≠0 取 a，否则取 b。",
    "AND": "AND(a, b): 元素级逻辑与，a、b 同时非零返回 1，否则 0。",
    "OR": "OR(a, b): 元素级逻辑或，任一非零返回 1，否则 0。",
    "NOT": "NOT(a): 元素级逻辑非，a 为 0 返回 1，否则 0。",
    "GT": "GT(x, y): 大于比较，返回 {0,1}；布尔比较请统一使用 GT/GE/LT/LE/EQ，不要直接写 >、<、>=、<=。",
    "GE": "GE(x, y): 大于等于比较，返回 {0,1}。",
    "LT": "LT(x, y): 小于比较，返回 {0,1}。",
    "LE": "LE(x, y): 小于等于比较，返回 {0,1}。",
    "EQ": "EQ(x, y): 相等比较，返回 {0,1}。",
    "MAX": "MAX(x, y): 元素级最大值，可用于上下限保护。",
    "MIN": "MIN(x, y): 元素级最小值。",
    "REPLACE_NAN_INF": "REPLACE_NAN_INF(x, v=0): 将 x 中的 NaN/Inf 替换为常数 v（默认 0）。",
    "ADV": "ADV(volume, n): 最近 n 日平均成交量/金额（具体由输入决定），可衡量流动性。",
    "SAFE_DIV": "SAFE_DIV(n, d, eps=1e-4, fill=0): 对 n/d 做除法，若分母接近 0 用 eps 代替，再用 fill 填补 NaN/Inf；建议使用位置参数形式。",
    "CLIP": "CLIP(x, eps=1): 将 x 限制在 [-eps, eps]，常用于抑制极值（默认 eps=1）。",
    "EMA": "EMA(x, n): 指数加权移动平均（span=n），与 pandas ewm(span=n) 一致。",
    "EXP_MOVING_AVG": "EXP_MOVING_AVG(x, n): 同 EMA，指数加权滚动平均。",
    "POW": "POW(x, y): 元素级幂运算，支持常数指数。",
    "FAST_TS_SUM": "FAST_TS_SUM(x, n): 高性能实现的滚动求和，语义等同 TS_SUM。",
    "TS_QUANTILE": "TS_QUANTILE(x, n, q): 滚动分位数，返回窗口内分位点 q（0~1）。",
    "TS_KURT": "TS_KURT(x, n): 窗口内无偏峰度，衡量尖峰程度。",
    "TS_SKEW": "TS_SKEW(x, n): 窗口内无偏偏度，衡量对称性。",
    "TS_MAXDRAWDOWN": "TS_MAXDRAWDOWN(x, n): 窗口内最大回撤 (max drawdown) 值。",
    "TS_LINEAR_REGRESSION_R2": "TS_LINEAR_REGRESSION_R2(x, n, y=None): 滚动线性回归的 R^2（拟合优度）。",
    "TS_LINEAR_REGRESSION_SLOPE": "TS_LINEAR_REGRESSION_SLOPE(x, n, y=None): 滚动线性回归斜率。",
    "TS_LINEAR_REGRESSION_RESI": "TS_LINEAR_REGRESSION_RESI(x, n, y=None): 滚动线性回归残差序列。",
    # "DIFF_WITH_WEIGHTED_SUM": "DIFF_WITH_WEIGHTED_SUM(v, w): 先计算截面加权和 v⋅w，再返回 v - 加权和，常用于剔除市场/行业暴露。",
}


def render_cards(ops: list[str]) -> str:
    """将算子白名单渲染为多段文本。"""

    lines: list[str] = []
    for op in ops:
        card = OP_CARDS.get(op)
        if not card:
            continue
        lines.append(f"### {op}\n{card}")
    return "\n\n".join(lines)
