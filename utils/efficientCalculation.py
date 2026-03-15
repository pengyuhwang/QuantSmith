import pandas as pd
import numpy as np
import numba
from typing import Union


class EfficientCalculator:
    def __init__(self):
        pass

    def efficient_cal_ic(self, x, y, method="pearson", handle_nan=True):
        """
        Universal function to calculate Information Coefficient (correlation)
        for pandas Series or NumPy arrays with various methods.

        Args:
            x (pd.Series or np.ndarray): The first series.
            y (pd.Series or np.ndarray): The second series.
            method (str): Correlation method - "spearman" (default) or "pearson"
            handle_nan (bool): Whether to ignore NaN values pair-wise (default True)

        Returns:
            float: The correlation coefficient (IC).
        """
        # Convert pandas Series to NumPy arrays if necessary
        x_vals = x.values if isinstance(x, pd.Series) else x
        y_vals = y.values if isinstance(y, pd.Series) else y

        return _calculate_correlation_numba(x_vals, y_vals, method, handle_nan)

    def efficent_cal_ric(self, x, y, handle_nan=True):
        """
        Legacy function name for backward compatibility.
        Calculates Spearman correlation (RIC) with NaN handling.
        """
        return self.efficient_cal_ic(x, y, method="spearman", handle_nan=handle_nan)

    def efficient_expanding_rank_pct_series(self, series: pd.Series) -> pd.Series:
        value = expanding_rank_pct_numba(series.to_numpy())
        return pd.Series(value, index=series.index)

    def efficient_expanding_rank_pct_window_series(self, series: pd.Series, window: int) -> pd.Series:
        if window < 0:
            return self.efficient_expanding_rank_pct_series(series)
        value = expanding_rank_pct_window_numba(series.to_numpy(), window)
        return pd.Series(value, index=series.index)

    def efficient_rank_with_ties(
            self, arr: Union[pd.DataFrame, pd.Series, np.ndarray], reverse: bool = False, axis: int = 0, pct=False
    ) -> np.ndarray:
        """
        Calculate ranks with ties using Numba for acceleration.

        reverse: Whether to reverse the ranks (default False, ie ascending order)
        """
        if isinstance(arr, pd.Series):
            arr = arr.to_numpy()

        return _rank_with_ties_along_axis(arr, reverse=reverse, axis=axis, pct=pct)

    def efficient_bin_equal_width(self, arr: Union[pd.DataFrame, pd.Series, np.ndarray], bins: int):
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            arr = arr.to_numpy()
        return _bin_equal_width(arr, bins)

    def efficient_cal_icir(
        self,
        x,
        y,
        method="pearson",
        handle_nan=True,
        window: int = 60,
        min_periods: int = 20,
        ddof: int = 1,
    ):
        """基于滚动局部 IC 序列计算 ICIR，返回 mean(IC_t) / std(IC_t)。"""
        x_vals = x.values if isinstance(x, pd.Series) else np.asarray(x)
        y_vals = y.values if isinstance(y, pd.Series) else np.asarray(y)
        x_vals = np.asarray(x_vals, dtype=np.float64).reshape(-1)
        y_vals = np.asarray(y_vals, dtype=np.float64).reshape(-1)

        if x_vals.shape[0] != y_vals.shape[0]:
            raise ValueError("Input arrays must have the same length")

        if handle_nan:
            mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            x_vals = x_vals[mask]
            y_vals = y_vals[mask]

        n = len(x_vals)
        if n < 2:
            return np.nan

        if window <= 1:
            raise ValueError(f"window must be greater than 1, got {window}")
        if min_periods <= 1:
            raise ValueError(f"min_periods must be greater than 1, got {min_periods}")

        window = min(window, n)
        min_periods = min(min_periods, window)

        local_ics = []
        for end in range(min_periods, n + 1):
            start = max(0, end - window)
            corr = _calculate_correlation_numba(x_vals[start:end], y_vals[start:end], method, False)
            if not np.isnan(corr):
                local_ics.append(corr)

        if len(local_ics) <= ddof:
            return np.nan

        ic_series = np.asarray(local_ics, dtype=np.float64)
        ic_std = np.nanstd(ic_series, ddof=ddof)
        if np.isnan(ic_std) or ic_std == 0:
            return np.nan

        return float(np.nanmean(ic_series) / ic_std)


@numba.jit(nopython=True)
def _bin_equal_width(arr, bins):
    """
    对 NumPy 数组进行等宽分箱。

    参数:
        arr (np.ndarray): 输入的 NumPy 数组，可以是多维的。
        bins (int): 需要划分的箱数。

    返回:
        tuple: 一个包含两个元素的元组:
            - bin_values (np.ndarray): 形状为 (bins,) 的数组，代表每个箱的中点值。
            - h_bin_indices (np.ndarray): 与 arr 形状相同的数组，包含每个元素对应的分箱索引 (0 到 bins-1)。
    """
    # 处理空数组的情况
    if arr.size == 0:
        return np.empty(bins, dtype=np.float64), np.empty(arr.shape, dtype=np.int64)

    min_val = np.min(arr)
    max_val = np.max(arr)

    # 先将索引数组展平以便于循环赋值，最后再恢复原状
    bin_indices = np.empty(arr.shape, dtype=np.int64).ravel()
    bin_values = np.empty(bins, dtype=np.float64)
    flat_arr = arr.ravel()

    # 特殊情况：如果数组中所有值都相等
    if min_val == max_val:
        bin_indices[:] = 0  # 所有元素的索引都为 0
        bin_values[:] = min_val  # 所有箱的值都为该单一值
        return bin_values, bin_indices.reshape(arr.shape)

    # 1. 计算每个箱的宽度
    bin_width = (max_val - min_val) / bins

    # 2. 计算每个箱的代表值（使用区间中点）
    for i in range(bins):
        bin_edge_start = min_val + i * bin_width
        bin_edge_end = min_val + (i + 1) * bin_width
        bin_values[i] = (bin_edge_start + bin_edge_end) / 2.0

    # 3. 为输入数组中的每个元素分配分箱索引
    for i in range(flat_arr.shape[0]):
        val = flat_arr[i]

        # 特殊处理最大值，确保它能正确归入最后一个箱
        if val == max_val:
            idx = bins - 1
        else:
            # 根据值计算其所在的箱索引
            # Numba 会将这个循环编译成高效的机器码
            idx = int((val - min_val) / bin_width)

        bin_indices[i] = idx

    # 将索引数组的形状恢复为与输入数组一致
    return bin_values, bin_indices.reshape(arr.shape)


@numba.jit(nopython=True)
def _rank_with_ties_along_axis(arr, reverse=False, axis=0, pct=False):
    """
    沿多维数组的指定维度(axis)计算排名。

    参数:
        arr (np.ndarray): 输入的多维数组。
        axis (int):        要沿着哪个维度进行排名。默认为 0。
        reverse (bool):    是否按降序排名。False为升序，True为降序。

    返回:
        np.ndarray: 一个与 arr 形状相同的数组，包含了排名结果。
    """
    # 1. 交换 (Swap): 将指定维度换到最后
    swapped = np.swapaxes(arr, axis, -1)

    # 2. 重塑 (Reshape): 将数据变为二维，其中每一行是需要排名的一维向量
    original_shape = swapped.shape
    reshaped = swapped.copy().reshape(-1, original_shape[-1])

    # 准备一个空的二维数组来存放排名结果
    ranks_reshaped = np.empty(reshaped.shape, dtype=np.float64)

    # 3. 计算 (Compute): 遍历每一行，调用1D排名函数
    for i in range(reshaped.shape[0]):
        ranks_reshaped[i] = _rank_with_ties(reshaped[i], reverse, pct=pct)

    # 4. 还原 (Un-reshape & Un-swap): 将结果恢复到原始形状并换回轴
    ranks_swapped = ranks_reshaped.reshape(original_shape)
    final_ranks = np.swapaxes(ranks_swapped, -1, axis)

    return final_ranks


@numba.jit(nopython=True)
def _rank_with_ties(arr: np.ndarray, reverse: bool = False, pct=False) -> np.ndarray:
    """
    Calculate ranks with average method for ties (same as pandas default).
    """
    if reverse:
        # Reverse the array for descending ranks
        arr = -arr

    n = len(arr)
    ranks = np.empty(n, dtype=np.float64)

    # Get sorted indices
    sorted_indices = np.argsort(arr)

    i = 0
    while i < n:
        # Find the range of equal values
        j = i
        while j < n and arr[sorted_indices[j]] == arr[sorted_indices[i]]:
            j += 1

        # Calculate average rank for this group of tied values
        # Ranks are 1-based, so we add 1
        avg_rank = (i + j + 1) / 2.0

        # Assign average rank to all tied values
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank

        i = j

    if pct:
        ranks = ranks / ranks.max()

    return ranks


@numba.jit(nopython=True)
def _pearson_correlation_numba(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient using Numba for acceleration.
    """
    n = len(x)
    if n < 2:
        return np.nan

    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate correlation
    numerator = np.sum((x - x_mean) * (y - y_mean))

    x_std_dev = np.sqrt(np.sum((x - x_mean) ** 2))
    y_std_dev = np.sqrt(np.sum((y - y_mean) ** 2))

    denominator = x_std_dev * y_std_dev

    if denominator == 0:
        return np.nan

    return numerator / denominator


@numba.jit(nopython=True)
def _arr_flatten_and_handle(x: np.ndarray, y: np.ndarray, handle_nan: bool = True):
    """
    Preprocess input arrays: flatten, convert to float64, and optionally handle NaNs.

    Returns:
        tuple: (x_processed, y_processed, n_valid)
    """
    # Ensure arrays are 1D float64
    x_flat = x.flatten().astype(np.float64)
    y_flat = y.flatten().astype(np.float64)

    if x_flat.shape[0] != y_flat.shape[0]:
        raise ValueError("Input arrays must have the same length")

    if handle_nan:
        # Filter out corresponding NaN values
        valid_mask = ~np.isnan(x_flat) & ~np.isnan(y_flat)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        return x_valid, y_valid, len(x_valid)
    else:
        return x_flat, y_flat, len(x_flat)


@numba.jit(nopython=True)
def _calculate_correlation_numba(
        x: np.ndarray, y: np.ndarray, correlation_type: str = "spearman", handle_nan: bool = True
) -> float:
    """
    Calculate correlation coefficient for two NumPy arrays using Numba.

    Args:
        x, y: Input arrays
        correlation_type: "spearman" for rank correlation, "pearson" for linear correlation
        handle_nan: Whether to ignore NaN values pair-wise

    Returns:
        float: Correlation coefficient
    """
    x_processed, y_processed, n = _arr_flatten_and_handle(x, y, handle_nan)

    if n < 2:
        return np.nan

    if correlation_type == "spearman":
        # Calculate ranks with average method for ties
        x_rank = _rank_with_ties(x_processed)
        y_rank = _rank_with_ties(y_processed)
        return _pearson_correlation_numba(x_rank, y_rank)
    elif correlation_type == "pearson":
        return _pearson_correlation_numba(x_processed, y_processed)
    else:
        raise ValueError("correlation_type must be 'spearman' or 'pearson'")


@numba.jit(nopython=True)
def expanding_rank_pct_numba(arr: np.ndarray) -> np.ndarray:
    """
    使用 Numba JIT 加速计算扩展窗口的百分比排名。
    这与 pandas.Series.expanding().apply(lambda x: x.rank().iloc[-1] / len(x)) 的逻辑等效。

    参数:
    - arr: 输入的 NumPy 数组。

    返回:
    - 一个新的 NumPy 数组，包含每个点的扩展百分比排名。
    """
    n = len(arr)
    # 初始化一个空的浮点数数组来存储结果
    output = np.empty(n, dtype=np.float64)

    # 遍历输入数组的每个元素
    for i in range(n):
        # 当前窗口是 arr[0] 到 arr[i]
        window_size = i + 1
        current_value = arr[i]
        # 在当前窗口内，计算比 current_value 小和相等的元素数量
        less_count = 0
        equal_count = 0
        for j in range(window_size):
            if arr[j] < current_value:
                less_count += 1
            elif arr[j] == current_value:
                equal_count += 1
        # 计算排名(method='average')
        # 排名 = (小于当前值的数量) + 1 (这是最小排名) + (等于当前值的数量 - 1) / 2
        rank = (less_count + 1) + (equal_count - 1) / 2.0
        # 计算百分比排名并存入输出数组
        output[i] = rank / window_size
    return output


@numba.jit(nopython=True)
def expanding_rank_pct_window_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    使用 Numba JIT 加速计算滚动窗口的百分比排名。
    计算当前数值处于最近window窗口内数值的什么分位数。

    参数:
    - arr: 输入的 NumPy 数组。
    - window: 滚动窗口大小。

    返回:
    - 一个新的 NumPy 数组，包含每个点在滚动窗口内的百分比排名。前window-1个点返回NaN。
    """
    n = len(arr)
    # 初始化一个空的浮点数数组来存储结果
    output = np.empty(n, dtype=np.float64)

    # 前window-1个点设置为NaN
    for i in range(min(window - 1, n)):
        output[i] = np.nan

    # 从第window个点开始计算
    for i in range(window - 1, n):
        # 当前窗口是 arr[i-window+1] 到 arr[i]
        window_start = i - window + 1
        current_value = arr[i]

        # 在当前窗口内，计算比 current_value 小和相等的元素数量
        less_count = 0
        equal_count = 0
        for j in range(window_start, i + 1):
            if arr[j] < current_value:
                less_count += 1
            elif arr[j] == current_value:
                equal_count += 1

        # 计算排名(method='average')
        # 排名 = (小于当前值的数量) + 1 (这是最小排名) + (等于当前值的数量 - 1) / 2
        rank = (less_count + 1) + (equal_count - 1) / 2.0
        # 计算百分比排名并存入输出数组
        output[i] = rank / window

    return output
