"""README 中描述的 CSS 聚类辅助函数。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_factors_kmeans(
    factor_matrix: "np.ndarray", k: int
) -> tuple[list[list[int]], "np.ndarray", "np.ndarray"]:
    """使用 KMeans 对因子暴露做聚类（README “CSS Context Assembly”）。

    Args:
        factor_matrix: README CSS 小节中提到的因子矩阵。
        k: 目标聚类数量，对应 defaults.yaml 中的 ``k``。

    Returns:
        (簇列表, 簇心, 标准化后的因子矩阵)。
    """

    if factor_matrix.size == 0:
        empty = np.zeros((0, 0))
        return [], empty, empty

    safe_matrix = _prepare_matrix(factor_matrix)
    n_obs = safe_matrix.shape[1]
    n_clusters = max(1, min(k, n_obs))
    if n_clusters == 1 or n_obs == 1:
        clusters = [list(range(n_obs))]
        centers = safe_matrix[:, :1].T if safe_matrix.size else np.zeros((1, safe_matrix.shape[0]))
        return clusters, centers, safe_matrix

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(safe_matrix.T)
    raw_clusters = [np.where(labels == idx)[0].tolist() for idx in range(n_clusters)]
    clusters = [cluster for cluster in raw_clusters if cluster]
    centers = []
    for members in clusters:
        cluster_matrix = safe_matrix[:, members]
        centers.append(cluster_matrix.mean(axis=1))
    centers_array = np.vstack(centers) if centers else np.zeros((0, safe_matrix.shape[0]))
    centers = centers_array
    return clusters, centers, safe_matrix


def select_cross_samples(
    clusters: list[list[int]],
    n_select: int,
    *,
    seed: int | None = None,
    ric_scores: list[float] | None = None,
) -> list[int]:
    """执行论文中的随机化跨簇采样流程。

    先从每个簇随机抽取一个因子，得到组合 ``FC``；再从中随机挑选 ``l`` 个作为 CSS 上下文。
    """

    if not clusters or n_select <= 0:
        return []

    rng = np.random.default_rng(seed)
    factor_combo: list[int] = []
    for members in clusters:
        if not members:
            continue
        if ric_scores is not None:
            # 先按 RIC 值排序，截取簇内 RIC 最大的 Top3，再随机抽 1 个
            scored = [(ric_scores[idx] if idx < len(ric_scores) else None, idx) for idx in members]
            scored = [(s, i) for s, i in scored if s is not None]
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                top = [i for _, i in scored[:3]]
                picked = int(rng.choice(top))
                if picked not in factor_combo:
                    factor_combo.append(picked)
                continue
        # 原先的全随机逻辑（保留注释以便回滚）
        # idx = int(rng.integers(0, len(members)))
        # picked = members[idx]
        # if picked not in factor_combo:
        #     factor_combo.append(picked)
        idx = int(rng.integers(0, len(members)))
        picked = members[idx]
        if picked not in factor_combo:
            factor_combo.append(picked)

    if not factor_combo:
        return []

    l = min(n_select, len(factor_combo))
    if l == len(factor_combo):
        return factor_combo

    selected_idx = rng.choice(len(factor_combo), size=l, replace=False)
    return [factor_combo[int(i)] for i in selected_idx]


def _prepare_matrix(matrix: "np.ndarray") -> "np.ndarray":
    """对因子矩阵进行裁剪与标准化，避免 KMeans 出现溢出。"""

    safe = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    safe = np.clip(safe, -1e6, 1e6)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(safe)
    return np.nan_to_num(scaled, nan=0.0)
