#!/usr/bin/env python
"""Render per-round generation/pass summary as a PNG bar+line chart."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from fama.utils.io import ensure_dir
except ModuleNotFoundError:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from fama.utils.io import ensure_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepare_round_frame(round_memory_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(round_memory_csv)
    if frame.empty:
        raise ValueError(f"round memory csv is empty: {round_memory_csv}")
    required = {"generated_count", "success_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"round memory csv missing columns: {sorted(missing)}")

    work = frame.copy()
    work["generated_count"] = pd.to_numeric(work["generated_count"], errors="coerce").fillna(0)
    work["success_count"] = pd.to_numeric(work["success_count"], errors="coerce").fillna(0)
    if "pass_rate" in work.columns:
        work["pass_rate"] = pd.to_numeric(work["pass_rate"], errors="coerce")
    else:
        work["pass_rate"] = pd.NA
    computed_pass_rate = work["success_count"] / work["generated_count"].replace(0, pd.NA)
    work["pass_rate"] = work["pass_rate"].fillna(computed_pass_rate).fillna(0.0)
    work = work.reset_index(drop=True)
    work["seq_index"] = work.index + 1
    if work.empty:
        raise ValueError("No valid rows found in round memory csv.")
    return work


def _sparse_round_ticks(seq_index: list[int], every: int = 5) -> tuple[list[int], list[str]]:
    if not seq_index:
        return [], []
    positions: list[int] = []
    labels: list[str] = []
    for idx, seq_num in enumerate(seq_index):
        if idx == 0 or idx == len(seq_index) - 1 or seq_num % every == 0:
            positions.append(idx)
            labels.append(str(seq_num))
    deduped_positions: list[int] = []
    deduped_labels: list[str] = []
    seen: set[int] = set()
    for pos, label in zip(positions, labels):
        if pos in seen:
            continue
        seen.add(pos)
        deduped_positions.append(pos)
        deduped_labels.append(label)
    return deduped_positions, deduped_labels


def build_round_memory_progress_graph(
    *,
    round_memory_csv: Path,
    output_png: Path,
) -> tuple[Path, int]:
    try:
        mpl_cache_dir = PROJECT_ROOT / "tmp" / "mplconfig"
        ensure_dir(str(mpl_cache_dir))
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to import matplotlib: {exc}") from exc

    frame = _prepare_round_frame(round_memory_csv)
    ensure_dir(str(output_png.parent))

    seq_index = frame["seq_index"].tolist()
    generated = frame["generated_count"].tolist()
    success = frame["success_count"].tolist()
    pass_rate = frame["pass_rate"].tolist()

    x = np.arange(len(frame))
    width = 0.38
    fig_w = max(10.0, min(24.0, 6.5 + len(frame) * 0.34))
    fig, ax_count = plt.subplots(figsize=(fig_w, 6.0))
    fig.patch.set_facecolor("#FFFFFF")
    ax_count.set_facecolor("#FFFFFF")

    bars_generated = ax_count.bar(
        x - width / 2,
        generated,
        width=width,
        color="#BFD7EA",
        edgecolor="#8AA9C4",
        linewidth=0.8,
        label="Generated",
        zorder=2,
    )
    bars_success = ax_count.bar(
        x + width / 2,
        success,
        width=width,
        color="#4F6FA5",
        edgecolor="#35527E",
        linewidth=0.8,
        label="Passed",
        zorder=2.1,
    )

    ax_rate = ax_count.twinx()
    line_pass_rate = ax_rate.plot(
        x,
        pass_rate,
        color="#D1495B",
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        label="Pass Rate",
        zorder=3,
    )[0]

    ax_count.set_title("Per-Record Factor Generation Summary", fontsize=12, pad=12, color="#1A2B44")
    ax_count.set_xlabel("Record Index")
    ax_count.set_ylabel("Factor Count")
    ax_rate.set_ylabel("Pass Rate")
    ax_rate.set_ylim(0.0, max(1.0, max(pass_rate) * 1.15 if pass_rate else 1.0))
    tick_positions, tick_labels = _sparse_round_ticks(seq_index, every=5)
    ax_count.set_xticks(tick_positions)
    ax_count.set_xticklabels(tick_labels, rotation=0)
    ax_count.grid(axis="y", color="#E7EDF5", linewidth=0.8, zorder=0)

    handles = [bars_generated, bars_success, line_pass_rate]
    labels = ["Generated", "Passed", "Pass Rate"]
    ax_count.legend(handles, labels, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(output_png, format="png", dpi=220)
    plt.close(fig)
    return output_png, len(frame)


def build_cumulative_success_graph(
    *,
    round_memory_csv: Path,
    output_png: Path,
) -> tuple[Path, int, int]:
    try:
        mpl_cache_dir = PROJECT_ROOT / "tmp" / "mplconfig"
        ensure_dir(str(mpl_cache_dir))
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to import matplotlib: {exc}") from exc

    frame = _prepare_round_frame(round_memory_csv)
    ensure_dir(str(output_png.parent))

    seq_index = frame["seq_index"].tolist()
    cumulative_success = frame["success_count"].cumsum().tolist()
    total_success = int(frame["success_count"].sum())
    x = np.arange(len(frame))

    fig_w = max(10.0, min(24.0, 6.5 + len(frame) * 0.34))
    fig, ax = plt.subplots(figsize=(fig_w, 5.6))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.plot(
        x,
        cumulative_success,
        color="#2A9D8F",
        marker="o",
        markersize=4.2,
        linewidth=2.0,
        zorder=2,
    )
    ax.fill_between(x, cumulative_success, color="#2A9D8F", alpha=0.12, zorder=1)

    ax.set_title("Cumulative Passed Factors by Record Index", fontsize=12, pad=12, color="#1A2B44")
    ax.set_xlabel("Record Index")
    ax.set_ylabel("Cumulative Passed Factors")
    tick_positions, tick_labels = _sparse_round_ticks(seq_index, every=5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.grid(axis="y", color="#E7EDF5", linewidth=0.8, zorder=0)

    fig.tight_layout()
    fig.savefig(output_png, format="png", dpi=220)
    plt.close(fig)
    return output_png, len(frame), total_success


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render round-memory generation summary PNG graph.")
    parser.add_argument(
        "--round-memory-csv",
        default=str(PROJECT_ROOT / "data" / "memory" / "round_memory.csv"),
        help="Path to round_memory.csv.",
    )
    parser.add_argument(
        "--output-png",
        default=str(PROJECT_ROOT / "data" / "memory" / "round_generation_summary.png"),
        help="Output PNG path for per-round generation summary.",
    )
    parser.add_argument(
        "--cumulative-output-png",
        default=str(PROJECT_ROOT / "data" / "memory" / "round_cumulative_success.png"),
        help="Output PNG path for cumulative passed-factor curve.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    round_memory_csv = Path(args.round_memory_csv).expanduser().resolve()
    output_png = Path(args.output_png).expanduser().resolve()
    cumulative_output_png = Path(args.cumulative_output_png).expanduser().resolve()

    if not round_memory_csv.exists():
        raise FileNotFoundError(f"round memory csv not found: {round_memory_csv}")

    png_path, round_count = build_round_memory_progress_graph(
        round_memory_csv=round_memory_csv,
        output_png=output_png,
    )
    cumulative_png_path, _, total_success = build_cumulative_success_graph(
        round_memory_csv=round_memory_csv,
        output_png=cumulative_output_png,
    )
    print(f"[round-progress] png={png_path}")
    print(f"[round-progress] cumulative_png={cumulative_png_path}")
    print(f"[round-progress] rounds={round_count}")
    print(f"[round-progress] cumulative_success={total_success}")


if __name__ == "__main__":
    main()
