#!/usr/bin/env python
"""Render full success-factor lineage as a compact PNG graph."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
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
_LLM_RE = re.compile(r"^LLM_Factor(\d+)$")
_DIGIT_RE = re.compile(r"(\d+)")


def _is_llm_factor(name: str) -> bool:
    return bool(_LLM_RE.match(str(name).strip()))


def _llm_num(name: str) -> int | None:
    match = _LLM_RE.match(str(name).strip())
    if not match:
        return None
    return int(match.group(1))


def _parse_round_id(value: Any) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    match = _DIGIT_RE.search(text)
    return int(match.group(1)) if match else 0


def _parse_references(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            if isinstance(parsed, str):
                return [parsed.strip()] if parsed.strip() else []
        except Exception:
            continue
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _select_main_parent(
    factor_id: str,
    refs: list[str],
    selected_ids: set[str],
) -> tuple[str, str]:
    """Return (parent_id, edge_kind). edge_kind in {'llm','base','history'}."""

    fid_num = _llm_num(factor_id)
    llm_refs = [ref for ref in refs if _is_llm_factor(ref)]
    if llm_refs and fid_num is not None:
        older_llm: list[tuple[int, str]] = []
        for ref in llm_refs:
            ref_num = _llm_num(ref)
            if ref_num is None:
                continue
            if ref_num < fid_num:
                older_llm.append((ref_num, ref))
        if older_llm:
            older_llm.sort(key=lambda item: item[0], reverse=True)
            main_llm = older_llm[0][1]
            if main_llm in selected_ids:
                return main_llm, "llm"
            return "LLM_HISTORY", "history"
        main_llm = sorted(llm_refs)[0]
        if main_llm in selected_ids:
            return main_llm, "llm"
        return "LLM_HISTORY", "history"

    base_refs = [str(ref).strip() for ref in refs if str(ref).strip() and not _is_llm_factor(str(ref).strip())]
    if base_refs:
        def _base_key(name: str) -> tuple[int, str]:
            low = name.lower()
            if low.startswith("alpha158"):
                return (0, low)
            if low.startswith("alpha101"):
                return (1, low)
            if low.startswith("alpha"):
                return (2, low)
            return (3, low)

        return sorted(base_refs, key=_base_key)[0], "base"
    return "BASE_LIBRARY", "base"


def _centered(count: int, spacing: float) -> list[float]:
    if count <= 0:
        return []
    mid = (count - 1) / 2.0
    return [(idx - mid) * spacing for idx in range(count)]


def _build_positions(
    node_rows: list[dict[str, Any]],
    *,
    round_order: list[int],
    parent_map: dict[str, str],
    base_order: list[str],
    history_needed: bool,
) -> dict[str, tuple[float, float]]:
    by_round: dict[int, list[dict[str, Any]]] = {}
    for row in node_rows:
        by_round.setdefault(int(row["round_int"]), []).append(row)

    positions: dict[str, tuple[float, float]] = {}
    round_step = 0.72
    round_x = {rnd: float(1.6 + i * round_step) for i, rnd in enumerate(round_order)}

    base_spacing = 0.62 if len(base_order) <= 24 else 0.54
    for base_id, y in zip(base_order, _centered(len(base_order), spacing=base_spacing)):
        positions[base_id] = (0.0, y)
    if history_needed:
        history_y = min([positions[item][1] for item in base_order], default=0.0) - 0.95
        positions["LLM_HISTORY"] = (0.82, history_y)

    for rnd in round_order:
        rows = by_round.get(rnd, [])

        def _parent_y(row: dict[str, Any]) -> float:
            fid = str(row["factor_id"])
            parent = parent_map.get(fid, "")
            if parent in positions:
                return float(positions[parent][1])
            return 0.0

        rows.sort(key=lambda row: (_parent_y(row), -float(row.get("score", 0.0)), str(row["factor_id"])))
        n_rows = len(rows)
        if n_rows <= 8:
            spacing = 0.88
        elif n_rows <= 14:
            spacing = 0.74
        else:
            spacing = 0.62
        for row, y in zip(rows, _centered(n_rows, spacing=spacing)):
            positions[str(row["factor_id"])] = (round_x[rnd], y)
    return positions


def _draw_mainline(
    *,
    node_rows: list[dict[str, Any]],
    edges: list[dict[str, str]],
    output_png: Path,
) -> bool:
    try:
        mpl_cache_dir = PROJECT_ROOT / "tmp" / "mplconfig"
        ensure_dir(str(mpl_cache_dir))
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not node_rows:
        return False
    ensure_dir(str(output_png.parent))

    round_order = sorted({int(row["round_int"]) for row in node_rows if int(row["round_int"]) > 0})
    parent_map = {str(edge["dst"]): str(edge["src"]) for edge in edges}
    base_counts = Counter(str(edge["src"]) for edge in edges if str(edge.get("kind")) == "base")
    base_order = sorted(base_counts.keys(), key=lambda name: (-int(base_counts[name]), str(name)))
    history_needed = any(str(edge.get("src")) == "LLM_HISTORY" for edge in edges)

    positions = _build_positions(
        node_rows,
        round_order=round_order,
        parent_map=parent_map,
        base_order=base_order,
        history_needed=history_needed,
    )

    fig_w = max(9.0, min(20.0, 5.8 + len(round_order) * 0.40))
    fig_h = max(7.2, min(15.0, 4.6 + max(10, len(node_rows)) * 0.12))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_title("Successful Factor Lineage (Full)", fontsize=11, pad=8, color="#1A2B44")

    round_to_x = {rnd: float(1.6 + i * 0.72) for i, rnd in enumerate(round_order)}
    y_values = [pos[1] for pos in positions.values()]
    y_min = min(y_values) - 1.0
    y_max = max(y_values) + 1.0

    for rnd in round_order:
        x = round_to_x[rnd]
        ax.plot([x, x], [y_min, y_max], color="#EEF2F8", lw=0.62, zorder=0.1)
        ax.text(x, y_max + 0.08, f"R{rnd}", ha="center", va="bottom", fontsize=5.8, color="#6A7D98")

    llm_edges: list[dict[str, str]] = []
    routed_edges: list[dict[str, str]] = []
    for edge in edges:
        if edge["kind"] == "llm":
            llm_edges.append(edge)
        else:
            routed_edges.append(edge)

    for edge in llm_edges:
        src = str(edge["src"])
        dst = str(edge["dst"])
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        rad = 0.04 if abs(y2 - y1) < 1.4 else 0.10
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "color": "#4F6FA5",
                "lw": 1.28,
                "linestyle": "-",
                "alpha": 0.58,
                "shrinkA": 8,
                "shrinkB": 8,
                "connectionstyle": f"arc3,rad={rad}",
            },
            zorder=1,
        )

    factor_round = {str(row["factor_id"]): int(row["round_int"]) for row in node_rows}
    grouped_routes: dict[tuple[str, int, str], list[str]] = {}
    for edge in routed_edges:
        src = str(edge["src"])
        dst = str(edge["dst"])
        kind = str(edge["kind"])
        rnd = factor_round.get(dst)
        if rnd is None:
            continue
        grouped_routes.setdefault((src, rnd, kind), []).append(dst)

    for (src, rnd, kind), dst_list in sorted(grouped_routes.items(), key=lambda item: (item[0][1], item[0][0])):
        if src not in positions or rnd not in round_to_x:
            continue
        x_src, y_src = positions[src]
        hub_x = round_to_x[rnd] - 0.48
        ax.plot([x_src, hub_x], [y_src, y_src], color="#999999", lw=0.62, ls="--", alpha=0.22, zorder=0.9)
        for dst in sorted(dst_list):
            dst = str(dst)
            if dst not in positions:
                continue
            x_dst, y_dst = positions[dst]
            color = "#7A3FC8" if kind == "history" else "#8A8A8A"
            ax.plot([hub_x, hub_x], [y_src, y_dst], color=color, lw=0.56, ls="--", alpha=0.20, zorder=0.9)
            ax.annotate(
                "",
                xy=(x_dst, y_dst),
                xytext=(hub_x, y_dst),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "lw": 0.58,
                    "linestyle": "--",
                    "alpha": 0.24,
                    "shrinkA": 2,
                    "shrinkB": 7,
                    "connectionstyle": "arc3,rad=0.0",
                },
                zorder=1.0,
            )

    base_font = 4.6 if len(base_order) <= 28 else 4.2
    for node_id in base_order:
        x, y = positions[node_id]
        ax.text(
            x,
            y,
            node_id,
            ha="center",
            va="center",
            fontsize=base_font,
            color="#4A4A4A",
            bbox={"boxstyle": "round,pad=0.14", "fc": "#F4F4F4", "ec": "#A0A0A0", "lw": 0.55},
            zorder=3,
        )

    if history_needed and "LLM_HISTORY" in positions:
        hx, hy = positions["LLM_HISTORY"]
        ax.text(
            hx,
            hy,
            "history_llm",
            ha="center",
            va="center",
            fontsize=4.6,
            color="#4F3D73",
            bbox={"boxstyle": "round,pad=0.14", "fc": "#EFE7FA", "ec": "#8468B3", "lw": 0.55},
            zorder=3,
        )

    max_score = max([float(row.get("score", 0.0)) for row in node_rows] + [1.0])
    cmap = plt.get_cmap("Blues")
    label_fs = 4.7 if len(node_rows) <= 160 else 4.3

    for row in node_rows:
        fid = str(row["factor_id"])
        x, y = positions[fid]
        score = float(row.get("score", 0.0))
        ratio = max(0.0, min(1.0, score / max_score))
        fill_color = cmap(0.20 + 0.52 * ratio)
        ax.text(
            x,
            y,
            fid,
            ha="center",
            va="center",
            fontsize=label_fs,
            color="#11243A",
            bbox={"boxstyle": "round,pad=0.14", "fc": fill_color, "ec": "#4F6FA5", "lw": 0.55},
            zorder=4,
        )

    x_vals = [xy[0] for xy in positions.values()]
    x_min = min(x_vals) - 0.45
    x_max = max(x_vals) + 0.45
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max + 0.25)
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    fig.savefig(output_png, format="png", dpi=220)
    plt.close(fig)
    return True


def build_success_mainline_graph(
    *,
    success_csv: Path,
    output_png: Path,
    topk_per_round: int = 0,
    last_rounds: int = 0,
) -> tuple[Path, int, int]:
    df = pd.read_csv(success_csv)
    if df.empty:
        raise ValueError(f"success csv is empty: {success_csv}")

    if "final_status" in df.columns:
        df = df[df["final_status"].astype(str).str.lower() == "success"]
    df = df[df["factor_id"].astype(str).map(_is_llm_factor)].copy()
    if df.empty:
        raise ValueError("No successful LLM factors found in csv.")

    df["round_int"] = df["round_id"].map(_parse_round_id).astype(int)
    df["train_ric_num"] = pd.to_numeric(df.get("train_ric"), errors="coerce")
    df["score"] = df["train_ric_num"].abs().fillna(0.0)
    df["refs"] = df.get("reference").map(_parse_references)

    rounds = sorted([int(item) for item in df["round_int"].unique().tolist() if int(item) > 0])
    if rounds and last_rounds > 0 and len(rounds) > last_rounds:
        keep = set(rounds[-last_rounds:])
        df = df[df["round_int"].isin(keep)]
    if topk_per_round > 0:
        df = (
            df.sort_values(["round_int", "score", "factor_id"], ascending=[True, False, True])
            .groupby("round_int", as_index=False, group_keys=False)
            .head(topk_per_round)
        )

    rows = df.to_dict("records")
    selected_ids = {str(row["factor_id"]) for row in rows}
    edges: list[dict[str, str]] = []
    for row in rows:
        factor_id = str(row["factor_id"])
        refs = row.get("refs") or []
        parent, kind = _select_main_parent(factor_id, refs, selected_ids)
        edges.append({"src": parent, "dst": factor_id, "kind": kind})

    ok = _draw_mainline(node_rows=rows, edges=edges, output_png=output_png)
    if not ok:
        raise RuntimeError("Failed to render PNG mainline graph.")
    return output_png, len(rows), len(edges)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render full success-factor lineage PNG graph.")
    parser.add_argument(
        "--success-csv",
        default=str(PROJECT_ROOT / "data" / "memory" / "factor_success_cases.csv"),
        help="Path to success memory csv.",
    )
    parser.add_argument(
        "--output-png",
        default=str(PROJECT_ROOT / "data" / "memory" / "llm_success_mainline_all.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--topk-per-round",
        type=int,
        default=0,
        help="Keep top-K successful factors per round by |train_ric|. <=0 means keep all.",
    )
    parser.add_argument(
        "--last-rounds",
        type=int,
        default=0,
        help="Keep only latest N rounds. <=0 means keep all.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    success_csv = Path(args.success_csv).expanduser().resolve()
    output_png = Path(args.output_png).expanduser().resolve()

    if not success_csv.exists():
        raise FileNotFoundError(f"success csv not found: {success_csv}")

    png_path, node_cnt, edge_cnt = build_success_mainline_graph(
        success_csv=success_csv,
        output_png=output_png,
        topk_per_round=int(args.topk_per_round),
        last_rounds=int(args.last_rounds),
    )
    print(f"[mainline] nodes={node_cnt} | edges={edge_cnt}")
    print(f"[mainline] png={png_path}")


if __name__ == "__main__":
    main()
