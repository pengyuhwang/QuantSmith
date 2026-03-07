from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from fama.data.factor_space import deserialize_factor_set, FactorSet
from fama.utils.io import ensure_dir
from utils.factor_catalog import resolve_base_factor_cache

from .patterns import estimate_nesting_depth, extract_ops, pattern_key

ROUND_MEMORY_COLUMNS = [
    "round_id",
    "batch_id",
    "generated_count",
    "success_count",
    "pass_rate",
    "weak_signal_fail_count",
    "stability_fail_count",
    "duplicate_fail_count",
    "complexity_fail_count",
    "success_pattern_dist_json",
    "failure_pattern_dist_json",
    "round_packet_json",
    "round_reflection_json",
]


def _safe_float(value: Any) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_load(value: Any, default: Any) -> Any:
    if value is None or value is pd.NA:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _meta_field(meta: Mapping[str, Any] | None, field: str) -> Any:
    if not isinstance(meta, Mapping):
        return None
    return meta.get(field)


def _normalize_references(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            loaded = _json_load(stripped, [])
            if isinstance(loaded, list):
                return [str(item).strip() for item in loaded if str(item).strip()]
        return [stripped]
    return [str(value).strip()]


def _factor_meta_card(name: str, meta: Mapping[str, Any] | None, source: str | None = None) -> dict[str, Any]:
    references = _normalize_references(_meta_field(meta, "references"))
    return {
        "factor_name": name,
        "expression": str(_meta_field(meta, "expression") or ""),
        "explanation": str(_meta_field(meta, "explanation") or ""),
        "references": references,
        "source": source or "",
    }


def _reference_card(meta_map: Mapping[str, Mapping[str, Any]], factor_id: Any) -> dict[str, Any] | None:
    if factor_id is None:
        return None
    try:
        if pd.isna(factor_id):
            return None
    except Exception:
        pass
    key = str(factor_id).strip()
    if not key:
        return None
    meta = meta_map.get(key)
    card = _factor_meta_card(key, meta)
    if not card["expression"] and not card["explanation"] and not card["references"]:
        return {"factor_name": key, "expression": "", "explanation": "", "references": [], "source": ""}
    return card


def _classify_failure_bucket(final_status: Any, failure_reason: Any) -> str:
    if str(final_status or "").strip().lower() == "success":
        return "success"
    reasons = [item.strip() for item in str(failure_reason or "").split(";") if item.strip()]
    reason_set = set(reasons)
    if any("corr" in reason for reason in reason_set):
        return "duplicate_fail"
    if "max_operator_count" in reason_set or "max_nesting_depth" in reason_set:
        return "complexity_fail"
    if any(reason.startswith("valid_") for reason in reason_set) and "train_min_abs_ric" not in reason_set:
        return "stability_fail"
    if "train_min_abs_ric" in reason_set or not reason_set:
        return "weak_signal_fail"
    return "other_fail"


def build_round_packet(
    metrics_df: pd.DataFrame,
    *,
    round_id: str,
    batch_id: str,
    factor_meta_map: Mapping[str, Mapping[str, Any]],
    reference_meta_map: Mapping[str, Mapping[str, Any]],
    recent_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    success_patterns: Counter[str] = Counter()
    failure_patterns: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()

    if metrics_df is None:
        metrics_df = pd.DataFrame()

    for row in metrics_df.sort_values("factor_id").itertuples(index=False):
        factor_name = str(getattr(row, "factor_id"))
        meta = factor_meta_map.get(factor_name, {})
        references = _normalize_references(_meta_field(meta, "references"))
        operator_count = _safe_int(getattr(row, "operator_count", None))
        nesting_depth = _safe_int(getattr(row, "nesting_depth", None))
        patt = pattern_key(
            _meta_field(meta, "expression"),
            references,
            operator_count,
            nesting_depth,
        )
        bucket = _classify_failure_bucket(
            getattr(row, "final_status", None),
            getattr(row, "failure_reason", None),
        )
        bucket_counts[bucket] += 1
        if bucket == "success":
            success_patterns[patt] += 1
        else:
            failure_patterns[patt] += 1

        cards.append(
            {
                "factor_name": factor_name,
                "expression": str(_meta_field(meta, "expression") or ""),
                "explanation": str(_meta_field(meta, "explanation") or ""),
                "references": references,
                "train_ic": _safe_float(getattr(row, "train_ic", None)),
                "train_ric": _safe_float(getattr(row, "train_ric", None)),
                "train_icir": _safe_float(getattr(row, "train_icir", None)),
                "valid_ic": _safe_float(getattr(row, "valid_ic", None)),
                "valid_ric": _safe_float(getattr(row, "valid_ric", None)),
                "valid_icir": _safe_float(getattr(row, "valid_icir", None)),
                "train_corr_base": _safe_float(getattr(row, "train_base_max_corr", None)),
                "train_corr_old_llm": _safe_float(getattr(row, "train_old_llm_max_corr", None)),
                "train_corr_new_llm": _safe_float(getattr(row, "train_new_llm_max_corr", None)),
                "valid_corr_base": _safe_float(getattr(row, "valid_base_max_corr", None)),
                "valid_corr_old_llm": _safe_float(getattr(row, "valid_old_llm_max_corr", None)),
                "valid_corr_new_llm": _safe_float(getattr(row, "valid_new_llm_max_corr", None)),
                "operator_count": operator_count,
                "nesting_depth": nesting_depth,
                "expression_size": _safe_int(getattr(row, "expression_size", None)),
                "final_status": str(getattr(row, "final_status", "") or ""),
                "failure_stage": str(getattr(row, "failure_stage", "") or ""),
                "failure_reasons": [
                    item.strip() for item in str(getattr(row, "failure_reason", "") or "").split(";") if item.strip()
                ],
                "failure_bucket": bucket,
                "pattern": patt,
                "train_max_corr_factor": _reference_card(reference_meta_map, getattr(row, "train_max_corr_factor_id", None)),
                "valid_max_corr_factor": _reference_card(reference_meta_map, getattr(row, "valid_max_corr_factor_id", None)),
            }
        )

    generated_count = len(cards)
    success_count = bucket_counts.get("success", 0)
    pass_rate = (success_count / generated_count) if generated_count else 0.0

    return {
        "round_id": str(round_id),
        "batch_id": str(batch_id),
        "round_metrics": {
            "generated_count": generated_count,
            "success_count": success_count,
            "pass_rate": pass_rate,
            "weak_signal_fail_count": bucket_counts.get("weak_signal_fail", 0),
            "stability_fail_count": bucket_counts.get("stability_fail", 0),
            "duplicate_fail_count": bucket_counts.get("duplicate_fail", 0),
            "complexity_fail_count": bucket_counts.get("complexity_fail", 0),
        },
        "success_pattern_dist": dict(success_patterns.most_common()),
        "failure_pattern_dist": dict(failure_patterns.most_common()),
        "all_factor_cards": cards,
        "recent_context": recent_context or [],
    }


def build_round_memory_row(
    round_packet: Mapping[str, Any],
    round_reflection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metrics = round_packet.get("round_metrics", {}) if isinstance(round_packet, Mapping) else {}
    return {
        "round_id": str(round_packet.get("round_id", "")),
        "batch_id": str(round_packet.get("batch_id", "")),
        "generated_count": int(metrics.get("generated_count", 0) or 0),
        "success_count": int(metrics.get("success_count", 0) or 0),
        "pass_rate": float(metrics.get("pass_rate", 0.0) or 0.0),
        "weak_signal_fail_count": int(metrics.get("weak_signal_fail_count", 0) or 0),
        "stability_fail_count": int(metrics.get("stability_fail_count", 0) or 0),
        "duplicate_fail_count": int(metrics.get("duplicate_fail_count", 0) or 0),
        "complexity_fail_count": int(metrics.get("complexity_fail_count", 0) or 0),
        "success_pattern_dist_json": _json_text(round_packet.get("success_pattern_dist", {})),
        "failure_pattern_dist_json": _json_text(round_packet.get("failure_pattern_dist", {})),
        "round_packet_json": _json_text(round_packet),
        "round_reflection_json": _json_text(round_reflection or {}),
    }


def append_round_memory_csv(path: str | Path, row: Mapping[str, Any]) -> None:
    output_path = Path(path)
    ensure_dir(str(output_path.parent))
    frame = pd.DataFrame([{col: row.get(col) for col in ROUND_MEMORY_COLUMNS}])
    if output_path.exists():
        existing_cols = pd.read_csv(output_path, nrows=0).columns.tolist()
        if existing_cols != ROUND_MEMORY_COLUMNS:
            backup_path = output_path.with_suffix(output_path.suffix + ".legacy")
            output_path.replace(backup_path)
            frame.to_csv(output_path, index=False)
            return
        frame.to_csv(output_path, mode="a", header=False, index=False)
        return
    frame.to_csv(output_path, index=False)


def load_round_memory_rows(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    frame = pd.read_csv(input_path)
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        parsed = dict(row)
        parsed["success_pattern_dist"] = _json_load(parsed.get("success_pattern_dist_json"), {})
        parsed["failure_pattern_dist"] = _json_load(parsed.get("failure_pattern_dist_json"), {})
        parsed["round_packet"] = _json_load(parsed.get("round_packet_json"), {})
        parsed["round_reflection"] = _json_load(parsed.get("round_reflection_json"), {})
        rows.append(parsed)
    rows.sort(key=lambda item: int(item.get("round_id") or 0))
    return rows


def load_recent_round_context(path: str | Path, limit: int = 3) -> list[dict[str, Any]]:
    rows = load_round_memory_rows(path)
    if limit <= 0:
        return []
    trimmed = rows[-limit:]
    context: list[dict[str, Any]] = []
    for row in trimmed:
        reflection = row.get("round_reflection") or {}
        context.append(
            {
                "round_id": row.get("round_id"),
                "round_overview": reflection.get("round_overview", ""),
                "next_round_guidance": reflection.get("next_round_guidance", {}),
            }
        )
    return context


class FactorLibraryIndex:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self._by_name = {str(item.get("factor_name")): item for item in records if item.get("factor_name")}

    @classmethod
    def from_paths(
        cls,
        *,
        base_factor_cache_path: str | Path | None,
        llm_factor_library_path: str | Path | None,
    ) -> "FactorLibraryIndex":
        records: list[dict[str, Any]] = []
        records.extend(_load_factor_records(base_factor_cache_path, source="base"))
        records.extend(_load_factor_records(llm_factor_library_path, source="llm_library"))
        deduped: dict[str, dict[str, Any]] = {}
        for item in records:
            deduped[str(item.get("factor_name"))] = item
        return cls(list(deduped.values()))

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "FactorLibraryIndex":
        base_resolved_path, _ = resolve_base_factor_cache(dict(cfg))
        paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), Mapping) else {}
        return cls.from_paths(
            base_factor_cache_path=base_resolved_path,
            llm_factor_library_path=paths.get("llm_factor_library"),
        )

    def query(self, query: str, mode: str = "auto", limit: int = 5) -> list[dict[str, Any]]:
        limit = max(1, int(limit or 5))
        text = str(query or "").strip()
        if not text:
            return []
        mode_text = str(mode or "auto").strip().lower()
        if mode_text in {"auto", "factor_name"}:
            exact = self._by_name.get(text)
            if exact is not None:
                return [exact]
        lowered = text.lower()
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in self.records:
            name = str(item.get("factor_name") or "")
            expr = str(item.get("expression") or "")
            expl = str(item.get("explanation") or "")
            patt = str(item.get("pattern") or "")
            haystack = "\n".join([name.lower(), expr.lower(), expl.lower(), patt.lower()])
            score = 0
            if mode_text in {"auto", "factor_name"} and lowered in name.lower():
                score = max(score, 120 if lowered == name.lower() else 90)
            if mode_text in {"auto", "pattern"} and lowered in patt.lower():
                score = max(score, 100 if lowered == patt.lower() else 80)
            if mode_text in {"auto", "keyword"} and lowered in haystack:
                score = max(score, 60)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], str(pair[1].get("factor_name") or "")), reverse=True)
        return [item for _, item in scored[:limit]]


def _load_factor_records(path: str | Path | None, *, source: str) -> list[dict[str, Any]]:
    if path is None:
        return []
    input_path = Path(path).expanduser()
    if not input_path.exists():
        return []
    try:
        factor_set = deserialize_factor_set(str(input_path))
    except Exception:
        factor_set = FactorSet([])
    records: list[dict[str, Any]] = []
    for factor in getattr(factor_set, "factors", []) or []:
        refs = [str(item).strip() for item in (getattr(factor, "references", None) or []) if str(item).strip()]
        expr = str(getattr(factor, "expression", "") or "")
        op_count = len(extract_ops(expr))
        depth = estimate_nesting_depth(expr)
        record = {
            "factor_name": str(getattr(factor, "name", "") or ""),
            "expression": expr,
            "explanation": str(getattr(factor, "explanation", "") or ""),
            "references": refs,
            "source": source,
        }
        record["pattern"] = pattern_key(record["expression"], refs, op_count, depth)
        if record["factor_name"]:
            records.append(record)
    return records


def build_retrieval_packet(
    path: str | Path,
    *,
    recent_rounds: int = 10,
    top_pass_rounds: int = 3,
    top_duplicate_rounds: int = 3,
) -> dict[str, Any]:
    rows = load_round_memory_rows(path)
    if not rows:
        return {
            "recent_round_memories": [],
            "top_pass_rounds": [],
            "top_duplicate_rounds": [],
            "aggregate_stats": {},
        }
    recent = rows[-max(1, int(recent_rounds)):]
    pass_sorted = sorted(recent, key=lambda item: float(item.get("pass_rate") or 0.0), reverse=True)
    duplicate_sorted = sorted(recent, key=lambda item: int(item.get("duplicate_fail_count") or 0), reverse=True)

    success_patterns: Counter[str] = Counter()
    failure_patterns: Counter[str] = Counter()
    overviews: list[str] = []
    for row in recent:
        success_patterns.update(row.get("success_pattern_dist") or {})
        failure_patterns.update(row.get("failure_pattern_dist") or {})
        reflection = row.get("round_reflection") or {}
        overview = str(reflection.get("round_overview") or "").strip()
        if overview:
            overviews.append(overview)

    def _compact_round(row: Mapping[str, Any]) -> dict[str, Any]:
        reflection = row.get("round_reflection") or {}
        packet = row.get("round_packet") or {}
        cards = list(packet.get("all_factor_cards") or [])
        cards.sort(
            key=lambda item: (
                0 if str(item.get("final_status") or "").lower() == "success" else 1,
                str(item.get("factor_name") or ""),
            )
        )
        return {
            "round_id": row.get("round_id"),
            "generated_count": int(row.get("generated_count") or 0),
            "success_count": int(row.get("success_count") or 0),
            "pass_rate": float(row.get("pass_rate") or 0.0),
            "duplicate_fail_count": int(row.get("duplicate_fail_count") or 0),
            "stability_fail_count": int(row.get("stability_fail_count") or 0),
            "success_pattern_dist": row.get("success_pattern_dist") or {},
            "failure_pattern_dist": row.get("failure_pattern_dist") or {},
            "round_overview": reflection.get("round_overview", ""),
            "next_round_guidance": reflection.get("next_round_guidance", {}),
            "sample_factors": cards[: min(4, len(cards))],
        }

    avg_pass_rate = sum(float(item.get("pass_rate") or 0.0) for item in recent) / len(recent)
    return {
        "recent_round_memories": [_compact_round(item) for item in recent],
        "top_pass_rounds": [_compact_round(item) for item in pass_sorted[: max(1, int(top_pass_rounds))]],
        "top_duplicate_rounds": [_compact_round(item) for item in duplicate_sorted[: max(1, int(top_duplicate_rounds))]],
        "aggregate_stats": {
            "recent_round_count": len(recent),
            "avg_pass_rate": avg_pass_rate,
            "dominant_success_patterns": [name for name, _ in success_patterns.most_common(5)],
            "dominant_failure_patterns": [name for name, _ in failure_patterns.most_common(5)],
            "recent_round_overviews": overviews[-5:],
        },
    }


def render_retrieval_guidance(plan: Mapping[str, Any] | None) -> str:
    if not isinstance(plan, Mapping) or not plan:
        return ""
    lines: list[str] = []
    market_summary = str(plan.get("market_summary") or "").strip()
    if market_summary:
        lines.append("[记忆检索总结]")
        lines.append(market_summary)

    preferred_patterns = plan.get("preferred_patterns") or []
    if preferred_patterns:
        lines.append("[优先结构模式]")
        for item in preferred_patterns:
            if not isinstance(item, Mapping):
                continue
            pattern = str(item.get("pattern") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if pattern:
                lines.append(f"- {pattern}: {reason}" if reason else f"- {pattern}")

    avoid_patterns = plan.get("avoid_patterns") or []
    if avoid_patterns:
        lines.append("[避免结构模式]")
        for item in avoid_patterns:
            if not isinstance(item, Mapping):
                continue
            pattern = str(item.get("pattern") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if pattern:
                lines.append(f"- {pattern}: {reason}" if reason else f"- {pattern}")

    preferred_semantics = [str(item).strip() for item in (plan.get("preferred_semantics") or []) if str(item).strip()]
    if preferred_semantics:
        lines.append("[优先语义方向]")
        for item in preferred_semantics:
            lines.append(f"- {item}")

    repair_directions = [str(item).strip() for item in (plan.get("repair_directions") or []) if str(item).strip()]
    if repair_directions:
        lines.append("[修复方向]")
        for item in repair_directions:
            lines.append(f"- {item}")

    evidence_factors = plan.get("evidence_factors") or []
    if evidence_factors:
        lines.append("[参考因子与含义]")
        for item in evidence_factors:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("factor_name") or "").strip()
            expr = str(item.get("expression") or "").strip()
            expl = str(item.get("explanation") or "").strip()
            if name and expr:
                line = f"- {name}: {expr}"
                if expl:
                    line += f" | 含义: {expl}"
                lines.append(line)

    prompt_memo = str(plan.get("prompt_memo") or "").strip()
    if prompt_memo:
        lines.append("[本轮挖掘提醒]")
        lines.append(prompt_memo)

    return "\n".join(lines).strip()


def extract_reference_names_from_plan(plan: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(plan, Mapping):
        return []
    names: list[str] = []
    for item in plan.get("evidence_factors") or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("factor_name") or "").strip()
        if name:
            names.append(name)
    return list(dict.fromkeys(names))
