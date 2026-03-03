from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from fama.utils.io import ensure_dir

MEMORY_COLUMNS = [
    "factor_id",
    "round_id",
    "batch_id",
    "formula",
    "formula_hash",
    "reference",
    "economic_rationale",
    "train_ic",
    "train_ric",
    "train_icir",
    "train_max_corr",
    "train_max_corr_factor_id",
    "train_max_corr_factor_formula",
    "train_max_corr_factor_rationale",
    "train_max_corr_factor_reference",
    "valid_ic",
    "valid_ric",
    "valid_icir",
    "valid_max_corr",
    "valid_max_corr_factor_id",
    "valid_max_corr_factor_formula",
    "valid_max_corr_factor_rationale",
    "valid_max_corr_factor_reference",
    "operator_count",
    "nesting_depth",
    "expression_size",
    "final_status",
    "failure_stage",
    "failure_reason",
]


def _meta_field(meta: Mapping[str, Any] | None, field: str) -> Any:
    if not isinstance(meta, Mapping):
        return None
    return meta.get(field)


def _to_json_list(value: Any) -> str:
    if value is None:
        return "[]"
    if isinstance(value, list):
        return json.dumps([str(item) for item in value], ensure_ascii=False)
    if isinstance(value, tuple):
        return json.dumps([str(item) for item in value], ensure_ascii=False)
    if isinstance(value, str):
        if not value.strip():
            return "[]"
        return json.dumps([value], ensure_ascii=False)
    return json.dumps([str(value)], ensure_ascii=False)


def _hash_formula(formula: str) -> str:
    return hashlib.sha256(formula.encode("utf-8")).hexdigest()


def _enrich_reference(meta_map: Mapping[str, Mapping[str, Any]], factor_id: Any) -> tuple[str, str, str]:
    if factor_id is None or (isinstance(factor_id, float) and pd.isna(factor_id)):
        return "", "", "[]"
    key = str(factor_id)
    meta = meta_map.get(key, {})
    formula = _meta_field(meta, "expression") or ""
    rationale = _meta_field(meta, "explanation") or ""
    refs = _to_json_list(_meta_field(meta, "references"))
    return str(formula), str(rationale), refs


def build_memory_records(
    metrics_df: pd.DataFrame,
    *,
    round_id: str,
    batch_id: str,
    factor_meta_map: Mapping[str, Mapping[str, Any]],
    reference_meta_map: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(columns=MEMORY_COLUMNS)

    records: list[dict[str, Any]] = []
    for row in metrics_df.itertuples(index=False):
        factor_id = str(getattr(row, "factor_id"))
        factor_meta = factor_meta_map.get(factor_id, {})
        formula = str(_meta_field(factor_meta, "expression") or "")
        rationale = str(_meta_field(factor_meta, "explanation") or "")
        references = _to_json_list(_meta_field(factor_meta, "references"))

        train_ref_formula, train_ref_rationale, train_ref_reference = _enrich_reference(
            reference_meta_map,
            getattr(row, "train_max_corr_factor_id", None),
        )
        valid_ref_formula, valid_ref_rationale, valid_ref_reference = _enrich_reference(
            reference_meta_map,
            getattr(row, "valid_max_corr_factor_id", None),
        )

        record = {
            "factor_id": factor_id,
            "round_id": str(round_id),
            "batch_id": str(batch_id),
            "formula": formula,
            "formula_hash": _hash_formula(formula) if formula else "",
            "reference": references,
            "economic_rationale": rationale,
            "train_ic": getattr(row, "train_ic", pd.NA),
            "train_ric": getattr(row, "train_ric", pd.NA),
            "train_icir": getattr(row, "train_icir", pd.NA),
            "train_max_corr": getattr(row, "train_max_corr", pd.NA),
            "train_max_corr_factor_id": getattr(row, "train_max_corr_factor_id", ""),
            "train_max_corr_factor_formula": train_ref_formula,
            "train_max_corr_factor_rationale": train_ref_rationale,
            "train_max_corr_factor_reference": train_ref_reference,
            "valid_ic": getattr(row, "valid_ic", pd.NA),
            "valid_ric": getattr(row, "valid_ric", pd.NA),
            "valid_icir": getattr(row, "valid_icir", pd.NA),
            "valid_max_corr": getattr(row, "valid_max_corr", pd.NA),
            "valid_max_corr_factor_id": getattr(row, "valid_max_corr_factor_id", ""),
            "valid_max_corr_factor_formula": valid_ref_formula,
            "valid_max_corr_factor_rationale": valid_ref_rationale,
            "valid_max_corr_factor_reference": valid_ref_reference,
            "operator_count": getattr(row, "operator_count", pd.NA),
            "nesting_depth": getattr(row, "nesting_depth", pd.NA),
            "expression_size": getattr(row, "expression_size", pd.NA),
            "final_status": getattr(row, "final_status", ""),
            "failure_stage": getattr(row, "failure_stage", ""),
            "failure_reason": getattr(row, "failure_reason", ""),
        }
        records.append(record)

    out = pd.DataFrame(records)
    for col in MEMORY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[MEMORY_COLUMNS]


def append_memory_csv(path: str | Path, frame: pd.DataFrame) -> None:
    output_path = Path(path)
    ensure_dir(str(output_path.parent))
    if frame is None or frame.empty:
        return

    if output_path.exists():
        existing_cols = pd.read_csv(output_path, nrows=0).columns.tolist()
        if existing_cols != MEMORY_COLUMNS:
            backup_path = output_path.with_suffix(output_path.suffix + ".legacy")
            output_path.replace(backup_path)
            frame.to_csv(output_path, index=False)
            return
        frame.to_csv(output_path, mode="a", header=False, index=False)
        return
    frame.to_csv(output_path, index=False)
