from __future__ import annotations

import re
from typing import Iterable

OPS_REGEX = re.compile(
    r"\b(RANK|DELTA|DELAY|TS_MEAN|TS_SUM|TS_STDDEV|TS_MIN|TS_MAX|TS_PRODUCT|TS_ARGMAX|TS_ARGMIN|TS_RANK|CORREL|COVAR|SIGN|ABS|DECAY_LINEAR|SCALE|IF|AND|OR|NOT|GT|GE|LT|LE|EQ|REPLACE_NAN_INF|ADV|SAFE_DIV|CLIP|EMA|EXP_MOVING_AVG|FAST_TS_SUM|TS_QUANTILE|TS_KURT|TS_SKEW|TS_MAXDRAWDOWN|TS_LINEAR_REGRESSION_R2|TS_LINEAR_REGRESSION_SLOPE|TS_LINEAR_REGRESSION_RESI|DIFF_WITH_WEIGHTED_SUM|LOG|EXP|POW|MAX|MIN)\b"
)


def extract_ops(expr: str | None) -> list[str]:
    if not expr:
        return []
    ops: set[str] = set()
    for match in OPS_REGEX.finditer(str(expr).upper()):
        ops.add(match.group(1))
    return sorted(ops)


def estimate_nesting_depth(expr: str | None) -> int:
    text = str(expr or "")
    max_depth = 0
    current = 0
    for char in text:
        if char == "(":
            current += 1
            max_depth = max(max_depth, current)
        elif char == ")":
            current = max(0, current - 1)
    return max_depth


def reference_shape(references: Iterable[str] | None) -> str:
    refs = [str(item).strip() for item in (references or []) if str(item).strip()]
    if not refs:
        return "none"
    llm_count = sum(1 for item in refs if item.upper().startswith("LLM_"))
    base_count = len(refs) - llm_count
    if llm_count and base_count:
        return "hybrid"
    if llm_count:
        return "single_llm" if llm_count == 1 else "multi_llm"
    return "single_base" if base_count == 1 else "multi_base"


def complexity_band(operator_count: int | float | None, nesting_depth: int | float | None) -> str:
    try:
        ops = int(operator_count) if operator_count is not None else None
    except Exception:
        ops = None
    try:
        depth = int(nesting_depth) if nesting_depth is not None else None
    except Exception:
        depth = None

    if ops is None or depth is None:
        return "unknown"
    if ops <= 5 and depth <= 2:
        return "low"
    if ops <= 10 and depth <= 4:
        return "mid"
    return "high"


def operator_family(expr: str | None) -> str:
    text = str(expr or "").upper()
    ops = set(extract_ops(text))

    if {"CORREL", "COVAR"} & ops:
        return "correlation"
    if {"TS_STDDEV", "TS_KURT", "TS_SKEW", "TS_MAXDRAWDOWN"} & ops:
        return "volatility"
    if {"TS_MAX", "TS_MIN", "TS_ARGMAX", "TS_ARGMIN"} & ops:
        return "range_extrema"
    if any(token in text for token in ("VOLUME", "AMOUNT", "VWAP", "ADV(")):
        if {"DELTA", "DECAY_LINEAR", "EMA", "EXP_MOVING_AVG", "TS_LINEAR_REGRESSION_SLOPE"} & ops:
            return "liquidity_flow"
        return "volume_price"
    if {"DELTA", "DECAY_LINEAR", "EMA", "EXP_MOVING_AVG", "TS_LINEAR_REGRESSION_SLOPE"} & ops:
        return "trend_momentum"
    if {"TS_RANK", "RANK", "SIGN", "ABS"} & ops:
        return "price_structure"
    return "other"


def pattern_key(
    expr: str | None,
    references: Iterable[str] | None,
    operator_count: int | float | None,
    nesting_depth: int | float | None,
) -> str:
    return "|".join(
        [
            operator_family(expr),
            reference_shape(references),
            complexity_band(operator_count, nesting_depth),
        ]
    )
