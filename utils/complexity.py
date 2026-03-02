from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable


_OP_NODES = (ast.Call, ast.BinOp, ast.UnaryOp)


@dataclass(frozen=True)
class ComplexityRecord:
    factor: str
    ops: int | None
    depth: int | None
    reason: str


def compute_expression_complexity(expr: str) -> tuple[int, int]:
    """Return (ops, depth) for a DSL expression.

    - ops: count of operator nodes = Call + BinOp + UnaryOp
    - depth: max nesting depth measured only on operator nodes
    """

    tree = ast.parse(expr, mode="eval")
    return _scan(tree.body)


def _scan(node: ast.AST) -> tuple[int, int]:
    child_nodes = _children(node)
    total_ops = 0
    max_child_depth = 0
    for child in child_nodes:
        child_ops, child_depth = _scan(child)
        total_ops += child_ops
        if child_depth > max_child_depth:
            max_child_depth = child_depth

    if isinstance(node, _OP_NODES):
        return total_ops + 1, max_child_depth + 1
    return total_ops, max_child_depth


def _children(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Call):
        # For Call, only inspect arguments/keyword values; function name is metadata.
        out: list[ast.AST] = list(node.args)
        out.extend(kw.value for kw in node.keywords)
        return out
    return list(ast.iter_child_nodes(node))


def apply_complexity_gate(
    candidates: Iterable[str],
    expr_map: dict[str, str],
    *,
    enabled: bool,
    max_ops: int,
    max_depth: int,
) -> tuple[list[str], list[ComplexityRecord]]:
    if not enabled:
        return sorted({str(name) for name in candidates}), []

    kept: list[str] = []
    dropped: list[ComplexityRecord] = []
    for factor in sorted({str(name) for name in candidates}):
        expr = expr_map.get(factor)
        if not isinstance(expr, str) or not expr.strip():
            dropped.append(
                ComplexityRecord(
                    factor=factor,
                    ops=None,
                    depth=None,
                    reason="missing_expression",
                )
            )
            continue
        try:
            ops, depth = compute_expression_complexity(expr)
        except Exception as exc:
            dropped.append(
                ComplexityRecord(
                    factor=factor,
                    ops=None,
                    depth=None,
                    reason=f"parse_error:{type(exc).__name__}",
                )
            )
            continue

        reasons: list[str] = []
        if ops > int(max_ops):
            reasons.append(f"ops={ops}>{int(max_ops)}")
        if depth > int(max_depth):
            reasons.append(f"depth={depth}>{int(max_depth)}")
        if reasons:
            dropped.append(
                ComplexityRecord(
                    factor=factor,
                    ops=ops,
                    depth=depth,
                    reason="; ".join(reasons),
                )
            )
            continue
        kept.append(factor)

    return kept, dropped

