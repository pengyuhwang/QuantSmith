"""KunQuant DSL 后端（新版）：补充宏语义与算子守护。"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import BoolOpTrait, Builder, ConstantOp, Input, Output, Rank, Scale
from KunQuant.Stage import Function
from KunQuant.jit import cfake
from KunQuant.ops.CompOp import (
    Clip,
    DecayLinear,
    Pow as OpPow,
    TsArgMax,
    TsArgMin,
    TsRank,
    WindowedAvg,
    WindowedCorrelation,
    WindowedCovariance,
    WindowedKurt,
    WindowedLinearRegressionRSqaure,
    WindowedLinearRegressionResi,
    WindowedLinearRegressionSlope,
    WindowedMax,
    WindowedMaxDrawdown,
    WindowedMin,
    WindowedProduct,
    WindowedSkew,
    WindowedStddev,
    WindowedSum,
)
from KunQuant.ops.ElewiseOp import (
    Abs as OpAbs,
    Add as OpAdd,
    AddConst,
    And as OpAnd,
    Div as OpDiv,
    Equals as OpEquals,
    Exp as OpExp,
    GreaterEqual as OpGreaterEqual,
    GreaterThan as OpGreaterThan,
    LessEqual as OpLessEqual,
    LessThan as OpLessThan,
    Log as OpLog,
    Max as OpMax,
    Min as OpMin,
    Mul as OpMul,
    Not as OpNot,
    Or as OpOr,
    Select,
    SetInfOrNanToValue,
    Sign as OpSign,
    Sub as OpSub,
)
from KunQuant.ops.MiscOp import (
    BackRef,
    DiffWithWeightedSum,
    ExpMovingAvg,
    FastWindowedSum,
    WindowedQuantile,
)
from KunQuant.runner import KunRunner as kr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "fama" / "config" / "defaults.yaml"
DEFAULT_KUN_BATCH_SIZE = 500


FIELD_MAP: Dict[str, str] = {
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "VOLUME": "volume",
    "AMOUNT": "amount",
}

# 常见 ADV 窗口，便于 LLM 直接引用 ADV20 等变量。
ADV_WINDOWS = (5, 10, 15, 20, 30, 40, 50, 60, 120, 180)

DSL_FUNCTIONS: Dict[str, callable] = {}


def _ensure_expr(value):
    if isinstance(value, (int, float)):
        return ConstantOp(float(value))
    return value


def _ensure_numeric(value):
    """Cast numbers to ConstantOp and boolean masks to 0/1 float."""
    expr = _ensure_expr(value)
    if isinstance(expr, BoolOpTrait):
        return Select(expr, ConstantOp(1.0), ConstantOp(0.0))
    return expr


def _ensure_condition(value):
    expr = _ensure_expr(value)
    if isinstance(expr, BoolOpTrait):
        return expr
    # Treat any non-zero value as True to satisfy KunQuant's bool mask.
    return OpNot(OpEquals(expr, ConstantOp(0.0)))


def _register_ops() -> None:

    def _if_then_else(cond, a, b):
        cond_mask = _ensure_condition(cond)
        true_v = _ensure_expr(a)
        false_v = _ensure_expr(b)
        mask_float = Select(cond_mask, ConstantOp(1.0), ConstantOp(0.0))
        inv_mask = OpSub(ConstantOp(1.0), mask_float)
        return OpAdd(OpMul(mask_float, true_v), OpMul(inv_mask, false_v))

    def _logical_and(x, y):
        return OpAnd(_ensure_condition(x), _ensure_condition(y))

    def _logical_or(x, y):
        return OpOr(_ensure_condition(x), _ensure_condition(y))

    def _logical_not(x):
        return OpNot(_ensure_condition(x))

    def _ts_rank(expr, window):
        return TsRank(_ensure_numeric(expr), _to_int(window))

    def _decay_linear(expr, window):
        return DecayLinear(_ensure_numeric(expr), _to_int(window))

    def _dsl_safe_div(*args):
        if len(args) < 2:
            raise ValueError("SAFE_DIV 至少需要两个参数")
        numerator, denominator = args[0], args[1]
        eps = _to_float(args[2]) if len(args) >= 3 else 1e-4
        fill = _to_float(args[3]) if len(args) >= 4 else 0.0
        num_expr = _ensure_numeric(numerator)
        denom_expr = _ensure_numeric(denominator)
        eps_const = ConstantOp(float(eps))
        safe_denom = Select(OpEquals(denom_expr, ConstantOp(0.0)), eps_const, denom_expr)
        ratio = OpDiv(num_expr, safe_denom)
        return SetInfOrNanToValue(ratio, float(fill))

    DSL_FUNCTIONS.update(
        {
            "RANK": lambda x: Rank(_ensure_numeric(x)),
            "DELTA": lambda x, n=1: OpSub(_ensure_numeric(x), BackRef(_ensure_numeric(x), _to_int(n))),
            "TS_MEAN": lambda x, n: WindowedAvg(_ensure_numeric(x), _to_int(n)),
            "TS_STDDEV": lambda x, n: WindowedStddev(_ensure_numeric(x), _to_int(n)),
            "CORREL": lambda x, y, n: WindowedCorrelation(_ensure_numeric(x), _to_int(n), _ensure_numeric(y)),
            "SIGN": lambda x: OpSign(_ensure_numeric(x)),
            "ABS": lambda x: OpAbs(_ensure_numeric(x)),
            "DELAY": lambda x, n=1: BackRef(_ensure_numeric(x), _to_int(n)),
            "TS_SUM": lambda x, n: WindowedSum(_ensure_numeric(x), _to_int(n)),
            "TS_MIN": lambda x, n: WindowedMin(_ensure_numeric(x), _to_int(n)),
            "TS_MAX": lambda x, n: WindowedMax(_ensure_numeric(x), _to_int(n)),
            "TS_PRODUCT": lambda x, n: WindowedProduct(_ensure_numeric(x), _to_int(n)),
            "TS_ARGMAX": lambda x, n: TsArgMax(_ensure_numeric(x), _to_int(n)),
            "TS_ARGMIN": lambda x, n: TsArgMin(_ensure_numeric(x), _to_int(n)),
            "TS_RANK": _ts_rank,
            "DECAY_LINEAR": _decay_linear,
            "SCALE": lambda x: Scale(_ensure_numeric(x)),
            "IF": _if_then_else,
            "AND": _logical_and,
            "OR": _logical_or,
            "NOT": _logical_not,
            "GT": lambda x, y: OpGreaterThan(_ensure_expr(x), _ensure_expr(y)),
            "GE": lambda x, y: OpGreaterEqual(_ensure_expr(x), _ensure_expr(y)),
            "LT": lambda x, y: OpLessThan(_ensure_expr(x), _ensure_expr(y)),
            "LE": lambda x, y: OpLessEqual(_ensure_expr(x), _ensure_expr(y)),
            "EQ": lambda x, y: OpEquals(_ensure_expr(x), _ensure_expr(y)),
            "MAX": lambda x, y: OpMax(_ensure_numeric(x), _ensure_numeric(y)),
            "MIN": lambda x, y: OpMin(_ensure_numeric(x), _ensure_numeric(y)),
            "LOG": lambda x: OpLog(_ensure_numeric(x)),
            "EXP": lambda x: OpExp(_ensure_numeric(x)),
            "POW": lambda x, y: OpPow(_ensure_numeric(x), _ensure_numeric(y)),
            "REPLACE_NAN_INF": lambda x, value=0.0: SetInfOrNanToValue(_ensure_numeric(x), _to_float(value)),
            "COVAR": lambda x, y, n: WindowedCovariance(_ensure_numeric(x), _to_int(n), _ensure_numeric(y)),
            "ADV": lambda x, n: WindowedAvg(_ensure_numeric(x), _to_int(n)),
            "SAFE_DIV": _dsl_safe_div,
            "CLIP": lambda x, eps=1.0: Clip(_ensure_numeric(x), _to_float(eps)),
            "EMA": lambda x, n: ExpMovingAvg(_ensure_numeric(x), _to_int(n)),
            "EXP_MOVING_AVG": lambda x, n: ExpMovingAvg(_ensure_numeric(x), _to_int(n)),
            "FAST_TS_SUM": lambda x, n: FastWindowedSum(_ensure_numeric(x), _to_int(n)),
            "TS_QUANTILE": lambda x, n, q: WindowedQuantile(_ensure_numeric(x), _to_int(n), _to_float(q)),
            "TS_KURT": lambda x, n: WindowedKurt(_ensure_numeric(x), _to_int(n)),
            "TS_SKEW": lambda x, n: WindowedSkew(_ensure_numeric(x), _to_int(n)),
            "TS_MAXDRAWDOWN": lambda x, n: WindowedMaxDrawdown(_ensure_numeric(x), _to_int(n)),
            "TS_LINEAR_REGRESSION_R2": lambda x, n, y=None: WindowedLinearRegressionRSqaure(
                _ensure_numeric(x), _to_int(n), None if y is None else _ensure_numeric(y)
            ),
            "TS_LINEAR_REGRESSION_SLOPE": lambda x, n, y=None: WindowedLinearRegressionSlope(
                _ensure_numeric(x), _to_int(n), None if y is None else _ensure_numeric(y)
            ),
            "TS_LINEAR_REGRESSION_RESI": lambda x, n, y=None: WindowedLinearRegressionResi(
                _ensure_numeric(x), _to_int(n), None if y is None else _ensure_numeric(y)
            ),
            "DIFF_WITH_WEIGHTED_SUM": lambda v, w: DiffWithWeightedSum(_ensure_numeric(v), _ensure_numeric(w)),
        }
    )


_register_ops()


def _load_defaults(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    cfg_path = Path(config_path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _resolve_batch_size(batch_size: int | None) -> int:
    if batch_size is None:
        cfg = _load_defaults(DEFAULT_CONFIG_PATH)
        compute_cfg = cfg.get("compute", {}) if isinstance(cfg, dict) else {}
        batch_size = int(compute_cfg.get("kun_batch_size", DEFAULT_KUN_BATCH_SIZE))
    else:
        batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    return batch_size


def compute_factor_values_kunquant_new(
    market_data: pd.DataFrame,
    expr_list: List[str],
    *,
    threads: int = 4,
    layout: str = "TS",
    batch_size: int | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """解析 DSL 并调用 KunQuant 计算因子值。"""

    if not expr_list:
        empty = pd.DataFrame(index=market_data.index)
        empty.index.names = ["date", "symbol"]
        return empty, []

    inputs_np, dates, symbols = _build_ts_inputs(market_data)
    batch_size = _resolve_batch_size(batch_size)

    results: list[pd.DataFrame] = []
    fallback_exprs: list[str] = []
    num_dates = len(dates)
    num_symbols = len(symbols)
    executor = kr.createMultiThreadExecutor(max(1, int(threads)))
    first = next(iter(inputs_np.values()))
    length = first.shape[0]
    num_stocks = first.shape[1]

    total_batches = (len(expr_list) + batch_size - 1) // batch_size
    batch_id = 0
    for offset in range(0, len(expr_list), batch_size):
        batch_id += 1
        chunk = expr_list[offset : offset + batch_size]
        print(f"[DSL-New] KunQuant 编译批次 {batch_id}/{total_batches}（{len(chunk)} 条）...")
        builder = Builder()
        compiled_exprs: list[str] = []
        with builder:
            inp = {name: Input(name) for name in inputs_np.keys()}
            env = _build_env(inp)
            counter = 0
            for expr in chunk:
                try:
                    ir = _compile_expression(expr, env)
                except Exception:
                    fallback_exprs.append(expr)
                    continue
                if isinstance(ir, BoolOpTrait):
                    ir = Select(ir, ConstantOp(1.0), ConstantOp(0.0))
                counter += 1
                compiled_exprs.append(expr)
                Output(ir, f"f_{counter}")

        if not compiled_exprs:
            continue

        func = Function(builder.ops)
        try:
            lib = cfake.compileit(
                [(f"fama_graph_new_{offset}", func, KunCompilerConfig(input_layout=layout, output_layout=layout))],
                f"fama_graph_new_lib_{offset}",
                cfake.CppCompilerConfig(),
            )
        except Exception:
            fallback_exprs.extend(compiled_exprs)
            print(f"[DSL-New] 批次 {batch_id} 编译失败，跳过 {len(compiled_exprs)} 条。")
            continue

        module = lib.getModule(f"fama_graph_new_{offset}")
        try:
            out = kr.runGraph(
                executor,
                module,
                inputs_np,
                0,
                length,
                {},
                True,
                num_stocks=num_stocks,
            )
        except Exception:
            fallback_exprs.extend(compiled_exprs)
            print(f"[DSL-New] 批次 {batch_id} 执行失败，跳过 {len(compiled_exprs)} 条。")
            continue

        stacked: Dict[str, pd.Series] = {}
        for idx, expr in enumerate(compiled_exprs, 1):
            raw = np.asarray(out[f"f_{idx}"])
            if raw.shape == (num_symbols, num_dates):
                matrix = raw.T
            elif raw.shape == (num_dates, num_symbols):
                matrix = raw
            else:
                if raw.size != num_dates * num_symbols:
                    raise ValueError(
                        f"Unexpected KunQuant output shape {raw.shape}; expected "
                        f"({num_dates}, {num_symbols}) or ({num_symbols}, {num_dates})."
                    )
                matrix = raw.reshape(num_dates, num_symbols)
            df = pd.DataFrame(matrix, index=dates, columns=symbols)
            try:
                stacked_series = df.stack(future_stack=True)
            except TypeError:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="The previous implementation of stack is deprecated",
                        category=FutureWarning,
                    )
                    stacked_series = df.stack(dropna=False)
            stacked[expr] = stacked_series

        chunk_df = pd.concat(stacked, axis=1)
        chunk_df.index.names = ["date", "symbol"]
        results.append(chunk_df.sort_index())
        print(f"[DSL-New] 批次 {batch_id} 完成，成功 {len(compiled_exprs)} 条。")

    if not results:
        empty = pd.DataFrame(index=market_data.index)
        empty.index.names = ["date", "symbol"]
        # 若所有批次都失败，将剩余未标记的表达式加入 fallback
        if not fallback_exprs:
            fallback_exprs = expr_list.copy()
        return empty, fallback_exprs

    result = pd.concat(results, axis=1)
    result.index.names = ["date", "symbol"]
    return result.sort_index(), fallback_exprs


def _compile_expression(expr: str, env: Dict[str, Input]):
    tree = ast.parse(expr, mode="eval")
    return _eval_ast(tree.body, env)


def _eval_ast(node: ast.AST, env: Dict[str, Input]):
    if isinstance(node, ast.BinOp):
        left = _ensure_numeric(_eval_ast(node.left, env))
        right = _ensure_numeric(_eval_ast(node.right, env))
        if isinstance(node.op, ast.Add):
            return OpAdd(left, right)
        if isinstance(node.op, ast.Sub):
            return OpSub(left, right)
        if isinstance(node.op, ast.Mult):
            return OpMul(left, right)
        if isinstance(node.op, ast.Div):
            return OpDiv(left, right)
        if isinstance(node.op, ast.Pow):
            return OpPow(left, right)
        raise NotImplementedError(f"Unsupported operator {node.op}")
    if isinstance(node, ast.UnaryOp):
        operand = _ensure_numeric(_eval_ast(node.operand, env))
        if isinstance(node.op, ast.USub):
            return OpSub(ConstantOp(0.0), operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        raise NotImplementedError(f"Unsupported unary {node.op}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id.upper()
        func = DSL_FUNCTIONS.get(func_name)
        if func is None:
            raise NotImplementedError(f"Unsupported function {func_name}")
        args = [_eval_ast(arg, env) for arg in node.args]
        return func(*args)
    if isinstance(node, ast.Name):
        key = node.id.upper()
        if key not in env:
            raise NotImplementedError(f"Unknown variable {key}")
        return env[key]
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)):
            return ConstantOp(float(value))
        raise NotImplementedError(f"Unsupported constant {value}")
    raise NotImplementedError(f"Unsupported AST node {ast.dump(node)}")


def _power_expr(base, exponent):
    base_expr = _ensure_numeric(base)
    exp_value = _extract_constant(exponent)
    if exp_value is None:
        exponent_expr = _ensure_numeric(exponent)
        return _signed_power(base_expr, exponent_expr)
    if abs(exp_value) < 1e-12:
        return ConstantOp(1.0)
    if abs(exp_value - 1.0) < 1e-12:
        return base_expr
    if float(exp_value).is_integer():
        n = int(round(exp_value))
        if n < 0:
            raise NotImplementedError("Negative exponents are not supported.")
        result = base_expr
        for _ in range(n - 1):
            result = OpMul(result, base_expr)
        return result
    exponent_expr = ConstantOp(float(exp_value))
    return _signed_power(base_expr, exponent_expr)


def _extract_constant(value):
    if isinstance(value, ConstantOp):
        return float(value.attrs.get("value", 0.0))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_int(value):
    if isinstance(value, ConstantOp):
        return int(value.attrs.get("value", 0))
    return int(value)


def _to_float(value):
    if isinstance(value, ConstantOp):
        return float(value.attrs.get("value", 0.0))
    return float(value)


def _signed_power(base_expr, exponent_expr):
    base_expr = _ensure_numeric(base_expr)
    exponent_expr = _ensure_numeric(exponent_expr)
    abs_base = OpAbs(base_expr)
    safe_base = OpMax(abs_base, ConstantOp(1e-6))
    magnitude = OpExp(OpMul(exponent_expr, OpLog(safe_base)))
    sign = OpSign(base_expr)
    return OpMul(sign, magnitude)


def _build_ts_inputs(mkt_df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], List[pd.Timestamp], List[str]]:
    if not isinstance(mkt_df.index, pd.MultiIndex) or mkt_df.index.nlevels != 2:
        raise ValueError("market_data 必须使用 (date, symbol) MultiIndex")

    dates = sorted(mkt_df.index.get_level_values(0).unique())
    symbols = sorted(mkt_df.index.get_level_values(1).unique())
    inputs: Dict[str, np.ndarray] = {}

    for famaf, alias in FIELD_MAP.items():
        col = _find_column(mkt_df, famaf)
        if col is None:
            col = _find_column(mkt_df, alias)
        if col is None:
            continue
        slice_df = mkt_df[col].unstack(level=1)
        slice_df = slice_df.reindex(index=dates, columns=symbols)
        arr = slice_df.to_numpy(dtype=np.float32)
        inputs[alias] = np.ascontiguousarray(arr)

    missing = [field for field in FIELD_MAP.values() if field not in inputs]
    if missing:
        raise ValueError(f"KunQuant 后端缺少字段: {missing}")
    return inputs, dates, symbols


def _find_column(df: pd.DataFrame, name: str) -> str | None:
    target = name.lower()
    for col in df.columns:
        if col.lower() == target:
            return col
    return None


def _build_env(inputs: Dict[str, Input]) -> Dict[str, Input]:
    env = {key.upper(): inputs[key] for key in FIELD_MAP.values()}
    close = env["CLOSE"]
    volume = env["VOLUME"]
    amount = env["AMOUNT"]

    lag_close = BackRef(close, 1)
    env["RET"] = OpSub(OpDiv(close, lag_close), ConstantOp(1.0))
    env["VWAP"] = OpDiv(amount, OpAdd(volume, ConstantOp(1e-7)))

    for window in ADV_WINDOWS:
        env[f"ADV{window}"] = WindowedAvg(volume, window)
    return env
