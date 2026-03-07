"""README “LLM Prompting” 部分所述的提示词工具。"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Optional

from fama.factors.opcards import render_cards

OPS_REGEX = re.compile(
    r"\b(RANK|DELTA|DELAY|TS_MEAN|TS_SUM|TS_STDDEV|TS_MIN|TS_MAX|TS_PRODUCT|TS_ARGMAX|TS_ARGMIN|TS_RANK|CORREL|COVAR|SIGN|ABS|DECAY_LINEAR|SCALE|IF|AND|OR|NOT|GT|GE|LT|LE|EQ|REPLACE_NAN_INF|ADV|SAFE_DIV|CLIP|EMA|EXP_MOVING_AVG|FAST_TS_SUM|TS_QUANTILE|TS_KURT|TS_SKEW|TS_MAXDRAWDOWN|TS_LINEAR_REGRESSION_R2|TS_LINEAR_REGRESSION_SLOPE|TS_LINEAR_REGRESSION_RESI|DIFF_WITH_WEIGHTED_SUM|LOG|EXP|POW|MAX|MIN)\b"
)


def _extract_ops(exprs: Iterable[str]) -> list[str]:
    ops: set[str] = set()
    for expr in exprs:
        for match in OPS_REGEX.finditer(str(expr)):
            ops.add(match.group(1))
    return sorted(ops)


def _checksum(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def build_prompt(
    css_examples: list[str],
    coe_path: list[str],
    constraints: dict,
    *,
    available_fields: list[str],
    max_references: Optional[int] = None,
    memory_guidance: str | None = None,
) -> str:
    """拼装 LLM 挖掘阶段所需的结构化提示词。"""

    whitelist = set(constraints.get("operator_whitelist", []))
    context_ops = _extract_ops(list(css_examples) + list(coe_path))
    # 旧逻辑：只展示上下文出现过的算子与白名单的交集
    # ops = sorted((set(context_ops) & whitelist) if whitelist else set(context_ops))
    # 新逻辑：有 whitelist 时总是展示 whitelist（保持模型可用全量允许算子）
    ops = sorted(whitelist) if whitelist else sorted(set(context_ops))
    cards = render_cards(ops)
    checksum = _checksum(cards)
    css_block = "\n".join(f"- {expr}" for expr in css_examples) or "- (none)"
    coe_block = "\n".join(f"- {entry}" for entry in coe_path) or "- (none)"
    memory_block = (memory_guidance or "").strip()
    fields_block = ", ".join(available_fields)
    max_new = constraints.get("max_new_factors", 5)
    ref_cap = max_references or constraints.get("max_reference_factors") or 3

    # 4.生成的因子公式使用的算子嵌套不得大于5层，总算子使用数量不得超过20个，尽可能确保每个因子的经济学意义

    prompt = f"""OPS-CHECKSUM: {checksum}
    你是一名量化研究员，请用中文完成任务，并严格遵守以下要求：

    [总体目标]
    1. 在现有因子库的基础上，生成 **与当前因子库整体相关性更低、但预期 RankIC 更优** 的新阿尔法因子。
    2. 新因子应在“结构和经济含义”上体现**新的风格或机制**，而不是对示例因子做轻微扰动（如只改窗口长度、简单替换 HIGH/LOW 等）或线性组合。
    3. 你必须设计具有明确的经济或行为原理的因素，避免那些只能通过“它可以很好地回测”来证明的因素没有任何经济解释
    4. 尽量多借鉴给到的因子示例与进化方向，给出质量更高、具有经济意义的因子
    5. 虽然本次提供的样例中没出现qlib和alpha360的因子，但在你的生成过程中不应出现与这两个库以及其他已开源因子库中高度相似的因子，务必保证每个因子由思考诞生而非直接照抄

    [算子与表达式约束]
    - 仅可使用下述允许算子与字段构造表达式；
    - 严禁使用未在算子说明中出现的算子或函数。
    - 布尔比较必须使用 GT/GE/LT/LE/EQ/AND/OR/NOT 等函数，严禁直接写 >、<、>=、<=、&、|。
    - 所有时间序列函数的窗口参数必须是整数常量，绝不能传入序列或表达式；
    
    [允许算子说明]
    {cards or '(no operators detected—fallback to whitelist)'}

    [如何利用 CSS 示例]
    - 下方 CSS 示例来自 **不同簇**，它们之间两两相关性较低，是当前因子库中“跨风格、低相关”的代表性强因子；括号中的 ric/ic/icir 仅供参考，优先考虑能提升这些指标的结构创新。
    - 你可以从中学习：
      - 使用了哪些算子（如秩、相关、滑窗统计、量价组合等）；
      - 它们是如何刻画不同市场行为/风格的。
    - 但必须避免：
      - 直接复刻这些表达式；
      - 只对其做参数级的小改动；
      - 简单线性组合多个 CSS 示例因子。
    - 目标：在这些“低相关强风格”的启发下，构造**在风格上有差异、与当前因子库整体相关性更低**的新因子，生成的因子公式在经济学意义上也尽量与示例错开，避免高相关性。

    # CSS 示例（跨簇低相关的代表性因子）
    {css_block}

    [如何利用 CoE 经验链]
    - 下方 CoE 为同一簇内按 **RankIC 由弱到强排序** 的因子演化链（括号中给出 ric/ic/icir 及基准资产）：
      - 链头部：相对较弱/早期版本；
      - 链尾部：同簇内 RankIC 更强、结构更成熟的版本。
    - 你需要：
      - 观察链中因子是如何从“简单”演化到“复杂”以提升 RankIC 的（例如加入条件过滤、引入波动率或成交量、增加滞后维度、使用更加稳健的归一化等）；
      - 在此基础上进行进一步**结构创新**，提出有望在该簇中取得更高 RankIC/IC/ICIR 的新变体。
    - 同时要注意：
      - 新因子不能只是链尾因子的微小改写；
      - 在全局上也应与当前因子库保持较低相关性；
      - 生成的因子公式在经济学意义上也尽量与示例错开，避免高相关性。

    # Chain-of-Experience（同簇内按 RankIC 由弱到强的经验链）
    {coe_block}

    [如何利用记忆检索引导]
    - 下方内容来自近期轮次复盘与检索总结，目标是告诉你最近哪些结构更值得继续挖、哪些结构应避免。
    - 优先吸收其中关于因子机制、经济学含义、市场行为解释的内容，而不是只关注指标高低。
    - 如果记忆中给出了参考因子，请优先学习其结构机制与含义，再做有明确创新的延展。

    # 记忆检索引导
    {memory_block or '- (none)'}
    
    [输出格式与字段约束]
    - 仅允许使用以下字段（必须大写）：{fields_block}
    - 你必须输出一个 **JSON 数组**，共 {max_new} 个元素；
    - 每个元素必须严格为以下格式（字段名固定，均为字符串）：
    {{
      "expression": "<合法 DSL 表达式，满足算子/字段/长度/嵌套约束>",
      "explanation": "<一句中文经济学解释，仅一句，简洁说明该因子的经济含义>",
      "references": ["<参考因子名1>", "<参考因子名2>", ...]  // 最多 {ref_cap} 个，应从上文 CSS/CoE/记忆检索引导中明确出现的因子名中选择；如无可引用则输出空数组 []
    }}
    - JSON 外 **不得包含任何额外文本、注释、编号、说明或反引号**；
    - 不要出现```json[]```包裹的形式
    - 不要输出自然语言解释、指南或多余字段，只输出 JSON 数组本身。
    - 尽可能捕捉更多提供因子的特点生成新因子

    请基于以上 CSS 示例与 CoE 经验链信息以及坏因子示例，在控制与现有因子库相关性较低的前提下，发挥你对金融量化因子的理解，尽可能提升新因子的预期 RankIC，直接输出符合要求的 JSON。""".strip()

    return prompt


def parse_llm_output(text: str, max_references: Optional[int] = None) -> list[dict]:
    """将 LLM 的输出解析成包含 expression / explanation / references 的列表。"""

    import json

    cleaned = text.strip()
    if not cleaned:
        return []

    def _normalize_refs(raw) -> list[str] | None:
        if raw is None:
            return None
        refs: list[str] = []
        if isinstance(raw, str):
            candidates = raw.replace(";", ",").split(",")
            refs = [item.strip() for item in candidates if item.strip()]
        elif isinstance(raw, list):
            for item in raw:
                if item is None:
                    continue
                refs.append(str(item).strip())
            refs = [item for item in refs if item]
        if not refs:
            return None
        if max_references and max_references > 0:
            refs = refs[:max_references]
        return refs

    try:
        data = json.loads(cleaned)
        parsed: list[dict] = []
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                expr = item.get("expression")
                expl = item.get("explanation")
                refs = _normalize_refs(item.get("references"))
                if not expr:
                    continue
                parsed.append(
                    {
                        "expression": str(expr).strip(),
                        "explanation": expl.strip() if isinstance(expl, str) else None,
                        "references": refs,
                    }
                )
            if parsed:
                return parsed
    except Exception:
        pass

    # 兼容旧格式：每行一个表达式
    parsed: list[dict] = []
    for line in cleaned.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if ":" in candidate:
            _, candidate = candidate.split(":", 1)
        candidate = candidate.strip().strip("`")
        if candidate:
            parsed.append({"expression": candidate, "explanation": None, "references": None})
    return parsed
