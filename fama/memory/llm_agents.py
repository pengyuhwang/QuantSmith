from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[misc]

from fama.utils.io import ensure_dir

from .round_memory import FactorLibraryIndex


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    temperature: float | None
    thinking: str | None


def _resolve_settings(cfg: Mapping[str, Any], section_name: str) -> LLMSettings:
    base_cfg = cfg.get("llm", {}) if isinstance(cfg.get("llm"), Mapping) else {}
    memory_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), Mapping) else {}
    section_cfg = memory_cfg.get(section_name, {}) if isinstance(memory_cfg.get(section_name), Mapping) else {}
    api_key_env = section_cfg.get("api_key_env") or base_cfg.get("api_key_env") or "LLM_API_KEY"
    api_key = os.getenv(str(api_key_env)) or section_cfg.get("api_key") or base_cfg.get("api_key")
    return LLMSettings(
        provider=str(section_cfg.get("provider") or base_cfg.get("provider") or "mock"),
        model=str(section_cfg.get("model") or base_cfg.get("model") or "mock"),
        api_key=api_key,
        base_url=section_cfg.get("base_url") or base_cfg.get("base_url"),
        temperature=section_cfg.get("temperature", base_cfg.get("temperature")),
        thinking=section_cfg.get("thinking", base_cfg.get("thinking")),
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last > first:
        try:
            data = json.loads(cleaned[first : last + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def _chat_text(messages: list[dict[str, str]], settings: LLMSettings, logger=None) -> str:
    provider = settings.provider.lower()
    if provider not in {"openai", "siliconflow", "deepseek"} or OpenAI is None or not settings.api_key:
        raise RuntimeError("fallback")
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url) if settings.base_url else OpenAI(api_key=settings.api_key)
    resp = client.chat.completions.create(
        model=settings.model,
        temperature=float(settings.temperature or 0.2),
        messages=messages,
    )
    content = (resp.choices[0].message.content or "").strip()
    if logger is not None:
        logger.info("Memory LLM raw output: %s", content)
    return content


def _dump_prompt_input(path: str | Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    output_path = Path(path)
    ensure_dir(str(output_path.parent))
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_round_analyst(
    cfg: Mapping[str, Any],
    round_packet: Mapping[str, Any],
    *,
    logger=None,
    dump_path: str | Path | None = None,
) -> dict[str, Any]:
    settings = _resolve_settings(cfg, "round_analyst")
    messages = [
        {
            "role": "system",
            "content": (
                "你是量化研究复盘分析师。你必须依据输入证据总结本轮因子挖掘经验。"
                "成功原因必须以结构模式、经济学含义、市场行为机制为主，RIC、相关性、复杂度只能作为辅助证据，"
                "禁止把'因为RIC高/相关性低/复杂度低所以成功'写成主要原因。"
                "输出必须是严格 JSON 对象，不要包含任何额外文本。"
            ),
        },
        {
            "role": "user",
            "content": f"""请阅读以下本轮挖掘证据包，并输出严格 JSON。
输出 schema:
{{
  "round_id": "string_or_number",
  "round_overview": "string",
  "success_patterns": [
    {{
      "pattern": "string",
      "structure_summary": "string",
      "economic_reason": "string",
      "market_hypothesis": "string",
      "evidence_factors": [
        {{"factor_name": "string", "expression": "string", "explanation": "string"}}
      ]
    }}
  ],
  "failure_patterns": [
    {{
      "pattern": "string",
      "structure_summary": "string",
      "failure_reasoning": "string",
      "evidence_factors": [
        {{"factor_name": "string", "expression": "string", "explanation": "string"}}
      ]
    }}
  ],
  "stability_observations": [
    {{
      "factor": {{"factor_name": "string", "expression": "string", "explanation": "string"}},
      "interpretation": "string",
      "suggested_fix": "string"
    }}
  ],
  "next_round_guidance": {{
    "preferred_patterns": ["string"],
    "avoid_patterns": ["string"],
    "preferred_semantics": ["string"],
    "avoid_semantics": ["string"],
    "factor_direction": "string"
  }}
}}

要求：
1. `success_patterns` 和 `failure_patterns` 必须引用输入中的具体因子作为证据。
2. 可以结合 train/valid 指标讨论稳定性，但重点仍是因子结构和含义。
3. `market_hypothesis` 必须写成推断，不能写成确定事实。
4. `all_factor_cards` 已包含本轮全部因子，请综合分析，不要只看少数样本。

本轮证据包如下:
{json.dumps(round_packet, ensure_ascii=False)}""",
        },
    ]
    _dump_prompt_input(
        dump_path,
        {
            "agent": "round_analyst",
            "round_id": round_packet.get("round_id"),
            "messages": messages,
        },
    )
    try:
        content = _chat_text(messages, settings, logger=logger)
        parsed = _extract_json_object(content)
        if parsed:
            return parsed
    except Exception as exc:  # pragma: no cover
        if logger is not None:
            logger.warning("Round analyst failed (%s); using deterministic fallback.", exc)
    return _round_analyst_fallback(round_packet)


def _round_analyst_fallback(round_packet: Mapping[str, Any]) -> dict[str, Any]:
    cards = list(round_packet.get("all_factor_cards") or [])
    success_cards = [item for item in cards if str(item.get("final_status") or "").lower() == "success"]
    stability_cards = [item for item in cards if item.get("failure_bucket") == "stability_fail"]
    failure_cards = [item for item in cards if item.get("failure_bucket") != "success"]

    def _factor_stub(item: Mapping[str, Any]) -> dict[str, str]:
        return {
            "factor_name": str(item.get("factor_name") or ""),
            "expression": str(item.get("expression") or ""),
            "explanation": str(item.get("explanation") or ""),
        }

    success_pattern = ""
    failure_pattern = ""
    success_dist = round_packet.get("success_pattern_dist") or {}
    failure_dist = round_packet.get("failure_pattern_dist") or {}
    if success_dist:
        success_pattern = next(iter(success_dist.keys()))
    if failure_dist:
        failure_pattern = next(iter(failure_dist.keys()))

    return {
        "round_id": round_packet.get("round_id"),
        "round_overview": "本轮成功样本较少，请优先参考成功样本的结构含义，同时回避重复失败最集中的模式。",
        "success_patterns": [
            {
                "pattern": success_pattern,
                "structure_summary": "当前成功样本更接近已有成功结构。",
                "economic_reason": "成功样本可能更贴近价格与成交信息的有效错配刻画。",
                "market_hypothesis": "当前有效信号可能更来自局部交易行为，而非普遍趋势。",
                "evidence_factors": [_factor_stub(item) for item in success_cards[:3]],
            }
        ] if success_cards else [],
        "failure_patterns": [
            {
                "pattern": failure_pattern,
                "structure_summary": "当前失败样本集中在重复度更高的结构。",
                "failure_reasoning": "这些结构更可能接近库存量表达，难以提供新的经济含义。",
                "evidence_factors": [_factor_stub(item) for item in failure_cards[:3]],
            }
        ] if failure_cards else [],
        "stability_observations": [
            {
                "factor": _factor_stub(item),
                "interpretation": "该因子训练期有效但验证期转弱，说明结构稳定性仍需验证。",
                "suggested_fix": "保留主驱动结构，减少附加条件。",
            }
            for item in stability_cards[:3]
        ],
        "next_round_guidance": {
            "preferred_patterns": [success_pattern] if success_pattern else [],
            "avoid_patterns": [failure_pattern] if failure_pattern else [],
            "preferred_semantics": ["保留成功样本的主机制表达"],
            "avoid_semantics": ["重复已有库存量表达"] if failure_pattern else [],
            "factor_direction": "优先围绕成功样本的结构机制做延展，并避免重复失败模式。",
        },
    }


def run_retrieval_planner(
    cfg: Mapping[str, Any],
    retrieval_packet: Mapping[str, Any],
    *,
    factor_library: FactorLibraryIndex,
    logger=None,
    dump_path: str | Path | None = None,
) -> dict[str, Any]:
    recent_rounds = retrieval_packet.get("recent_round_memories") or []
    if not recent_rounds:
        _dump_prompt_input(
            dump_path,
            {
                "agent": "retrieval_planner",
                "status": "skipped",
                "reason": "no_recent_round_memories",
                "messages": [],
            },
        )
        return {}
    settings = _resolve_settings(cfg, "retrieval_planner")
    memory_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), Mapping) else {}
    retrieval_cfg = memory_cfg.get("retrieval_planner", {}) if isinstance(memory_cfg.get("retrieval_planner"), Mapping) else {}
    max_tool_calls = max(1, int(retrieval_cfg.get("tool_max_calls", 4) or 4))
    tool_result_limit = max(1, int(retrieval_cfg.get("tool_result_limit", 5) or 5))

    system_prompt = (
        "你是量化研究检索规划助手。你的任务是在下一轮主 LLM 挖因子前，"
        "基于近期轮次记忆提出结构偏好、回避方向、语义方向、修复方向，并给出可作为参考的具体因子。"
        "你可以使用一个本地工具 query_factor_library 来查询因子库。"
        "如果需要查库，请输出严格 JSON："
        '{"tool_call":{"name":"query_factor_library","arguments":{"query":"...","mode":"factor_name|pattern|keyword","limit":3}}}'
        "。如果已经足够，请直接输出最终 JSON。"
        '最终 JSON schema: {"market_summary":"string","preferred_patterns":[{"pattern":"string","reason":"string"}],"avoid_patterns":[{"pattern":"string","reason":"string"}],"preferred_semantics":["string"],"avoid_semantics":["string"],"repair_directions":["string"],"evidence_factors":[{"factor_name":"string","expression":"string","explanation":"string","references":["string"]}],"prompt_memo":"string"}.'
        "理由必须以结构机制和经济含义为主，不能把指标高低当作主要理由。"
        "输出必须是严格 JSON 对象，不要包含任何额外文本。"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "以下是近期轮次记忆包，请在必要时调用 query_factor_library，"
                "然后输出下一轮主 LLM 的记忆检索结果。\n"
                f"{json.dumps(retrieval_packet, ensure_ascii=False)}"
            ),
        },
    ]
    trace: dict[str, Any] = {
        "agent": "retrieval_planner",
        "status": "running",
        "recent_round_count": len(recent_rounds),
        "call_inputs": [],
    }

    for _ in range(max_tool_calls + 1):
        trace["call_inputs"].append(
            {
                "messages": [{"role": item["role"], "content": item["content"]} for item in messages],
            }
        )
        try:
            content = _chat_text(messages, settings, logger=logger)
            parsed = _extract_json_object(content)
        except Exception as exc:  # pragma: no cover
            if logger is not None:
                logger.warning("Retrieval planner failed (%s); using deterministic fallback.", exc)
            trace["status"] = "fallback"
            trace["error"] = str(exc)
            _dump_prompt_input(dump_path, trace)
            return _retrieval_planner_fallback(retrieval_packet)

        if not parsed:
            break
        tool_call = parsed.get("tool_call")
        if isinstance(tool_call, Mapping):
            args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), Mapping) else {}
            results = factor_library.query(
                str(args.get("query") or ""),
                mode=str(args.get("mode") or "auto"),
                limit=min(tool_result_limit, int(args.get("limit") or tool_result_limit)),
            )
            trace.setdefault("tool_calls", []).append(
                {
                    "tool_name": "query_factor_library",
                    "arguments": dict(args),
                    "result_count": len(results),
                    "results": results,
                }
            )
            messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": "TOOL_RESULT query_factor_library:\n" + json.dumps({"results": results}, ensure_ascii=False),
                }
            )
            continue
        trace["status"] = "completed"
        trace["final_output"] = parsed
        _dump_prompt_input(dump_path, trace)
        return parsed

    trace["status"] = "fallback"
    _dump_prompt_input(dump_path, trace)
    return _retrieval_planner_fallback(retrieval_packet)


def _retrieval_planner_fallback(retrieval_packet: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = retrieval_packet.get("aggregate_stats") or {}
    preferred_patterns = [
        {"pattern": item, "reason": "近期多轮记忆中反复出现成功证据。"}
        for item in list(aggregate.get("dominant_success_patterns") or [])[:3]
    ]
    avoid_patterns = [
        {"pattern": item, "reason": "近期多轮记忆中反复出现失败证据。"}
        for item in list(aggregate.get("dominant_failure_patterns") or [])[:3]
    ]
    evidence_factors: list[dict[str, Any]] = []
    for round_item in retrieval_packet.get("top_pass_rounds") or []:
        for factor in round_item.get("sample_factors") or []:
            evidence_factors.append(
                {
                    "factor_name": factor.get("factor_name", ""),
                    "expression": factor.get("expression", ""),
                    "explanation": factor.get("explanation", ""),
                    "references": factor.get("references", []),
                }
            )
            if len(evidence_factors) >= 5:
                break
        if len(evidence_factors) >= 5:
            break
    return {
        "market_summary": "近期应优先沿着最近成功模式延展，并回避重复失败最集中的结构。",
        "preferred_patterns": preferred_patterns,
        "avoid_patterns": avoid_patterns,
        "preferred_semantics": ["沿着近期成功样本的主机制继续展开"],
        "avoid_semantics": ["重复已有库存量表达"] if avoid_patterns else [],
        "repair_directions": ["对训练期有效但验证期转弱的结构，优先减少附加条件。"],
        "evidence_factors": evidence_factors,
        "prompt_memo": "优先参考近期成功样本的结构与含义，避免重复失败模式。",
    }
