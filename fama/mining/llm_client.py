"""负责向 LLM 请求新因子的客户端适配层。"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[misc]

from fama.mining.prompt_builder import parse_llm_output


def request_new_factors(
    prompt: str,
    provider: str,
    model: str,
    api_key: str | None = None,
    temperature: float | None = None,
    thinking: str | None = None,
    base_url: str | None = None,
    allowed_fields: list[str] | None = None,
    max_references: int | None = None,
    reference_names: list[str] | None = None,
    parallel_calls: int | None = None,
    logger=None,
) -> list[dict]:
    """将提示词发送给配置的 LLM 服务端。

    Args:
        prompt: 编排器构造的提示词。
        provider: defaults.yaml 中的服务商标识，目前支持 ``openai``、``siliconflow``（DeepSeek 模型）。
        model: 目标模型名称。
        api_key: API 密钥，缺失时将使用回退逻辑。
        temperature: 采样温度，会直接传递给 OpenAI 接口。
        thinking: reasoning/“thinking”力度设置。
        base_url: 可选自定义 API 基础地址（例如 SiliconFlow 的 OpenAI 兼容端点）。
        allowed_fields: 允许引用的字段列表，用于 fallback 生成时保障安全。
        max_references: references 字段允许的最大长度。
        reference_names: 可供参考的因子名列表，用于 fallback 输出。
        logger: 可选日志记录器，用于输出原始响应。

    Returns:
        因子表达式的字典列表；没有密钥或 SDK 不可用时退回确定性样本。
    """

    parallel = max(1, int(parallel_calls or 1))

    def _call_once() -> list[dict]:
        provider_l = provider.lower()
        # OpenAI 兼容分支：原生 OpenAI 或 SiliconFlow/OpenRouter 之类的兼容接口
        if provider_l in {"openai", "siliconflow", "deepseek"}:
            if OpenAI is None:
                raise RuntimeError(
                    "未安装 openai SDK，无法调用真实 LLM。请 `pip install openai` 或提供其他提供商。"
                )
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=float(temperature or 0.2),
                    messages=[
                        {"role": "system", "content": "You are a quant researcher. Return strict JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                )
            except Exception as exc:  # pragma: no cover - 网络/SDK 异常
                if logger is not None:
                    logger.warning("LLM call failed (%s); falling back to deterministic output.", exc)
                return _fallback_generation(
                    prompt,
                    allowed_fields=allowed_fields,
                    max_references=max_references,
                    reference_names=reference_names,
                    logger=logger,
                )
            content = (resp.choices[0].message.content or "").strip()
            if logger is not None:
                logger.info("LLM raw output: %s", content)
            return parse_llm_output(content, max_references=max_references)

        # 未知 provider 走可重复的伪造输出
        if OpenAI is None:
            raise RuntimeError(
                "未安装 openai SDK，无法调用真实 LLM。请 `pip install openai` 或提供其他提供商。"
            )
        return _fallback_generation(
            prompt,
            allowed_fields=allowed_fields,
            max_references=max_references,
            reference_names=reference_names,
            logger=logger,
        )

    if parallel == 1:
        return _call_once()

    if logger is not None:
        logger.info("Launching %d parallel LLM calls for diversity.", parallel)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [executor.submit(_call_once) for _ in range(parallel)]
        for fut in as_completed(futures):
            try:
                batch = fut.result() or []
                results.extend(batch)
            except Exception as exc:  # pragma: no cover - 防御并发异常
                if logger is not None:
                    logger.warning("One parallel LLM call failed: %s", exc)

    deduped: list[dict] = []
    seen: set[str] = set()
    for item in results:
        expr = None
        if isinstance(item, dict):
            expr = item.get("expression")
        if not expr or expr in seen:
            continue
        seen.add(expr)
        deduped.append(item)
    return deduped


def _fallback_generation(
    prompt: str,
    allowed_fields: list[str] | None = None,
    max_references: int | None = None,
    reference_names: list[str] | None = None,
    logger=None,
) -> list[str]:
    """在无法访问真实 LLM 时生成可复现且受限于字段列表的伪造因子。"""

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    seeds = [digest[i : i + 8] for i in range(0, len(digest), 8)][:3]
    fields = allowed_fields or ["CLOSE", "VWAP", "RET"]
    responses = []
    ref_names = reference_names or []
    for i, seed in enumerate(seeds):
        field = fields[i % len(fields)]
        refs = ref_names[i : i + 1] if ref_names else []
        if max_references and max_references > 0:
            refs = refs[:max_references]
        responses.append(
            {
                "expression": f"RANK({field}) + 0.{seed[:3]} * DELTA({field}, {i + 1})",
                "explanation": None,
                "references": refs,
            }
        )
    import json

    text = json.dumps(responses, ensure_ascii=False)
    if logger is not None:
        logger.info("Fallback raw output: %s", text)
    return parse_llm_output(text, max_references=max_references)
