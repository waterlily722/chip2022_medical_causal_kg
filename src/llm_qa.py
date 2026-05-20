#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM QA with three modes:
- llm_only: ask Qwen directly
- kg_only: answer with graph templates
- kg_augmented: retrieve graph evidence, then ask Qwen with grounded prompt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

from config import load_qwen_settings
from graph_retrieval import format_evidence, retrieve_evidence
from reasoning import KGReasoner


class QwenChatClient:
    def __init__(self) -> None:
        settings = load_qwen_settings()
        status = "yes" if settings.configured else "no"
        print(f"Qwen API key configured: {status} (config: {settings.config_path})")
        self.api_key = settings.api_key
        self.base_url = settings.base_url
        self.model = settings.model
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.enabled = bool(self.api_key and OpenAI)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.enabled else None

    def chat(self, prompt: str) -> str:
        if not self.enabled or self.client is None:
            raise RuntimeError(
                "Qwen API is not configured. Set QWEN_API_KEY in .env."
            )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是严谨的医学知识问答助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


def kg_only_answer(question: str, triples_path: str = "data/triples.csv") -> Dict[str, Any]:
    r = KGReasoner(triples_path)
    result = r.answer_query(question)
    evidence_lines = []
    for t in result.get("evidence", [])[:10]:
        evidence_lines.append(f"{t['head']} --{t['relation']}--> {t['tail']}")
    text = (
        f"答案：{result['answer']}\n"
        f"证据链：" + ("；".join(evidence_lines) if evidence_lines else "无") + "\n"
        f"图谱不足或注意事项：本回答仅基于当前知识图谱，不构成医疗建议。"
    )
    return {"mode": "kg_only", "answer": text, "evidence": result}


def llm_only_answer(question: str) -> Dict[str, Any]:
    client = QwenChatClient()
    prompt = (
        "请回答下面的医学知识问题。注意：本回答仅用于学习，不构成医疗建议。\n\n"
        f"问题：{question}"
    )
    answer = client.chat(prompt)
    return {"mode": "llm_only", "answer": answer, "evidence": None}


def kg_augmented_answer(
    question: str,
    triples_path: str = "data/triples.csv",
    prompt_path: str = "prompts/kg_augmented_prompt.txt",
    fallback_to_kg_only: bool = True,
) -> Dict[str, Any]:
    evidence = retrieve_evidence(question, triples_path=triples_path)
    evidence_text = format_evidence(evidence)
    prompt_tpl = Path(prompt_path).read_text(encoding="utf-8")
    prompt = prompt_tpl.replace("{question}", question).replace("{evidence}", evidence_text)

    client = QwenChatClient()
    try:
        answer = client.chat(prompt)
        mode = "kg_augmented"
    except Exception as e:
        if not fallback_to_kg_only:
            raise
        # This makes the framework runnable without API keys.
        kg_result = kg_only_answer(question, triples_path)
        answer = (
            kg_result["answer"]
            + f"\n\n[提示] 未调用 Qwen，原因：{e}。当前返回 KG-only 模板答案。"
        )
        mode = "kg_augmented_fallback_kg_only"
    return {"mode": mode, "answer": answer, "evidence": evidence, "prompt": prompt}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True)
    p.add_argument("--mode", choices=["llm_only", "kg_only", "kg_augmented"], default="kg_augmented")
    p.add_argument("--triples", default="data/triples.csv")
    p.add_argument("--prompt", default="prompts/kg_augmented_prompt.txt")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.mode == "llm_only":
        result = llm_only_answer(args.question)
    elif args.mode == "kg_only":
        result = kg_only_answer(args.question, args.triples)
    else:
        result = kg_augmented_answer(args.question, args.triples, args.prompt)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])
        if result.get("evidence"):
            print("\n--- 检索证据 ---")
            print(format_evidence(result["evidence"]))


if __name__ == "__main__":
    main()
