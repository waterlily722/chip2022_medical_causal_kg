#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample-based relation discovery using Qwen."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

from config import load_qwen_settings
from build_kg import extract_json_array, make_doc_id, read_json


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl_row(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def sample_indices(n: int, k: int, seed: int) -> List[int]:
    if n <= k:
        return list(range(n))
    rng = random.Random(seed)
    return rng.sample(range(n), k)


class QwenDiscoveryClient:
    def __init__(self, settings) -> None:
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
                {"role": "system", "content": "You are a careful medical IE assistant. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


def discover_relations(
    raw_dir: Path,
    prompt_path: Path,
    files: List[str],
    sample_size: int,
    seed: int,
    sleep_seconds: float,
    output_path: Path,
    client: QwenDiscoveryClient,
) -> Dict[str, Any]:
    prompt_tpl = prompt_path.read_text(encoding="utf-8")
    if not client.enabled:
        raise RuntimeError("Qwen API is not configured. Provide QWEN_API_KEY in .env first.")

    relation_counts = Counter()
    per_file_counts: Dict[str, Counter] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for file_name in files:
            path = raw_dir / file_name
            if not path.exists():
                append_jsonl_row(handle, {"source_file": file_name, "error": "missing_file"})
                continue
            data = read_json(path)
            indices = sample_indices(len(data), sample_size, seed)
            per_file_counts[file_name] = Counter()
            for idx in indices:
                item = data[idx] if idx < len(data) else None
                text = item.get("text", "") if isinstance(item, dict) else ""
                doc_id = make_doc_id(file_name, idx)
                if not text:
                    append_jsonl_row(
                        handle,
                        {"source_file": file_name, "source_doc_id": doc_id, "error": "empty_text"},
                    )
                    continue
                prompt = prompt_tpl.replace("{text}", text)
                try:
                    content = client.chat(prompt)
                    relations = extract_json_array(content)
                except Exception as exc:
                    append_jsonl_row(
                        handle,
                        {
                            "source_file": file_name,
                            "source_doc_id": doc_id,
                            "error": str(exc),
                        },
                    )
                    continue

                for rel in relations:
                    label = rel.get("relation") or rel.get("rel") or rel.get("type")
                    if label:
                        relation_counts[label] += 1
                        per_file_counts[file_name][label] += 1

                append_jsonl_row(
                    handle,
                    {
                        "source_file": file_name,
                        "source_doc_id": doc_id,
                        "relations": relations,
                    },
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

    return {
        "outputs": outputs,
        "relation_counts": relation_counts,
        "per_file_counts": per_file_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--files", default="unlabel.json,testA.json,testB.json")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="prompts/relation_discovery_prompt.txt")
    parser.add_argument("--sleep_seconds", type=float, default=0.2)
    parser.add_argument("--output", default="results/relation_discovery.jsonl")
    parser.add_argument("--summary", default="results/relation_discovery_summary.json")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    prompt_path = Path(args.prompt)
    files = [x.strip() for x in args.files.split(",") if x.strip()]

    settings = load_qwen_settings()
    status = "yes" if settings.configured else "no"
    print(f"Qwen API key configured: {status} (config: {settings.config_path})")
    client = QwenDiscoveryClient(settings)

    result = discover_relations(
        raw_dir=raw_dir,
        prompt_path=prompt_path,
        files=files,
        sample_size=args.sample,
        seed=args.seed,
        sleep_seconds=args.sleep_seconds,
        output_path=Path(args.output),
        client=client,
    )

    summary = {
        "relation_counts": dict(result["relation_counts"].most_common()),
        "per_file_counts": {
            k: dict(v.most_common()) for k, v in result["per_file_counts"].items()
        },
    }
    write_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
