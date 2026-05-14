#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph retrieval for GraphRAG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from reasoning import KGReasoner


def retrieve_evidence(question: str, triples_path: str = "data/triples.csv", max_hops: int = 3, limit: int = 12) -> Dict[str, Any]:
    r = KGReasoner(triples_path)
    matched = r.match_entities(question, top_k=8)
    triples: List[Dict[str, Any]] = []
    paths: List[Dict[str, Any]] = []
    conditions: List[Dict[str, Any]] = []

    # One-hop evidence around matched entities.
    for ent in matched:
        for t in r.out_edges.get(ent, []):
            if t["relation"] in {"causes", "is_a", "condition_of"}:
                triples.append(t)
        for t in r.in_edges.get(ent, []):
            if t["relation"] in {"causes", "is_a"}:
                triples.append(t)

    # Multi-hop causal paths between matched entity pairs.
    for i, src in enumerate(matched):
        for dst in matched[i + 1 :]:
            for a, b in [(src, dst), (dst, src)]:
                for path in r.causal_paths(a, b, max_hops=max_hops, limit=3):
                    edges = r.path_edges(path)
                    paths.append({"path": path, "edges": edges})
                    triples.extend(edges)

    # Conditional events related to matched entities.
    for ent in matched:
        events = r.condition_events(cause=ent) + r.condition_events(effect=ent) + r.condition_events(condition=ent)
        for ev in events[:5]:
            conditions.append(ev)
            triples.extend(ev["causes"] + ev["effects"] + ev["conditions"])

    # Deduplicate triples.
    seen = set()
    dedup = []
    for t in triples:
        key = (t.get("head"), t.get("relation"), t.get("tail"), t.get("source_doc_id"))
        if key not in seen:
            seen.add(key)
            dedup.append(t)
        if len(dedup) >= limit:
            break

    return {
        "question": question,
        "matched_entities": matched,
        "triples": dedup,
        "paths": paths[:5],
        "conditional_events": conditions[:5],
    }


def format_evidence(evidence: Dict[str, Any]) -> str:
    lines: List[str] = []
    if evidence.get("triples"):
        lines.append("[三元组证据]")
        for i, t in enumerate(evidence["triples"], 1):
            lines.append(
                f"{i}. {t['head']} --{t['relation']}--> {t['tail']} "
                f"(source={t.get('source_doc_id','')}, type={t.get('source_type','')}, evidence={t.get('evidence','')})"
            )
    if evidence.get("paths"):
        lines.append("\n[路径证据]")
        for i, p in enumerate(evidence["paths"], 1):
            lines.append(f"{i}. {' --causes--> '.join(p['path'])}")
    if evidence.get("conditional_events"):
        lines.append("\n[条件事件证据]")
        for i, ev in enumerate(evidence["conditional_events"], 1):
            cs = "、".join(t["tail"] for t in ev["causes"])
            es = "、".join(t["tail"] for t in ev["effects"])
            conds = "、".join(t["tail"] for t in ev["conditions"])
            lines.append(f"{i}. Event={ev['event_id']}: condition={conds}; cause={cs}; effect={es}")
    return "\n".join(lines) if lines else "知识图谱中没有检索到相关证据。"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--triples", default="data/triples.csv")
    p.add_argument("--question", required=True)
    p.add_argument("--max_hops", type=int, default=3)
    args = p.parse_args()
    ev = retrieve_evidence(args.question, args.triples, args.max_hops)
    print(json.dumps(ev, ensure_ascii=False, indent=2))
    print("\n--- formatted evidence ---")
    print(format_evidence(ev))


if __name__ == "__main__":
    main()
