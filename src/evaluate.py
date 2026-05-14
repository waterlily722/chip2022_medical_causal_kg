#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation utilities for the course project.

This script generates:
- graph statistics
- automatic QA cases from KG
- extraction metrics if qwen_extracted_clean.json and gold_triples.csv exist
- a markdown case report
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from graph_retrieval import format_evidence, retrieve_evidence
from reasoning import KGReasoner


def read_csv(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def graph_stats(triples: List[Dict[str, Any]]) -> Dict[str, Any]:
    entities = set()
    for t in triples:
        entities.add(t["head"])
        entities.add(t["tail"])
    return {
        "num_entities": len(entities),
        "num_triples": len(triples),
        "num_entity_types": len({t["head_type"] for t in triples} | {t["tail_type"] for t in triples}),
        "num_relation_types": len({t["relation"] for t in triples}),
        "relation_counts": dict(Counter(t["relation"] for t in triples)),
        "source_type_counts": dict(Counter(t.get("source_type", "") for t in triples)),
    }


def triple_key(t: Dict[str, Any]) -> Tuple[str, str, str]:
    return (t.get("head", ""), t.get("relation", ""), t.get("tail", ""))


def extraction_metrics(gold_path: str | Path, qwen_clean_path: str | Path) -> Dict[str, Any]:
    gold = [t for t in read_csv(gold_path) if t.get("relation") in {"causes", "is_a", "condition_of"}]
    pred_raw = []
    p = Path(qwen_clean_path)
    if p.exists():
        try:
            pred_raw = json.load(p.open("r", encoding="utf-8"))
        except Exception:
            pred_raw = []
    pred = [t for t in pred_raw if t.get("relation") in {"causes", "is_a", "condition_of"}]
    gold_set = {triple_key(t) for t in gold}
    pred_set = {triple_key(t) for t in pred}
    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    rel_scores = {}
    for rel in ["causes", "condition_of", "is_a"]:
        g = {triple_key(t) for t in gold if t.get("relation") == rel}
        pr = {triple_key(t) for t in pred if t.get("relation") == rel}
        rel_tp = len(g & pr)
        p_rel = rel_tp / len(pr) if pr else 0.0
        r_rel = rel_tp / len(g) if g else 0.0
        f_rel = 2 * p_rel * r_rel / (p_rel + r_rel) if p_rel + r_rel else 0.0
        rel_scores[rel] = {"precision": p_rel, "recall": r_rel, "f1": f_rel, "gold": len(g), "pred": len(pr), "tp": rel_tp}

    return {
        "note": "Exact-match metrics. Meaningful only if Qwen extraction was run on comparable train texts.",
        "gold_count": len(gold_set),
        "pred_count": len(pred_set),
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "relation_wise": rel_scores,
    }


def generate_questions(triples: List[Dict[str, Any]], max_questions: int = 30, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    causes = [t for t in triples if t["relation"] == "causes"]
    isa = [t for t in triples if t["relation"] == "is_a"]
    event_cause = defaultdict(list)
    event_effect = defaultdict(list)
    event_condition = defaultdict(list)
    for t in triples:
        if t["relation"] == "event_cause":
            event_cause[t["head"]].append(t)
        elif t["relation"] == "event_effect":
            event_effect[t["head"]].append(t)
        elif t["relation"] == "has_condition":
            event_condition[t["head"]].append(t)

    questions: List[Dict[str, Any]] = []
    qid = 1

    for t in causes[:8]:
        questions.append({
            "id": f"Q{qid:03d}",
            "question": f"{t['head']}可能导致什么？",
            "type": "direct_causal",
            "gold_evidence": [[t["head"], t["relation"], t["tail"]]],
            "gold_answer": f"{t['head']}可能导致{t['tail']}。",
            "source": "kg_auto_generated",
        })
        qid += 1
        questions.append({
            "id": f"Q{qid:03d}",
            "question": f"{t['tail']}可能由什么引起？",
            "type": "reverse_causal",
            "gold_evidence": [[t["head"], t["relation"], t["tail"]]],
            "gold_answer": f"{t['tail']}可能由{t['head']}引起。",
            "source": "kg_auto_generated",
        })
        qid += 1
        if qid > max_questions:
            break

    # Multi-hop: A->B and B->C.
    out = defaultdict(list)
    for t in causes:
        out[t["head"]].append(t)
    for t1 in causes:
        for t2 in out.get(t1["tail"], [])[:3]:
            questions.append({
                "id": f"Q{qid:03d}",
                "question": f"为什么{t1['head']}可能和{t2['tail']}有关？",
                "type": "multi_hop_causal",
                "gold_evidence": [[t1["head"], "causes", t1["tail"]], [t2["head"], "causes", t2["tail"]]],
                "gold_answer": f"{t1['head']}可能通过{t1['tail']}间接关联{t2['tail']}。",
                "source": "kg_path_generated",
            })
            qid += 1
            break
        if qid > max_questions:
            break

    for t in isa[:5]:
        questions.append({
            "id": f"Q{qid:03d}",
            "question": f"{t['head']}属于什么类型？",
            "type": "is_a",
            "gold_evidence": [[t["head"], "is_a", t["tail"]]],
            "gold_answer": f"{t['head']}属于{t['tail']}。",
            "source": "kg_auto_generated",
        })
        qid += 1
        if qid > max_questions:
            break

    for eid in list(set(event_cause) & set(event_effect) & set(event_condition))[:5]:
        c = event_cause[eid][0]["tail"]
        e = event_effect[eid][0]["tail"]
        cond = event_condition[eid][0]["tail"]
        questions.append({
            "id": f"Q{qid:03d}",
            "question": f"在什么条件下{c}可能导致{e}？",
            "type": "conditional_causal",
            "gold_evidence": [[eid, "event_cause", c], [eid, "event_effect", e], [eid, "has_condition", cond]],
            "gold_answer": f"在{cond}条件下，{c}可能导致{e}。",
            "source": "causal_event_generated",
        })
        qid += 1
        if qid > max_questions:
            break

    # Negative hallucination checks.
    ents = list({t["head"] for t in causes[:200]} | {t["tail"] for t in causes[:200]})
    existing = {(t["head"], t["tail"]) for t in causes}
    attempts = 0
    while qid <= max_questions and len(ents) >= 2 and attempts < 200:
        a, b = random.sample(ents, 2)
        attempts += 1
        if (a, b) in existing:
            continue
        questions.append({
            "id": f"Q{qid:03d}",
            "question": f"图谱中有没有证据说明{a}会导致{b}？",
            "type": "negative_hallucination_check",
            "gold_evidence": [],
            "gold_answer": "知识图谱中没有足够证据支持该关系。",
            "source": "negative_generated",
        })
        qid += 1

    return questions[:max_questions]


def evaluate_retrieval(questions: List[Dict[str, Any]], triples_path: str | Path) -> Dict[str, Any]:
    total = len(questions)
    hit = 0
    evidence_nonempty = 0
    for q in questions:
        ev = retrieve_evidence(q["question"], str(triples_path))
        retrieved = {(t["head"], t["relation"], t["tail"]) for t in ev.get("triples", [])}
        gold = {tuple(x) for x in q.get("gold_evidence", [])}
        if not gold:
            # For negative questions, success means no direct matching evidence.
            if not retrieved:
                hit += 1
        elif gold & retrieved or ev.get("paths"):
            hit += 1
        if retrieved or ev.get("paths") or ev.get("conditional_events"):
            evidence_nonempty += 1
    return {
        "num_questions": total,
        "retrieval_hit_rate": hit / total if total else 0.0,
        "evidence_nonempty_rate": evidence_nonempty / total if total else 0.0,
    }


def write_cases_md(path: str | Path, questions: List[Dict[str, Any]], triples_path: str | Path, max_cases: int = 12) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# GraphRAG Cases", ""]
    for q in questions[:max_cases]:
        ev = retrieve_evidence(q["question"], str(triples_path))
        lines.append(f"## {q['id']} {q['type']}")
        lines.append(f"**Question:** {q['question']}")
        lines.append("")
        lines.append(f"**Gold answer:** {q.get('gold_answer','')}")
        lines.append("")
        lines.append("**Retrieved evidence:**")
        lines.append("```text")
        lines.append(format_evidence(ev))
        lines.append("```")
        lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--triples", default="data/triples.csv")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--max_questions", type=int, default=30)
    args = p.parse_args()

    processed = Path(args.processed_dir)
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)

    triples = read_csv(args.triples)
    stats = graph_stats(triples)
    questions = generate_questions(triples, max_questions=args.max_questions)
    write_json(processed / "test_questions.json", questions)

    retrieval_eval = evaluate_retrieval(questions, args.triples)
    extr_eval = extraction_metrics(processed / "gold_triples.csv", processed / "qwen_extracted_clean.json")

    metrics = {
        "graph_stats": stats,
        "retrieval_eval": retrieval_eval,
        "extraction_eval": extr_eval,
    }
    write_json(results / "metrics.json", metrics)
    write_json(results / "extraction_metrics.json", extr_eval)
    write_cases_md(results / "cases.md", questions, args.triples)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[INFO] Test questions saved to {processed / 'test_questions.json'}")
    print(f"[INFO] Cases saved to {results / 'cases.md'}")


if __name__ == "__main__":
    main()
