#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate comprehensive test questions covering all relation types and reasoning types.

Outputs:
  data/processed/test_questions.json
  results/experiment_results.json
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_csv(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_indexes(triples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out_edges: Dict[str, List[Dict]] = defaultdict(list)
    in_edges: Dict[str, List[Dict]] = defaultdict(list)
    causal_adj: Dict[str, List[str]] = defaultdict(list)
    by_relation: Dict[str, List[Dict]] = defaultdict(list)
    for t in triples:
        h, r, tail = t["head"], t["relation"], t["tail"]
        out_edges[h].append(t)
        in_edges[tail].append(t)
        by_relation[r].append(t)
        if r == "causes":
            causal_adj[h].append(tail)
    return {"out_edges": out_edges, "in_edges": in_edges, "causal_adj": causal_adj, "by_relation": by_relation}


def find_2hop_paths(causal_adj: Dict[str, List[str]], max_paths: int = 50) -> List[Tuple[str, str, str]]:
    paths = []
    seen = set()
    for a, bs in causal_adj.items():
        for b in bs:
            for c in causal_adj.get(b, []):
                if c != a and (a, c) not in seen:
                    seen.add((a, c))
                    paths.append((a, b, c))
    random.shuffle(paths)
    return paths[:max_paths]


def generate_questions(triples: List[Dict[str, Any]], num_questions: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    idx = build_indexes(triples)
    by_rel = idx["by_relation"]
    causal_adj = idx["causal_adj"]
    out_edges = idx["out_edges"]
    in_edges = idx["in_edges"]

    questions: List[Dict[str, Any]] = []
    qid = 1

    def add(q: str, qtype: str, gold_evidence: list, gold_answer: str, tested_relations: list):
        nonlocal qid
        questions.append({
            "id": f"Q{qid:03d}",
            "question": q,
            "type": qtype,
            "gold_evidence": gold_evidence,
            "gold_answer": gold_answer,
            "tested_relations": tested_relations,
        })
        qid += 1

    # ── 1. Direct causal (causes) - 5 questions ──
    causes = by_rel["causes"]
    samples = random.sample(causes, min(5, len(causes)))
    for t in samples:
        add(
            f"{t['head']}可能导致什么？",
            "direct_causal",
            [[t["head"], "causes", t["tail"]]],
            f"{t['head']}可能导致{t['tail']}。",
            ["causes"],
        )

    # ── 2. Reverse causal (causes) - 5 questions ──
    samples = random.sample(causes, min(5, len(causes)))
    for t in samples:
        add(
            f"{t['tail']}可能由什么引起？",
            "reverse_causal",
            [[t["head"], "causes", t["tail"]]],
            f"{t['tail']}可能由{t['head']}引起。",
            ["causes"],
        )

    # ── 3. Multi-hop causal chain (2-hop, causes) - 5 questions ──
    hop2 = find_2hop_paths(causal_adj, max_paths=200)
    seen_heads = set()
    for a, b, c in hop2:
        if a not in seen_heads and len(a) < 20 and len(c) < 20:
            add(
                f"为什么{a}可能和{c}有关？",
                "multi_hop_causal",
                [[a, "causes", b], [b, "causes", c]],
                f"{a}可能通过{b}间接关联{c}。",
                ["causes"],
            )
            seen_heads.add(a)
        if len(seen_heads) >= 5:
            break

    # ── 4. Condition reasoning (condition_of) - 5 questions ──
    cond_rels = by_rel.get("condition_of", [])
    cond_by_cause = defaultdict(list)
    for t in cond_rels:
        cond_by_cause[t["tail"]].append(t)
    cond_items = [(cause, conds) for cause, conds in cond_by_cause.items() if len(cause) < 20]
    random.shuffle(cond_items)
    for cause, conds in cond_items[:5]:
        cond_names = "、".join(t["head"] for t in conds[:3])
        evidence = [[t["head"], "condition_of", t["tail"]] for t in conds[:3]]
        add(
            f"在什么条件下{cause}可能发生？",
            "condition_reasoning",
            evidence,
            f"{cause}的条件包括：{cond_names}。",
            ["condition_of"],
        )

    # ── 5. Hypernym (is_a) - 5 questions ──
    isa = by_rel.get("is_a", [])
    samples = random.sample(isa, min(5, len(isa)))
    for t in samples:
        add(
            f"{t['head']}属于什么类型？",
            "is_a",
            [[t["head"], "is_a", t["tail"]]],
            f"{t['head']}属于{t['tail']}。",
            ["is_a"],
        )

    # ── 6. Symptom (symptom_of) - 5 questions ──
    symp = by_rel.get("symptom_of", [])
    symp_by_disease = defaultdict(list)
    for t in symp:
        symp_by_disease[t["tail"]].append(t)
    symp_items = [(d, s) for d, s in symp_by_disease.items() if len(d) < 15]
    random.shuffle(symp_items)
    for disease, symptoms in symp_items[:5]:
        symp_names = "、".join(t["head"] for t in symptoms[:4])
        evidence = [[t["head"], "symptom_of", t["tail"]] for t in symptoms[:4]]
        add(
            f"{disease}有哪些症状？",
            "symptom_query",
            evidence,
            f"{disease}的症状包括：{symp_names}。",
            ["symptom_of"],
        )

    # ── 7. Treatment (treated_by) - 5 questions ──
    treat = by_rel.get("treated_by", [])
    treat_by_disease = defaultdict(list)
    for t in treat:
        treat_by_disease[t["head"]].append(t)
    treat_items = [(d, t) for d, t in treat_by_disease.items() if len(d) < 15]
    random.shuffle(treat_items)
    for disease, treatments in treat_items[:5]:
        treat_names = "、".join(t["tail"] for t in treatments[:4])
        evidence = [[t["head"], "treated_by", t["tail"]] for t in treatments[:4]]
        add(
            f"{disease}如何治疗？",
            "treatment_query",
            evidence,
            f"{disease}的治疗方式包括：{treat_names}。",
            ["treated_by"],
        )

    # ── 8. Located_in - 5 questions ──
    loc = by_rel.get("located_in", [])
    samples = random.sample(loc, min(5, len(loc)))
    for t in samples:
        add(
            f"{t['head']}发生在什么部位？",
            "location_query",
            [[t["head"], "located_in", t["tail"]]],
            f"{t['head']}发生在{t['tail']}。",
            ["located_in"],
        )

    # ── 9. Diagnosed_by - 5 questions ──
    diag = by_rel.get("diagnosed_by", [])
    samples = random.sample(diag, min(5, len(diag)))
    for t in samples:
        add(
            f"{t['head']}如何诊断？",
            "diagnosis_query",
            [[t["head"], "diagnosed_by", t["tail"]]],
            f"{t['head']}可通过{t['tail']}诊断。",
            ["diagnosed_by"],
        )

    # ── 10. Negative / hallucination check - 3 questions ──
    existing_pairs = {(t["head"], t["relation"], t["tail"]) for t in triples}
    ents = list({t["head"] for t in triples[:500]} | {t["tail"] for t in triples[:500]})
    attempts = 0
    count = 0
    while count < 3 and attempts < 300:
        a, b = random.sample(ents, 2)
        attempts += 1
        if (a, "causes", b) not in existing_pairs and len(a) < 15 and len(b) < 15:
            add(
                f"图谱中有没有证据说明{a}会导致{b}？",
                "negative_check",
                [],
                "知识图谱中没有足够证据支持该关系。",
                ["causes"],
            )
            count += 1

    # ── 11. Cross-relation reasoning: disease → symptoms + causes - 2 questions ──
    diseases_with_symp = {t["tail"] for t in symp}
    diseases_with_cause = {t["tail"] for t in causes}
    common_diseases = list(diseases_with_symp & diseases_with_cause)
    random.shuffle(common_diseases)
    for d in common_diseases[:2]:
        if len(d) < 15:
            d_symps = [t["head"] for t in symp_by_disease.get(d, [])[:3]]
            d_causes = [t["head"] for t in in_edges.get(d, []) if t["relation"] == "causes"][:3]
            if d_symps and d_causes:
                add(
                    f"{d}有哪些症状？可能由什么引起？",
                    "cross_relation",
                    [],
                    f"{d}的症状包括：{'、'.join(d_symps)}。可能由{'、'.join(d_causes)}引起。",
                    ["symptom_of", "causes"],
                )

    return questions[:num_questions]


def main():
    base = Path(__file__).resolve().parents[1]
    triples = read_csv(base / "data" / "triples.csv")
    questions = generate_questions(triples, num_questions=50)

    out_dir = base / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_questions.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    # Print summary
    from collections import Counter
    type_counts = Counter(q["type"] for q in questions)
    rel_counts = Counter()
    for q in questions:
        for r in q.get("tested_relations", []):
            rel_counts[r] += 1

    print(f"Generated {len(questions)} test questions")
    print(f"\nBy type:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")
    print(f"\nBy tested relation:")
    for r, c in rel_counts.most_common():
        print(f"  {r}: {c}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
