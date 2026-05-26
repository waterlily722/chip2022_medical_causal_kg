#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reasoning over the Medical Causal Event KG.

Supported reasoning:
1. Direct causal reasoning: X -> ?
2. Reverse causal reasoning: ? -> Y
3. Multi-hop causal path reasoning: X -> ... -> Y
4. Condition reasoning: condition_of (condition -> cause)
5. Hypernym reasoning: X is_a ?
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def load_triples(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class KGReasoner:
    def __init__(self, triples_path: str | Path = "data/triples.csv") -> None:
        self.triples = load_triples(triples_path)
        self.entities: Set[str] = set()
        self.out_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.in_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.causal_adj: Dict[str, List[str]] = defaultdict(list)
        self.causal_edge_lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self._build_indexes()

    def _build_indexes(self) -> None:
        for t in self.triples:
            h, r, tail = t["head"], t["relation"], t["tail"]
            self.entities.add(h)
            self.entities.add(tail)
            self.out_edges[h].append(t)
            self.in_edges[tail].append(t)
            if r == "causes":
                self.causal_adj[h].append(tail)
                self.causal_edge_lookup[(h, tail)].append(t)

    def match_entities(self, query: str, top_k: int = 8) -> List[str]:
        """Exact substring matcher, longest mentions first."""
        candidates = [e for e in self.entities if e and not e.startswith("Doc_") and e in query]
        candidates.sort(key=lambda x: (-len(x), x))
        dedup = []
        for c in candidates:
            if not any(c in d and c != d for d in dedup):
                dedup.append(c)
        return dedup[:top_k]

    def direct_effects(self, cause: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [t for t in self.out_edges.get(cause, []) if t["relation"] == "causes"][:limit]

    def direct_causes(self, effect: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [t for t in self.in_edges.get(effect, []) if t["relation"] == "causes"][:limit]

    def hypernyms(self, entity: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [t for t in self.out_edges.get(entity, []) if t["relation"] == "is_a"][:limit]

    def hyponyms(self, category: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [t for t in self.in_edges.get(category, []) if t["relation"] == "is_a"][:limit]

    def causal_paths(self, source: str, target: str, max_hops: int = 3, limit: int = 5) -> List[List[str]]:
        if source == target:
            return [[source]]
        paths: List[List[str]] = []
        q = deque([[source]])
        visited_depth = {source: 0}
        while q and len(paths) < limit:
            path = q.popleft()
            node = path[-1]
            if len(path) - 1 >= max_hops:
                continue
            for nxt in self.causal_adj.get(node, []):
                if nxt in path:
                    continue
                new_path = path + [nxt]
                if nxt == target:
                    paths.append(new_path)
                elif visited_depth.get(nxt, 999) > len(new_path):
                    visited_depth[nxt] = len(new_path)
                    q.append(new_path)
        return paths

    def path_edges(self, path: List[str]) -> List[Dict[str, Any]]:
        edges = []
        for a, b in zip(path, path[1:]):
            candidates = self.causal_edge_lookup.get((a, b), [])
            if candidates:
                edges.append(candidates[0])
        return edges

    def conditions_for_cause(self, cause: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [t for t in self.in_edges.get(cause, []) if t["relation"] == "condition_of"][:limit]

    def symptoms_of(self, disease: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [t for t in self.in_edges.get(disease, []) if t["relation"] == "symptom_of"][:limit]

    def treatments_of(self, disease: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [t for t in self.out_edges.get(disease, []) if t["relation"] == "treated_by"][:limit]

    def location_of(self, disease: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [t for t in self.out_edges.get(disease, []) if t["relation"] == "located_in"][:limit]

    def diagnosis_of(self, disease: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [t for t in self.out_edges.get(disease, []) if t["relation"] == "diagnosed_by"][:limit]

    @staticmethod
    def triple_to_text(t: Dict[str, Any]) -> str:
        return f"{t['head']} --{t['relation']}--> {t['tail']}"

    def _fmt(self, triples: List[Dict[str, Any]], key: str) -> str:
        return "、".join(t[key] for t in triples[:10])

    def answer_query(self, query: str) -> Dict[str, Any]:
        ents = self.match_entities(query)
        answer_lines: List[str] = []
        evidence: List[Dict[str, Any]] = []
        q = query
        ent = ents[0] if ents else None

        # Multi-hop: if at least two entities appear and the question asks why/relationship/indirect.
        if len(ents) >= 2 and any(k in q for k in ["为什么", "是否", "间接", "有关", "关系", "路径"]):
            for src in ents:
                for dst in ents:
                    if src == dst:
                        continue
                    paths = self.causal_paths(src, dst, max_hops=3, limit=3)
                    if paths:
                        for path in paths:
                            edges = self.path_edges(path)
                            evidence.extend(edges)
                            answer_lines.append(f"存在因果路径：{' --causes--> '.join(path)}")
                        return {"query": query, "matched_entities": ents, "answer": "\n".join(answer_lines), "evidence": evidence}

        # Condition reasoning (condition_of: condition -> cause).
        if ent and any(k in q for k in ["条件", "情况下", "条件下"]):
            cond_edges = self.conditions_for_cause(ent)
            if cond_edges:
                conditions = "、".join(t["head"] for t in cond_edges[:10])
                answer_lines.append(f"可能的条件包括：{conditions}")
                evidence.extend(cond_edges)
                return {"query": query, "matched_entities": ents, "answer": "\n".join(answer_lines), "evidence": evidence}

        # Hypernym reasoning.
        if ent and any(k in q for k in ["属于", "类型", "哪类", "类别"]):
            h = self.hypernyms(ent)
            if h:
                evidence.extend(h)
                answer_lines.append(f"{ent} 属于：" + "、".join(t["tail"] for t in h[:10]))
                return {"query": query, "matched_entities": ents, "answer": "\n".join(answer_lines), "evidence": evidence}
            hypo = self.hyponyms(ent)
            if hypo:
                evidence.extend(hypo)
                answer_lines.append(f"{ent} 包括：" + "、".join(t["head"] for t in hypo[:10]))
                return {"query": query, "matched_entities": ents, "answer": "\n".join(answer_lines), "evidence": evidence}

        # Symptom reasoning.
        if ent and any(k in q for k in ["症状", "表现", "体征"]):
            s = self.symptoms_of(ent)
            if s:
                evidence.extend(s)
                answer_lines.append(f'“{ent}”的症状包括：' + self._fmt(s, "head"))
                return {"query": query, "matched_entities": ents, "answer": "\n".join(answer_lines), "evidence": evidence}

        # Treatment reasoning.
        if ent and any(k in q for k in ["治疗", "如何治疗", "怎么治", "用药", "干预"]):
            t = self.treatments_of(ent)
            if t:
                evidence.extend(t)
                answer_lines.append(f'"{ent}"的治疗方式包括：' + self._fmt(t, "tail"))
                return {"query": query, "matched_entities": ents, "answer": "\n".join(answer_lines), "evidence": evidence}

        # Location reasoning.
        if ent and any(k in q for k in ["部位", "位置", "在哪", "发生"]):
            loc = self.location_of(ent)
            if loc:
                evidence.extend(loc)
                answer_lines.append(f'"{ent}"发生在：' + self._fmt(loc, "tail"))
                return {"query": query, "matched_entities": ents, "answer": "\n".join(answer_lines), "evidence": evidence}

        # Diagnosis reasoning.
        if ent and any(k in q for k in ["诊断", "检查", "怎么确诊", "如何确诊"]):
            d = self.diagnosis_of(ent)
            if d:
                evidence.extend(d)
                answer_lines.append(f'"{ent}"可通过以下方式诊断：' + self._fmt(d, "tail"))
                return {"query": query, "matched_entities": ents, "answer": "\n".join(answer_lines), "evidence": evidence}

        # Reverse causal.
        if ent and any(k in q for k in ["由什么", "什么引起", "原因", "导致它", "造成"]):
            c = self.direct_causes(ent)
            if c:
                evidence.extend(c)
                answer_lines.append(f'可能导致"{ent}"的因素包括：' + "、".join(t["head"] for t in c[:10]))
                return {"query": query, "matched_entities": ents, "answer": "\n".join(answer_lines), "evidence": evidence}

        # Direct causal.
        if ent:
            e = self.direct_effects(ent)
            if e:
                evidence.extend(e)
                answer_lines.append(f'"{ent}"可能导致：' + "、".join(t["tail"] for t in e[:10]))
                return {"query": query, "matched_entities": ents, "answer": "\n".join(answer_lines), "evidence": evidence}

        return {"query": query, "matched_entities": ents, "answer": "知识图谱中没有足够证据。", "evidence": []}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--triples", default="data/triples.csv")
    p.add_argument("--query", required=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    r = KGReasoner(args.triples)
    result = r.answer_query(args.query)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("问题：", result["query"])
        print("匹配实体：", "、".join(result["matched_entities"]) or "无")
        print("回答：")
        print(result["answer"])
        if result["evidence"]:
            print("证据链：")
            for t in result["evidence"][:10]:
                print("-", KGReasoner.triple_to_text(t), "|", t.get("evidence", "")[:120])


if __name__ == "__main__":
    main()
