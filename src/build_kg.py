#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Medical Causal Event Knowledge Graph from CHIP2022 data.

Design:
- train_0717.json labels -> Gold Seed KG
- unlabel/testA/testB texts -> Qwen-extracted KG (optional, enabled with --extract_qwen)
- condition_of relations are converted to CausalEvent nodes.

Outputs:
- data/triples.csv
- data/processed/entities.csv
- data/processed/relations.csv
- data/processed/causal_events.csv
- data/processed/kg.json
- results/kg_visualization.html
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

try:
    from pyvis.network import Network
except Exception:  # pragma: no cover
    Network = None

from config import load_qwen_settings


# CHIP2022 relation ids in example_code.txt
CAUSAL_RELATION = 1
CONDITIONAL_RELATION = 2
HYPERNYM_RELATION = 3

ALLOWED_ENTITY_TYPES = {
    "Disease",
    "Symptom",
    "ClinicalSign",
    "PathologicalState",
    "RiskFactor",
    "TestResult",
    "ExamProcedure",
    "Treatment",
    "AnatomicalSite",
    "MedicalCategory",
    "CausalEvent",
    "Document",
    "Other",
}

ALLOWED_RELATIONS = {
    "causes",
    "risk_factor_for",
    "condition_of",
    "is_a",
    "symptom_of",
    "treated_by",
    "located_in",
    "diagnosed_by",
}

CORE_RELATIONS = {
    "causes",
    "risk_factor_for",
    "condition_of",
    "is_a",
    "symptom_of",
    "treated_by",
    "located_in",
    "diagnosed_by",
}
EVENT_RELATIONS = {"event_cause", "event_effect", "has_condition"}


@dataclass(frozen=True)
class Triple:
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str
    evidence: str
    source_doc_id: str
    source_file: str
    source_type: str  # gold / qwen / inferred
    confidence: float = 1.0


def read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    return s


def safe_float(x: Any, default: float = 1.0) -> float:
    try:
        v = float(x)
        if v < 0:
            return 0.0
        if v > 1:
            return 1.0
        return v
    except Exception:
        return default


def make_doc_id(source_file: str, idx: int) -> str:
    stem = source_file.replace(".json", "")
    return f"Doc_{stem}_{idx:04d}"


def event_id_for(cause: str, effect: str, condition: str, doc_id: str) -> str:
    raw = f"{doc_id}|{cause}|{effect}|{condition}".encode("utf-8")
    return "Event_" + hashlib.md5(raw).hexdigest()[:12]


def evidence_span(text: str, mentions: Iterable[str], window: int = 35) -> str:
    """Return a compact evidence span containing as many mentions as possible."""
    mentions = [m for m in mentions if m]
    if not text or not mentions:
        return ""
    positions = []
    for m in mentions:
        pos = text.find(m)
        if pos >= 0:
            positions.append((pos, pos + len(m)))
    if not positions:
        return text[:120]
    start = max(0, min(p[0] for p in positions) - window)
    end = min(len(text), max(p[1] for p in positions) + window)
    return text[start:end]


def infer_entity_type(name: str, relation: Optional[str] = None, role: Optional[str] = None) -> str:
    """A lightweight rule-based medical mention classifier.

    CHIP2022 labels do not include entity types, so we use heuristic rules.
    Qwen extraction can override this by providing *_type fields.
    """
    n = norm_text(name)
    if not n:
        return "Other"
    if n.startswith("Doc_"):
        return "Document"
    if n.startswith("Event_"):
        return "CausalEvent"

    disease_keywords = ["病", "炎", "癌", "症", "瘤", "感染", "梗死", "卒中", "高血压", "糖尿病", "癫痫", "感冒", "青光眼", "肝硬化", "肺炎", "胃溃疡", "肠炎"]
    symptom_keywords = ["痛", "疼", "咳", "发热", "发烧", "呕吐", "恶心", "腹泻", "便血", "乏力", "瘙痒", "肿", "麻木", "头晕", "失眠", "耳鸣", "出血", "减少", "增多", "丧失", "偏瘫", "呼吸", "心跳"]
    pathology_keywords = ["硬化", "粘连", "衰退", "受压", "增高", "下降", "损害", "坏死", "水肿", "潴留", "破裂", "狭窄", "异常", "功能", "病变", "空洞", "梗阻"]
    risk_keywords = ["病毒", "细菌", "劳累", "饮水", "饮食", "肥胖", "过敏", "受凉", "外伤", "吸烟", "饮酒", "辛辣", "感染", "用力", "年龄", "抵抗力"]
    test_keywords = ["激素", "血压", "眼压", "指标", "阳性", "阴性", "水平", "结果", "数值", "浓度"]
    exam_keywords = ["检查", "检测", "化验", "影像", "内镜", "超声", "ct", "CT", "b超", "B超", "x光", "X光", "核磁", "MRI", "磁共振", "纤维镜", "结直肠镜"]
    treatment_keywords = ["治疗", "手术", "服用", "用药", "药", "注射", "放疗", "化疗", "接种", "人工流产", "激素治疗", "康复", "输注", "止血", "输血", "营养"]
    anatomical_keywords = ["脑", "心", "肺", "肝", "脾", "肾", "胃", "肠", "直肠", "结肠", "胰", "胆", "子宫", "卵巢", "前列腺", "乳房", "眼", "角膜", "瞳孔", "颅内", "腹部", "胸部", "皮肤", "口腔", "咽", "喉", "鼻", "四肢", "肢体"]
    clinical_sign_keywords = ["意识", "心率", "呼吸", "血压", "瞳孔", "体征", "颅内压", "发绀", "水肿", "休克", "偏瘫", "麻木", "无力", "呼吸暂停", "呼吸急促", "心率减慢", "心率增快"]
    category_keywords = ["症状", "表现", "疾病", "病变", "类型", "类别", "现象", "部位"]

    # Relation-aware hints.
    if relation == "is_a":
        if role == "tail":
            return "MedicalCategory"
        if role == "head":
            # subclass may still be disease/symptom; fall through after common checks
            pass
    if role == "condition" and any(k in n for k in ["如果", "情况下", "非常好", "早期", "晚期", "患者", "期间"]):
        return "TestResult" if any(k in n for k in test_keywords) else "Other"

    if any(k in n for k in exam_keywords):
        return "ExamProcedure"
    if any(k in n for k in treatment_keywords):
        return "Treatment"
    if any(k in n for k in test_keywords):
        return "TestResult"
    if any(k in n for k in disease_keywords):
        return "Disease"
    if any(k in n for k in clinical_sign_keywords):
        return "ClinicalSign"
    if any(k in n for k in symptom_keywords):
        return "Symptom"
    if any(k in n for k in pathology_keywords):
        return "PathologicalState"
    if any(k in n for k in risk_keywords):
        return "RiskFactor"
    if any(k in n for k in anatomical_keywords):
        return "AnatomicalSite"
    if any(k in n for k in category_keywords):
        return "MedicalCategory"
    return "Other"


def normalize_entity_type(t: Any, name: str, relation: Optional[str] = None, role: Optional[str] = None) -> str:
    t = str(t).strip() if t else ""
    aliases = {
        "Cause": "RiskFactor",
        "CauseEntity": "RiskFactor",
        "Effect": "Other",
        "Finding": "Symptom",
        "ClinicalFinding": "Symptom",
        "Check": "TestResult",
        "Drug": "Treatment",
        "Treatment": "Treatment",
        "Operation": "Treatment",
        "Category": "MedicalCategory",
        "Exam": "ExamProcedure",
        "Procedure": "ExamProcedure",
        "Anatomy": "AnatomicalSite",
    }
    t = aliases.get(t, t)
    if t in ALLOWED_ENTITY_TYPES:
        return t
    return infer_entity_type(name, relation=relation, role=role)


def normalize_relation(r: Any) -> Optional[str]:
    if r in (1, "1", "causal", "casual", "cause", "causes", "因果关系", "因果"):
        return "causes"
    if r in (2, "2", "conditional", "condition", "condition_of", "条件关系", "条件"):
        return "condition_of"
    if r in (3, "3", "hypernym", "hypernmy", "hyponym", "is_a", "上下位关系", "上下位"):
        return "is_a"
    if isinstance(r, str) and r.strip() in ALLOWED_RELATIONS:
        return r.strip()
    return None


def add_triple(triples: List[Triple], triple: Triple) -> None:
    if not triple.head or not triple.tail:
        return
    if triple.relation not in ALLOWED_RELATIONS:
        return
    if triple.head_type not in ALLOWED_ENTITY_TYPES or triple.tail_type not in ALLOWED_ENTITY_TYPES:
        return
    triples.append(triple)


def add_mentioned_in(triples: List[Triple], entity: str, entity_type: str, doc_id: str, source_file: str, source_type: str) -> None:
    if not entity or entity.startswith("Doc_"):
        return
    add_triple(
        triples,
        Triple(
            head=entity,
            head_type=entity_type,
            relation="mentioned_in",
            tail=doc_id,
            tail_type="Document",
            evidence=doc_id,
            source_doc_id=doc_id,
            source_file=source_file,
            source_type=source_type,
            confidence=1.0,
        ),
    )


def parse_gold_train(train_path: Path) -> Tuple[List[Triple], List[Dict[str, Any]]]:
    data = read_json(train_path)
    triples: List[Triple] = []
    causal_events: List[Dict[str, Any]] = []

    for i, block in enumerate(tqdm(data, desc="Parsing gold train labels")):
        text = block.get("text", "")
        doc_id = make_doc_id("train_0717.json", i)
        for rel_idx, rel in enumerate(block.get("relation_of_mention", [])):
            relation = normalize_relation(rel.get("relation"))
            head_obj = rel.get("head", {})
            tail_obj = rel.get("tail", {})
            if not relation:
                continue

            if relation == "causes":
                head = norm_text(head_obj.get("mention"))
                tail = norm_text(tail_obj.get("mention"))
                ev = evidence_span(text, [head, tail])
                ht = infer_entity_type(head, relation="causes", role="head")
                tt = infer_entity_type(tail, relation="causes", role="tail")
                add_triple(triples, Triple(head, ht, "causes", tail, tt, ev, doc_id, "train_0717", "gold", 1.0))

            elif relation == "is_a":
                # CHIP relation=3: head is hypernym, tail is hyponym.
                super_concept = norm_text(head_obj.get("mention"))
                sub_concept = norm_text(tail_obj.get("mention"))
                ev = evidence_span(text, [super_concept, sub_concept])
                ht = infer_entity_type(sub_concept, relation="is_a", role="head")
                tt = infer_entity_type(super_concept, relation="is_a", role="tail")
                add_triple(triples, Triple(sub_concept, ht, "is_a", super_concept, tt, ev, doc_id, "train_0717", "gold", 1.0))

            elif relation == "condition_of":
                condition = norm_text(head_obj.get("mention"))
                nested = tail_obj if isinstance(tail_obj, dict) else {}
                if normalize_relation(nested.get("relation")) != "causes":
                    continue
                cause = norm_text(nested.get("head", {}).get("mention"))
                effect = norm_text(nested.get("tail", {}).get("mention"))
                if not cause or not effect or not condition:
                    continue
                ev = evidence_span(text, [condition, cause, effect])
                cause_t = infer_entity_type(cause, relation="causes", role="head")
                effect_t = infer_entity_type(effect, relation="causes", role="tail")
                cond_t = infer_entity_type(condition, relation="condition_of", role="condition")
                # Direct causal edge plus condition->cause relation.
                add_triple(triples, Triple(cause, cause_t, "causes", effect, effect_t, ev, doc_id, "train_0717", "gold", 1.0))
                add_triple(triples, Triple(condition, cond_t, "condition_of", cause, cause_t, ev, doc_id, "train_0717", "gold", 1.0))
                causal_events.append(
                    {
                        "event_id": f"{doc_id}_{rel_idx}",
                        "cause": cause,
                        "effect": effect,
                        "condition": condition,
                        "evidence": ev,
                        "source_doc_id": doc_id,
                        "source_file": "train_0717",
                        "source_type": "gold",
                        "confidence": 1.0,
                    }
                )
    return triples, causal_events


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Parse a JSON array from an LLM response."""
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, flags=re.S)
        if not match:
            return []
        try:
            obj = json.loads(match.group(0))
        except Exception:
            return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


class QwenClient:
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
        self.client = None
        if self.enabled:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, prompt: str) -> str:
        if not self.enabled or self.client is None:
            raise RuntimeError(
                "Qwen API is not configured. Set QWEN_API_KEY in .env."
            )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是严谨的医学信息抽取助手，只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


def qwen_extract_from_files(
    raw_dir: Path,
    prompt_path: Path,
    files: List[str],
    max_docs: Optional[int] = None,
    sleep_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    prompt_tpl = prompt_path.read_text(encoding="utf-8")
    client = QwenClient()
    if not client.enabled:
        print("[WARN] Qwen API not configured. Skipping Qwen extraction.")
        return []

    all_outputs: List[Dict[str, Any]] = []
    for file_name in files:
        path = raw_dir / file_name
        if not path.exists():
            print(f"[WARN] Missing {path}, skipped.")
            continue
        blocks = read_json(path)
        if max_docs is not None:
            blocks = blocks[:max_docs]
        for i, block in enumerate(tqdm(blocks, desc=f"Qwen extracting {file_name}")):
            text = block.get("text", "")
            doc_id = make_doc_id(file_name, i)
            prompt = prompt_tpl.replace("{text}", text)
            try:
                content = client.chat(prompt)
                items = extract_json_array(content)
                all_outputs.append(
                    {
                        "source_file": file_name.replace(".json", ""),
                        "source_doc_id": doc_id,
                        "text": text,
                        "raw_response": content,
                        "items": items,
                    }
                )
            except Exception as e:
                all_outputs.append(
                    {
                        "source_file": file_name.replace(".json", ""),
                        "source_doc_id": doc_id,
                        "text": text,
                        "error": str(e),
                        "items": [],
                    }
                )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    return all_outputs


def clean_qwen_outputs(raw_outputs: List[Dict[str, Any]], min_confidence: float = 0.65) -> Tuple[List[Triple], List[Dict[str, Any]]]:
    triples: List[Triple] = []
    events: List[Dict[str, Any]] = []
    for block in raw_outputs:
        text = block.get("text", "")
        doc_id = block.get("source_doc_id", "")
        source_file = block.get("source_file", "qwen")
        for item_idx, item in enumerate(block.get("items", [])):
            relation = normalize_relation(item.get("relation"))
            if relation not in CORE_RELATIONS:
                continue
            conf = safe_float(item.get("confidence"), default=0.75)
            if conf < min_confidence:
                continue
            ev = norm_text(item.get("evidence")) or text[:160]

            if relation in {"causes", "is_a", "symptom_of", "treated_by", "located_in", "diagnosed_by"}:
                head = norm_text(item.get("head"))
                tail = norm_text(item.get("tail"))
                if not head or not tail:
                    continue
                # Require entities to be at least plausible mentions in text, unless evidence was returned.
                if head not in text and tail not in text and not ev:
                    continue
                ht = normalize_entity_type(item.get("head_type"), head, relation, "head")
                tt = normalize_entity_type(item.get("tail_type"), tail, relation, "tail")
                add_triple(triples, Triple(head, ht, relation, tail, tt, ev, doc_id, source_file, "qwen", conf))

            elif relation == "condition_of":
                condition = norm_text(item.get("condition") or item.get("head"))
                cause = norm_text(item.get("cause"))
                effect = norm_text(item.get("effect"))
                # Accept an alternative nested shape if model returns tail object.
                if not cause or not effect:
                    tail_obj = item.get("tail")
                    if isinstance(tail_obj, dict):
                        cause = norm_text(tail_obj.get("head") or tail_obj.get("cause"))
                        effect = norm_text(tail_obj.get("tail") or tail_obj.get("effect"))
                if not condition or not cause or not effect:
                    continue
                cond_t = normalize_entity_type(item.get("condition_type"), condition, relation, "condition")
                cause_t = normalize_entity_type(item.get("cause_type"), cause, "causes", "head")
                effect_t = normalize_entity_type(item.get("effect_type"), effect, "causes", "tail")
                add_triple(triples, Triple(cause, cause_t, "causes", effect, effect_t, ev, doc_id, source_file, "qwen", conf))
                add_triple(triples, Triple(condition, cond_t, "condition_of", cause, cause_t, ev, doc_id, source_file, "qwen", conf))
                events.append(
                    {
                        "event_id": f"{doc_id}_{item_idx}",
                        "cause": cause,
                        "effect": effect,
                        "condition": condition,
                        "evidence": ev,
                        "source_doc_id": doc_id,
                        "source_file": source_file,
                        "source_type": "qwen",
                        "confidence": conf,
                    }
                )
    return triples, events


def deduplicate_triples(triples: List[Triple]) -> List[Triple]:
    best: Dict[Tuple[str, str, str, str], Triple] = {}
    priority = {"gold": 3, "qwen": 2, "inferred": 1}
    for t in triples:
        key = (t.head, t.relation, t.tail, t.source_doc_id)
        old = best.get(key)
        if old is None:
            best[key] = t
            continue
        old_score = (priority.get(old.source_type, 0), old.confidence, len(old.evidence))
        new_score = (priority.get(t.source_type, 0), t.confidence, len(t.evidence))
        if new_score > old_score:
            best[key] = t
    return list(best.values())


def write_triples_csv(path: Path, triples: List[Triple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(triples[0]).keys()) if triples else [
        "head", "head_type", "relation", "tail", "tail_type", "evidence", "source_doc_id", "source_file", "source_type", "confidence"
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in triples:
            writer.writerow(asdict(t))


def write_entities_relations(processed_dir: Path, triples: List[Triple]) -> None:
    entity_types: Dict[str, Counter] = defaultdict(Counter)
    relation_counts = Counter()
    entity_sources: Dict[str, Counter] = defaultdict(Counter)
    for t in triples:
        entity_types[t.head][t.head_type] += 1
        entity_types[t.tail][t.tail_type] += 1
        entity_sources[t.head][t.source_type] += 1
        entity_sources[t.tail][t.source_type] += 1
        relation_counts[t.relation] += 1

    entities_path = processed_dir / "entities.csv"
    with entities_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["entity_id", "name", "type", "degree_count", "main_source"])
        writer.writeheader()
        for i, (name, c) in enumerate(sorted(entity_types.items(), key=lambda x: (-sum(x[1].values()), x[0]))):
            etype = c.most_common(1)[0][0]
            main_source = entity_sources[name].most_common(1)[0][0]
            writer.writerow({"entity_id": f"E{i:06d}", "name": name, "type": etype, "degree_count": sum(c.values()), "main_source": main_source})

    relations_path = processed_dir / "relations.csv"
    with relations_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relation", "count", "description"])
        writer.writeheader()
        desc = {
            "causes": "原因导致结果",
            "condition_of": "条件修饰某个因果事件",
            "is_a": "下位概念属于上位概念",
            "event_cause": "因果事件的原因",
            "event_effect": "因果事件的结果",
            "has_condition": "因果事件的条件",
            "mentioned_in": "实体或事件来自某篇文本",
        }
        for rel, count in relation_counts.most_common():
            writer.writerow({"relation": rel, "count": count, "description": desc.get(rel, "")})

    write_json(processed_dir / "entity_types.json", {k: dict(v) for k, v in entity_types.items()})


def write_causal_events(path: Path, events: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["event_id", "cause", "effect", "condition", "evidence", "source_doc_id", "source_file", "source_type", "confidence"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        seen = set()
        for e in events:
            if e.get("event_id") in seen:
                continue
            seen.add(e.get("event_id"))
            writer.writerow({k: e.get(k, "") for k in fields})


def write_kg_json(path: Path, triples: List[Triple]) -> None:
    nodes: Dict[str, Dict[str, Any]] = {}
    links: List[Dict[str, Any]] = []
    for t in triples:
        nodes.setdefault(t.head, {"id": t.head, "label": t.head, "type": t.head_type})
        nodes.setdefault(t.tail, {"id": t.tail, "label": t.tail, "type": t.tail_type})
        links.append(
            {
                "source": t.head,
                "target": t.tail,
                "relation": t.relation,
                "evidence": t.evidence,
                "source_doc_id": t.source_doc_id,
                "source_file": t.source_file,
                "source_type": t.source_type,
                "confidence": t.confidence,
            }
        )
    write_json(path, {"nodes": list(nodes.values()), "links": links})


def visualize(triples: List[Triple], out_path: Path, max_edges: int = 800) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample = [t for t in triples if t.relation != "mentioned_in"][:max_edges]
    if Network is None:
        html = "<html><body><h1>KG Visualization</h1><p>Install pyvis to view interactive graph.</p><ul>"
        for t in sample[:200]:
            html += f"<li>{t.head} --{t.relation}--&gt; {t.tail}</li>"
        html += "</ul></body></html>"
        out_path.write_text(html, encoding="utf-8")
        return
    net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
    type_colors = {
        "Disease": "#ffb3ba",
        "Symptom": "#bae1ff",
        "PathologicalState": "#baffc9",
        "RiskFactor": "#ffffba",
        "TestResult": "#e2c2ff",
        "TreatmentOrOperation": "#ffd6a5",
        "MedicalCategory": "#caffbf",
        "CausalEvent": "#d0d0d0",
        "Document": "#eeeeee",
        "Other": "#f2f2f2",
    }
    added = set()
    for t in sample:
        for name, typ in [(t.head, t.head_type), (t.tail, t.tail_type)]:
            if name not in added:
                net.add_node(name, label=name[:30], title=f"{name}\n{typ}", color=type_colors.get(typ, "#f2f2f2"))
                added.add(name)
        net.add_edge(t.head, t.tail, label=t.relation, title=t.evidence[:120])
    net.write_html(str(out_path), notebook=False)


def build_project(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    triples_path = Path(args.triples_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / "train_0717.json"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    gold_triples, gold_events = parse_gold_train(train_path)
    write_triples_csv(processed_dir / "gold_triples.csv", deduplicate_triples(gold_triples))
    print(f"[INFO] Gold triples parsed: {len(gold_triples)}")

    qwen_triples: List[Triple] = []
    qwen_events: List[Dict[str, Any]] = []
    if args.extract_qwen:
        raw_outputs = qwen_extract_from_files(
            raw_dir=raw_dir,
            prompt_path=Path(args.extraction_prompt),
            files=args.qwen_files.split(","),
            max_docs=args.max_qwen_docs,
            sleep_seconds=args.sleep_seconds,
        )
        write_json(processed_dir / "qwen_extracted_raw.json", raw_outputs)
        qwen_triples, qwen_events = clean_qwen_outputs(raw_outputs, min_confidence=args.min_confidence)
        write_json(processed_dir / "qwen_extracted_clean.json", [asdict(t) for t in qwen_triples])
        print(f"[INFO] Qwen triples parsed: {len(qwen_triples)}")
    else:
        # Keep empty files so downstream scripts have stable paths.
        write_json(processed_dir / "qwen_extracted_raw.json", [])
        write_json(processed_dir / "qwen_extracted_clean.json", [])
        print("[INFO] Qwen extraction disabled. Use --extract_qwen to enable.")

    all_triples = deduplicate_triples(gold_triples + qwen_triples)
    all_events = gold_events + qwen_events

    write_triples_csv(triples_path, all_triples)
    write_entities_relations(processed_dir, all_triples)
    write_causal_events(processed_dir / "causal_events.csv", all_events)
    write_kg_json(processed_dir / "kg.json", all_triples)
    visualize(all_triples, results_dir / "kg_visualization.html", max_edges=args.visualize_max_edges)

    stats = {
        "num_triples": len(all_triples),
        "num_entities": len({t.head for t in all_triples} | {t.tail for t in all_triples}),
        "relation_counts": dict(Counter(t.relation for t in all_triples)),
        "source_type_counts": dict(Counter(t.source_type for t in all_triples)),
        "num_causal_events": len({e["event_id"] for e in all_events if e.get("event_id")}),
    }
    write_json(processed_dir / "build_stats.json", stats)
    print("[INFO] Build finished:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CHIP2022 Medical Causal Event KG")
    p.add_argument("--raw_dir", default="data/raw")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--triples_path", default="data/triples.csv")
    p.add_argument("--extraction_prompt", default="prompts/qwen_extraction_prompt.txt")
    p.add_argument("--extract_qwen", action="store_true", help="Enable Qwen extraction for unlabel/testA/testB")
    p.add_argument("--qwen_files", default="unlabel.json,testA.json,testB.json")
    p.add_argument("--max_qwen_docs", type=int, default=None, help="Limit Qwen extraction docs per file for quick tests")
    p.add_argument("--sleep_seconds", type=float, default=0.0)
    p.add_argument("--min_confidence", type=float, default=0.65)
    p.add_argument("--visualize_max_edges", type=int, default=800)
    return p.parse_args()


if __name__ == "__main__":
    build_project(parse_args())
