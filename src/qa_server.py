#!/usr/bin/env python3
"""Gradio-based interactive QA system for the Medical Causal Event KG."""

import csv
import html
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from llm_qa import kg_only_answer, llm_only_answer, kg_augmented_answer
from reasoning import KGReasoner
from graph_retrieval import retrieve_evidence, format_evidence

TRIPLES_PATH = str(Path(__file__).resolve().parents[1] / "data" / "triples.csv")
PROMPT_PATH = str(Path(__file__).resolve().parents[1] / "prompts" / "kg_augmented_prompt.txt")
VIS_PATH = str(Path(__file__).resolve().parents[1] / "results" / "kg_visualization.html")
RESULTS_DIR = str(Path(__file__).resolve().parents[1] / "results")
VIS_JS_PATH = Path(__file__).resolve().parents[1] / "lib" / "vis-9.1.2" / "vis-network.min.js"
VIS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"

RELATION_COLORS = {
    "causes": "#e74c3c",
    "risk_factor_for": "#e67e22",
    "condition_of": "#9b59b6",
    "is_a": "#3498db",
    "symptom_of": "#2ecc71",
    "treated_by": "#1abc9c",
    "located_in": "#f39c12",
    "diagnosed_by": "#34495e",
}

TYPE_COLORS = {
    "Disease": "#ffb3ba",
    "Symptom": "#bae1ff",
    "ClinicalSign": "#b5ead7",
    "PathologicalState": "#baffc9",
    "RiskFactor": "#ffffba",
    "TestResult": "#e2c2ff",
    "ExamProcedure": "#c7ceea",
    "Treatment": "#ffd6a5",
    "AnatomicalSite": "#ffdac1",
    "MedicalCategory": "#caffbf",
    "Other": "#f2f2f2",
}

VIS_JS = VIS_JS_PATH.read_text(encoding="utf-8") if VIS_JS_PATH.exists() else ""


def iframe_srcdoc(inner_html: str, height: int = 680) -> str:
    escaped = html.escape(inner_html, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" width="100%" height="{height}" '
        'style="border:1px solid #ddd;border-radius:8px;background:white;"></iframe>'
    )


def graph_html(triples: List[Dict[str, Any]], height: int = 680) -> str:
    if not triples:
        return iframe_srcdoc(
            "<!doctype html><html><body style='font-family:sans-serif;color:#666;padding:18px;'>暂无可视化证据。</body></html>",
            height=height,
        )

    degrees = Counter()
    for t in triples:
        degrees[t["head"]] += 1
        degrees[t["tail"]] += 1

    nodes: Dict[str, Dict[str, Any]] = {}
    edges = []
    for t in triples:
        for key, type_key in [("head", "head_type"), ("tail", "tail_type")]:
            name = t.get(key, "")
            typ = t.get(type_key, "Other")
            if name and name not in nodes:
                nodes[name] = {
                    "id": name,
                    "label": name[:28],
                    "title": f"{name} [{typ}]",
                    "color": TYPE_COLORS.get(typ, "#f2f2f2"),
                    "size": min(26, 9 + degrees[name] * 2),
                }
        relation = t.get("relation", "")
        edges.append(
            {
                "from": t.get("head", ""),
                "to": t.get("tail", ""),
                "label": relation,
                "title": f"{t.get('head','')} --{relation}--> {t.get('tail','')}\n{t.get('evidence','')[:120]}",
                "evidence": t.get("evidence", ""),
                "arrows": "to",
                "color": RELATION_COLORS.get(relation, "#999999"),
            }
        )

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body {{ width: 100%; height: 100%; margin: 0; font-family: sans-serif; }}
    body {{ background: #fff; }}
    #mynetwork {{ width: 100%; height: calc(100% - 78px); min-height: 390px; }}
    #edgeEvidence {{
      box-sizing: border-box;
      height: 78px;
      border-top: 1px solid #e5e7eb;
      padding: 8px 10px;
      color: #374151;
      font-size: 12px;
      line-height: 1.45;
      overflow: auto;
      background: #fafafa;
    }}
  </style>
  <script src="{VIS_CDN}"></script>
  <script>
    if (typeof vis === "undefined") {{
      document.write('<script src="../lib/vis-9.1.2/vis-network.min.js"><\\/script>');
    }}
  </script>
</head>
<body>
  <div id="mynetwork"></div>
  <div id="edgeEvidence">点击或触碰边可查看对应证据</div>
  <script>
    const nodes = new vis.DataSet({json.dumps(list(nodes.values()), ensure_ascii=False)});
    const edges = new vis.DataSet({json.dumps(edges, ensure_ascii=False)});
    const options = {{
      interaction: {{ hover: true, navigationButtons: true }},
      nodes: {{ shape: "dot", font: {{ size: 13 }} }},
      edges: {{ smooth: {{ enabled: true, type: "dynamic" }}, font: {{ size: 10 }} }},
      physics: {{
        timestep: 0.05,
        adaptiveTimestep: false,
        minVelocity: 0.4,
        stabilization: {{ iterations: 300, updateInterval: 80 }},
        barnesHut: {{
          gravitationalConstant: -18000,
          springLength: 170,
          springConstant: 0.0015,
          damping: 0.60,
          avoidOverlap: 0.15
        }}
      }}
    }};
    const network = new vis.Network(document.getElementById("mynetwork"), {{ nodes, edges }}, options);
    network.on("selectEdge", function(params) {{
      const edge = edges.get(params.edges[0]);
      if (!edge) return;
      document.getElementById("edgeEvidence").innerText =
        `${{edge.from}} --${{edge.label}}--> ${{edge.to}}\\n证据：${{edge.evidence || edge.title || "无"}}`;
    }});
  </script>
</body>
</html>"""
    return iframe_srcdoc(doc, height=height)


def overview_triples(limit: int = 360) -> List[Dict[str, Any]]:
    by_relation: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with Path(TRIPLES_PATH).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("relation") in RELATION_COLORS:
                by_relation[row["relation"]].append(row)

    sample: List[Dict[str, Any]] = []
    per_relation = max(1, limit // max(1, len(by_relation)))
    for rel in sorted(by_relation):
        sample.extend(by_relation[rel][:per_relation])
    return sample[:limit]


def evidence_triples(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    triples = list(evidence.get("triples", []))
    for path in evidence.get("paths", []):
        triples.extend(path.get("edges", []))
    triples.extend(evidence.get("conditional_events", []))

    seen = set()
    dedup = []
    for t in triples:
        key = (t.get("head"), t.get("relation"), t.get("tail"), t.get("source_doc_id"))
        if key not in seen:
            seen.add(key)
            dedup.append(t)
    return dedup[:60]


def answer_question(question: str, mode: str) -> tuple[str, str]:
    if not question.strip():
        return "请输入问题。", graph_html([])
    try:
        if mode == "LLM-only":
            result = llm_only_answer(question)
            return result["answer"], graph_html([])
        elif mode == "KG-augmented":
            result = kg_augmented_answer(question, TRIPLES_PATH, PROMPT_PATH)
            ev = format_evidence(result.get("evidence", {}))
            graph = graph_html(evidence_triples(result.get("evidence", {})))
            return f"{result['answer']}\n\n--- 检索证据 ---\n{ev}", graph
        else:  # KG-only
            result = kg_only_answer(question, TRIPLES_PATH)
            triples = result.get("evidence", {}).get("evidence", [])
            return result["answer"], graph_html(triples)
    except Exception as e:
        return f"[错误] {e}", graph_html([])


def show_evidence(question: str) -> tuple[str, str]:
    if not question.strip():
        return "请输入问题。", graph_html([])
    ev = retrieve_evidence(question, TRIPLES_PATH)
    return format_evidence(ev), graph_html(evidence_triples(ev))


EXAMPLES = [
    "宫腔粘连可能导致什么？",
    "月经量少可能由什么引起？",
    "为什么高血压可能和心肌梗死有关？",
    "在什么条件下支气管哮喘可能发生？",
    "肺源性心脏病有哪些症状？",
    "胃溃疡属于什么类型？",
    "双硫仑反应有哪些症状？",
    "脑出血发生在什么部位？",
    "肠癌如何诊断？",
    "乳腺癌有哪些风险因素？",
]


def build_app():
    with gr.Blocks(
        title="医疗因果事件知识图谱问答系统",
    ) as app:
        gr.Markdown(
            "# 医疗因果事件知识图谱问答系统\n"
            "> 本系统仅用于医学知识学习和知识图谱课程实验，不构成医疗诊断或用药建议。"
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## 提问")
                question = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入医学知识问题...",
                    lines=2,
                )
                mode = gr.Radio(
                    choices=["KG-only", "KG-augmented", "LLM-only"],
                    value="KG-only",
                    label="问答模式",
                )
                btn = gr.Button("提问", variant="primary")

                gr.Markdown("### 示例问题")
                gr.Examples(EXAMPLES, question)

                answer = gr.Textbox(label="回答", lines=12, interactive=False)

                gr.Markdown("## 图谱检索证据")
                ev_btn = gr.Button("检索证据", variant="secondary")
                evidence = gr.Textbox(label="证据", lines=10, interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("## 图谱可视化")
                graph = gr.HTML(value=graph_html(overview_triples()))

                with gr.Accordion("关系类型颜色", open=False):
                    gr.Markdown(
                        "- 🔴 causes\n"
                        "- 🟠 risk_factor_for\n"
                        "- 🟣 condition_of\n"
                        "- 🔵 is_a\n"
                        "- 🟢 symptom_of\n"
                        "- 🟦 treated_by\n"
                        "- 🟡 located_in\n"
                        "- ⚫ diagnosed_by"
                    )

                gr.Markdown("## 图谱统计")
                stats = gr.JSON(
                    value={
                        "实体数量": 14762,
                        "三元组数量": 18835,
                        "实体类型": 11,
                        "关系类型": 8,
                        "Gold 三元组": 9126,
                        "Qwen 三元组": 9709,
                        "因果事件": 783,
                    },
                    label="",
                )

        btn.click(answer_question, [question, mode], [answer, graph])
        ev_btn.click(show_evidence, [question], [evidence, graph])

    return app


if __name__ == "__main__":
    app = build_app()
    print("启动 Gradio QA 系统...")
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        theme=gr.themes.Soft(),
        allowed_paths=[RESULTS_DIR],
    )
