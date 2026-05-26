#!/usr/bin/env python3
"""Gradio-based interactive QA system for the Medical Causal Event KG."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from llm_qa import kg_only_answer, llm_only_answer, kg_augmented_answer
from reasoning import KGReasoner
from graph_retrieval import retrieve_evidence, format_evidence

TRIPLES_PATH = str(Path(__file__).resolve().parents[1] / "data" / "triples.csv")
PROMPT_PATH = str(Path(__file__).resolve().parents[1] / "prompts" / "kg_augmented_prompt.txt")
VIS_PATH = str(Path(__file__).resolve().parents[1] / "results" / "kg_visualization.html")


def answer_question(question: str, mode: str) -> str:
    if not question.strip():
        return "请输入问题。"
    try:
        if mode == "LLM-only":
            result = llm_only_answer(question)
            return result["answer"]
        elif mode == "KG-augmented":
            result = kg_augmented_answer(question, TRIPLES_PATH, PROMPT_PATH)
            ev = format_evidence(result.get("evidence", {}))
            return f"{result['answer']}\n\n--- 检索证据 ---\n{ev}"
        else:  # KG-only
            result = kg_only_answer(question, TRIPLES_PATH)
            return result["answer"]
    except Exception as e:
        return f"[错误] {e}"


def show_evidence(question: str) -> str:
    if not question.strip():
        return "请输入问题。"
    ev = retrieve_evidence(question, TRIPLES_PATH)
    return format_evidence(ev)


EXAMPLES = [
    "宫腔粘连可能导致什么？",
    "月经量少可能由什么引起？",
    "为什么高血压可能和心肌梗死有关？",
    "在什么条件下支气管哮喘可能发生？",
    "胃溃疡属于什么类型？",
    "双硫仑反应有哪些症状？",
    "瘫痪如何治疗？",
    "脑出血发生在什么部位？",
    "肠癌如何诊断？",
]


def build_app():
    with gr.Blocks(
        title="医疗因果事件知识图谱问答系统",
    ) as app:
        gr.Markdown(
            "# 医疗因果事件知识图谱问答系统\n"
            "基于 CHIP2022 数据集 | GraphRAG 增强问答\n\n"
            "> 本系统仅用于医学知识学习和知识图谱课程实验，不构成医疗诊断或用药建议。"
        )

        with gr.Row():
            with gr.Column(scale=2):
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
                answer = gr.Textbox(label="回答", lines=12, interactive=False)

                gr.Markdown("## 图谱检索证据")
                ev_btn = gr.Button("检索证据", variant="secondary")
                evidence = gr.Textbox(label="证据", lines=10, interactive=False)

                btn.click(answer_question, [question, mode], answer)
                ev_btn.click(show_evidence, [question], evidence)

                gr.Markdown("### 示例问题")
                gr.Examples(EXAMPLES, question)

            with gr.Column(scale=1):
                gr.Markdown("## 图谱统计")
                stats = gr.JSON(
                    value={
                        "实体数量": 14596,
                        "三元组数量": 18539,
                        "实体类型": 11,
                        "关系类型": 7,
                        "Gold 三元组": 9126,
                        "Qwen 三元组": 9413,
                        "因果事件": 783,
                    },
                    label="",
                )

                gr.Markdown(
                    "### 关系类型颜色\n"
                    "- 🔴 causes\n"
                    "- 🟣 condition_of\n"
                    "- 🔵 is_a\n"
                    "- 🟢 symptom_of\n"
                    "- 🟦 treated_by\n"
                    "- 🟡 located_in\n"
                    "- ⚫ diagnosed_by"
                )

                gr.Markdown("## 图谱可视化")
                gr.HTML(
                    f'<iframe src="file://{VIS_PATH}" width="100%" height="500" '
                    f'style="border:1px solid #ddd;border-radius:8px;"></iframe>'
                )

    return app


if __name__ == "__main__":
    app = build_app()
    print("启动 Gradio QA 系统...")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
