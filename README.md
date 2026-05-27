# CHIP2022 Medical Causal Event KG

基于 CHIP2022 医疗因果实体关系抽取数据的医疗因果事件知识图谱、推理与 GraphRAG 增强问答系统。

本项目面向知识图谱课程作业要求，包含：

- 知识图谱构建：Gold Seed KG + Qwen 抽取扩展。
- 知识推理：直接因果、反向因果、多跳路径、条件约束、上下位、症状、治疗、部位、诊断和风险因素查询。
- 大模型增强：GraphRAG 检索三元组、路径和条件证据，再注入 Qwen Prompt。
- 可视化系统：独立 HTML 图谱和 Gradio 问答界面中的动态图谱子图。
- 实验评估：图谱规模、检索命中、问答对比和案例分析。

> 本系统仅用于知识图谱课程实验和医学知识学习，不构成医疗诊断或用药建议。

## 1. 数据与 Schema

### 1.1 数据文件

当前实际使用的原始数据：

```text
data/raw/
├── train_0717.json   # 人工标注数据，用于构建 Gold Seed KG
├── unlabel.json      # 无标注文本，用于 Qwen 抽取扩展
├── testA.json        # 可选扩展抽取数据
└── testB.json        # 可选扩展抽取数据
```

当前最终图谱主要由 `train_0717.json` 和 `unlabel.json` 构建。`testA.json`、`testB.json` 保留为可选扩展数据，不是复现当前结果的必需输入。

### 1.2 关系类型

CHIP2022 原始标注关系：

| 原始标签 | 项目关系 | 说明 |
|---:|---|---|
| 1 | `causes` | 因果关系 |
| 2 | `condition_of` | 条件修饰因果关系 |
| 3 | `is_a` | 上下位关系 |

项目最终 schema 扩展为 8 种关系：

```text
causes
risk_factor_for
condition_of
is_a
symptom_of
treated_by
located_in
diagnosed_by
```

其中 `condition_of` 的处理方式是保留因果边，并把条件连接到原因实体：

```text
宫腔粘连 --causes--> 月经量少
女性激素非常好 --condition_of--> 宫腔粘连
```

### 1.3 实体类型

```text
Disease
Symptom
ClinicalSign
PathologicalState
RiskFactor
TestResult
ExamProcedure
Treatment
AnatomicalSite
MedicalCategory
Other
```

## 2. 环境配置

```bash
conda create -n chipkg python=3.10
conda activate chipkg
pip install -r requirements.txt
```

Qwen API 只在运行 Qwen 抽取、LLM-only 或 KG-augmented 生成式回答时需要。KG-only、推理、GraphRAG 检索、评估和 Gradio 界面可以先不配置 API。

如需调用 Qwen：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
QWEN_API_KEY=your_api_key_here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
QWEN_TEMPERATURE=0
QWEN_MAX_TOKENS=2048
```

## 3. 直接使用已有结果

仓库已经包含当前实验使用的最终图谱和结果文件，可以直接运行推理、检索、评估和交互系统。

```bash
# 查看图谱规模
cat data/processed/build_stats.json

# 查看评估结果
cat results/metrics.json

# 查看前几条三元组
head -n 5 data/triples.csv
```

浏览独立图谱 HTML：

```bash
python -m http.server 8000
# 浏览器访问 http://localhost:8000/results/kg_visualization.html
```

## 4. 构建图谱

### 4.1 构建 Gold Seed KG

不需要 API，只使用 `train_0717.json`，适合快速验证构建流程：

```bash
python src/build_kg.py
```

注意：该命令只会构建 Gold Seed KG，不合并已有 Qwen 抽取结果，会覆盖 `data/triples.csv`、`data/processed/` 和 `results/kg_visualization.html` 中的部分结果。课程最终结果不建议只运行这一条。

### 4.2 复用已有 Qwen 抽取结果重建混合图谱

当前仓库已经有提取后的 Qwen 结果：

```text
data/processed/qwen_extracted_raw.json
data/processed/qwen_extracted_clean.json
```

如果只是想重建 Gold + Qwen 的最终混合图谱，不需要重新调用 API，直接运行：

```bash
python src/build_kg.py --reuse_qwen_outputs
```

这会重新解析 `train_0717.json`，并复用 `data/processed/qwen_extracted_raw.json` 生成 Qwen 三元组和条件因果事件，输出当前最终规模的混合图谱。

### 4.3 重新调用 Qwen 抽取扩展 KG

只有当你想重新从原文调用 Qwen 抽取时，才使用 `--extract_qwen`。该模式需要配置 API Key，并会覆盖已有的 `qwen_extracted_raw.json` 和 `qwen_extracted_clean.json`。

小规模测试：

```bash
python src/build_kg.py --extract_qwen --max_qwen_docs 5 --sleep_seconds 0.2
```

重新抽取 `unlabel.json` 的命令：

```bash
python src/build_kg.py --extract_qwen --qwen_files unlabel.json --sleep_seconds 0.2
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--qwen_files` | 指定抽取文件，多个文件用逗号分隔 |
| `--max_qwen_docs` | 限制每个文件抽取篇数，调试时使用 |
| `--sleep_seconds` | API 调用间隔 |
| `--min_confidence` | Qwen 抽取三元组最低置信度，默认 0.65 |
| `--visualize_max_edges` | 可视化 HTML 中最多写入的边数 |
| `--reuse_qwen_outputs` | 复用已有 Qwen 抽取结果，不调用 API |

输出核心文件：

```text
data/triples.csv
data/processed/entities.csv
data/processed/relations.csv
data/processed/causal_events.csv
data/processed/kg.json
data/processed/build_stats.json
results/kg_visualization.html
```

## 5. 运行推理与问答

### 5.1 知识图谱推理

```bash
python src/reasoning.py --query "高血压是否可能间接导致心肌梗死？"
python src/reasoning.py --query "胃溃疡属于什么类型？"
python src/reasoning.py --query "在什么条件下宫腔粘连可能导致月经量少？"
python src/reasoning.py --query "乳腺癌有哪些风险因素？"
python src/reasoning.py --query "肠癌如何诊断？"
python src/reasoning.py --query "瘫痪如何治疗？"
```

如需 JSON 输出：

```bash
python src/reasoning.py --query "乳腺癌有哪些风险因素？" --json
```

### 5.2 GraphRAG 证据检索

```bash
python src/graph_retrieval.py --question "为什么高血压可能和心肌梗死有关？"
python src/graph_retrieval.py --question "在什么条件下宫腔粘连可能导致月经量少？"
```

GraphRAG 返回三类结构化证据：

| 证据类型 | 对应 schema | 用途 |
|---|---|---|
| 三元组证据 | `causes`、`risk_factor_for`、`is_a`、`symptom_of`、`treated_by`、`located_in`、`diagnosed_by` 等基本边 | 回答直接事实问题 |
| 路径证据 | 多条 `causes` 边组成的因果链 | 回答“为什么/是否间接导致/是否有关”问题 |
| 条件证据 | `condition_of` 边 | 回答“在什么条件下”问题 |

`matched_entities`、`source_doc_id`、`source_type`、`confidence` 和原文 `evidence` 属于证据元信息，不作为单独证据类型。

### 5.3 大模型增强问答

KG-only，不需要 API：

```bash
python src/llm_qa.py --question "为什么高血压可能和心肌梗死有关？" --mode kg_only
```

KG-augmented，优先调用 Qwen；未配置 API 时会退化为 KG-only：

```bash
python src/llm_qa.py --question "为什么高血压可能和心肌梗死有关？" --mode kg_augmented
```

LLM-only，需要 Qwen API：

```bash
python src/llm_qa.py --question "为什么高血压可能和心肌梗死有关？" --mode llm_only
```

JSON 输出：

```bash
python src/llm_qa.py --question "宫腔粘连可能导致什么？" --mode kg_augmented --json
```

## 6. Gradio 交互系统

启动：

```bash
python src/qa_server.py
```

浏览器访问：

```text
http://localhost:7860
```

如果端口被占用：

```bash
GRADIO_SERVER_PORT=7861 python src/qa_server.py
```

交互系统功能：

- 左侧输入问题，支持 KG-only、KG-augmented、LLM-only 三种问答模式。
- “提问”按钮下方提供示例问题，点击后自动填入输入框。
- “检索证据”按钮单独返回 GraphRAG 三类证据。
- 右侧初始显示图谱概览；提问或检索后自动更新为本次相关证据子图。
- 点击或触碰图谱边，可查看对应三元组和原文证据。
- 图谱统计显示实体数、三元组数、实体类型数、关系类型数和来源分布。

## 7. 运行评估

```bash
python src/evaluate.py
```

可限制测试问题数量：

```bash
python src/evaluate.py --max_questions 30
```

输出：

```text
results/metrics.json
results/extraction_metrics.json
results/cases.md
data/processed/test_questions.json
```

## 8. 当前主要结果

| 指标 | 数值 | 课程最低要求 |
|---|---:|---:|
| 实体数量 | 14,762 | 100 |
| 三元组数量 | 18,835 | 300 |
| 实体类型数量 | 11 | 3 |
| 关系类型数量 | 8 | 3 |
| Gold Seed 三元组 | 9,126 | - |
| Qwen 抽取三元组 | 9,709 | - |
| CausalEvent 数量 | 783 | - |

关系分布：

| 关系 | 数量 |
|---|---:|
| `causes` | 10,709 |
| `treated_by` | 2,283 |
| `symptom_of` | 1,970 |
| `is_a` | 1,852 |
| `condition_of` | 584 |
| `diagnosed_by` | 575 |
| `located_in` | 566 |
| `risk_factor_for` | 296 |

评估结果见 `results/metrics.json`，案例分析见 `results/cases.md`。

## 9. 项目结构


```text
project/
├── README.md
├── requirements.txt
├── .env.example
├── course.txt
├── report.md
├── data/
│   ├── raw/
│   │   ├── train_0717.json
│   │   ├── unlabel.json
│   │   ├── testA.json
│   │   └── testB.json
│   ├── processed/
│   │   ├── build_stats.json
│   │   ├── gold_triples.csv
│   │   ├── qwen_extracted_raw.json
│   │   ├── qwen_extracted_clean.json
│   │   ├── entities.csv
│   │   ├── relations.csv
│   │   ├── causal_events.csv
│   │   ├── kg.json
│   │   └── test_questions.json
│   └── triples.csv
├── src/
│   ├── build_kg.py
│   ├── config.py
│   ├── reasoning.py
│   ├── graph_retrieval.py
│   ├── llm_qa.py
│   ├── qa_server.py
│   └── evaluate.py
├── prompts/
│   ├── qwen_extraction_prompt.txt
│   └── kg_augmented_prompt.txt
└── results/
    ├── kg_visualization.html
    ├── metrics.json
    └── cases.md
```
