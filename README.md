# CHIP2022 Medical Causal Event KG

基于 CHIP2022 医疗因果实体关系抽取数据的**医疗因果事件知识图谱构建、推理与 GraphRAG 增强问答系统**。

## 1. 项目简介

本项目采用“人工标注种子图谱 + Qwen 无标签文本抽取扩展”的混合构建方式：

- `train_0717.json`：使用人工标注 `relation_of_mention` 构建高质量 **Gold Seed KG**。
- `unlabel.json`、`testA.json`、`testB.json`：调用 Qwen API 从原始医学文本中抽取 `causes`、`condition_of`、`is_a`、`symptom_of`、`treated_by`、`located_in`、`diagnosed_by` 关系，构建 **Qwen-extracted KG**。
- 最终融合两部分图谱，形成医疗因果事件知识图谱，并实现因果链推理、条件约束推理、上下位推理和 GraphRAG 增强问答。

> 注意：本系统仅用于知识图谱课程实验和医学知识学习，不构成医疗诊断或用药建议。

## 2. 数据说明

原始数据放在：

```text
data/raw/
├── train_0717.json
├── unlabel.json
├── testA.json
├── testB.json
└── example_code.txt
```

CHIP2022 标注关系：

| 原始标签 | 项目关系 | 说明 |
|---:|---|---|
| 1 | `causes` | 因果关系，原因导致结果 |
| 2 | `condition_of` | 条件关系，条件修饰一条因果关系 |
| 3 | `is_a` | 上下位关系，构图时统一为“下位概念 -> 上位概念” |

当前抽取关系包括：`causes`、`condition_of`、`is_a`、`symptom_of`、`treated_by`、`located_in`、`diagnosed_by`。

实体类型来源说明：
- **train 标签**：不包含实体类别，构图时使用规则推断实体类型。
- **Qwen 抽取**：要求模型输出 `head_type/tail_type`，`condition_of` 还需 `condition_type/cause_type/effect_type`；若缺失会回退到规则推断。

## 3. Schema 设计

### 实体类型

```text
Disease
Symptom
PathologicalState
RiskFactor
TestResult
TreatmentOrOperation
MedicalCategory
CausalEvent
Document
Other
```

### 关系类型

```text
causes
condition_of
is_a
symptom_of
treated_by
located_in
diagnosed_by
```

其中 `condition_of` 表示“条件 -> 原因”，条件会修饰该原因与其后续因果关系。

示例：

```text
女性激素非常好 condition_of [宫腔粘连 causes 月经量少]
```

转换为（保留因果边，同时将条件关联到原因）：

```text
宫腔粘连 --causes--> 月经量少
女性激素非常好 --condition_of--> 宫腔粘连
```

## 4. 环境配置

创建并激活虚拟环境（Linux/macOS）：

```bash
conda create -n chipkg python=3.10
conda activate chipkg
```

```bash
pip install -r requirements.txt
```

配置 Qwen API：

```bash
cp .env.example .env
```

编辑 `.env`，填写 API Key：

```env
QWEN_API_KEY=your_api_key_here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
QWEN_TEMPERATURE=0
QWEN_MAX_TOKENS=2048
```

运行 `src/build_kg.py` 或 `src/relation_discovery.py` 时会打印：
`Qwen API key configured: yes/no`，用来确认 token 是否配置成功。


## 5. 运行方式

### 5.1 仅使用 train 标签构建 Gold Seed KG

不需要 API Key，适合先快速跑通项目：

```bash
python src/build_kg.py
```

输出：

```text
data/triples.csv
data/processed/entities.csv
data/processed/relations.csv
data/processed/causal_events.csv
data/processed/kg.json
results/kg_visualization.html
```

### 5.2 启用 Qwen 抽取扩展 KG

先小规模测试：

```bash
python src/build_kg.py --extract_qwen --max_qwen_docs 5 --sleep_seconds 0.2
```

正式抽取：

```bash
python src/build_kg.py --extract_qwen --qwen_files unlabel.json,testA.json,testB.json
```

如果 API 调用成本较高，建议先只对 `unlabel.json` 的前 50—100 条做抽取演示。

### 5.3 运行知识推理

```bash
python src/reasoning.py --query "高血压是否可能间接导致心肌梗死？"
python src/reasoning.py --query "胃溃疡属于什么类型？"
python src/reasoning.py --query "在什么条件下宫腔粘连可能导致月经量少？"
```

### 5.4 GraphRAG 证据检索

```bash
python src/graph_retrieval.py --question "为什么高血压可能和心肌梗死有关？"
```

### 5.5 大模型增强问答

KG-only 模式，不需要 API：

```bash
python src/llm_qa.py --question "为什么高血压可能和心肌梗死有关？" --mode kg_only
```

KG-enhanced 模式，优先调用 Qwen；若未配置 API，会自动退化为 KG-only 模板答案：

```bash
python src/llm_qa.py --question "为什么高血压可能和心肌梗死有关？" --mode kg_augmented
```

LLM-only 模式，需要 Qwen API：

```bash
python src/llm_qa.py --question "为什么高血压可能和心肌梗死有关？" --mode llm_only
```

### 5.6 运行评估

```bash
python src/evaluate.py
```

输出：

```text
results/metrics.json
results/extraction_metrics.json
results/cases.md
data/processed/test_questions.json
```

## 6. 代码结构

```text
project/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/
│   │   ├── train_0717.json
│   │   ├── unlabel.json
│   │   ├── testA.json
│   │   ├── testB.json
│   │   └── example_code.txt
│   ├── processed/
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
│   ├── reasoning.py
│   ├── graph_retrieval.py
│   ├── llm_qa.py
│   └── evaluate.py
├── prompts/
│   ├── qwen_extraction_prompt.txt
│   └── kg_augmented_prompt.txt
├── results/
│   ├── cases.md
│   ├── metrics.json
│   ├── extraction_metrics.json
│   ├── reasoning_cases.md
│   └── kg_visualization.html
├── report.pdf
└── slides.pdf
```

## 7. 实验设计建议

### 实验 1：图谱规模与质量

统计：

- 实体数量
- 三元组数量
- 实体类型数量
- 关系类型数量
- Gold Seed KG 三元组数量
- Qwen-extracted KG 三元组数量
- `causes` / `condition_of` / `is_a` 分布
- `CausalEvent` 数量

### 实验 2：Qwen 抽取质量评估

可从 `train_0717.json` 抽样一部分文本，让 Qwen 抽取，再和 gold 标签比较。

对比方法：

- Rule baseline
- Qwen zero-shot
- Qwen few-shot
- Qwen few-shot + schema check

指标：

- Precision
- Recall
- F1
- relation-wise F1

### 实验 3：推理评估

测试问题类型：

- 直接因果推理
- 反向因果推理
- 多跳因果链推理
- 条件约束推理
- 上下位概念推理

### 实验 4：GraphRAG 问答对比

对比：

- LLM-only
- KG-only
- KG-enhanced

指标：

- Accuracy
- Evidence Traceability
- Hallucination Rate
- Refusal Accuracy
- Multi-hop Correctness

## 8. 小组分工示例

| 成员 | 任务 |
|---|---|
| A | 数据解析、Gold Seed KG 构建、Schema 设计 |
| B | Qwen 抽取 Prompt、API 调用、抽取结果清洗 |
| C | 推理模块、GraphRAG 检索模块 |
| D | 实验评估、报告撰写、PPT 制作 |

## 9. 注意事项

- `train_0717.json` 的人工标签直接构建 Gold Seed KG，是高质量种子图谱。
- `unlabel.json/testA.json/testB.json` 通过 Qwen 抽取补充关系，体现信息抽取模块。
- 医学回答必须带证据链，并说明不构成医疗建议。
- Qwen 抽取结果需要人工抽样审核，报告中建议加入错误分析。
