# CHIP2022 Medical Causal Event KG

本项目基于 CHIP2022 医疗因果实体关系抽取数据构建医疗因果知识图谱，并实现图谱推理、GraphRAG 证据检索和知识图谱增强问答。系统主要包含四部分：知识图谱构建、规则/路径推理、图谱增强问答、实验评估与可视化。

## 1. 课程要求对应关系

| 课程要求 | 本项目实现 | 对应文件 |
|---|---|---|
| 知识图谱构建 | 从 CHIP2022 标注数据构建 Gold Seed KG，并使用 Qwen 从无标注医学文本中扩展三元组 | `src/build_kg.py`、`data/triples.csv` |
| 数据来源 | 使用 CHIP2022 原始数据，包括人工标注文本和无标注文本 | `data/raw/` |
| Schema 设计 | 设计 11 类实体和 8 类关系，覆盖疾病、症状、检查、治疗、部位、风险因素和因果条件 | README 第 2 节 |
| 信息抽取 | 解析人工标注关系，并可选调用 Qwen 进行三元组抽取 | `src/build_kg.py`、`prompts/qwen_extraction_prompt.txt` |
| 图谱存储 | 使用 CSV 和 JSON 保存完整图谱 | `data/triples.csv`、`data/processed/kg.json` |
| 图谱可视化 | 生成 HTML 抽样可视化，并在 Gradio 页面中展示问答证据子图 | `results/kg_visualization.html`、`src/qa_server.py` |
| 知识推理 | 支持直接因果、反向因果、多跳因果、条件约束、上下位、症状、治疗、部位、诊断和风险因素查询 | `src/reasoning.py` |
| 大模型增强 | 先从图谱中检索三元组、路径和条件证据，再注入 Prompt 生成回答 | `src/graph_retrieval.py`、`src/llm_qa.py` |
| 实验与分析 | 生成测试问题，输出评估指标和 GraphRAG 案例 | `src/evaluate.py`、`results/metrics.json`、`results/cases.md` |

## 2. 数据与 Schema

原始数据位于：

```text
data/raw/
├── train_0717.json   # 人工标注数据，用于构建 Gold Seed KG
├── unlabel.json      # 无标注文本，用于 Qwen 抽取扩展
├── testA.json        # 可选扩展数据
└── testB.json        # 可选扩展数据
```

实体类型：

```text
Disease, Symptom, ClinicalSign, PathologicalState, RiskFactor,
TestResult, ExamProcedure, Treatment, AnatomicalSite,
MedicalCategory, Other
```

关系类型：

```text
causes, risk_factor_for, condition_of, is_a,
symptom_of, treated_by, located_in, diagnosed_by
```

CHIP2022 原始标签和本项目关系的对应：

| 原始标签 | 项目关系 | 含义 |
|---:|---|---|
| 1 | `causes` | 因果关系 |
| 2 | `condition_of` | 条件修饰关系 |
| 3 | `is_a` | 上下位关系 |

`condition_of` 保存为“条件 -> 原因实体”。例如：

```text
宫腔粘连 --causes--> 月经量少
女性激素非常好 --condition_of--> 宫腔粘连
```

## 3. 项目结构

```text
src/
├── build_kg.py           # 构建知识图谱
├── config.py             # 读取 Qwen API 配置
├── reasoning.py          # 图谱推理
├── graph_retrieval.py    # GraphRAG 证据检索
├── llm_qa.py             # LLM-only / KG-only / KG-augmented 问答
├── qa_server.py          # Gradio 交互界面
├── evaluate.py           # 实验评估
└── relation_discovery.py # 关系发现探索脚本，主流程不依赖

prompts/
├── qwen_extraction_prompt.txt
├── kg_augmented_prompt.txt
└── relation_discovery_prompt.txt

results/
├── kg_visualization.html
├── metrics.json
├── cases.md
├── reasoning_results.json
└── llm_qa_comparison.json
```

整体流程：

```text
data/raw/*.json
  -> src/build_kg.py
  -> data/triples.csv + data/processed/kg.json
  -> src/reasoning.py / src/graph_retrieval.py / src/llm_qa.py
  -> src/evaluate.py
  -> results/metrics.json + results/cases.md
```

## 4. 环境配置

```bash
conda create -n chipkg python=3.10
conda activate chipkg
pip install -r requirements.txt
```

Qwen API 只在重新抽取三元组、运行 `llm_only` 或真实 `kg_augmented` 生成式回答时需要。图谱构建、KG-only、推理、检索、评估和界面可以在没有 API 的情况下运行。

如需调用 Qwen：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
QWEN_API_KEY=your_api_key_here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.5-flash
QWEN_TEMPERATURE=0
QWEN_MAX_TOKENS=32k
```

## 5. 构建知识图谱

只使用人工标注数据构建 Gold Seed KG：

```bash
python src/build_kg.py
```

使用已有 Qwen 抽取结果构建 Gold + Qwen 混合图谱：

```bash
python src/build_kg.py --reuse_qwen_outputs
```

重新调用 Qwen 从无标注文本抽取三元组：

```bash
python src/build_kg.py --extract_qwen --qwen_files unlabel.json --sleep_seconds 0.2
```

小规模调试：

```bash
python src/build_kg.py --extract_qwen --max_qwen_docs 5 --sleep_seconds 0.2
```

构建输出：

| 输出文件 | 说明 |
|---|---|
| `data/triples.csv` | 完整三元组表，是完整图谱的主要文件 |
| `data/processed/kg.json` | JSON 格式完整图谱，供程序读取 |
| `data/processed/entities.csv` | 实体列表和实体类型 |
| `data/processed/relations.csv` | 关系类型和数量 |
| `data/processed/causal_events.csv` | 条件因果事件 |
| `data/processed/build_stats.json` | 图谱规模统计 |
| `results/kg_visualization.html` | 抽样图谱可视化，默认最多 800 条边 |

## 6. 运行推理

```bash
python src/reasoning.py --query "高血压是否可能间接导致心肌梗死？"
python src/reasoning.py --query "胃溃疡属于什么类型？"
python src/reasoning.py --query "在什么条件下宫腔粘连可能导致月经量少？"
python src/reasoning.py --query "乳腺癌有哪些风险因素？"
python src/reasoning.py --query "肠癌如何诊断？"
python src/reasoning.py --query "瘫痪如何治疗？"
```

JSON 输出：

```bash
python src/reasoning.py --query "乳腺癌有哪些风险因素？" --json
```

## 7. 运行 GraphRAG 检索

```bash
python src/graph_retrieval.py --question "为什么高血压可能和心肌梗死有关？"
python src/graph_retrieval.py --question "在什么条件下宫腔粘连可能导致月经量少？"
```

检索结果包含：

- `triples`：相关三元组证据
- `paths`：多跳因果路径
- `conditional_events`：条件证据
- `matched_entities`：问题命中的图谱实体

## 8. 运行问答

KG-only，不需要 API：

```bash
python src/llm_qa.py --question "为什么高血压可能和心肌梗死有关？" --mode kg_only
```

KG-augmented，优先调用 Qwen；未配置 API 时退化为 KG-only：

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

## 9. 运行交互界面

```bash
python src/qa_server.py
```

浏览器打开：

```text
http://localhost:7860
```

如果端口被占用，可以换一个空闲端口：

```bash
GRADIO_SERVER_PORT=7861 python src/qa_server.py
```

如果环境设置了 SOCKS 代理，Gradio 导入 `httpx` 时可能需要 `socksio`。依赖已写入 `requirements.txt`，可重新安装：

```bash
pip install -r requirements.txt
```

## 10. 图谱可视化

启动本地静态服务：

```bash
python -m http.server 8000
```

浏览器打开：

```text
http://localhost:8000/results/kg_visualization.html
```

`results/kg_visualization.html` 是抽样可视化，不是完整图谱。完整图谱有 18,835 条三元组，全部写入 HTML 会影响浏览器加载和交互，因此 `src/build_kg.py` 默认通过 `--visualize_max_edges 800` 最多写入 800 条边。

抽样逻辑是先按关系类型分组，每种关系取一部分边，保证 `causes`、`is_a`、`symptom_of`、`treated_by` 等关系都能被看到；如果还没达到上限，再继续补充其他三元组。

完整图谱以以下文件为准：

```text
data/triples.csv
data/processed/kg.json
```

如需生成包含更多边的 HTML：

```bash
python src/build_kg.py --reuse_qwen_outputs --visualize_max_edges 2000
```

## 11. 运行评估

```bash
python src/evaluate.py
```

限制测试问题数量：

```bash
python src/evaluate.py --max_questions 50
```

评估输出：

| 输出文件 | 说明 |
|---|---|
| `results/metrics.json` | 总评估指标，包括图谱规模、检索命中率、证据非空率和抽取 exact-match 指标 |
| `results/extraction_metrics.json` | Qwen 抽取与 Gold 三元组的精确匹配结果 |
| `results/cases.md` | GraphRAG 案例，包括问题、标准答案和检索证据 |
| `results/error_cases.json` | 错误 case 的结构化 JSON |
| `data/processed/test_questions.json` | 自动生成的测试问题 |

## 12. 主要结果

当前图谱规模：

| 指标 | 数值 |
|---|---:|
| 实体数量 | 14,762 |
| 三元组数量 | 18,835 |
| 实体类型数量 | 11 |
| 关系类型数量 | 8 |
| Gold Seed 三元组 | 9,126 |
| Qwen 抽取三元组 | 9,709 |
| CausalEvent 数量 | 783 |

50 case 评估结果：

| 指标 | 数值 |
|---|---:|
| 测试问题数 | 50 |
| GraphRAG 检索命中率 | 0.88 |
| 证据非空率 | 1.00 |
| Gold 证据召回率 | 0.92 |
| 推理证据覆盖率 | 0.94 |
| 平均证据数 | 2.80 |
| 错误 case 数量 | 6 |

50 case 覆盖直接因果、反向因果、多跳因果、条件推理、上下位、症状、治疗、部位、诊断、负例检查和跨关系问题，结果见 `results/metrics.json`、`results/reasoning_results.json`、`results/llm_qa_comparison.json` 和 `results/error_cases.json`。

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
