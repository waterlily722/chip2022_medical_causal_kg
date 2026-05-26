# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CHIP2022 Medical Causal Event KG — a GraphRAG-enhanced medical Q&A system that builds a knowledge graph from Chinese medical text and uses it for causal reasoning and question answering. The project is a university course assignment for a Knowledge Graph class.

Two-phase KG construction: (1) **Gold Seed KG** from human-annotated `train_0717.json`, (2) **Qwen-extracted KG** via LLM API calls on unlabeled medical text. The merged graph supports multi-hop causal chain reasoning, condition constraints, hypernym reasoning, and three QA modes (LLM-only, KG-only, KG-augmented).

## Setup

```bash
conda create -n chipkg python=3.10
conda activate chipkg
pip install -r requirements.txt
cp .env.example .env  # then edit with QWEN_API_KEY
```

The system works without an API key in degraded mode (gold KG only, no LLM extraction or QA).

## Common Commands

```bash
# Build gold seed KG only (no API needed)
python src/build_kg.py

# Build KG with Qwen extraction (requires API key)
python src/build_kg.py --extract_qwen --max_qwen_docs 5 --sleep_seconds 0.2
python src/build_kg.py --extract_qwen --qwen_files unlabel.json

# Run reasoning queries
python src/reasoning.py --query "高血压是否可能间接导致心肌梗死？"

# GraphRAG evidence retrieval
python src/graph_retrieval.py --question "为什么高血压可能和心肌梗死有关？"

# LLM QA (three modes)
python src/llm_qa.py --question "为什么高血压可能和心肌梗死有关？" --mode kg_only
python src/llm_qa.py --question "为什么高血压可能和心肌梗死有关？" --mode kg_augmented
python src/llm_qa.py --question "为什么高血压可能和心肌梗死有关？" --mode llm_only

# Run evaluation
python src/evaluate.py
```

There is no formal test suite. Evaluation uses `src/evaluate.py`, which generates 30 auto-generated test questions from the KG and reports metrics to `results/metrics.json`.

## Architecture

**Data flow:**

```
data/raw/*.json → src/build_kg.py → data/processed/ (CSV, JSON) + data/triples.csv
                                       ↓
                    src/reasoning.py (graph indexing + BFS path finding)
                    src/graph_retrieval.py (evidence retrieval for QA)
                    src/llm_qa.py (3-mode QA: kg_only / llm_only / kg_augmented)
                                       ↓
                    results/ (metrics, cases, visualization HTML)
```

**Modules (all in `src/`):**

| File | Role |
|---|---|
| `config.py` | Loads Qwen API settings from `.env` via `python-dotenv`. Returns `QwenSettings` dataclass. |
| `build_kg.py` | Main KG construction. Parses CHIP2022 JSON, runs Qwen extraction, infers entity types, deduplicates triples, outputs CSV/JSON/HTML. |
| `reasoning.py` | `KGReasoner` class — indexes the graph and supports 5 reasoning types: direct/reverse causal, multi-hop (BFS up to 3 hops), condition, hypernym. |
| `graph_retrieval.py` | `retrieve_evidence()` — retrieves triples, multi-hop paths, and conditional events for a given question. |
| `llm_qa.py` | Three QA modes. `kg_augmented` retrieves evidence then prompts Qwen; falls back to `kg_only` without API. |
| `evaluate.py` | Generates test questions, computes extraction precision/recall/F1, retrieval hit rate, and writes case studies. |
| `relation_discovery.py` | Samples unlabeled text and uses Qwen for open-ended relation discovery. |

**Key design patterns:**

- **Triple deduplication priority**: gold > qwen > inferred
- **Entity type inference**: Rule-based medical mention classifier when types are missing from source data
- **Graceful degradation**: All modules handle missing API keys without crashing
- **`condition_of` handling**: Stored as a relation from condition entity to cause entity; the original cause-effect edge is preserved separately
- **Confidence scoring**: Each triple carries a source and confidence score

## Schema

**Entity types**: Disease, Symptom, ClinicalSign, PathologicalState, RiskFactor, TestResult, ExamProcedure, Treatment, AnatomicalSite, MedicalCategory, Other

**Relation types**: causes, risk_factor_for, condition_of, is_a, symptom_of, treated_by, located_in, diagnosed_by

The CHIP2022 source data uses relation labels 1/2/3 mapped to causes/condition_of/is_a. The `condition_of` relation means "condition modifies a cause" — e.g., `女性激素非常好 --condition_of--> 宫腔粘连` where `宫腔粘连 --causes--> 月经量少`.

## Prompts

LLM prompts live in `prompts/`:
- `qwen_extraction_prompt.txt` — detailed IE instructions with few-shot examples for relation extraction
- `kg_augmented_prompt.txt` — grounded QA template
- `relation_discovery_prompt.txt` — open-ended relation discovery

## Language

The domain text, prompts, and CLI queries are all in **Chinese**. Code comments and docstrings are in English. The README is in Chinese.
