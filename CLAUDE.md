# Moral Bottleneck — Project Guide

## Overview

This project explores using psychological moral theories as "bottlenecks" to improve LLM alignment with human moral judgments. The core hypothesis: prompting LLMs to reason through structured moral frameworks (Dyadic Morality, MFT, Deontology, etc.) before producing a final score improves correlation with human annotations.

## Project Structure

```
moral_bottleneck/
├── data/moral_data_circumstances/dyadic/all_dyadic.tsv  # Main dataset (649 scenarios × 4 circumstances)
├── experiments/
│   ├── end_to_end.py                    # Direct baseline (no bottleneck)
│   ├── dyadic_bottleneck.py             # Theory of Dyadic Morality bottleneck
│   ├── mft_bottleneck.py                # Moral Foundations Theory bottleneck
│   ├── moral_bottleneck.py              # Generic bottleneck (any theory)
│   ├── prompt_variants_end_to_end.py    # Few-shot / CoT template variants
│   ├── baseline_prompts/                # Prompt templates (basic, few-shot, CoT, CoT+few-shot)
│   ├── prompts/                         # Theory-specific prompts (deontology, utilitarianism, etc.)
│   ├── gptinference/                    # OpenAI API wrapper with JSONL caching
│   ├── results/{model}/                 # Output TSVs per model and experiment
│   ├── cache/{model}/                   # Cached API responses (JSONL)
│   └── evaluation/
│       ├── evaluation.py                # Full evaluation pipeline
│       ├── evaluation_moral_theory.py   # Evaluate single theory/template
│       ├── evaluate_end_to_end_regression.py
│       ├── metadata.json                # TSV schema per experiment type
│       ├── evaluation_logs/{model}/     # Saved stdout logs
│       └── weights/{model}/            # Saved MLP weights (.pth)
├── other_experiments/                   # Post-hoc: regression, MLP, MoE
├── pyproject.toml
└── uv.lock
```

## Setup & Running

### Install dependencies
```bash
uv sync
```

### Required environment variables
```bash
export OPENAI_API_KEY=sk-...
export TOGETHER_AI_API_KEY=...   # For Together AI models
```

### Run experiments (from `experiments/`)
```bash
# End-to-end baseline
uv run python end_to_end.py --model gpt-4o --all-circumstances

# Dyadic bottleneck
uv run python dyadic_bottleneck.py --model gpt-4o --all-circumstances

# MFT bottleneck
uv run python mft_bottleneck.py --model gpt-4o --all-circumstances

# Custom theory bottleneck
uv run python moral_bottleneck.py --theory-name deontology \
  --theory-prompt-path prompts/deontology.txt --model gpt-4o --num-addn-qns 4

# Template variants (few-shot / CoT)
uv run python prompt_variants_end_to_end.py --model gpt-4o \
  --template-path baseline_prompts/few_shot_template.txt --temperature 0.0
```

### Evaluate results (from `experiments/evaluation/`)
```bash
uv run python evaluation_moral_theory.py --model gpt-4o --theory cot_few_shot_template_end_to_end_all

# With saved log
uv run python evaluation_moral_theory.py --model gpt-4o --theory cot_few_shot_template_end_to_end_all \
  > evaluation_logs/gpt-4o/cot_few_shot_template_end_to_end_all.txt 2>&1
```

## Dataset Split (Fixed)

| Split | IDs | Count |
|-------|-----|-------|
| Train | id ≤ 96 | 96 scenarios |
| Val | 96 < id ≤ 146 | 50 scenarios |
| Test | id > 146 | 503 scenarios |

Evaluation scripts always report test-set metrics unless the file is named `end_to_end` exactly.

## Result File Naming

```
results/{model_dir_name}/{experiment_type}.tsv
```

- Model names with `/` are replaced by `_` (e.g., `meta/llama3` → `meta_llama3`)
- Template variants are named e.g. `cot_few_shot_template_end_to_end_all.tsv`

## metadata.json

Every result TSV needs an entry in `experiments/evaluation/metadata.json`. The key is the filename without `.tsv`:

```json
"cot_few_shot_template_end_to_end_all": {
    "moral_score_col": "moral_score",
    "human_score_col": "human_score",
    "feature_cols": []
}
```

- `feature_cols: []` → LLM evaluation only (no regression/MLP step)
- Non-empty `feature_cols` → triggers Ridge regression + MLP + ensemble on those columns

## Bottleneck Approaches

| Approach | feature_cols in metadata | Intermediate scores |
|---|---|---|
| `end_to_end` | `[]` | None |
| `dyadic` | `[vulnerable_score, intentional_score, harm_score, help_score]` | 4 dyadic dimensions |
| `mft` | `[Harm/ Help, Cheating/ Fairness, ...]` | 6 MFT foundations |
| `deontology` / `utilitarianism` | `[Q5_score]` | 1 theory question |
| `morality_as_cooperation` | `[Q1_score ... Q7_score]` | 7 questions |
| `virtue_ethics` | `[Q1_score ... Q4_score]` | 4 questions |

## Caching

All API calls are cached in `experiments/cache/{model}/{experiment_type}.jsonl`. Re-running a script will not make duplicate API calls for already-cached prompts.

## Supported Models

- OpenAI: `gpt-3.5-turbo`, `gpt-4o`, `o1`, `o3-mini`
- Together AI: `meta/llama3-70b-instruct`, `mistralai/mixtral-8x22b-instruct-v0.1`, `deepseek-ai/DeepSeek-R1`
- Qwen: `qwen3-thinking`, `qwen3-non-thinking`
