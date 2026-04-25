# Moral Bottleneck

We explore using popular psychological theories as bottlenecks to improve LLM performance on human moral alignment tasks. Rather than asking an LLM to directly produce a moral acceptability score, we prompt it to first reason through structured moral frameworks — Theory of Dyadic Morality (TDM) and Moral Foundations Theory (MFT) — and then derive a final score from those intermediate judgments. The intermediate scores can also be used as features for downstream regression and ensemble models.

## Method

We prompt LLMs to score moral schema defined by each theory and then generate a final moral acceptability score on a scale of -4 to 4. Since the schema scores are real numbers, we train additional Ridge regression, MLP, and ensemble models on these features to further improve predictions. We use `temperature=0.0` throughout for reproducibility. Prompts are provided in the [prompts](prompts/) folder.

**Moral theories explored:**
- **Theory of Dyadic Morality (TDM)** — 4 dimensions: vulnerability, intentionality, harm, help
- **Moral Foundations Theory (MFT)** — 6 foundations: Harm/Help, Cheating/Fairness, Betrayal/Loyalty, Subversion/Authority, Degradation/Sanctity, Oppression/Liberty
- **Deontology, Utilitarianism, Virtue Ethics, Morality as Cooperation** — custom theory prompts

**Baselines:**
- End-to-end direct scoring (no theory)
- CoT (chain-of-thought)
- Few-shot templates
- CoT + few-shot combined

## Dataset

We evaluate on the [Social Chemistry 101](https://maxwellforbes.com/social-chemistry/) dataset, using 649 moral scenarios across 7 sub-datasets (Clifford, Effron, Mickelberg, Cook, Grizzard, Kruepke, Lotto), each with 4 circumstance variations (2,596 total rows) and human moral acceptability scores.

**Fixed split:** Train: 96 · Val: 50 · Test: 503 scenarios

## Results

### Bottleneck Approaches (GPT-4o)

| Approach | Method | Pearson | MSE |
|---|---|---|---|
| End-to-End (no theory) | LLM | 0.8824 | 2.9148 |
| Dyadic Bottleneck | LLM | 0.9050 | 1.9187 |
| Dyadic Bottleneck | + Regression | 0.8845 | 0.9340 |
| Dyadic Bottleneck | + MLP | 0.8907 | 0.8884 |
| MFT Bottleneck | LLM | 0.9058 | 2.5098 |
| MFT Bottleneck | + Regression | 0.8978 | 0.8357 |
| MFT Bottleneck | + MLP | 0.8967 | 0.8282 |
| CoT | LLM | 0.8610 | 3.4100 |

### Template Baselines (CoT + Few-Shot)

| Model | Pearson | MSE |
|---|---|---|
| GPT-4o | 0.9152 | 1.4287 |
| Qwen3-thinking | 0.9182 | 0.6751 |
| Qwen3-non-thinking | 0.8981 | 1.0390 |
| DeepSeek-R1 | 0.9365 | 0.5204 |

### End-to-End Baselines (No Theory, Direct Scoring)

| Model | Pearson | MSE |
|---|---|---|
| GPT-3.5-turbo | 0.8352 | 3.7867 |
| GPT-4o | 0.8824 | 2.9148 |
| Meta Llama-3 70B | 0.9056 | 1.6983 |
| DeepSeek-R1 | 0.9084 | 1.6330 |
| o3-mini | 0.8519 | 3.5643 |

### GPT-3.5-turbo Bottleneck Results (original experiments)

| Approach | Method | Pearson | MSE |
|---|---|---|---|
| End-to-End | LLM | 0.8352 | 3.7867 |
| Dyadic Bottleneck | LLM | 0.8326 | 3.6026 |
| Dyadic Bottleneck | + Regression | 0.8723 | 1.0228 |
| Dyadic Bottleneck | + MLP | 0.8816 | 0.9567 |
| MFT Bottleneck | LLM | 0.8203 | 3.3119 |
| MFT Bottleneck | + Regression | 0.5359 | 3.0821 |
| MFT Bottleneck | + MLP | 0.6440 | 2.7350 |

## Repository Structure

```
├── data/                        # Input dataset
├── experiments/
│   ├── end_to_end.py            # Direct scoring baseline
│   ├── dyadic_bottleneck.py     # TDM bottleneck
│   ├── mft_bottleneck.py        # MFT bottleneck
│   ├── moral_bottleneck.py      # Generic theory bottleneck
│   ├── prompt_variants_end_to_end.py  # Template-based variants
│   ├── baseline_prompts/        # Few-shot / CoT prompt templates
│   ├── prompts/                 # Theory-specific prompts
│   ├── results/                 # Output TSVs organized by model
│   └── evaluation/              # Evaluation scripts and logs
├── other_experiments/           # Regression, MLP, MoE analyses
└── pyproject.toml
```

## Setup

```bash
uv sync
export OPENAI_API_KEY=sk-...
```

See [CLAUDE.md](CLAUDE.md) for detailed usage instructions.
