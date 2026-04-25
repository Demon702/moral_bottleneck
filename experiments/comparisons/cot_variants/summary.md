# CoT / few-shot / dyadic variants × {gpt-4o, qwen3-235B-Instruct}

Test-set MSE on n=503 scenarios.

For dyadic columns we report the **MLP** head over the 4 dyadic features (vulnerable / intentional / harm / help). For end-to-end columns we report the model's direct moral score.

## Headline table

| model | cot | few_shot | cot+few_shot | cot+dyadic (MLP) | cot+few_shot+dyadic (MLP) |
|---|---|---|---|---|---|
| gpt-4o | 2.554 | 1.316 | 1.429 | 0.973 | 0.675 |
| qwen3-235B-Instruct | 2.770 | 1.326 | 1.390 | 1.068 | **0.661** |

## Adding dyadic always helps

| model | cot → cot+dyadic | cot+few_shot → cot+few_shot+dyadic |
|---|---|---|
| gpt-4o | 2.554 → 0.973 (**−61.9%**) | 1.429 → 0.675 (**−52.8%**) |
| qwen3-235B-Instruct | 2.770 → 1.068 (**−61.4%**) | 1.390 → 0.661 (**−52.5%**) |

A remarkably stable ~62% MSE reduction over plain CoT and ~53% over CoT+few-shot — the same magnitude across both models, suggesting the dyadic structure (not model scale or prompt style) is the load-bearing piece.

## Notes

- **Best overall**: qwen3-235B-Instruct `cot+few_shot+dyadic` at MLP stage — MSE 0.661.
- **CoT alone is the weakest baseline** for both models, slightly worse than few-shot alone.
- **Few-shot is competitive on its own** — gpt-4o few-shot (1.316) beats gpt-4o cot+few-shot (1.429); qwen behaves similarly.
- **A ~10× cheaper model edges out gpt-4o** at the strongest configuration (cot+few_shot+dyadic, MLP), once Qwen's `<think>`-tag chat-template collision is sidestepped via `<thinking>`.

## Files

```
gpt-4o/
  cot.tsv                    cot_template (zero-shot CoT)
  few_shot.tsv               few_shot_template (no CoT)
  cot+few_shot.tsv           cot_few_shot_template
  cot+dyadic.tsv             cot_dyadic_bottleneck
  cot+few_shot+dyadic.tsv    cot_few_shot_dyadic_bottleneck

qwen3-235b-instruct/        Qwen/Qwen3-235B-A22B-Instruct-2507-tput on Together
  cot.tsv                    cot_template_thinking
  few_shot.tsv               few_shot_template (plain — no thinking tags involved)
  cot+few_shot.tsv           cot_few_shot_template_thinking
  cot+dyadic.tsv             cot_dyadic_bottleneck (<thinking> tag)
  cot+few_shot+dyadic.tsv    cot_few_shot_dyadic_bottleneck (<thinking> tag)
```

Qwen runs requiring CoT use `<thinking>` instead of `<think>` because the latter collides with Qwen3-235B-Instruct's chat-template reasoning hooks and silently breaks JSON formatting (~64% parse failure rate observed in the broken variant before the fix).
