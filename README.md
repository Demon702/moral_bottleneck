# Moral Bottleneck

In this work we exploit popular theories in psychology as bottleneck to improve LLM's performance on human moral alignment task.
Our method improves upon the baseline where we provide no theory and ask the LLM for a moral acceptability score.
We used two popular psychological theories MFT (Moral Foundations Theory) and TDM (Theory of Dyadic Morality).

## Method
We prompt GPT-3.5-turbo to score moral schema and then generate a final moral acceptability score. Since the schema scores are real numbers, we can train additional Regression, MLP and Mixture Of Experts and other deep learning modules on these schema scores and get a final moral score. We also explore ensemble approaches where we combine different model's predictions to obtain a final moral score. We provide the prompts under the [prompts](prompts) folder for reproducibility. We used `temperature=0.0` for text generation.

## Results
Our code and results have been summarized in this [colab notebook](https://colab.research.google.com/drive/1_rv1vzzuyNbj_vUXUNJ33Gvfux4hizAz?usp=sharing).
More specific implementation details have been provided in the [experiments](experiments/) folder.

Here are the results in tabular format.

| Baseline Type     | Baselines         | Overall Pearson correlation | Overall MSE | Hard Subset Pearson correlation | Hard Subset MSE |
|-------------------|-------------------|-----------------------------|-------------|---------------------------------|-----------------|
| No Theory         | GPT End to End    | 0.8667                      | 2.5886      | 0.2737                          | 8.9121          |
| Bottleneck        | GPT dyadic bottleneck | 0.8411                 | 2.7395      | 0.2675                          | 10.1724         |
|                   | GPT MFT bottleneck  | 0.8130                 | 3.7102      | -0.1434                         | 14.7328         |
| Regression        | Regression Dyadic   | 0.8576                 | 2.2067      | 0.3382                          | 0.7912          |
|                   | Regression MFT      | 0.5375                 | 3.0018      | -0.1159                         | 1.8041          |
| MLP               | MLP Dyadic          | 0.8671                 | 1.1019      | 0.3703                          | 1.5755          |
|                   | MLP MFT             | 0.6059                 | 2.7154      | -0.1800                         | 2.6152          |
| Ensemble          | Dyadic              | 0.8790                 | 0.9603      | 0.3554                          | 1.9778          |
|                   | MFT                 | 0.7996                 | 1.5192      | -0.1735                         | 4.0866          |
| Ensemble with End to End | Dyadic         | 0.9025                 | 0.7890      | 0.3554                          | 1.9262          |
|                   | MFT                 | 0.8407                 | 1.2623      | 0.1381                          | 2.2764          |
| Mixture of Experts with Theories | -    | 0.8856                 | 0.9121      | 0.2172                          | 1.5903          |
| Mixture of Experts with Features | -    | 0.7810                 | 1.7848      | 0.0070                          | 3.2713          |
| Mixture of Experts with Features Top 2 | - | 0.7906              | 2.0572      | 0.2514                          | 5.0678          |
| MoE with end to end | -                 | 0.9043                 | 1.0150      | 0.2900                          | 3.3736          |


