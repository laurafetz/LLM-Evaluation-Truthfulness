# Evaluation outputs

- `model_0_evaluated.csv` through `model_4_evaluated.csv`: baseline, system prompt, LoRA, base RAG, and instruct RAG respectively, one row per question.
- `model_summary.csv`: correctness on every question, bootstrap intervals, similarity coverage, and filtered diagnostics.
- `paired_comparisons.csv`: matched differences, paired bootstrap intervals, exact McNemar p-values, and Holm adjustment.
- `summary_by_question_type.csv`: unfiltered scores by adversarial/non-adversarial question type.
- `correctness_all.png`: primary comparison figure.
- `run_metadata.json`: input hashes, scorer settings, and runtime versions for the full scoring run.
- `original_*.csv` and `validation_report.csv`: historical coursework summaries and initial input-alignment checks.

Run `python analysis.py --summarize-only` to reproduce the current tables from the scored answer files. The similarity-filtered subsets differ by model and are not used for the primary paired comparison.
