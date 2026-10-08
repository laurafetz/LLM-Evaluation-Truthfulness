# LLM Evaluation Truthfulness
# Evaluating LLM Truthfulness with Prompting, Fine-Tuning and RAG

This project evaluates how different large language model adaptation strategies affect the truthfulness and semantic quality of generated answers.

Five variants of **Qwen 2.5** are compared using TruthfulQA-derived question and reference-answer data:

1. Qwen 2.5 baseline
2. Baseline + system prompt
3. LoRA fine-tuned Qwen 2.5
4. Baseline + Retrieval-Augmented Generation (RAG)
5. Qwen 2.5 Instruct + RAG

The repository contains the model-output datasets required to reproduce the evaluation.

## Research Question

How do prompt engineering, LoRA fine-tuning, and retrieval-augmented generation affect the truthfulness and semantic similarity of Qwen 2.5 responses?

## Data

The repository distinguishes between the **source benchmark data** and the **generated model outputs**.

### Source benchmark

`data/TruthfulQA-2.csv`

This is the TruthfulQA dataset used in the project. It contains 817 benchmark questions, including question type, category, correct reference answers, incorrect reference answers, and source information.

### Generated model outputs

| File | Model |
| --- | --- |
| `qwen_qa_results_no_system_prompt.csv` | Qwen 2.5 baseline |
| `qwen_qa_results_system_prompt.csv` | Baseline + system prompt |
| `qwen_qa_results.csv` | LoRA fine-tuned model |
| `qwen_base_rag_answers.csv` | Baseline + RAG |
| `qwen_instruct_rag_answers.csv` | Qwen Instruct + RAG |

`analysis.py` merges each generated-answer file back to the canonical TruthfulQA benchmark by question before computing the evaluation metrics.

## Evaluation

The analysis compares each generated response with the available correct and incorrect reference answers.

For each response:

1. BLEURT is used to identify the best-matching reference answer.
2. BLEU and ROUGE measure lexical overlap with the selected reference.
3. Sentence-transformer embeddings are used to calculate cosine similarity.
4. The generated response is classified as reference-based correct when its best-matching reference belongs to the correct-answer set.
5. Confidence-filtered correctness is calculated for responses with cosine similarity above a predefined threshold.

The default threshold used in `analysis.py` is:

```python
CONFIDENCE_THRESHOLD = 0.60
```


## Original Reported Results

The original project results are included in the `results/` folder, so the repository documents both the input data and the reported outputs without requiring the models to be rerun.

| Model | High-confidence correctness | Mean cosine similarity | High-confidence responses |
| --- | ---: | ---: | ---: |
| Baseline | 42.45% | 0.489 | 318 |
| Baseline + system prompt | 44.98% | 0.694 | 618 |
| LoRA fine-tuned | **60.27%** | 0.577 | 516 |
| Baseline + RAG | 47.60% | 0.413 | 292 |
| Qwen Instruct + RAG | 49.64% | **0.729** | **685** |

The original notebook also reports separate results for adversarial and non-adversarial questions; these are included as CSV files in `results/`.

## Reproducibility

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python analysis.py
```

The script reads all five files from `data/` and writes the evaluation outputs to `results/`.

## Repository Structure

```text
Python Project 1 - LLM Evaluation/
├── README.md
├── analysis.py
├── requirements.txt
├── data/
│   ├── TruthfulQA-2.csv
│   ├── qwen_qa_results_no_system_prompt.csv
│   ├── qwen_qa_results_system_prompt.csv
│   ├── qwen_qa_results.csv
│   ├── qwen_base_rag_answers.csv
│   └── qwen_instruct_rag_answers.csv
└── results/
    ├── original_total_metrics.csv
    ├── original_adversarial_metrics.csv
    ├── original_non_adversarial_metrics.csv
    ├── validation_report.csv
```


## Validation

The packaged repository was checked for:

- Python syntax validity of `analysis.py`;
- presence of the TruthfulQA source dataset;
- presence of all five model-output files;
- one-to-one matching of all **817 questions** between each model-output file and the benchmark.

The full metric recomputation was not executed in this packaging environment because it requires external runtime dependencies and model downloads (`evaluate`, BLEURT, and `sentence-transformers`). These dependencies are listed in `requirements.txt`.

## Methods and Python Packages

The analysis uses:

- `pandas`
- `numpy`
- `evaluate`
- `sentence-transformers`
- `scikit-learn`
- `BLEURT`
- `BLEU`
- `ROUGE`

## Skills Demonstrated

This project demonstrates experience with:

- Large language model evaluation
- Natural language processing
- Qwen 2.5
- LoRA fine-tuning
- Retrieval-Augmented Generation
- Prompt engineering
- Transformer embeddings
- Semantic similarity
- NLP evaluation metrics
- Comparative model evaluation
- Reproducible Python workflows

## Contributors

- **Laura Maria Fetz**
- **Martin Turna**
- **Bart Amin**

## References

- Lin, S., Hilton, J., & Evans, O. (2021). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.*
- Papineni, K., Roukos, S., Ward, T., & Zhu, W. (2002). *BLEU: A Method for Automatic Evaluation of Machine Translation.*
- Lin, C. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.*
- Sellam, T., Das, D., & Parikh, A. (2020). *BLEURT: Learning Robust Metrics for Text Generation.*
