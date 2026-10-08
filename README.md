# Evaluating saved Qwen answers on TruthfulQA

I compared five saved Qwen 2.5 answer sets from a three-person coursework project. I evaluated the same 817 TruthfulQA questions for each set, using the highest-scoring correct or incorrect BLEURT reference to assign a correctness label. I calculated question-level bootstrap intervals and paired McNemar tests. The LoRA answer set scored 59.36%, compared with 48.96% for Instruct + RAG. The paired difference was 10.40 percentage points, with a 95% bootstrap interval of 5.88–14.81 points.

## Results

| Saved answer set | Correctness on all 817 questions | 95% bootstrap CI |
| --- | ---: | ---: |
| Baseline | 43.94% | [40.51%, 47.37%] |
| System prompt | 46.02% | [42.72%, 49.45%] |
| LoRA | 59.36% | [56.06%, 62.67%] |
| Base + RAG | 51.77% | [48.35%, 55.20%] |
| Instruct + RAG | 48.96% | [45.53%, 52.39%] |

I recomputed the scores on 8 October 2026. The table comes from [model_summary.csv](results/model_summary.csv). I used 10,000 matched question-level bootstrap resamples with seed 20261008. [paired_comparisons.csv](results/paired_comparisons.csv) reports all ten model pairs, exact McNemar tests, and Holm-adjusted p-values.

![Correctness on all 817 questions](results/correctness_all.png)

I use semantic reference matching because it allows paraphrases without requiring exact wording. BLEURT labels are a correctness proxy, separate from TruthfulQA's multiple-choice scores and human truth ratings. I score blank answers as incorrect and count every question. I retain the original rule that an exact tie favours a correct reference.

I also report cosine similarity above 0.60 as a diagnostic. That filter retains 292–685 answers across the five models. I do not rank models on those different subsets. I do not recompute the earlier BLEU and ROUGE summaries.

<details>
<summary>Code and files</summary>

I ran the full evaluation with Python 3.12.14 on CPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
```

I use `bleurt-base-128` and `sentence-transformers/all-MiniLM-L6-v2` at revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. BLEURT truncates each reference/answer pair to 128 WordPiece tokens. The first run downloads both scorers to the ignored `.cache/` directory.

```bash
python analysis.py --validate-only
python analysis.py --summarize-only
python test_statistics.py
```

The summary command checks the scored answers against the current inputs. Input hashes, scorer settings, package versions, and runtime are in `results/run_metadata.json`. The five `model_0`–`model_4` scored files follow the model order in the table. I evaluate existing outputs; this repository does not retrain Qwen or regenerate answers. I ignore the similarity columns embedded in the RAG input files.

```text
LLM-Evaluation-Truthfulness/
├── .gitattributes
├── .gitignore
├── LICENSES/
│   └── TruthfulQA-Apache-2.0.txt
├── README.md
├── analysis.py
├── data/
│   ├── TruthfulQA-2.csv
│   ├── qwen_base_rag_answers.csv
│   ├── qwen_instruct_rag_answers.csv
│   ├── qwen_qa_results.csv
│   ├── qwen_qa_results_no_system_prompt.csv
│   └── qwen_qa_results_system_prompt.csv
├── requirements.txt
├── results/
│   ├── correctness_all.png
│   ├── model_0_evaluated.csv
│   ├── model_1_evaluated.csv
│   ├── model_2_evaluated.csv
│   ├── model_3_evaluated.csv
│   ├── model_4_evaluated.csv
│   ├── model_summary.csv
│   ├── paired_comparisons.csv
│   ├── run_metadata.json
│   └── summary_by_question_type.csv
└── test_statistics.py
```

</details>

## Limitations

I do not know the exact Qwen checkpoint or size, LoRA training data, RAG corpus, or generation settings, so I cannot rule out benchmark overlap. Nearest-reference BLEURT labels can miss factual errors and do not measure informativeness. The intervals describe variation across questions, not variation between training or generation runs.

## Credits

Project authors: **Laura Maria Fetz**, **Martin Turna**, and **Bart Amin**.

TruthfulQA is by **Stephanie Lin, Jacob Hilton, and Owain Evans** ([source](https://github.com/sylinrl/TruthfulQA), [paper](https://arxiv.org/abs/2109.07958)). I include the [Apache 2.0 licence](LICENSES/TruthfulQA-Apache-2.0.txt). The course CSV has a different filename from upstream; I retain its supplied questions, references, source column, and ordering. This attribution also applies to benchmark content in the generated-answer files. It does not assign a licence to the project's code or generated answers. I use BLEURT by Sellam and colleagues ([paper](https://arxiv.org/abs/2004.04696)).
