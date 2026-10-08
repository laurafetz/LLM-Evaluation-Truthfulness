# Model provenance and interpretation

This repository evaluates five existing answer files. It does not contain the generation, training, or retrieval pipeline.

| Item | Status |
| --- | --- |
| Model family | Qwen 2.5, according to the project documentation |
| Exact checkpoint, parameter count, and revision | Unknown |
| LoRA training dataset, splits, and hyperparameters | Unknown |
| LoRA training overlap with TruthfulQA | Not assessed; contamination cannot be ruled out |
| RAG corpus, document versions, and retrieval configuration | Unknown |
| Whether benchmark questions or reference answers were available to retrieval | Unknown |
| Full generation prompts and settings | Not included |

Consequently, differences between the answer files are descriptive. They do not demonstrate a leakage-free improvement caused by fine-tuning or RAG. The embedded similarity columns in two RAG files are not used in this evaluation.

The original project credits **Laura Maria Fetz, Martin Turna, and Bart Amin**. Individual task responsibilities are not documented, so the repository does not attribute model training or retrieval implementation to a particular contributor. The current repository supplies a revised evaluation workflow and its audit outputs.

The revision uses `bleurt-base-128`, matching the default checkpoint of the original `evaluate.load("bleurt")` call, and pinned `all-MiniLM-L6-v2` embeddings. Checkpoint settings of the historical notebook metrics are not documented. The current run therefore replaces historical headline figures with fully specified rerun figures, rather than claiming an exact reproduction of unspecified scoring settings.
