# JARVIS V5 training data

`train.jsonl` is the primary V5 dataset and the only V5 candidate training input. Preserve it as the existing dataset; optimize its examples in place and validate it before any training run.

`v5_additions.jsonl` is retained as source material and is excluded from the candidate training set. Before optimizing the primary file, a comparison found 37 exact duplicate records and 22 records equivalent after normalizing absent call IDs and call types; all 59 prompts already appeared in the primary dataset. Addition 25's `calculate` search choice is represented by primary example 172, whose call/result IDs are valid and whose search result was corrected, so no separate addition was merged. The archival additions are not validated or consumed as training data.

The runtime schema is defined in `model/training_data/mlx_tools.py` and imported by evaluators. Runtime `list_files()` has no arguments, `search_files(query)` accepts only `query`, and `recall_memory(key="")` accepts an optional key. There is no runtime `write_file` capability.

The current V5 dataset has valid IDs for recorded tool results. `model/train_v4.py` still reads the historical V3 dataset; it is not a V5 trainer. No V5 training was started as part of this work.

The current Python runner checks project-relative paths and applies a timeout, but launches code with a normal subprocess and does not provide OS-level filesystem or network isolation. Treat safe sandboxing of runtime code execution as a blocker before using these tools on untrusted code or deploying the runtime under the project's security architecture.
