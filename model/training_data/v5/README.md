# JARVIS V5 training data

`train.jsonl` is the primary V5 dataset and the only V5 candidate training input. Preserve it as the existing dataset; optimize its examples in place and validate it before any training run.

`v5_additions.jsonl` is retained as source material and is excluded from the candidate training set. The 59-row repository-root `v5_additions.jsonl` file is archival source material and is not consumed by the candidate training set. Comparison against the primary dataset found 37 exact duplicate no-tool records and 6 additional records equivalent after normalizing call IDs/types. The remaining 16 additions were reviewed but not incorporated because they do not justify changing the curated primary dataset; Addition 25 is already represented by primary example 172. The source file remains recoverable and unchanged.

The canonical tool schemas are defined in `tools/schemas.py`. The training helper re-exports them from `model/training_data/mlx_tools.py`, and the evaluator consumes the same schemas. Runtime `list_files()` has no arguments, `search_files(query)` accepts only `query`, and `recall_memory(key="")` accepts an optional key. There is no runtime `write_file` capability.

The current V5 dataset has valid IDs for recorded tool results. `model/train_v4.py` still reads the historical V3 dataset; it is not a V5 trainer. No V5 training was started as part of this work.

The current Python execution adapter fails closed because OS-level filesystem/network isolation and resource controls are not configured. Treat sandboxed Python execution as unavailable until a compliant host adapter is configured. `write_file`, package installation, and deletion are not runtime capabilities. `web_search` is exposed through a host-provider interface, requires Security Gate approval, and is unavailable until a provider is configured; returned web content is untrusted.
