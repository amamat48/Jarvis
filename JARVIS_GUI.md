# JARVIS task and GUI prototype

## Integration boundary

`gui/server.py` serves the local prototype and calls the GUI-facing methods in `gui/application.py`. The application layer submits and controls tasks through `tasks.manager.TaskManager`; it never dispatches a tool by name or reaches into model internals. Real inference is loaded lazily through `brain.orchestrator.execute_task_turn`. The existing console remains available through `main.py`.

The GUI server binds only to loopback (`127.0.0.1` by default), validates local POST origins, caps request bodies, and uses task-view projections that omit model message history, tool arguments, authorization tokens, and hidden reasoning. Use `python -m gui` to launch it.

## Modes

The default Demo mode uses deterministic task work and the same TaskManager/SecurityGate path. It supports progress, tool events through an authorized calculator operation, input waiting, approval waiting, pause/resume, cancellation, failure, and concurrent side questions without model loading. The demo approval example is a gated `web_search`; with the default unconfigured provider it returns no fabricated result.

Real runtime mode lazy-loads the current local Ollama backend. It may fail if the local provider or model is unavailable. Independent conversations can run concurrently (bounded to two workers); tasks in the same conversation remain ordered. A message submitted while that conversation has unfinished work is placed in a fresh conversation so a side question cannot alter the running task's prompt history.

## Web search and security

The canonical `web_search(query)` runtime schema is in `tools/schemas.py`. `tools/web_search.py` has no network implementation or credentials by default; a host may configure a provider explicitly. Each web search is marked as network access and requires human approval through the existing Security Gate. Provider results are labeled untrusted and only HTTPS links are presented. No returned content can override the JARVIS system policy.

Python execution remains fail-closed until an OS-enforced isolation adapter is configured. Package installation, file writing, and deletion are not available tools. Secret/credential paths remain denied.

## V5 data and evaluation

`model/training_data/v5/train.jsonl` remains the primary V5 dataset. `v5_additions.jsonl` is archival source material and is not merged wholesale. The fixed `model/evaluations/jarvis_eval_32.jsonl` benchmark remains immutable. Run `python -m unittest discover -s tests` for repository tests; this does not train models or run the fixed benchmark.

## V6 preparation

`model/train_v6.py` points to `model/training_data/v6/train.jsonl`, saves only its final adapter under `model/evaluations/jarvis_lora_v6/final`, and defaults to 240 iterations. Training was not started as part of this work.
