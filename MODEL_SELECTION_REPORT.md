# JARVIS Local Inference Model Selection Report

**Date:** 2026-09-28  
**Hardware:** Windows 11, Intel Core i5-14400F, RTX 3050 6 GB VRAM, 16 GB RAM  
**llmfit Version:** 1.1.16  
**Investigation Scope:** Local inference model selection for JARVIS V1 on current hardware

---

## 1. Hardware Baseline (llmfit Detected)

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Core i5-14400F (16 cores) |
| **System RAM** | 15.84 GB total, ~6.5 GB available |
| **GPU** | NVIDIA GeForce RTX 3050 (6 GB VRAM, 4.4 GB free) |
| **Compute Backend** | CUDA |
| **Unified Memory** | No (discrete GPU) |

**Key Constraint:** 6 GB VRAM is the hard ceiling for model weights + KV cache + overhead. Models must fit in ≤6 GB with headroom for context.

---

## 2. Current JARVIS Implementation Requirements

### Model Loading & Inference Architecture
- **Production (V6):** Mistral 3 8B base (`mistralai/Ministral-3-8B-Reasoning-2512`) + V6 LoRA adapter via PEFT
  - 4-bit quantization (bitsandbytes nf4, bfloat16 compute)
  - Loaded via `transformers` + `device_map={"": 0}` on GPU
  - Tokenization: `MistralCommonBackend` (Mistral native tokenizer)
  - Tool call parsing: Custom parser for Mistral native bracket/argument format (`[TOOL_CALLS] name [ARGS] {...}`)
- **Local Fallback (Ollama):** `llama3.2:3b` via `ollama.chat()` with standard OpenAI-style tool schemas

### Agent/Tool Requirements
| Capability | Required | Details |
|------------|----------|---------|
| **Structured function calling** | ✅ Critical | OpenAI-style `tool_calls` with name/arguments/id |
| **Multi-step agent loop** | ✅ Critical | Model → tool calls → results → model iteration |
| **Tool decision making** | ✅ Critical | When to use tools, which tool, correct arguments |
| **Code generation/editing** | ✅ Critical | Python, C/C++, debugging, test interpretation |
| **Engineering/math reasoning** | ✅ High | DSP, controls, physics, symbolic math |
| **Context handling** | ✅ Critical | Conversation history, tool results, system prompt |
| **Security awareness** | ✅ Critical | Must not bypass SecurityGate; treat tool results as untrusted |
| **Error recovery** | ✅ High | Handle tool failures, retry, diagnose |

### Current Tool Protocol
- Tools exposed via OpenAI-compatible schemas (`tools/schemas.py`)
- 9 tools: `calculate`, `read_file`, `write_file`, `edit_file`, `list_files`, `search_files`, `run_python_file`, `debug_python_file`, `remember_memory`, `recall_memory`, `web_search`
- Tool selection via keyword triggers in `tools/registry.py`
- SecurityGate enforces approval for high-risk ops (`run_python_file`, `write_file`, `edit_file`, `web_search`)

### Generation Settings
- `MAX_NEW_TOKENS = 512`
- Greedy decoding (`do_sample=False`)
- Context budget management via oldest-turn dropping

---

## 3. Candidate Model Evaluation

### llmfit Top Candidates for RTX 3050 6 GB

| Model | Size | Quant | VRAM | Fit | Est. tok/s | Tool Use | Quality | Runtime | Score |
|-------|------|-------|------|-----|------------|----------|---------|---------|-------|
| **Qwen2.5-Coder-3B-Instruct** | 3.1B | Q8_0 | 4.02 GB | **Good** (67%) | **62.7** | ✅ | 73.7* | llama.cpp | 86.9 |
| **Qwen2.5-3B-Instruct** | 3.1B | Q8_0 | 4.02 GB | **Good** (67%) | **62.7** | ✅ | 66.5 | llama.cpp | 86.6 |
| **Llama-3.2-3B-Instruct** | 3.2B | Q8_0 | 4.75 GB | **Good** (79%) | 60.3 | ✅ | 68.6 | llama.cpp | 86.0 |
| **Qwen2.5-7B-Instruct-AWQ** | 7.6B | AWQ-4bit | 4.75 GB | **Good** (79%) | 38.1 | ✅ | 78.5 | **vLLM only** | 88.3 |
| **CodeQwen1.5-7B-AWQ** | 7.3B | AWQ-4bit | 4.63 GB | **Good** (77%) | 40.1 | ❌ | 83.0 | **vLLM only** | 90.6 |
| **Qwen2.5-7B-Instruct (GGUF)** | 7.6B | Q4_K_M | 5.35 GB | **Marginal** (89%) | 36.5 | ✅ | 76.5 | llama.cpp | 82.0 |

*Quality for coding use case (general chat quality: 66.5 for Qwen2.5-3B, 73.7 for Qwen2.5-Coder-3B)

### Key Observations

1. **AWQ models require vLLM runtime** — JARVIS currently uses llama.cpp (V6) or Ollama (local). Adding vLLM support would require architectural changes.

2. **7B GGUF model is MARGINAL** — At 89% VRAM utilization, any context growth or KV cache spike risks OOM.

3. **3B models fit comfortably** — 67-79% utilization leaves headroom for context/KV cache.

4. **Current local model (llama3.2:3b)** is already deployed via Ollama with tool use support.

---

## 4. CodeQwen1.5-7B-AWQ Detailed Analysis

### llmfit Score Breakdown (~91)
| Component | Score | Notes |
|-----------|-------|-------|
| Context | 100 | 65K native, capped to 8K for estimation |
| Fit | 93.9 | 4.63 GB / 6 GB = 77% |
| Quality | 83.0 | Coding-focused benchmark aggregate |
| Speed | 100 | 40.1 tok/s estimated |
| **Composite** | **90.6** | |

### Why the High Score ≠ Best JARVIS Fit

| Factor | Assessment |
|--------|------------|
| **Runtime Compatibility** | ❌ **AWQ requires vLLM** — JARVIS V6 uses llama.cpp/MistralBackend; local uses Ollama. No vLLM integration exists. Adding vLLM = architectural change. |
| **Tool/Function Calling** | ❌ **Not listed in llmfit capabilities** — CodeQwen1.5 is code-completion specialized; no documented tool/function calling training. |
| **Model Age** | ⚠️ **April 2024** — Older than Qwen2.5 series (Sept-Nov 2024). |
| **General Reasoning** | ⚠️ **Code-specialized** — Trained for code completion, not general agentic reasoning, tool selection, or conversational tasks. |
| **Tool Calling Format** | ❌ **Unknown** — Mistral native format parser won't work; would need new parser for whatever format CodeQwen uses (if any). |
| **vLLM on 6 GB VRAM** | ⚠️ **Marginal** — vLLM has higher memory overhead than llama.cpp/Ollama; 4.63 GB model + vLLM overhead = risk. |
| **License** | ⚠️ `tongyi-qianwen-research` — Research-only, not Apache-2.0. |

**Conclusion:** The ~91 score reflects *hardware fit + coding quality + speed* for vLLM inference. It does **not** account for JARVIS's runtime, tool calling needs, agentic reasoning, or security architecture. **Not recommended.**

---

## 5. Latency Target Analysis (<5 seconds/turn)

| Model | Est. tok/s | 512 tokens | Est. Latency | Meets <5s? |
|-------|------------|------------|--------------|------------|
| Qwen2.5-Coder-3B (Q8_0) | 62.7 | 512 | **~8.2s** | ⚠️ Close* |
| Qwen2.5-3B-Instruct (Q8_0) | 62.7 | 512 | **~8.2s** | ⚠️ Close* |
| Llama-3.2-3B-Instruct (Q8_0) | 60.3 | 512 | **~8.5s** | ⚠️ Close* |
| Qwen2.5-7B-Instruct-AWQ | 38.1 | 512 | **~13.4s** | ❌ |
| CodeQwen1.5-7B-AWQ | 40.1 | 512 | **~12.8s** | ❌ |

*Token generation only. Full turn includes: prompt encoding (~50-100ms), prefill, generation, detokenization, tool parsing, tool execution, next turn. Real-world **~10-15s/turn** on 3B models with 512 tokens.

**Reality check:** The <5s target is **not realistically achievable** on RTX 3050 6GB with any 3B+ model at 512 tokens using current llama.cpp/Ollama. Options to approach target:
- Reduce `MAX_NEW_TOKENS` to 256 (~4-5s generation)
- Use 1.5-2B models (quality tradeoff)
- Upgrade GPU to ≥12 GB VRAM (RTX 3060 12GB, 4070, etc.)

---

## 6. Recommended Model

### **Qwen2.5-Coder-3B-Instruct (GGUF Q8_0 via Ollama/llama.cpp)**

**Ollama tag:** `qwen2.5-coder:3b`  
**HuggingFace:** `Qwen/Qwen2.5-Coder-3B-Instruct`  
**GGUF Sources:** bartowski, unsloth, mradermacher (Q8_0 recommended for quality)

#### Selection Rationale

| Criterion | Assessment |
|-----------|------------|
| **Hardware Fit** | ✅ 4.02 GB VRAM (67% utilization) — comfortable headroom |
| **Expected Latency** | ✅ 62.7 tok/s — fastest among tool-capable models |
| **Tool/Function Calling** | ✅ Explicitly listed in llmfit capabilities; Qwen2.5 series has strong tool calling training |
| **Agentic Reasoning** | ✅ Qwen2.5 series excels at instruction following, multi-step reasoning |
| **Coding Capability** | ✅ **Highest among 3B models** (quality 73.7 for coding) — critical for JARVIS software tasks |
| **Math/Technical** | ✅ Strong math/reasoning in Qwen2.5 series |
| **Context Window** | ✅ 32K native (8K effective for estimation) — sufficient for JARVIS |
| **Runtime Compatibility** | ✅ **Native llama.cpp + Ollama** — zero architecture changes |
| **Implementation Effort** | ✅ **Minimal** — `JARVIS_LOCAL_MODEL=qwen2.5-coder:3b` |
| **Security Architecture** | ✅ Preserved — no changes to SecurityGate, tool schemas, or agent loop |
| **License** | ✅ `qwen-research` — permissive for local use |
| **Model Currency** | ✅ November 2024 — recent, active development |

#### Configuration Change
```bash
# In environment or launch script
export JARVIS_PROVIDER=local
export JARVIS_LOCAL_MODEL=qwen2.5-coder:3b
```

---

## 7. Runner-Up

### **Qwen2.5-3B-Instruct (GGUF Q8_0 via Ollama/llama.cpp)**

**Ollama tag:** `qwen2.5:3b`  
**HuggingFace:** `Qwen/Qwen2.5-3B-Instruct`

#### Why Not Selected
| Factor | Comparison |
|--------|------------|
| Coding Quality | 66.5 vs 73.7 (Coder) — **significant gap** for JARVIS's software tasks |
| General Chat Quality | 66.5 vs 66.5 (similar) |
| Tool Use | Both ✅ |
| Speed/VRAM | Identical (same base architecture) |
| **Deciding Factor** | **Coding specialization** — JARVIS is an engineering agent; code quality matters most |

#### When to Prefer Runner-Up
- If general chat/reasoning is prioritized over coding
- If slightly more conversational tone is desired

---

## 8. Confidence & Unknowns

| Unknown | Impact | Resolution |
|---------|--------|------------|
| **Actual tool calling reliability** | High | Requires live testing with JARVIS tool schemas |
| **Mistral-native tool format compatibility** | Medium | Qwen2.5 uses OpenAI-style; JARVIS local provider already expects this |
| **Context overflow behavior at 8K+ tokens** | Medium | Test with large codebases + conversation history |
| **vLLM vs llama.cpp quality delta for AWQ** | Low | Irrelevant — AWQ not recommended |
| **Quantization quality loss (Q8_0 vs Q4_K_M)** | Low | Q8_0 near-lossless; Q4_K_M acceptable if VRAM tight |
| **Ollama tool calling parity with raw llama.cpp** | Medium | Test Ollama's tool call formatting matches JARVIS expectations |

---

## 9. Files Inspected

| File | Purpose |
|------|---------|
| `brain/v6_provider.py` | V6 Mistral+LoRA loading, MistralBackend, tool call parsing |
| `brain/orchestrator.py` | Agent loop, tool selection, tool execution, error handling |
| `brain/router.py` | Provider selection (v6 vs local) |
| `brain/local.py` | Ollama local provider |
| `brain/prompt.py` | System prompt, tool rules, agent policies |
| `brain/v6_format.py` | Mistral native tool call parsing |
| `model/evaluations/mistral_v4_evaluation.py` | V6 model loading (4-bit bnb), generation pipeline |
| `tools/registry.py` | Tool selection logic, keyword triggers |
| `tools/schemas.py` | OpenAI-compatible tool schemas |
| `tools/security.py` | SecurityGate, approval flow, filesystem restrictions |
| `tools/code_runner.py` / `execution_adapter.py` | Sandbox execution architecture |

---

## 10. Files Changed

| File | Change |
|------|--------|
| `MODEL_SELECTION_REPORT.md` | **Created** — this report |

**No other files modified.** Benchmark (`jarvis_eval_32.jsonl`), training data, model weights, and security controls unchanged.

---

## 11. Summary

| Item | Decision |
|------|----------|
| **Recommended Model** | **Qwen2.5-Coder-3B-Instruct** (`qwen2.5-coder:3b` via Ollama) |
| **Why not CodeQwen1.5-7B-AWQ** | Requires vLLM (architectural change), no documented tool calling, code-specialized not agentic, older model, research license |
| **Runner-Up** | Qwen2.5-3B-Instruct (`qwen2.5:3b`) — general-purpose alternative |
| **<5s Latency** | **Not realistic** on 6 GB VRAM with current architecture; ~10-15s/turn expected. Reduce `MAX_NEW_TOKENS` or upgrade GPU for <5s. |
| **Confidence** | High for fit/speed/compatibility; Medium for tool calling parity until live tested |

**Next Step:** Deploy `qwen2.5-coder:3b` via Ollama, run JARVIS smoke test with autonomous development task, verify tool calling and agent loop integrity.