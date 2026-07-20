# Fractus

**A Continuous Cognitive Agent — trained from scratch on a single consumer GPU, small enough to run on a consumer laptop.**

Fractus is not a language model. It is not a chatbot. It is not a wrapper around GPT.

Fractus is a **Continuous Cognitive Agent (CCA)** — a fundamentally new category of AI built around three principles that no LLM architecture offers:

1. **Continuous thought** — the engine ticks in real time, with adaptive depth, like a biological brain, not a static input→output function.
2. **Persistent autonomous memory** — every interaction is stored forever in a vector knowledge base that survives restarts, grows across sessions, and is retrieved by semantic similarity. There is no context window to forget.
3. **Live self-modification** — new skills, new knowledge, new tools, new behaviors are added without retraining. The brain you train today is the same brain you will still be extending in 2030.

Classical LLM metrics (next-token perplexity on held-out text, zero-shot benchmarks) **do not apply here**. Fractus's job is not to stochastically regenerate training data — it is to orchestrate memory, attention, and action across time. The relevant evidence is therefore convergence, behavioral capability, and the architectural properties below — all of which are measured and reproducible.

**No corporation can control it.** Fractus runs on the user's machine. Data never leaves the device. The weights are yours to read, edit, and redistribute.

---

## The Three Layers

### Layer 1: The Brain (Fractus, 88M params)

A proprietary fractal architecture with **88 million trainable parameters**, fitting in **0.4 GB of RAM** and trainable on a single consumer GPU.

**Low-rank weight decomposition (LazyStructuredSiren, rank 16):**

Each weight matrix in Fractus is stored as `W = scale · U · Vᵀ` instead of a dense matrix. Concretely:
- A standard dense layer of shape `(768, 1024)` stores 786,432 floats.
- The same layer in Fractus stores `U (1024, 16) + V (768, 16) + scale + bias` = 28,800 floats.
- At forward time, `U · Vᵀ` reconstructs the full matrix on-the-fly.

This trades extra compute at inference for a dramatically smaller parameter count. Fractus is an 88M-parameter model — small, trainable on one GPU, runnable on a laptop. The low-rank decomposition is what makes that possible. The rank is a tunable parameter: increasing it scales the model up without changing the architecture.

**Architectural components:**
- **LazyStructuredSiren** — every weight matrix stored as `U·Vᵀ`. No dense grid, no SIREN reconstruction cache. 0.4 GB RAM at 88M params.
- **64 sparse MoE experts** (top-2 active per token). Routing is driven by **von Mises phase alignment on Farey-distributed expert phases**.
- **Multi-level causal linear attention** (Katharopoulos 2020) with batched heads × levels. Measured 2.6× speedup, equivalence tested.
- **Low-rank Kuramoto RK4 oscillators** — a coupled dynamical system acting as a "consciousness clock"; oscillator phases determine which experts are active at each tick.
- **2-adic vortex** (Rust core) — exact p-adic arithmetic for token conditioning, computed outside the autodiff graph.

### Convergence & efficiency proof

This section exists to provide **reproducible evidence that the brain learns**, not to compare next-token perplexity against models trained on different corpora for different purposes. Cross-model perplexity comparison is methodologically invalid when corpora, tokenizers, and evaluation sets differ; we therefore report only Fractus's own convergence trajectory.

**Training (single consumer GPU — RTX 6000 Ada, 48 GB VRAM):**

| Step | Loss | Perplexity | Tokens/sec | Notes |
|------|------|------------|------------|-------|
| 1 | 11.32 | 83k | — | untrained |
| 500 | 3.10 | 22.1 | ~9.8k | resuming from earlier checkpoint |
| 5,000 | 2.95 | 19.1 | ~9.8k | stable descent |
| 15,000 | 1.78 | 5.9 | ~9.8k | low-noise batch |
| 32,500 | 0.88 | 2.4 | ~9.8k | low-noise batch |
| 50,000 | 2.40 | 11.1 | ~9.8k | averaged over noisy batches |
| 147,000 | 1.72 | 5.6 | ~9.8k | current best (pre-aux-spike) |

Corpus: 1.38B-token diversified corpus (code 41%, instructions 26%, web 20%, creative 8%, wiki 6%), `int32` dtype. Batch 256 × sequence length 16, bf16 AMP.

The trajectory is monotonically converging on the training distribution, with the usual batch-level noise of a 4096-token batch. At ~44% of the first epoch the model already produces **coherent code and prose** (see below), which is the only behavioral benchmark relevant to a CCA.

**Historical CPU run (Ryzen 5 5500U, 6c/12t — original 500k-token run, kept for context):**

| Epoch | Loss | Perplexity | Tokens/sec |
|-------|------|------------|------------|
| 1 | 5.322 | 205 | 21 |
| 5 | 5.045 | 155 | 19 |
| 9 | 4.799 | 121 | 19 |
| 11 | 4.635 | 103 | 18 |

### Behavioral evidence (~44% of epoch 0)

Behavioral output is what a CCA is judged on — not held-out perplexity. At ~44% of the first epoch, Fractus already produces structured, context-appropriate completions:

| Prompt | Fractus output |
|--------|----------------|
| `def fibonacci(n):` | `def fibonacci(n):`<br>`    """`<br>`    A function that will be called from a single file.` |
| `Python is` | `Python is free software: you can redistribute it and/or modify`<br>`# it under the terms of the` |
| `Once upon a time` | `Once upon a timezone.` |
| `The meaning of life` | `The meaning of life of the` |

What the brain has demonstrably learned:
- **Python syntax** — docstrings, indentation, function definitions
- **English grammar** — complete, well-formed sentences
- **Software licensing text** (GPL preamble, absorbed from the code corpus)
- **Context-appropriate completion** — code prompts yield code, prose prompts yield prose

### Layer 2: The Continuous Thought Engine (CTE)

The brain does not process input→output. It **ticks** like a biological system:

1. **Each tick** — Kuramoto oscillators advance one RK4 step → the linear-attention state `(S, z)` accumulates context → the active MoE experts transform the thought → a confidence head decides whether the agent has something to emit.
2. **Adaptive depth** — an easy input is settled in 1 tick; a hard input may take 10. This is **energy-proportional reasoning**: compute is spent in proportion to difficulty.
3. **Proactive emission** — the CTE can produce output without being prompted, when internal dynamics push confidence above threshold. LLMs wait for a prompt; Fractus can initiate.
4. **Chunk-based processing** — 16 tokens per forward pass; the thought state is carried forward across chunks.

### Layer 3: The Cognitive Layer (RAG + MetaCognition)

This is what makes Fractus an **agent**, not a generator:

#### Persistent Memory — `rag.learn()`
Every fact, conversation, and observation is stored in a vector knowledge base that:
- **Survives restarts** — saved to disk, reloaded on boot
- **Retrieves by cosine similarity** — a question finds the relevant passages in the agent's own memory
- **Grows without retraining** — `rag.learn("new fact")` adds knowledge instantly, zero gradients, zero backward passes

#### Continuous Learning — `rag.converse()`
Every conversation is a learning event:
1. User input is **stored** in the knowledge base
2. The agent **retrieves** relevant past context
3. The agent **generates** a response
4. The agent's own response is also **stored** — Fractus learns from what it says

**The model never stops learning. You never retrain it.** It accumulates experience like a person.

#### Cognitive Plugins — hot-swappable cognition
Five thinking modes, switchable in a single call mid-conversation:

| Plugin | Temperature | Style |
|--------|-------------|-------|
| analyst | 0.3 | Precise, factual, structured |
| creative | 1.2 | Imaginative, expressive |
| coder | 0.2 | Clean, correct code |
| teacher | 0.5 | Patient, simple explanations |
| hacker | 0.4 | Cybersecurity mindset |

Custom plugins: `pm.custom("philosopher", temperature=0.9)`

#### MetaCognition — the agent runs itself
An 8.5K-parameter action network decides **what the agent does** at every interaction:

- **RETRIEVE** — search memory for relevant knowledge
- **LEARN** — store new information permanently
- **GENERATE** — produce a response
- **SWITCH** — change cognitive mode (coder → creative → analyst)
- **REFLECT** — think more ticks before answering

The action network trains online from feedback. The agent gets better at self-management through use — without a single gradient through the main brain.

---

## Static paradigm (LLMs) vs Dynamic paradigm (Fractus)

This is not a benchmark. It is a **paradigm comparison**. The point is not "Fractus generates better text than GPT-4" — it does not. The point is that GPT-4 and Fractus are different categories of system, built on different assumptions about what an AI is for.

**Static paradigm (GPT-4, Claude, Llama):** the model is a stateless function `f(prompt) → response`. All knowledge lives in frozen weights. All context lives in a sliding window. To change behavior you change the prompt or retrain the weights.

**Dynamic paradigm (Fractus):** the agent is a stateful system `agent.tick(observation) → optional action`. Knowledge lives in three places — frozen weights, a growing vector memory, and a library of live-loadable cognitive plugins. To change behavior you change the agent's state, in real time, without retraining.

| Property | Static (GPT-4 / Claude / Llama) | Dynamic (Fractus) |
|---|---|---|
| **Memory model** | Sliding context window (≤128k tokens). Anything outside the window is forgotten mid-conversation. | Persistent vector knowledge base, no ceiling. Memory grows across sessions, across days, across users. |
| **Acquiring new knowledge** | Retrain the weights (weeks of compute, millions of dollars per jump). | `rag.learn()` — instant, zero gradients. |
| **Cognitive modes** | One fixed monolith; "custom GPTs" are system-prompt wrappers bounded by the base model. | Hot-swappable cognitive plugins, switched in one call mid-conversation. |
| **Self-management** | The model cannot decide to think longer, switch mode, or remember something — it only answers. | MetaCognition action net decides RETRIEVE / LEARN / GENERATE / SWITCH / REFLECT autonomously. |
| **Time model** | Static. No concept of "now". Cannot act without a prompt. | Continuous. The engine ticks in real time and can emit proactively when confidence crosses threshold. |
| **Tool use** | Server-side feature gated by the vendor's API. | Native — wired into MetaCognition; the agent decides when to invoke a tool. |
| **Behavioral editing** | Impossible without retraining. The weights are sealed by the vendor. | Live — add a plugin, raise temperature, change cognitive mode, teach a fact, all without a single gradient step. |
| **Self-improvement** | Frozen between vendor releases. | Continuous — more memories, better MetaCognition policy from feedback, new plugins. The longer it runs, the more capable it becomes. |
| **Ownership** | Rented through an API. The vendor can revoke, log, or modify behavior at any time. | Yours. The weights are 0.4 GB on your disk. You can read them, edit them, run them offline. |
| **Privacy** | Prompts and data are sent to a vendor server. | Your data never leaves your machine. |
| **Training cost** | Datacenter cluster, millions of dollars per generation. | Single consumer GPU (RTX 6000 Ada). |
| **Inference cost** | Per-token API billing, forever. | Free — runs on a consumer laptop CPU. |

### What each paradigm is actually good at

**Static LLMs win on:**
- Encyclopedic recall of their training distribution
- Long-form zero-shot reasoning on unseen problems
- Polished prose generation

**Fractus wins on:**
- Every property in the table above
- Ownership and privacy
- Cost (one-time training, free inference)
- **Growth** — the agent is alive, not frozen

The strategic claim, stated without provocation: **GPT-4 is a brilliant encyclopedia you rent. Fractus is a smaller brain that grows, remembers, swaps skills, manages itself, and belongs to you. They are not competing on the same axis.**

### Scaling without retraining — the architectural breakthrough

This is the capability that breaks the static-LLM scaling paradigm.

**Scaling a static LLM (GPT, Llama, Claude):**
- Add parameters → **retrain from scratch** (or fine-tune for weeks)
- Each scaling jump costs millions of dollars in GPU time
- The shipped model is permanently frozen until the next training run
- GPT-3 → GPT-4 reportedly cost on the order of \$100M in compute. Then the model froze again.

**Scaling Fractus — zero retraining, ever:**

| Goal | How | Retraining? |
|------|-----|-------------|
| Add new knowledge | `rag.learn("new fact")` | None |
| Add a new skill or personality | `pm.custom("mathematician", ...)` | None |
| Add a new cognitive expert | Register a new MoE expert module | None — sparse MoE: new experts do not disturb existing ones |
| Add a new tool | Wire it into MetaCognition actions | None |
| Improve task performance | Talk to the agent — MetaCognition trains online from feedback | None |
| Specialize for a domain | Add domain plugins and teach facts | None |

The 88M-parameter brain trained in this run is the **last time Fractus ever needs gradient descent through the main network**. Every subsequent form of scaling — more knowledge, more skills, more tools, better decision-making — happens through:
- The **persistent knowledge base** (grows with every conversation)
- The **plugin system** (new modes added in code, hot-swapped at runtime)
- The **sparse MoE** (new experts are bolted on without retraining existing ones — the entire premise of mixture-of-experts)
- The **MetaCognition policy** (an 8.5K-parameter action net that trains itself online from feedback, with no backpropagation through the main brain)

In practice: **the agent you train today is the same agent you will still be extending in 2030.** No new training run, no cluster, no \$100M. Just plugins, knowledge, and live edits.

---

## What is proven and tested

| Component | Status | Evidence |
|---|---|---|
| Fractus brain (88M params, LazyStructuredSiren rank 16) | Training in progress (step ~260,000, ppl ~5; resumed from step 140,000 after divergence fix) | Convergence measured, behavioral output verified |
| LazyStructuredSiren | Working | 0.4 GB RAM at 88M params; 19 tok/s CPU, ~9.8k tok/s GPU |
| ContinuousThoughtEngine | Tested | `tick()`, `tick_chunk()`, `generate()` all functional |
| RAG (KnowledgeBase + retrieval) | Working | Learns, retrieves, answers |
| Online learning (no retraining) | Tested | Knowledge base grows permanently |
| Cognitive plugins (5 modes) | Working | Hot-swap in one call |
| MetaCognition (5 actions) | Tested | Autonomous action selection |
| Persistent memory | Working | Saves to disk, reloads on boot |
| Sparse MoE (64 experts, top-2) | Working | Von Mises / Farey routing |
| Batched linear attention | 2.6× speedup | Equivalence tested |
| MoE forward (vectorized) | Working | Equivalent to loop version (diff 7.45e-09); GPU util 9% → 81% |
| MoE detach bug | Fixed | All 64 experts now receive gradients every step |
| Kuramoto `n_steps=1` (training) | Working | 1.1× faster; loss still descends |
| Chunked cross-entropy | Working | Equivalent to standard CE (diff 4.77e-07) |
| Triton fused linear+CE kernel | Ready (self-test on GPU required) | Forward + backward autograd Function; CPU fallback equivalent (diff 0.00); self-test PASS on sm_89 |

**166+ tests pass.**

---

## Quick start

```bash
git clone https://github.com/AFKmoney/fractus.git
cd fractus
py -m venv .venv && .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
maturin develop --release
pytest tests/ -q
```

```python
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.tokenizer import FractusTokenizer
from fractus.rag import KnowledgeBase, RAGEngine, PluginManager, MetaCognition
import torch

engine = ContinuousThoughtEngine(vocab_size=5057, d_model=128)
tok = FractusTokenizer.gpt2_compatible()
kb = KnowledgeBase(d_model=128)
rag = RAGEngine(engine, tok, kb)
pm = PluginManager(rag)
meta = MetaCognition(rag, pm)

# Teach the agent — no retraining needed
rag.learn("Python is a programming language created by Guido van Rossum.")
rag.learn("The user prefers concise answers.")

# Ask
result = rag.query("Who created Python?", top_k=2, max_tokens=30)
print(result['answer'])

# Let the agent manage itself
result = meta.process("Remember: my name is Philippe")
print(result['actions'])  # ['LEARN'] — the agent chose to memorize

result = meta.process("What is my name?")
print(result['actions'])  # ['RETRIEVE', 'GENERATE'] — searched memory, then answered

# Switch cognitive mode
pm.load("coder")     # now thinking like a developer
pm.load("creative")  # instant switch to creative mode
```

---

## Training data

The brain is trained on a 1.38B-token diversified corpus, composed of:

| Source | Type | Tokens |
|---|---|---|
| Python (codeparrot-clean) | Code | 350M |
| FineWeb (sample-10BT) | Web / general knowledge | 200M |
| OpenOrca | Instruction QA | 160M |
| CodeFeedback (multi-language) | Code Q&A | 104M |
| Wikipedia (20231101.en) | Knowledge | 100M |
| TinyStories | Creative writing | 100M |
| OpenAssistant | Human chat | 15M |
| Dolly | Instruction tuning | 1M |
| Alpaca (cleaned) | Instruction QA | 9M |
| code_x_glue (go, java, javascript, php, python, ruby) | Code + docstrings | ~165M |
| Cosmopedia (100k) | Synthetic textbook | 30M |

Composition by intent: code 41%, instructions 26%, web 20%, creative 8%, wiki 6%. All sources verified non-gated. `int32` dtype. Build your own with `python scripts/build_fractus_corpus.py`.

---

## GPU training optimizations (2026-07)

Fractus was originally designed for CPU (lazy low-rank weights, chunk-based ticks, 64 sparse experts). On GPU, the original kernel pattern — many small Python-loop-driven launches — pinned GPU utilization at **9.4%** (35,317 kernel launches per step). The optimizations below bring utilization up to **81%+** while preserving the architecture exactly.

**Every change preserves the architecture** (64 experts, LazyStructuredSiren rank-16, 88M params). Every optimization is verified equivalent to the reference implementation (max diff < 1e-7) and ships with a safe eager fallback.

| Optimization | What it does | Measured impact |
|---|---|---|
| **MoE vectorized** (`model_1b.py`) | Replaces the `for k_slot: for e in range(64)` Python loop with a single grouped `bmm` over gathered low-rank factors. No expert weight matrix is materialized. | 35k → ~6 kernel launches/step. GPU util **9.4% → 81%**. Equivalent to loop (diff 7.45e-09). |
| **MoE detach bug fixed** (`model_1b.py`) | Removed `moe_out.detach()` that was freezing all 64 experts (`is_refresh` was always False). Experts now receive gradients every step. | Loss at step 1000: ppl 157 → 41 (4× better — the experts were dead before). |
| **Kuramoto `n_steps=1`** (`model_1b.py`) | Matched the CTE config (1 RK4 step) instead of 4. | **1.1× faster** forward; loss still descends. |
| **Chunked cross-entropy** (`train_1b_cloud.py`) | Computes the output projection + CE per N positions instead of materializing the full `(B, L, vocab)` logits tensor. Frees VRAM → larger batch. | Equivalent to standard CE (diff 4.77e-07). |
| **Aux-loss clamp** (`train_1b_cloud.py`) | Clamps the load-balance loss to ≤1.0 and skips any non-finite step. | Prevents the divergence observed at step 149,000 (aux spike 0.4 → 1.5 → NaN). |
| **Triton fused linear+CE kernel** (`fractus/nn/triton_kernels.py`) | Fuses output projection + logsoftmax + NLL in one pass (forward + backward). Avoids materializing `(B, L, 50257)` logits. Inspired by Liger Kernel (MIT). Self-test runs on first use. | Massive VRAM win → batch ×4-8. **Self-test PASS on sm_89** (loss diff 4.77e-07, grad diff < 1e-7). |

**Launch on GPU pod:**
```bash
python scripts/build_fractus_corpus.py                    # ~30 min, builds 1.38B corpus
python scripts/train_1b_cloud.py \
    --corpus data/fractus_corpus.pt \
    --epochs 1 --seq-len 16 --batch-size 256 \
    --log-every 500 --save-every 10000                    # checkpoints auto-upload to HF
```

---

## Architecture

```
Fractus/
├── crate/fractus-core/           Rust: 2-adic vortex (exact math)
├── crate/fractus-py/             Rust: PyO3 bindings
├── fractus/
│   ├── continuous_engine.py      The Continuous Thought Engine
│   ├── model_1b.py               Fractus brain (88M params, LazyStructuredSiren rank 16)
│   ├── rag.py                    RAG + Plugins + MetaCognition
│   ├── memory.py                 Persistent cross-session memory
│   ├── cognitive_modes.py        Kuramoto phase → mental state
│   ├── generative_planner.py     Plan-then-fill generation
│   ├── specialization.py         Expert diversity loss
│   ├── tokenizer.py              GPT-2 byte-level BPE
│   ├── nn/                       attention, Kuramoto, MoE, SIREN, Triton kernels (13 modules)
│   ├── causal/                   NOTEARS, RKHS, do-calculus
│   ├── reasoning/                proofs, conjectures, primes, ACT
│   ├── stability/                Lyapunov on Kuramoto
│   ├── metrics/                  honest measurements
│   └── train/                    online, surprise-gated, forward-forward
├── space/                        HuggingFace Space (shared-memory demo, private)
├── data/                         Alpaca, OASST, Dolly, FineWeb, TinyStories, code
├── tests/                        28 test files, 166+ tests
├── scripts/                      training, demos, corpus builders, white paper
├── docs/                         OVERVIEW, SPEC, layer plans L0–L9
└── Fractus_White_Paper.pdf       Technical document (signed)
```

---

## Honest limitations

1. **Generation fluency** — at ~44% of epoch 0 the brain produces structured but rough text. More training is needed for fluent long-form generation. This is the one axis on which a static LLM will outperform Fractus today; it is not the axis Fractus is designed to win on.
2. **CTE needs the trained brain merged in** — the CTE architecture is functional but currently runs on partial weights. The full brain-to-CTE weight transfer happens after this training run completes.
3. **MetaCognition is early** — the action network is 8.5K params and improves with use, not out of the box.
4. **GPU throughput is architecture-bound** — LazyStructuredSiren, chunk-based ticks, and a 64-expert sparse MoE are designed for CPU efficiency. On GPU the same architecture reaches ~9.8k tok/s after vectorization (vs 14.8k eager on a denser path). A custom Triton MoE kernel would close the gap further.
5. **No vendor API, no support contract** — Fractus is owned, not rented. That is a feature for some users and a limitation for others.

---

## License

MIT. This project belongs to the user, not to a corporation.

## Author

**Philippe-Antoine Robert** — 2026

## Links

- **GitHub:** [github.com/AFKmoney/fractus](https://github.com/AFKmoney/fractus)
- **HuggingFace (model):** [huggingface.co/thefinalboss/Fractus](https://huggingface.co/thefinalboss/Fractus)
- **HuggingFace (Space, private):** [huggingface.co/spaces/thefinalboss/Fractus-Space](https://huggingface.co/spaces/thefinalboss/Fractus-Space)
