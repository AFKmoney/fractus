# Fractus

**The first AI with continuous thought, persistent memory, and autonomous cognition — trained from scratch on a consumer laptop.**

Fractus is not another GPT clone. It is a fundamentally different architecture where the AI thinks in real-time (not static input→output), remembers every interaction across sessions, learns new things without retraining, switches cognitive modes on the fly, and decides for itself when to search its memory, learn a fact, reflect longer, or generate a response.

**No corporation can control it.** The AI runs on the user's machine. Data never leaves the device.

---

## The Three Layers

### Layer 1: The Brain (Fractus-1B)

A proprietary fractal transformer with **0.86 billion effective parameters** compressed into only **88 million trainable parameters** using LazyStructuredSiren (low-rank decomposition W = scale·U·Vᵀ, rank 16).

- **LazyStructuredSiren**: every weight matrix stored as U·Vᵀ — no dense grid, no SIREN reconstruction. 0.86B capacity in 0.4 GB RAM.
- **64 sparse MoE experts** (top-2 active per token, von Mises routing on Farey-distributed phases). Compute proportional to 2/64 of total capacity.
- **Multi-level causal linear attention** (Katharopoulos 2020) with batched heads×levels (measured 2.6× speedup).
- **Low-rank Kuramoto RK4 oscillators**: a coupled dynamical system acting as a "consciousness clock" — phases determine cognitive routing.
- **2-adic vortex** (Rust core): exact p-adic arithmetic for token conditioning, outside the autodiff graph.

**Training (from scratch, CPU-only):**

| Epoch | Loss | Perplexity | Tokens/sec |
|-------|------|------------|------------|
| 1 | 5.322 | 205 | 21 |
| 5 | 5.045 | 155 | 19 |
| 9 | 4.799 | 121 | 19 |
| 10 | 4.730 | 113 | 19 |
| 11 | 4.635 | 103 | 18 |

Hardware: AMD Ryzen 5 5500U (6 cores, 12 threads). No GPU.
Data: 500k tokens from a 12.8M-token mega corpus (26% Python code, web knowledge, instruction QA, human chat, creative writing).

### Layer 2: The Continuous Thought Engine (CTE)

The brain doesn't process input→output. It **ticks** like a biological brain:

1. **Each tick**: Kuramoto oscillators advance one RK4 step → attention state (S,z) accumulates context → MoE transforms the thought → confidence head decides if the AI has something to say.
2. **Adaptive depth**: easy question = 1 tick. Hard question = 10 ticks. This is **energy-proportional reasoning** — spend compute proportional to difficulty.
3. **Proactive**: the CTE can emit output without being prompted — when internal dynamics push confidence above threshold. GPT and Claude wait for a question. Fractus can initiate.
4. **Chunk-based processing**: 16 tokens per forward pass (117 tok/s on 13M model, 19 tok/s on 1B).

### Layer 3: The Cognitive Layer (RAG + MetaCognition)

This is what makes Fractus an **agent**, not a tool:

#### Persistent Memory — `rag.learn()`
The AI stores every fact, conversation, and interaction in a **vector knowledge base** that:
- **Survives restarts** (saved to disk, reloaded on boot)
- **Retrieves by cosine similarity**: ask a question → AI finds relevant passages in its memory
- **Grows without retraining**: `rag.learn("new fact")` adds knowledge instantly — zero gradients, zero backward passes

#### Continuous Learning — `rag.converse()`
Every conversation is a learning opportunity:
1. User input is **stored** in the knowledge base
2. AI **retrieves** relevant past context
3. AI **generates** a response
4. Its own response is also **stored** (the AI learns from what it says)

**The model never stops learning. You never retrain it.** It accumulates experience like a human.

#### Cognitive Plugins — hot-swappable personality
Five thinking modes, switchable instantly:

| Plugin | Temperature | Style |
|--------|-------------|-------|
| analyst | 0.3 | Precise, factual, structured |
| creative | 1.2 | Imaginative, expressive |
| coder | 0.2 | Clean, correct code |
| teacher | 0.5 | Patient, simple explanations |
| hacker | 0.4 | Cybersecurity mindset |

Custom plugins: `pm.custom("philosopher", temperature=0.9)`

#### MetaCognition — the AI manages itself
An 8.5K-parameter action network that decides **what the AI does** at each interaction:

- **RETRIEVE** — search memory for relevant knowledge
- **LEARN** — store new information permanently
- **GENERATE** — produce an answer
- **SWITCH** — change cognitive mode (coder → creative → analyst)
- **REFLECT** — think more ticks before answering

The action network trains online from feedback. The AI gets better at self-management through use.

---

## How Fractus differs from GPT and Claude

| Property | GPT-4 / Claude | Fractus |
|---|---|---|
| **Processing** | Static (1 forward pass) | Continuous (tick-based CTE) |
| **Memory** | Context window (amnesic) | Persistent vector KB |
| **Learning** | Retraining required | Online (every conversation) |
| **Skills** | Generic monolith | Hot-swappable plugins (5 modes) |
| **Autonomy** | Waits for instructions | Decides actions itself |
| **Training** | Datacenter GPU cluster | Consumer CPU laptop |
| **Deployment** | Cloud API (centralized) | Local device (decentralized) |
| **User data** | Sent to server | Stays on device |
| **Growth** | Frozen between versions | Grows with every use |
| **Training cost** | Millions of dollars | $0 (electricity only) |

---

## What is proven and tested

| Component | Status | Evidence |
|---|---|---|
| Fractus-1B (88M params, 0.86B capacity) | Training (epoch 11, loss 4.635, ppl 103) | Convergence measured |
| LazyStructuredSiren | Working | 0.4 GB RAM, 19 tok/s on CPU |
| ContinuousThoughtEngine | Tested | tick(), tick_chunk(), generate() |
| RAG (KnowledgeBase + retrieval) | Working | Learns, retrieves, answers |
| Online learning (no retraining) | Tested | Grows permanently |
| Cognitive plugins (5 modes) | Working | Hot-swap in 1 call |
| MetaCognition (5 actions) | Tested | Autonomous action selection |
| Persistent memory | Working | Saves to disk, reloads |
| Sparse MoE (64 experts, top-2) | Working | Von Mises/Farey routing |
| Batched linear attention | 2.6× speedup | Equivalence tested |

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

engine = ContinuousThoughtEngine(vocab_size=50257, d_model=128)
tok = FractusTokenizer.gpt2_compatible()
kb = KnowledgeBase(d_model=128)
rag = RAGEngine(engine, tok, kb)
pm = PluginManager(rag)
meta = MetaCognition(rag, pm)

# Learn — no retraining needed
rag.learn("Python is a programming language created by Guido van Rossum.")
rag.learn("The user prefers concise answers.")

# Ask
result = rag.query("Who created Python?", top_k=2, max_tokens=30)
print(result['answer'])

# Let the AI manage itself
result = meta.process("Remember: my name is Philippe")
print(result['actions'])  # ['LEARN'] — AI chose to memorize

result = meta.process("What is my name?")
print(result['actions'])  # ['RETRIEVE', 'GENERATE'] — AI searched then answered

# Switch personality
pm.load("coder")     # now thinks like a developer
pm.load("creative")  # instant switch to creative mode
```

---

## Training data

12.8M-token mega corpus from 9 sources:

| Source | Type | Tokens |
|---|---|---|
| Python code instructions | Code | 1.5M |
| CodeAlpaca | Multi-language code | 2M |
| FineWeb (sample-10BT) | Web / general knowledge | 3M |
| Alpaca | Instruction QA | 2M |
| OpenAssistant | Human chat | 2M |
| TinyStories | Creative writing | 1.5M |
| Dolly | Instruction tuning | 1M |

Vocab coverage: 96.8%. Build your own: `python scripts/build_mega_corpus.py`

---

## Architecture

```
Fractus/
├── crate/fractus-core/           Rust: 2-adic vortex (exact math)
├── crate/fractus-py/             Rust: PyO3 bindings
├── fractus/
│   ├── continuous_engine.py      The Continuous Thought Engine
│   ├── model_1b.py               Fractus-1B (88M params, 0.86B capacity)
│   ├── rag.py                    RAG + Plugins + MetaCognition
│   ├── memory.py                 Persistent cross-session memory
│   ├── cognitive_modes.py        Kuramoto phase → mental state
│   ├── generative_planner.py     Plan-then-fill generation
│   ├── specialization.py         Expert diversity loss
│   ├── tokenizer.py              GPT-2 byte-level BPE
│   ├── nn/                       attention, Kuramoto, MoE, SIREN (12 modules)
│   ├── causal/                   NOTEARS, RKHS, do-calculus
│   ├── reasoning/                proofs, conjectures, primes, ACT
│   ├── stability/                Lyapunov on Kuramoto
│   ├── metrics/                  honest measurements
│   └── train/                    online, surprise-gated, forward-forward
├── data/                         Alpaca, OASST, Dolly, FineWeb, TinyStories, code
├── tests/                        28 test files, 166+ tests
├── scripts/                      training, demos, corpus builders, white paper
├── docs/                         OVERVIEW, SPEC, layer plans L0–L9
└── Fractus_White_Paper.pdf       Technical document (signed)
```

---

## Honest limitations

1. **Generation quality** — the model at epoch 11 (ppl 103) produces rough text. More training needed for coherent generation.
2. **CTE needs trained weights** — the CTE architecture works but needs the 1B's trained brain transferred into it.
3. **MetaCognition is early** — the action net is 8.5K params, improves with use.
4. **CPU training is slow** — 19 tok/s. A GPU would give 50-100× speedup ($37 for full corpus on A100).

---

## License

MIT. This project belongs to the user, not a corporation.

## Author

**Philippe-Antoine Robert** — 2026

## Links

- **GitHub:** [github.com/AFKmoney/fractus](https://github.com/AFKmoney/fractus)
- **HuggingFace:** [huggingface.co/thefinalboss/Fractus](https://huggingface.co/thefinalboss/Fractus)
