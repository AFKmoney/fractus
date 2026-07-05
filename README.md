# Fractus

**The first continuous-thought AI with persistent memory, cognitive modes, and autonomous self-management — running on a consumer laptop.**

Fractus is not a GPT clone. It is a fundamentally different architecture: a **dynamical system** that thinks in continuous time, remembers across sessions, learns without retraining, and decides its own actions. It runs and trains entirely on CPU — no datacenter required.

> **Status:** Training in progress (Fractus-1B, 88M trainable params, 0.86B capacity, 10 epochs on code+knowledge corpus). Current: epoch 3, loss 5.135, ppl 170.

---

## What is Fractus?

Fractus is built on three layers:

### Layer 1: The Brain (Fractus-1B)
A fractal transformer with:
- **88M trainable parameters** with **0.86B effective capacity** via LazyStructuredSiren (low-rank weight compression W = scale·U·Vᵀ)
- **64 sparse MoE experts** (top-2 active per token, von Mises/Farey routing)
- **Multi-level causal linear attention** (Katharopoulos 2020, batched heads×levels)
- **Kuramoto oscillators** (low-rank RK4) as a "consciousness clock"
- **2-adic vortex** (Rust core) for exact token conditioning
- Trains at **21 tokens/sec on a Ryzen 5 5500U** (from scratch, CPU-only)

### Layer 2: The Continuous Thought Engine (CTE)
The brain doesn't process input→output. It **ticks**:
- Each tick advances Kuramoto oscillators → updates attention state → routes through MoE → estimates confidence
- **Adaptive depth**: easy inputs = 1 tick, hard inputs = 10 ticks
- **Chunk-based processing**: 16 tokens per forward at 117 tok/s (13M) or 21 tok/s (1B)
- The CTE is **proactive**: it can emit output without being prompted

### Layer 3: The Cognitive Layer (RAG + MetaCognition)
This is what makes Fractus an **agent**, not a tool:

- **Persistent Memory** (`rag.learn()`): stores knowledge permanently without retraining. Every conversation adds to the knowledge base. Saved to disk, survives restarts.
- **Retrieval-Augmented Generation**: given a question, retrieves relevant knowledge from the KB and injects it as context before generating.
- **Cognitive Plugins** (hot-swappable personality): analyst, creative, coder, teacher, hacker. Change personality in one call, like installing an app.
- **MetaCognition** (autonomous self-management): an 8.5K-param action network decides whether to RETRIEVE, LEARN, GENERATE, SWITCH mode, or REFLECT. Trains online from feedback.
- **Online Learning**: `rag.converse()` learns from every interaction — the model grows permanently smarter through use.

---

## Why this is different from GPT/Claude

| Property | GPT-4 / Claude | Fractus |
|---|---|---|
| **Processing** | Static (1 forward pass) | Continuous (tick-based CTE) |
| **Memory** | Context window (amnesic) | Persistent KB (survives restarts) |
| **Learning** | Retraining required | Online (every conversation) |
| **Skills** | Generic monolith | Hot-swappable plugins (5 modes) |
| **Autonomy** | Waits for instructions | Decides actions itself (MetaCognition) |
| **Training** | Datacenter GPU cluster | Consumer CPU (Ryzen 5 5500U) |
| **Deployment** | Cloud API (centralized) | Local device (decentralized) |
| **Growth** | Frozen between versions | Grows with every use |

---

## Architecture

```
Fractus/
├── crate/fractus-core/           Rust: 2-adic vortex (exact math, off-graph)
├── crate/fractus-py/             Rust: PyO3 bindings
├── fractus/
│   ├── continuous_engine.py      The Continuous Thought Engine (tick-based)
│   ├── model_1b.py               Fractus-1B model (88M params, 0.86B capacity)
│   ├── rag.py                    RAG + KnowledgeBase + Plugins + MetaCognition
│   ├── memory.py                 Persistent cross-session memory
│   ├── cognitive_modes.py        Kuramoto phase → mental state classifier
│   ├── generative_planner.py     Plan-then-fill generation
│   ├── specialization.py         Expert diversity loss
│   ├── tokenizer.py              GPT-2 byte-level BPE
│   ├── nn/                       embedding, attention, Kuramoto, MoE, SIREN variants
│   │   ├── attention.py          Multi-level causal linear attention (L8 batched)
│   │   ├── phase_ode.py          Low-rank Kuramoto RK4 oscillators
│   │   ├── moe.py                Von Mises/Farey sparse MoE
│   │   ├── lazy_siren.py         LazyStructuredSirenLinear (W = scale·U·Vᵀ)
│   │   ├── structured_siren.py   StructuredSiren (low-rank + spectral residual)
│   │   ├── cached_siren.py       CachedStructuredSiren (grid caching variant)
│   │   ├── siren.py              True sin(ω₀·) SIREN (Sitzmann 2020)
│   │   ├── embedding.py          Fractal embedding (char + Fourier + vortex)
│   │   ├── block.py              FractalBlock + FractalBlockFull
│   │   └── farey.py              Farey sequence + expert phases
│   ├── causal/                   NOTEARS, RKHS, Pearl do-calculus
│   ├── reasoning/                proofs, conjectures, primes, ACT
│   ├── stability/                Lyapunov on Kuramoto
│   ├── metrics/                  honest measurements (compression, SHD, perplexity)
│   └── train/                    online, mini-batch, surprise-gated, forward-forward
├── data/                         datasets (Alpaca, OASST, Dolly, FineWeb, TinyStories, code)
├── tests/                        28 test files, 166+ tests
├── scripts/                      training, demos, benchmarks, dataset builders, white paper
├── docs/                         OVERVIEW, SPEC, layer plans L0–L9
├── Fractus_White_Paper.pdf       Technical document (signed)
└── MODEL_CARD.md                 Model card
```

---

## Training data

The model is trained on a 12.8M-token mega corpus built from:

| Source | Type | Tokens |
|---|---|---|
| Python code instructions | Code (Python) | 1.5M |
| CodeAlpaca | Code (multi-language) | 2M |
| FineWeb (sample-10BT) | Web text (general knowledge) | 3M |
| Alpaca | Instruction QA | 2M |
| OpenAssistant | Human chat | 2M |
| TinyStories | Creative writing | 1.5M |
| Dolly | Instruction tuning | 1M |

Build your own: `python scripts/build_mega_corpus.py` (produces 12.8M tokens, 51.5 MB)

---

## Training results (in progress)

| Epoch | Loss | Perplexity | Tok/s |
|-------|------|------------|-------|
| 1 | 5.322 | 205 | 21 |
| 2 | 5.182 | 178 | 19 |
| 3 | 5.135 | 170 | 19 |
| 4-10 | *training in progress* | | |

Config: 88M trainable params, 0.86B capacity (LazyStructuredSiren rank=16), d_model=768, 8 layers, 64 experts, top-2 routing.

---

## Quick start

```bash
git clone https://github.com/AFKmoney/fractus.git
cd fractus

# Setup
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
maturin develop --release

# Test everything
pytest tests/ -q

# Use the RAG system
python -c "
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

# Teach it — no retraining needed
rag.learn('Python is a programming language.')
rag.learn('Neural networks learn via backpropagation.')

# Ask it
result = rag.query('What is Python?', top_k=2, max_tokens=30)
print(result['answer'])

# Let it manage itself
result = meta.process('Remember: my name is Philippe')
print(result['actions'])  # AI chose: LEARN
"

# Train the 1B model
python scripts/train_1b_fast.py --epochs 10 --max-tokens 500000

# Build a custom corpus
python scripts/build_mega_corpus.py
```

**Prerequisites:** Rust (`cargo`), Python 3.10+, maturin.

---

## Measured performance

### Training throughput
| Config | Tokens/sec | Hardware |
|---|---|---|
| 13M CTE, single-token | 25 | Ryzen 5 5500U |
| 13M CTE, chunk-based (16) | 117 | Ryzen 5 5500U |
| 88M 1B, LazySiren (no checkpointing) | 21 | Ryzen 5 5500U |

### Training optimizations (all profile-measured)
| Optimization | Before → After | Speedup |
|---|---|---|
| Batch heads × levels in attention | 17.3ms → 6.6ms | 2.6× |
| Chunk-based tick (16 tokens) | 25 → 117 tok/s | 4.7× |
| LazyStructuredSiren (vs grid SIREN) | 43s → 1.3s/step | 33× |
| Remove gradient checkpointing | 5 → 21 tok/s | 4.2× |

---

## Documentation

| Document | Content |
|---|---|
| [Fractus_White_Paper.pdf](Fractus_White_Paper.pdf) | 10-page technical paper |
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | Complete A-to-Z walkthrough |
| [docs/SPEC.md](docs/SPEC.md) | Full specification (L0–L7) |
| [docs/2026-06-26-fractus-L9-continuous-thought-engine.md](docs/2026-06-26-fractus-L9-continuous-thought-engine.md) | The 5 innovations |
| [docs/2026-06-26-fractus-L8-lightweight-training.md](docs/2026-06-26-fractus-L8-lightweight-training.md) | Training optimizations |

---

## Honest limitations

1. **Generation quality** — the 1B model needs more epochs for fully coherent text. Training is ongoing.
2. **CTE needs trained weights** — the CTE architecture works but needs the 1B's trained brain transferred into it.
3. **MetaCognition is early** — the action net is 8.5K params, improves with use.
4. **1B training is slow** — 21 tok/s on CPU (GPU would give 50-100×).

---

## License

MIT. This project belongs to the user, not a corporation.

## Author

**Philippe-Antoine Robert** — 2026

## Links

- **GitHub:** [github.com/AFKmoney/fractus](https://github.com/AFKmoney/fractus)
- **HuggingFace:** [huggingface.co/thefinalboss/Fractus](https://huggingface.co/thefinalboss/Fractus)
