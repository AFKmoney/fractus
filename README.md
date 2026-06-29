# Fractus

**A continuous-thought AI engine and 1B-capacity language model, trained entirely on a consumer CPU laptop.**

Fractus is not another GPT clone. It is a fundamentally different architecture: a **dynamical system** that thinks in continuous time, remembers across sessions, exhibits cognitive modes, and generates structured output through planning. It runs and trains on any laptop — no datacenter required.

> **Start here:** [`docs/OVERVIEW.md`](docs/OVERVIEW.md) — the complete A-to-Z walkthrough.
> **Full paper:** [`Fractus_White_Paper.pdf`](Fractus_White_Paper.pdf) — 10-page technical document.

---

## What makes Fractus different

| Property | GPT / Claude | Fractus |
|---|---|---|
| **Processing** | Static function (1 forward pass) | Continuous thought engine (tick-based) |
| **Memory** | Context window (amnesic) | Persistent memory bank (survives restarts) |
| **Skills** | Generic monolith | Specialized MoE experts (1 expert = 1 domain) |
| **Mental state** | Stateless | Cognitive modes (Kuramoto phase patterns) |
| **Generation** | Token-by-token (no plan) | Plan-then-fill (generative planning) |
| **Training** | Datacenter GPU cluster | Consumer CPU (AMD Ryzen 5 5500U) |
| **Deployment** | Cloud API (centralized) | Local device (decentralized) |

---

## Architecture

Fractus combines a fractal transformer backbone with a continuous-time reasoning engine:

```
Rust Core (exact, off-graph)
└── 2-adic Vortex: valuation, ultrametric distance, Collatz hash

PyTorch Backbone (trainable, differentiable)
├── Fractal Embedding: 16 char features + Mandelbrot-decayed Fourier + vortex conditioning
├── Causal Linear Attention: multi-level, state-space, batched heads×levels
├── Kuramoto Oscillators: low-rank RK4, the "consciousness clock"
├── Sparse MoE: von Mises / Farey-routed, top-2 of 64 experts
└── LazyStructuredSiren: low-rank weight compression W = scale·U·Vᵀ

Continuous Thought Engine (the paradigm shift)
├── Tick-based reasoning: think → accumulate → emit when confident
├── Persistent Memory: recall + consolidate + save/load to disk
├── Cognitive Modes: Kuramoto phases → mental state classification
├── Expert Specialization: diversity loss forces distinct skill domains
└── Generative Planner: plan structural anchors, then fill content
```

### The key breakthrough: LazyStructuredSirenLinear

The 1B model fits on a CPU because each expert weight matrix is stored as a **low-rank decomposition** `W = scale · U·Vᵀ` (rank 16), not as a dense grid. This gives:

- **0.86B effective capacity** from **88M trainable parameters** (9.8× compression)
- **0.4 GB RAM** for the full 64-expert model
- **5.9s per training step** on a Ryzen 5 (no gradient checkpointing needed)
- **Zero grid memory** — the bottleneck that caused OOM in prior SIREN implementations

---

## What's in the repo

```
fractus/
├── crate/fractus-core/       Rust: 2-adic vortex (pure math, testable alone)
├── crate/fractus-py/         Rust: PyO3 bindings
├── fractus/
│   ├── nn/                   embedding, attention, Kuramoto, MoE, SIREN variants
│   ├── causal/               NOTEARS, RKHS, Pearl do-calculus
│   ├── reasoning/            proofs, conjectures, prime generation, ACT
│   ├── stability/            Lyapunov on Kuramoto
│   ├── metrics/              honest measurements (compression, SHD, perplexity)
│   ├── train/                online, mini-batch, surprise-gated, forward-forward trainers
│   ├── continuous_engine.py  the ContinuousThoughtEngine
│   ├── model_1b.py           the Fractus-1B model (0.86B capacity)
│   ├── memory.py             persistent cross-session memory
│   ├── cognitive_modes.py    phase-to-mode classifier
│   ├── generative_planner.py plan-then-fill generation
│   ├── specialization.py     expert diversity loss
│   └── tokenizer.py          GPT-2 byte-level BPE
├── data/                     tinyshakespeare + synthetic SCMs + fractus_corpus
├── tests/                    28 test files, 166+ tests
├── scripts/                  demos, training, benchmark, dataset builder, white paper
├── docs/                     OVERVIEW, SPEC, layer plans L0–L9
├── Fractus_White_Paper.pdf   10-page technical document
└── MODEL_CARD.md             model card for HF Hub
```

---

## Measured results

### Small model (13M params, ContinuousThoughtEngine, 30k tokens)
| Epoch | Loss | Perplexity | Accuracy |
|-------|------|------------|----------|
| 1     | 6.29 | 539        | 20.5%    |
| 10    | 2.37 | 10.7       | 47.0%    |
| 20    | 1.29 | 3.6        | 70.0%    |
| 30    | 1.00 | 2.7        | 77.0%    |

### Fractus-1B (88M trainable, 0.86B capacity)
- Config: d_model=768, 8 layers, 64 experts, top-2 routing, rank-16 LazyStructuredSiren
- Training: 500k-token multi-domain corpus (code, literature, math, knowledge)
- Throughput: 5–8 tokens/sec on Ryzen 5 5500U (CPU-only)
- Published: [huggingface.co/thefinalboss/Fractus-1B](https://huggingface.co/thefinalboss/Fractus-1B)

### Training optimizations (all profile-measured)
| Optimization | Before | After | Speedup |
|---|---|---|---|
| Attention: batch heads×levels | 17.3 ms | 6.6 ms | 2.6× |
| Chunk-based tick (16 tokens) | 25 tok/s | 117 tok/s | 4.7× |
| LazyStructuredSiren (vs CachedSiren) | 43 s/step | 5.9 s/step | 7.3× |
| LazyStructuredSiren (RAM) | 3.2 GB (OOM) | 0.4 GB | 8× |

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

# Run tests (166+ tests)
pytest tests/ -q

# Demo: continuous thought engine learns text
python scripts/demo_transformer.py

# Train the ContinuousThoughtEngine
python scripts/train_continuous.py --epochs 30 --d-model 128

# Train Fractus-1B
python scripts/train_1b_final.py --epochs 3 --d-model 768

# Benchmark training throughput
python scripts/bench_train.py --runs 3
```

**Prerequisites:** Rust (`cargo`), Python 3.10+ with the `py` launcher, maturin.

---

## Documentation

| Document | Content |
|---|---|
| [`docs/OVERVIEW.md`](docs/OVERVIEW.md) | **Start here.** Complete A-to-Z walkthrough. |
| [`Fractus_White_Paper.pdf`](Fractus_White_Paper.pdf) | 10-page technical paper (signed, dated). |
| [`docs/SPEC.md`](docs/SPEC.md) | Full specification (layers L0–L7). |
| [`docs/2026-06-26-fractus-L8-lightweight-training.md`](docs/2026-06-26-fractus-L8-lightweight-training.md) | Training optimizations (profile-driven). |
| [`docs/2026-06-26-fractus-L9-continuous-thought-engine.md`](docs/2026-06-26-fractus-L9-continuous-thought-engine.md) | The 5 innovations + paradigm shift. |
| [`docs/2026-06-19-fractus-L0-socle.md`](docs/2026-06-19-fractus-L0-socle.md) → [`L4-causal.md`](docs/2026-06-19-fractus-L4-causal.md) | Per-layer implementation plans. |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Model card for HuggingFace Hub. |

---

## Honest limitations

1. **Generation quality:** the trained model produces repetitive text. More data and epochs needed.
2. **1B training speed:** 5–8 tokens/sec on CPU is feasible but slow. GPU would yield 50–100×.
3. **Cognitive modes untrained:** the classifier exists but hasn't been trained on labeled data.
4. **Generative planner is proof-of-concept:** the plan/fill pipeline needs a well-trained engine.
5. **State-carry is attention-level:** carrying (S,z) through the full block stack is future work.
6. **No formal verification:** Lean 4 / ZK-SNARK are honestly absent.

---

## License

MIT. This project belongs to the user, not a corporation.

## Author

**Philippe-Antoine Robert** — 29 June 2026

## Links

- **GitHub:** [github.com/AFKmoney/fractus](https://github.com/AFKmoney/fractus)
- **HuggingFace:** [huggingface.co/thefinalboss/Fractus-1B](https://huggingface.co/thefinalboss/Fractus-1B)
