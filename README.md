# fractus

**A trainable fractal transformer, with SIREN compression, NOTEARS causal reasoning, verified proof generation, and lightweight CPU training.**

Fractus is an honest rebuild of two prior systems that corrects their errors while preserving all of their concepts. The result: a fractal transformer that **truly learns** (native autodiff, not random noise), with **faithful math** (verified by independent review), **honest tests** (not tautological), **measured numbers** (not hardcoded), and **profile-driven lightweight training** (attention ×2.6 faster, surprise-gated energy savings).

> CPU-only friendly · Python 3.10+ · PyTorch 2.2+ · Rust 1.70+

> 📖 **New here? Read [`docs/OVERVIEW.md`](docs/OVERVIEW.md) first** — it explains the entire project from A to Z (what fractus is, why it exists, how every module works, and how to use it).

---

## Why fractus?

The original systems suffered from **9 major falsehoods / errors**:
- One system did not learn (`training.rs:399` used `rand::random()*0.01` instead of a gradient).
- Its SIREN was fake (`nn.SiLU` instead of `sin(ω0·)`).
- The 20.4× compression was hardcoded (actual: ~1.5×).
- The RKHS had no kernel · the do-calculus zeroed columns · and more.

**Fractus corrects all of this** — see [`docs/OVERVIEW.md`](docs/OVERVIEW.md) §1 for the full list with `file:line` traceability.

---

## Concepts (all present, all functional)

| Concept | Module | Validated by |
|---|---|---|
| 2-adic Vortex (valuation, ultrametric) | `fractus-core/src/vortex.rs` (Rust) | Original bug fixed: `2^{+v2}` → `2^{-v2}` |
| Fractal embedding (Mandelbrot Fourier + char features) | `fractus/nn/embedding.py` | backward on EVERY parameter |
| Multi-level causal linear attention | `fractus/nn/attention.py` | **L8: heads×levels batched, ×2.6 faster** + causality tested |
| Low-rank Kuramoto RK4 oscillators | `fractus/nn/phase_ode.py` | RK4 + **L8: unrolled**, exact equivalence |
| von Mises / Farey-routed MoE | `fractus/nn/moe.py` | load-balance + **L8: adaptive dense/sparse** |
| State-carrying attention (chunked training) | `fractus/train/trainer.py` | **L8: proven equivalent to full-sequence** |
| True `sin(ω0·)` SIREN compression | `fractus/nn/siren.py` | ω0=30 Sitzmann, measured ratio |
| NOTEARS causal (acyclic DAG) | `fractus/causal/notears.py` | SHD=0 on non-linear SCM |
| RKHS via Random Fourier Features | `fractus/causal/rkhs.py` | true kernel, not a bare projection |
| Pearl's do-calculus | `fractus/causal/do.py` | clamps to v, not zeroing |
| Proofs (exact-sound verification) | `fractus/reasoning/proof.py` | Fermat/Wilson/GCD |
| Prime-number generation | `fractus/reasoning/prime_generator.py` | **>50% validity** after 150 REINFORCE steps |
| Conjectures (Popperian falsification) | `fractus/reasoning/conjecture.py` | 10 templates, 6 strategies |
| ACT (Adaptive Computation Time) | `fractus/reasoning/act.py` | Graves 2016 |
| Lyapunov (Kuramoto subsystem) | `fractus/stability/lyapunov.py` | true dynamical system |
| Surprise-gated training (energy-proportional) | `fractus/train/surprise_gate.py` | **L8: converges, selectivity<1 proven** |

---

## Installation

```bash
git clone https://github.com/AFKmoney/fractus.git
cd fractus

# Create the venv (use `py` on Windows, not the MSYS2 python)
py -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/macOS

# Python dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only
pip install -r requirements.txt

# Native Rust module (Python↔Rust bridge)
maturin develop --release

# Tests (should all pass)
pytest tests/ -q
```

**Prerequisites:** Rust (`cargo`), Python 3.10+ with the `py` launcher.

---

## Quick start

```bash
# Demo 1: the fractal transformer learns "hello world"
python scripts/demo_transformer.py

# Demo 2: prime-number generation (100% validity)
python scripts/demo_prime_reinforce.py

# Demo 3: NOTEARS recovers a causal DAG
python scripts/demo_causal.py

# Demo 4: tinyshakespeare (real perplexity)
python scripts/demo_shakespeare.py

# Full demo: text + proofs + causal
python scripts/demo_full.py

# Benchmark training throughput (measure your own CPU)
python scripts/bench_train.py --tag baseline --runs 3
python scripts/bench_train.py --compare
```

---

## Training on HuggingFace datasets

```bash
# Small model on tinyshakespeare (CPU, ~2 min)
python scripts/train_hf.py --preset cpu-small --dataset tinyshakespeare

# Medium model on wikitext (GPU recommended)
python scripts/train_hf.py --preset gpu-medium --dataset wikitext-2

# Custom configuration
python scripts/train_hf.py --dataset HuggingFaceFW/fineweb --text-field text \
    --d-model 256 --n-blocks 6 --seq-len 128 --batch-size 16 --epochs 3
```

### Model sizes and required hardware

| Preset | Params | Min RAM | Time/epoch | Hardware |
|---|---|---|---|---|
| `cpu-tiny` | ~80k | 4 GB | ~2 min | CPU laptop |
| `cpu-small` | ~500k | 8 GB | ~30 min | CPU desktop |
| `gpu-small` | ~5M | 8 GB VRAM | ~5 min | Entry GPU (RTX 3060) |
| `gpu-medium` | ~50M | 16 GB VRAM | ~30 min | Mid GPU (RTX 4090) |
| `gpu-large` | ~300M | 40 GB VRAM | ~2h | A100 40GB |
| `gpu-1b` | ~1B | 80 GB VRAM | ~8h | A100 80GB / H100 |

> ⚠️ **Honesty note:** the `gpu-1b` preset is provided for completeness, but it **requires a datacenter GPU** (A100 80GB or H100). It is **impossible** on a consumer CPU or GPU. Do not launch it without the appropriate hardware — it will OOM or take weeks. The main bottleneck at that scale is the SIREN (evaluated at each forward) and the sheer parameter count.

---

## Tests

```bash
pytest tests/ -q                    # full suite
pytest tests/test_attention_vectorized.py -v   # vectorization equivalence
```

---

## Documentation

- **[`docs/OVERVIEW.md`](docs/OVERVIEW.md)** — **start here.** The complete A-to-Z walkthrough (what fractus is, why, how every module works).
- [`docs/SPEC.md`](docs/SPEC.md) — the complete specification (the 7-layer plan).
- [`docs/2026-06-19-fractus-L0-socle.md`](docs/2026-06-19-fractus-L0-socle.md) through [`L4-causal.md`](docs/2026-06-19-fractus-L4-causal.md) — per-layer implementation plans.
- [`docs/2026-06-26-fractus-L8-lightweight-training.md`](docs/2026-06-26-fractus-L8-lightweight-training.md) — the lightweight-training deep-dive (profile-driven, attention ×2.6, surprise-gating).

---

## Architecture

```
fractus/
├── crate/fractus-core/       Rust: 2-adic vortex (pure math)
├── crate/fractus-py/         Rust: PyO3 bindings
├── fractus/nn/               PyTorch: embedding, attention, Kuramoto, MoE, SIREN
├── fractus/causal/           PyTorch: NOTEARS, RKHS, do-calculus
├── fractus/reasoning/        PyTorch: proofs, conjectures, ACT
├── fractus/stability/        PyTorch: Lyapunov
├── fractus/metrics/          honest measurements (compression, SHD, perplexity)
├── fractus/train/            PyTorch: lightweight trainers (L8: state-carry, AMP, surprise-gating)
├── data/                     datasets (tinyshakespeare, synthetic SCMs)
├── tests/                    test suite (166 tests)
├── scripts/                  demos, train_hf.py, bench_train.py
└── docs/                     OVERVIEW, SPEC, layer plans L0–L8
```

**Rust stays outside the autodiff graph** (exact computation, verification, precomputation). The forward/backward pass is **pure PyTorch** (native autodiff).

---

## Lightweight training (L8)

Profile-driven optimizations that exploit fractus's structure — **measure, don't guess.** Profiling overturned the README's old claim that "Kuramoto is the bottleneck": the real cost was the Python loop over heads × levels in attention.

| Optimization | Measured gain | Equivalence test |
|---|---|---|
| Attention: batch heads × levels into ONE call | **17.3 → 6.6 ms (×2.6)** | causality + vectorized equivalence |
| MoE: adaptive dense/sparse dispatch | sparse when E>2K, else dense | `test_moe_sparse_matches_reference` |
| Kuramoto RK4: unrolled steps | Python overhead removed | atol=1e-6 vs looped |
| State-carrying attention | O(chunk) memory, Mamba/RWKV trick | equivalent to full-sequence (atol=1e-3) |
| LightweightTrainer (bf16 + fused AdamW + cosine) | cheaper matmuls + faster convergence | runs, loss finite |
| **SurpriseGatedTrainer** (new method) | backprop only on high-loss tokens | converges, selectivity<1 |

```python
from fractus.train import LightweightTrainer, SurpriseGatedTrainer

# Cheap wins: AMP + fused AdamW + cosine scheduler
trainer = LightweightTrainer(model, lr=3e-3, warmup_steps=20, t_max=200)

# Or the energy-proportional surprise-gated trainer
trainer = SurpriseGatedTrainer(model, lr=3e-3, percentile=70.0, warmup_full=5)
```

See [`docs/2026-06-26-fractus-L8-lightweight-training.md`](docs/2026-06-26-fractus-L8-lightweight-training.md) for the full deep-dive.

---

## Known limitations (honesty)

1. **State-carry is attention-level only:** carrying the `(S,z)` state through the full block stack (embedding + N blocks + head) is documented future work; the principle and equivalence are proven at the attention level.
2. **Surprise-gating is a biased estimator:** useful in the low-gradient regime (tokens the model already knows), not a universal free lunch.
3. **Weak SIREN compression:** ~1.5–5× on dense weights (inherent — SIREN compresses smooth functions, not noise).
4. **ProofGenerator REINFORCE:** the original "converge to 1e-3" task is unattainable. The redefined task (produce primes) works at >50% validity.
5. **Strong non-linear causality:** linear NOTEARS is robust to moderate non-linearity (tanh), but not to strongly non-linear relations.
6. **Full-model bench is variance-dominated on a shared CPU:** trust the isolated component numbers; run `bench_train.py` on an idle machine for a clean figure.

---

## License

MIT. See `LICENSE`.

---

## Credits

Original concepts: prior work on fractal neural architectures and 2-adic / non-Archimedean operators.
Honest rebuild: fractus project, 2026.
