# fractus

**Un transformer fractal entrainable, with compression SIREN, raisonnement causal NOTEARS, and generation of proofs verifieses.**

Fractus est a refonte honestete of deux systems precedents (**the original architecture** and **the original design** of the original author) which correctede leurs errors all en preservant all the concepts. Le result : a transformer fractal which **apprend for of true** (autodiff natif, not of bruit aleatoire), with **maths faithfuls** (verifieses by reviewers independants), **tests honestetes** (156 tests, not tautologicals), and **chiffres mesures** (pas hardcodes).

> CPU-only friendly · Python 3.10+ · PyTorch 2.2+ · Rust 1.70+

---

## Pourquoi fractus ?

Les systems originaux (the original + the original) avaient **9 falsehoods/errors** majeurs :
- the original did not learn (`training.rs:399` utilisait `rand::random()*0.01` au lieu d'un gradient)
- La SIREN d'the original was fausse (`nn.SiLU` instead of `sin(ω0·)`)
- La compression 20.4× was hardcodee (real : 1.51×)
- Le RKHS n'avait not of noyau · Le do-computationus zerorait colonnes · etc.

**Fractus correctede all ca.** Voir `docs/CORRECTIONS.md` for the liste complete with tracabilite `file:line`.

---

## Concepts (all presents, all fonctionnels)

| Concept | Module | Valide by |
|---|---|---|
| Vortex 2-adique (valuation, ultrametrique) | `fractus-core/src/vortex.rs` (Rust) | Bug the original correctede : `2^{+v2}` → `2^{-v2}` |
| Embedding fractal (Fourier Mandelbrot + char features) | `fractus/nn/embedding.py` | backward CHAQUE param |
| Attention lineaire causale multi-niveaux | `fractus/nn/attention.py` | vectorisee 17× + causalite testee |
| Oscillateurs Kuramoto RK4 bas-rang | `fractus/nn/phase_ode.py` | RK4 + stateless |
| MoE a routing von Mises / Farey | `fractus/nn/moe.py` | load-balance + top-k |
| Compression SIREN `sin(ω0·)` | `fractus/nn/siren.py` | ω0=30 Sitzmann, ratio mesure |
| NOTEARS causal (DAG acyclique) | `fractus/causal/notears.py` | SHD=0 on SCM non-lineaire |
| RKHS via Random Fourier Features | `fractus/causal/rkhs.py` | true noyau, not projection nue |
| do-computationus of Pearl | `fractus/causal/do.py` | clamp a v, not zeroring |
| Preuves (verify exact sound) | `fractus/reasoning/proof.py` | Fermat/Wilson/GCD |
| Generation of numbers premiers | `fractus/reasoning/prime_generator.py` | **100% primalite** after 100 steps |
| Conjectures (Popperian falsification) | `fractus/reasoning/conjecture.py` | 10 templates, 6 strategies |
| ACT (Adaptive Computation Time) | `fractus/reasoning/act.py` | Graves 2016 |
| Lyapunov (under-system Kuramoto) | `fractus/stability/lyapunov.py` | true system dynamique |

---

## Installation

```bash
git clone https://github.com/AFKmoney/fractus.git
cd fractus

# Creer the venv (utiliser `py` on Windows, not python MSYS2)
py -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/macOS

# Dependances Python
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only
pip install -r requirements.txt

# Module natif Rust (pont Python↔Rust)
maturin develop --release

# Tests (must all passer)
pytest tests/ -q
```

**Prerequis** : Rust (`cargo`), Python 3.10+ with `py` launcher.

---

## Demarrage rapide

```bash
# Demo 1 : transformer fractal apprend "hello world"
python scripts/demo_transformer.py

# Demo 2 : generation of numbers premiers (100% validity)
python scripts/demo_prime_reinforce.py

# Demo 3 : NOTEARS recupere a DAG causal
python scripts/demo_causal.py

# Demo 4 : tinyshakespeare (perplexite real)
python scripts/demo_shakespeare.py

# Demo complete : texte + proofs + causal
python scripts/demo_full.py
```

---

## Entrainement on datasets HuggingFace

```bash
# Petit modele on tinyshakespeare (CPU, ~2 min)
python scripts/train_hf.py --preset cpu-small --dataset tinyshakespeare

# Modele medium on wikitext (GPU recommande)
python scripts/train_hf.py --preset gpu-medium --dataset wikitext-2

# Configuration personnalisee
python scripts/train_hf.py --dataset HuggingFaceFW/fineweb --text-field text \
    --d-model 256 --n-blocks 6 --seq-len 128 --batch-size 16 --epochs 3
```

### Tailles of modele and hardware requis

| Preset | Params | RAM min | Temps/epoch | Hardware |
|---|---|---|---|---|
| `cpu-tiny` | ~80k | 4 GB | ~2 min | CPU laptop |
| `cpu-small` | ~500k | 8 GB | ~30 min | CPU desktop |
| `gpu-small` | ~5M | 8 GB VRAM | ~5 min | GPU entry (RTX 3060) |
| `gpu-medium` | ~50M | 16 GB VRAM | ~30 min | GPU mid (RTX 4090) |
| `gpu-large` | ~300M | 40 GB VRAM | ~2h | A100 40GB |
| `gpu-1b` | ~1B | 80 GB VRAM | ~8h | A100 80GB / H100 |

> ⚠️ **Honnetete** : the preset `gpu-1b` est fourni for the completude, but il **necessite a GPU datacenter** (A100 80GB or H100). Il est **impossible** on CPU or GPU consumer. Ne the lancez not without the hardware approprie — ca OOM or prendra semaines. Le bottleneck main est the Kuramoto RK4 (non encore vectorise) and the SIREN (evaluation a each forward).

---

## Tests

```bash
pytest tests/ -q                    # 156 tests, all passants
pytest tests/test_attention_vectorized.py -v   # equivalence vectorisee
```

---

## Documentation

- `docs/CORRECTIONS.md` — the 9 falsehoods originaux correctedes (tracabilite file:line)
- `docs/superpowers/specs/2026-06-19-fractus-unified-design.md` — spec complete
- `docs/superpowers/plans/` — plans d'implementation by couche (L0→L7)

---

## Architecture

```
fractus/
├── crate/fractus-core/       Rust : vortex 2-adique (math pure)
├── crate/fractus-py/         Rust : bindings PyO3
├── fractus/nn/               PyTorch : embedding, attention, Kuramoto, MoE, SIREN
├── fractus/causal/           PyTorch : NOTEARS, RKHS, do-computationus
├── fractus/reasoning/        PyTorch : proofs, conjectures, ACT
├── fractus/stability/        PyTorch : Lyapunov
├── fractus/metrics/          mesures honestetes (compression, SHD, perplexite)
├── data/                     datasets (tinyshakespeare, SCM synthetiques)
├── tests/                    156 tests
├── scripts/                  8 demos + train_hf.py
└── docs/                     spec, plans, corrections
```

**Le Rust reste outside the autodiff graph** (computation exact, verification, precomputation). La forward/backward est **PyTorch pur** (autodiff natif).

---

## Limites connues (honestetes)

1. **Kuramoto RK4 non vectorise** : 4 under-steps Python, ~116ms/batch. Future work : vectoriser comme l'attention.
2. **SIREN compression faible** : 1.51× on poids denses (inherent — the SIREN compressent the functions lisses, not the bruit).
3. **ProofGenerator REINFORCE** : the tache the original originale (convergesr a 1e-3) est inatteignable. La tache redefinie (produire premiers) marche a 100%.
4. **Causal non-lineaire fort** : NOTEARS lineaire est robuste a the non-linearite moderee (tanh), but not aux relations fortement non-lineaires.

---

## Licence

MIT. Voir `LICENSE`.

---

## Credits

Concepts originaux : **the original author** (the original architecture, the original design).
Refonte honestete : fractus project, 2026.
