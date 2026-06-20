# fractus

**Un transformer fractal entrainable, with compression SIREN, raisonnement causal NOTEARS, et generation de preuves verifiees.**

Fractus est une refonte honnete de deux systems precedents (**the original architecture** et **the original design** de the original author) qui corrige leurs errors tout en preservant all les concepts. Le result : un transformer fractal qui **apprend for de true** (autodiff natif, pas de bruit aleatoire), with des **maths fideles** (verifiees par reviewers independants), des **tests honnetes** (156 tests, pas tautologiques), et des **chiffres mesures** (pas hardcodes).

> CPU-only friendly · Python 3.10+ · PyTorch 2.2+ · Rust 1.70+

---

## Pourquoi fractus ?

Les systems originaux (FNN + OMNI) avaient **9 mensonges/errors** majeurs :
- FNN n'apprenait pas (`training.rs:399` utilisait `rand::random()*0.01` au lieu d'un gradient)
- La SIREN d'OMNI was fausse (`nn.SiLU` au lieu de `sin(ω₀·)`)
- La compression 20.4× was hardcodee (real : 1.51×)
- Le RKHS n'avait pas de noyau · Le do-calculus zerorait des colonnes · etc.

**Fractus corrige tout ca.** Voir `docs/CORRECTIONS.md` for la liste complete with tracabilite `file:line`.

---

## Concepts (all presents, all fonctionnels)

| Concept | Module | Valide par |
|---|---|---|
| Vortex 2-adique (valuation, ultrametrique) | `fractus-core/src/vortex.rs` (Rust) | Bug OMNI corrige : `2^{+v₂}` → `2^{-v₂}` |
| Embedding fractal (Fourier Mandelbrot + char features) | `fractus/nn/embedding.py` | backward CHAQUE param |
| Attention lineaire causale multi-niveaux | `fractus/nn/attention.py` | vectorisee 17× + causalite testee |
| Oscillateurs Kuramoto RK4 bas-rang | `fractus/nn/phase_ode.py` | RK4 + stateless |
| MoE a routing von Mises / Farey | `fractus/nn/moe.py` | load-balance + top-k |
| Compression SIREN `sin(ω₀·)` | `fractus/nn/siren.py` | ω₀=30 Sitzmann, ratio mesure |
| NOTEARS causal (DAG acyclique) | `fractus/causal/notears.py` | SHD=0 sur SCM non-lineaire |
| RKHS via Random Fourier Features | `fractus/causal/rkhs.py` | true noyau, pas projection nue |
| do-calculus de Pearl | `fractus/causal/do.py` | clamp a v, pas zeroring |
| Preuves (verify exact sound) | `fractus/reasoning/proof.py` | Fermat/Wilson/GCD |
| Generation de numbers premiers | `fractus/reasoning/prime_generator.py` | **100% primalite** after 100 steps |
| Conjectures (falsification popperienne) | `fractus/reasoning/conjecture.py` | 10 templates, 6 strategies |
| ACT (Adaptive Computation Time) | `fractus/reasoning/act.py` | Graves 2016 |
| Lyapunov (under-system Kuramoto) | `fractus/stability/lyapunov.py` | true system dynamique |

---

## Installation

```bash
git clone https://github.com/AFKmoney/fractus.git
cd fractus

# Creer le venv (utiliser `py` sur Windows, pas python MSYS2)
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

# Demo 2 : generation de numbers premiers (100% validite)
python scripts/demo_prime_reinforce.py

# Demo 3 : NOTEARS recupere un DAG causal
python scripts/demo_causal.py

# Demo 4 : tinyshakespeare (perplexite real)
python scripts/demo_shakespeare.py

# Demo complete : texte + preuves + causal
python scripts/demo_full.py
```

---

## Entrainement sur datasets HuggingFace

```bash
# Petit modele sur tinyshakespeare (CPU, ~2 min)
python scripts/train_hf.py --preset cpu-small --dataset tinyshakespeare

# Modele medium sur wikitext (GPU recommande)
python scripts/train_hf.py --preset gpu-medium --dataset wikitext-2

# Configuration personnalisee
python scripts/train_hf.py --dataset HuggingFaceFW/fineweb --text-field text \
    --d-model 256 --n-blocks 6 --seq-len 128 --batch-size 16 --epochs 3
```

### Tailles de modele et hardware requis

| Preset | Params | RAM min | Temps/epoch | Hardware |
|---|---|---|---|---|
| `cpu-tiny` | ~80k | 4 GB | ~2 min | CPU laptop |
| `cpu-small` | ~500k | 8 GB | ~30 min | CPU desktop |
| `gpu-small` | ~5M | 8 GB VRAM | ~5 min | GPU entry (RTX 3060) |
| `gpu-medium` | ~50M | 16 GB VRAM | ~30 min | GPU mid (RTX 4090) |
| `gpu-large` | ~300M | 40 GB VRAM | ~2h | A100 40GB |
| `gpu-1b` | ~1B | 80 GB VRAM | ~8h | A100 80GB / H100 |

> ⚠️ **Honnetete** : le preset `gpu-1b` est fourni for la completude, but il **necessite un GPU datacenter** (A100 80GB ou H100). Il est **impossible** sur CPU ou GPU consumer. Ne le lancez pas without le hardware approprie — ca OOM ou prendra des semaines. Le bottleneck principal est le Kuramoto RK4 (non encore vectorise) et la SIREN (evaluation a each forward).

---

## Tests

```bash
pytest tests/ -q                    # 156 tests, all passants
pytest tests/test_attention_vectorized.py -v   # equivalence vectorisee
```

---

## Documentation

- `docs/CORRECTIONS.md` — les 9 mensonges originaux corriges (tracabilite file:line)
- `docs/superpowers/specs/2026-06-19-fractus-unified-design.md` — spec complete
- `docs/superpowers/plans/` — plans d'implementation par couche (L0→L7)

---

## Architecture

```
fractus/
├── crate/fractus-core/       Rust : vortex 2-adique (math pure)
├── crate/fractus-py/         Rust : bindings PyO3
├── fractus/nn/               PyTorch : embedding, attention, Kuramoto, MoE, SIREN
├── fractus/causal/           PyTorch : NOTEARS, RKHS, do-calculus
├── fractus/reasoning/        PyTorch : preuves, conjectures, ACT
├── fractus/stability/        PyTorch : Lyapunov
├── fractus/metrics/          mesures honnetes (compression, SHD, perplexite)
├── data/                     datasets (tinyshakespeare, SCM synthetiques)
├── tests/                    156 tests
├── scripts/                  8 demos + train_hf.py
└── docs/                     spec, plans, corrections
```

**Le Rust reste hors-graphe autodiff** (computation exact, verification, precalcul). La forward/backward est **PyTorch pur** (autodiff natif).

---

## Limites connues (honnetes)

1. **Kuramoto RK4 non vectorise** : 4 under-steps Python, ~116ms/batch. Future work : vectoriser comme l'attention.
2. **SIREN compression faible** : 1.51× sur poids denses (inherent — les SIREN compressent les functions lisses, pas le bruit).
3. **ProofGenerator REINFORCE** : la tâche FNN originale (converger a 1e-3) est inatteignable. La tâche redefinie (produire des premiers) marche a 100%.
4. **Causal non-lineaire fort** : NOTEARS lineaire est robuste a la non-linearite moderee (tanh), but pas aux relations fortement non-lineaires.

---

## Licence

MIT. Voir `LICENSE`.

---

## Credits

Concepts originaux : **the original author** (the original architecture, the original design).
Refonte honnete : fractus project, 2026.
