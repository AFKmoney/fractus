# fractus

**Un transformer fractal entraînable, avec compression SIREN, raisonnement causal NOTEARS, et génération de preuves vérifiées.**

Fractus est une réfonte honnête de deux systèmes précédents (**FNN v5.0** et **OMNI-FRACTAL** de Philippe-Antoine Robert) qui corrige leurs erreurs tout en préservant tous les concepts. Le résultat : un transformer fractal qui **apprend pour de vrai** (autodiff natif, pas de bruit aléatoire), avec des **maths fidèles** (vérifiées par reviewers indépendants), des **tests honnêtes** (156 tests, pas tautologiques), et des **chiffres mesurés** (pas hardcodés).

> CPU-only friendly · Python 3.10+ · PyTorch 2.2+ · Rust 1.70+

---

## Pourquoi fractus ?

Les systèmes originaux (FNN + OMNI) avaient **9 mensonges/erreurs** majeurs :
- FNN n'apprenait pas (`training.rs:399` utilisait `rand::random()*0.01` au lieu d'un gradient)
- La SIREN d'OMNI était fausse (`nn.SiLU` au lieu de `sin(ω₀·)`)
- La compression 20.4× était hardcodée (réelle : 1.51×)
- Le RKHS n'avait pas de noyau · Le do-calculus zerorait des colonnes · etc.

**Fractus corrige tout ça.** Voir `docs/CORRECTIONS.md` pour la liste complète avec traçabilité `file:line`.

---

## Concepts (tous présents, tous fonctionnels)

| Concept | Module | Validé par |
|---|---|---|
| Vortex 2-adique (valuation, ultramétrique) | `fractus-core/src/vortex.rs` (Rust) | Bug OMNI corrigé : `2^{+v₂}` → `2^{-v₂}` |
| Embedding fractal (Fourier Mandelbrot + char features) | `fractus/nn/embedding.py` | backward CHAQUE param |
| Attention linéaire causale multi-niveaux | `fractus/nn/attention.py` | vectorisée 17× + causalité testée |
| Oscillateurs Kuramoto RK4 bas-rang | `fractus/nn/phase_ode.py` | RK4 + stateless |
| MoE à routing von Mises / Farey | `fractus/nn/moe.py` | load-balance + top-k |
| Compression SIREN `sin(ω₀·)` | `fractus/nn/siren.py` | ω₀=30 Sitzmann, ratio mesuré |
| NOTEARS causal (DAG acyclique) | `fractus/causal/notears.py` | SHD=0 sur SCM non-linéaire |
| RKHS via Random Fourier Features | `fractus/causal/rkhs.py` | vrai noyau, pas projection nue |
| do-calculus de Pearl | `fractus/causal/do.py` | clamp à v, pas zeroring |
| Preuves (vérificateur exact sound) | `fractus/reasoning/proof.py` | Fermat/Wilson/GCD |
| Génération de nombres premiers | `fractus/reasoning/prime_generator.py` | **100% primalité** après 100 steps |
| Conjectures (falsification popperienne) | `fractus/reasoning/conjecture.py` | 10 templates, 6 stratégies |
| ACT (Adaptive Computation Time) | `fractus/reasoning/act.py` | Graves 2016 |
| Lyapunov (sous-système Kuramoto) | `fractus/stability/lyapunov.py` | vrai système dynamique |

---

## Installation

```bash
git clone https://github.com/AFKmoney/fractus.git
cd fractus

# Créer le venv (utiliser `py` sur Windows, pas python MSYS2)
py -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/macOS

# Dépendances Python
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only
pip install -r requirements.txt

# Module natif Rust (pont Python↔Rust)
maturin develop --release

# Tests (doivent tous passer)
pytest tests/ -q
```

**Prérequis** : Rust (`cargo`), Python 3.10+ avec `py` launcher.

---

## Démarrage rapide

```bash
# Démo 1 : transformer fractal apprend "hello world"
python scripts/demo_transformer.py

# Démo 2 : génération de nombres premiers (100% validité)
python scripts/demo_prime_reinforce.py

# Démo 3 : NOTEARS récupère un DAG causal
python scripts/demo_causal.py

# Démo 4 : tinyshakespeare (perplexité réelle)
python scripts/demo_shakespeare.py

# Démo complète : texte + preuves + causal
python scripts/demo_full.py
```

---

## Entraînement sur datasets HuggingFace

```bash
# Petit modèle sur tinyshakespeare (CPU, ~2 min)
python scripts/train_hf.py --preset cpu-small --dataset tinyshakespeare

# Modèle medium sur wikitext (GPU recommandé)
python scripts/train_hf.py --preset gpu-medium --dataset wikitext-2

# Configuration personnalisée
python scripts/train_hf.py --dataset HuggingFaceFW/fineweb --text-field text \
    --d-model 256 --n-blocks 6 --seq-len 128 --batch-size 16 --epochs 3
```

### Tailles de modèle et hardware requis

| Preset | Params | RAM min | Temps/epoch | Hardware |
|---|---|---|---|---|
| `cpu-tiny` | ~80k | 4 GB | ~2 min | CPU laptop |
| `cpu-small` | ~500k | 8 GB | ~30 min | CPU desktop |
| `gpu-small` | ~5M | 8 GB VRAM | ~5 min | GPU entry (RTX 3060) |
| `gpu-medium` | ~50M | 16 GB VRAM | ~30 min | GPU mid (RTX 4090) |
| `gpu-large` | ~300M | 40 GB VRAM | ~2h | A100 40GB |
| `gpu-1b` | ~1B | 80 GB VRAM | ~8h | A100 80GB / H100 |

> ⚠️ **Honnêteté** : le preset `gpu-1b` est fourni pour la complétude, mais il **nécessite un GPU datacenter** (A100 80GB ou H100). Il est **impossible** sur CPU ou GPU consumer. Ne le lancez pas sans le hardware approprié — ça OOM ou prendra des semaines. Le bottleneck principal est le Kuramoto RK4 (non encore vectorisé) et la SIREN (évaluation à chaque forward).

---

## Tests

```bash
pytest tests/ -q                    # 156 tests, tous passants
pytest tests/test_attention_vectorized.py -v   # équivalence vectorisée
```

---

## Documentation

- `docs/CORRECTIONS.md` — les 9 mensonges originaux corrigés (traçabilité file:line)
- `docs/superpowers/specs/2026-06-19-fractus-unified-design.md` — spec complet
- `docs/superpowers/plans/` — plans d'implémentation par couche (L0→L7)

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
├── fractus/metrics/          mesures honnêtes (compression, SHD, perplexité)
├── data/                     datasets (tinyshakespeare, SCM synthétiques)
├── tests/                    156 tests
├── scripts/                  8 démos + train_hf.py
└── docs/                     spec, plans, corrections
```

**Le Rust reste hors-graphe autodiff** (calcul exact, vérification, précalcul). La forward/backward est **PyTorch pur** (autodiff natif).

---

## Limites connues (honnêtes)

1. **Kuramoto RK4 non vectorisé** : 4 sous-steps Python, ~116ms/batch. Future work : vectoriser comme l'attention.
2. **SIREN compression faible** : 1.51× sur poids denses (inhérent — les SIREN compressent les fonctions lisses, pas le bruit).
3. **ProofGenerator REINFORCE** : la tâche FNN originale (converger à 1e-3) est inatteignable. La tâche redéfinie (produire des premiers) marche à 100%.
4. **Causal non-linéaire fort** : NOTEARS linéaire est robuste à la non-linéarité modérée (tanh), mais pas aux relations fortement non-linéaires.

---

## Licence

MIT. Voir `LICENSE`.

---

## Crédits

Concepts originaux : **Philippe-Antoine Robert** (FNN v5.0, OMNI-FRACTAL).
Réfonte honnête : fractus project, 2026.
