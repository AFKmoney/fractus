# fractus

Réfonte unifiée de **FNN v5.0** et **OMNI-FRACTAL** (Philippe-Antoine Robert) :
un transformer fractal **entraînable**, avec compression SIREN, raisonnement causal
NOTEARS, et génération/vérification de preuves. CPU-only.

> État : **L0 (socle technique)** — la plomberie Python↔Rust tient, aucune
> logique mathématique encore. Voir `docs/superpowers/specs/2026-06-19-fractus-unified-design.md`.

## Stack

- **Rust** (`crate/fractus-core`) : cœur mathématique pur (vortex 2-adique, SIREN, NOTEARS,
  vérificateur de preuves). Hors-graphe autodiff.
- **Python + PyTorch** (`fractus/`) : modèle entraînable, forward/backward, datasets.
- **maturin** : pont entre les deux.

## Setup dev (Windows)

Prérequis : Rust (`cargo`), le launcher `py` (Python 3.10+).

```powershell
cd C:\Users\PHIL\ZCodeProject\fractus

# 1. Créer le venv dédié (utiliser `py`, pas `python` MSYS2 qui n'a pas pip)
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Installer PyTorch CPU-only + outils dev
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

# 3. Construire et installer le module natif Rust dans le venv
maturin develop --release

# 4. Lancer les tests
pytest tests/ -v
```

Les 4 commandes `cargo build`, `maturin develop`, `import torch; import fractus`,
`pytest` doivent toutes réussir.

## Layout

```
crate/fractus-core/   Rust : cœur mathématique pur (testable seul)
crate/fractus-py/     Rust : bindings PyO3 (aucune logique)
fractus/              Python : modèle entraînable (L0 : juste le pont _core)
tests/                tests d'intégration
```

## Roadmap

Voir le spec : les couches L0 (socle) → L7 (démo). L0 = ce repo tel quel.
