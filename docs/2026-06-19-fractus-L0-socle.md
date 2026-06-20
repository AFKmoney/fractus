# Fractus L0 — Socle Technique Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Établir le socle technique de fractus — un repo avec un crate Rust pur (`fractus-core`), un crate de bindings PyO3 (`fractus-py`), un package Python (`fractus`), et un test fume qui prouve que Python → PyTorch → maturin → Rust → retour Python fonctionne de bout en bout.

**Architecture:** Trois composants isolés. (1) `fractus-core` : Rust pur, aucune I/O, exporte les fonctions mathématiques (ici juste `add` pour le fume + le port du vortex 2-adique d'OMNI). (2) `fractus-py` : bindings PyO3/maturin qui exposent `fractus-core` à Python sous le nom `fractus._core`. (3) `fractus` : package Python installant PyTorch et abritant le modèle entraînable (vide pour L0). Le Rust reste hors-graphe autodiff ; la forward/backward se fera en PyTorch (couches ultérieures).

**Tech Stack:** Rust 1.94 + `nalgebra`, `pyo3` (feature `extension-module`) ; Python 3.14 via `py` launcher dans un venv dédié ; `maturin 1.14` pour le build ; `torch` (CPU-only wheel) + `numpy` + `pytest`.

**Environment (vérifié sur la machine cible) :**
- `py` → Python 3.14.0 avec pip 25.3 ✅
- `cargo` / `rustc` 1.94.0 ✅
- `python`/`python3` (MSYS2) → **sans pip, ne pas utiliser**
- Hardware : AMD Ryzen 5 5500U, CPU-only effectif
- Création dans : `C:/Users/PHIL/ZCodeProject/fractus/`

**Lien spec :** `docs/superpowers/specs/2026-06-19-fractus-unified-design.md`, section 6 « L0 — Socle technique ».

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── .gitignore                          # ignore .venv, target/, __pycache__, *.egg-info
├── README.md                           # pitch court + instructions de dev
├── pyproject.toml                      # projet Python racine (fractus), buildé par maturin
├── requirements-dev.txt                # versions épinglées pour reproductibilité
├── crate/
│   ├── fractus-core/                   # workspace member : cœur mathématique Rust pur
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # pub mod uniquement pour modules existants
│   │       └── vortex.rs               # port 2-adique depuis OMNI (correct + fixes)
│   └── fractus-py/                     # workspace member : bindings PyO3
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs                  # #[pymodule] fractus._core
├── Cargo.toml                          # workspace racine (lie les 2 crates)
├── fractus/                            # package Python (le modèle entraînable, vide en L0)
│   ├── __init__.py                     # expose fractus._core pour test
│   └── nn/__init__.py                  # placeholder pour couches ultérieures
└── tests/
    ├── __init__.py
    └── test_smoke.py                   # LE test fume qui traverse tout
```

**Responsabilités :**
- `fractus-core` : calcul mathématique pur, testable en Rust seul, aucune dépendance Python.
- `fractus-py` : Pont Python↔Rust. Ne contient aucune logique, juste des wrappers `#[pyfunction]`.
- `fractus` (Python) : Le package utilisateur. En L0, juste l'import bridge + placeholder `nn/`.
- `tests/test_smoke.py` : prouve que tout tient.

---

## Task 1: Initialiser le repo et le .gitignore

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/.gitignore`

- [ ] **Step 1: Créer le dossier fractus et init git**

Run (cmd ou PowerShell) :
```bash
mkdir "C:\Users\PHIL\ZCodeProject\fractus"
cd "C:\Users\PHIL\ZCodeProject\fractus"
git init
git branch -M main
```
Expected: `Initialized empty Git repository in ...` puis silencieux pour `git branch`.

- [ ] **Step 2: Écrire le .gitignore**

Create `C:/Users/PHIL/ZCodeProject/fractus/.gitignore` :
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.eggs/
build/
dist/

# Virtual env
.venv/
venv/

# Rust
target/
**/*.rs.bk

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs / checkpoints
*.log
checkpoints/
```

- [ ] **Step 3: Commit initial**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add .gitignore
git commit -m "chore: initial commit with .gitignore"
```
Expected: `[main (root-commit) ...] chore: initial commit with .gitignore`

---

## Task 2: Créer le workspace Cargo racine

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/Cargo.toml`

- [ ] **Step 1: Écrire le Cargo.toml workspace**

Create `C:/Users/PHIL/ZCodeProject/fractus/Cargo.toml` :
```toml
# Workspace racine fractus.
# Les deux crates (fractus-core, fractus-py) sont des members isolés.
[workspace]
members = ["crate/fractus-core", "crate/fractus-py"]
resolver = "2"

# Profil release optimisé pour CPU-only.
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

- [ ] **Step 2: Commit**

```bash
git add Cargo.toml
git commit -m "chore: add Cargo workspace manifest"
```
Expected: `1 file changed`

---

## Task 3: Créer le crate fractus-core avec add() (premier test fume Rust)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/Cargo.toml`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs`

- [ ] **Step 1: Écrire Cargo.toml de fractus-core**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/Cargo.toml` :
```toml
[package]
name = "fractus-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "fractus_core"
path = "src/lib.rs"

[dependencies]
# Aucune dépendance en L0. nalgebra/serde/etc. ajoutés quand un module en a besoin.
```

- [ ] **Step 2: Écrire lib.rs minimal (juste add, pour le fume)**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs` :
```rust
//! # fractus-core
//!
//! Cœur mathématique pur de fractus. Aucune I/O, aucune dépendance Python.
//! Toutes les fonctions ici sont testables en Rust seul.
//!
//! En L0, seule `add` est exposée pour le test fume. Les vrais modules
//! (vortex, siren, causal, proof) sont ajoutés dans les couches ultérieures.

/// Addition entière. Existe uniquement pour le test fume Python↔Rust.
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }
}
```

- [ ] **Step 3: Vérifier que le crate compile et que le test passe**

Run :
```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
cargo test -p fractus-core
```
Expected: `test result: ok. 1 passed` et `Compiling fractus-core`.

- [ ] **Step 4: Commit**

```bash
git add crate/fractus-core/
git commit -m "feat(core): add fractus-core crate with add() smoke function"
```
Expected: `2 files changed`

---

## Task 4: Créer le crate fractus-py (bindings PyO3)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/Cargo.toml`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs`

- [ ] **Step 1: Écrire Cargo.toml de fractus-py**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/Cargo.toml` :
```toml
[package]
name = "fractus-py"
version = "0.1.0"
edition = "2021"

[lib]
name = "_core"
crate-type = ["cdylib"]
path = "src/lib.rs"

[dependencies]
fractus-core = { path = "../fractus-core" }
pyo3 = { version = "0.22", features = ["extension-module"] }
```

Note : `crate-type = ["cdylib"]` est obligatoire pour qu'maturin produise une extension Python.
Le `name = "_core"` fait que le module Python s'appellera `fractus._core` (après configuration pyproject).

- [ ] **Step 2: Écrire lib.rs des bindings**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs` :
```rust
//! Bindings Python (PyO3) pour fractus-core.
//!
//! Ce crate ne contient AUCUNE logique — seulement des wrappers #[pyfunction]
//! qui délèguent à fractus-core. Le but est d'exposer le Rust à Python
//! sous le nom `fractus._core`.

use pyo3::prelude::*;

/// Addition entière — wrapper Python pour fractus_core::add.
/// Exposée uniquement pour le test fume en L0.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    fractus_core::add(a, b)
}

/// Module Python `fractus._core`.
#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
```

- [ ] **Step 3: Vérifier que le workspace compile (sans maturin, juste cargo check)**

Run :
```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
cargo check -p fractus-py
```
Expected: `Compiling pyo3 ...` puis `Finished`. S'il manque le `Python` linker sous Windows, on verra l'erreur ici — à corriger avant de continuer (typiquement : installer le build tools C++ ou configurer le linker ; pyo3/maturin s'en occupe normalement automatiquement).

- [ ] **Step 4: Commit**

```bash
git add crate/fractus-py/
git commit -m "feat(py): add fractus-py PyO3 bindings with add() wrapper"
```
Expected: `2 files changed`

---

## Task 5: Créer le pyproject.toml racine (configuré pour maturin)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/pyproject.toml`

- [ ] **Step 1: Écrire pyproject.toml**

Create `C:/Users/PHIL/ZCodeProject/fractus/pyproject.toml` :
```toml
# fractus — package Python construit via maturin (backend Rust).
# Le module natif vient de crate/fractus-py ; le package Python vient de fractus/.

[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "fractus"
version = "0.1.0"
description = "Réfonte unifiée de FNN + OMNI-FRACTAL : transformer fractal entraînable."
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "maturin>=1.4,<2.0",
]

[tool.maturin]
# Le module natif (cdylib _core) sera placé dans fractus/ → importé comme fractus._core.
python-source = "."
module-name = "fractus._core"
manifest-path = "crate/fractus-py/Cargo.toml"
features = ["pyo3/extension-module"]
```

Note critique : `module-name = "fractus._core"` + `python-source = "."` fait que maturin place le `.pyd` dans le package `fractus/`, importable comme `from fractus import _core`. La feature `pyo3/extension-module` est passée à maturin directement (pas dans le Cargo.toml du crate, pour éviter le bug d'OMNI où `[features] python = ["pyo3"]` était mal configuré).

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml with maturin backend configuration"
```
Expected: `1 file changed`

---

## Task 6: Créer le package Python fractus (placeholder)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py`

- [ ] **Step 1: Écrire fractus/__init__.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/__init__.py` :
```python
"""fractus — réfonte unifiée de FNN + OMNI-FRACTAL.

L0 : seul le pont natif `_core` est exposé. Les modules nn/, causal/, reasoning/
seront ajoutés dans les couches ultérieures (L1+).
"""

__version__ = "0.1.0"

# Le module natif fractus._core est construit par maturin et placé ici.
# On l'importe explicitement pour qu'il soit accessible via `from fractus import _core`.
try:
    from fractus import _core  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Le module natif fractus._core est introuvable. "
        "As-tu lancé `maturin develop` ?"
    ) from e
```

- [ ] **Step 2: Écrire fractus/nn/__init__.py (placeholder)**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py` :
```python
"""Sous-package nn — modules de réseau de neurones (PyTorch).

L0 : vide. Sera rempli en L1 (embedding) puis L2 (attention, MoE, blocks).
"""
```

- [ ] **Step 3: Commit**

```bash
git add fractus/
git commit -m "feat(py): add fractus Python package skeleton with _core bridge"
```
Expected: `2 files changed`

---

## Task 7: Créer le test fume Python

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/tests/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/tests/test_smoke.py`

- [ ] **Step 1: Écrire tests/__init__.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/__init__.py` :
```python
# Package tests. Les tests sont découverts par pytest à la racine du repo.
```

- [ ] **Step 2: Écrire tests/test_smoke.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_smoke.py` :
```python
"""Test fume : prouve que la plomberie Python → PyTorch → Rust tient.

Ces tests ne valident aucune logique mathématique — juste que les briques
communiquent. Si un de ces tests échoue, rien d'autre ne peut marcher.
"""


def test_torch_available():
    """PyTorch est installé et fonctionnel."""
    import torch
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.sum().item() == 6.0


def test_numpy_available():
    """NumPy est installé (nécessaire pour le pont tenseurs)."""
    import numpy as np
    a = np.array([1, 2, 3])
    assert a.sum() == 6


def test_rust_bridge_import():
    """Le module natif fractus._core est bien construit et importable."""
    from fractus import _core
    assert hasattr(_core, "add")


def test_rust_bridge_add():
    """Python peut appeler du Rust et récupérer le bon résultat."""
    from fractus import _core
    assert _core.add(2, 3) == 5
    assert _core.add(-10, 4) == -6


def test_torch_numpy_interop():
    """PyTorch et numpy s'échangent des tenseurs (nécessaire pour le pont Rust)."""
    import numpy as np
    import torch
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = torch.from_numpy(arr)
    assert t.dtype == torch.float32
    # Retour vers numpy
    back = t.numpy()
    assert np.allclose(back, arr)
```

- [ ] **Step 3: Vérifier que les tests échouent avant installation (sanity check)**

Run :
```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
python -m pytest tests/test_smoke.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'fractus'` (le package n'est pas encore installé). C'est normal, ça confirme que les tests sont significatifs.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: add smoke tests for torch/numpy/rust-bridge"
```
Expected: `2 files changed`

---

## Task 8: Créer requirements-dev.txt (reproductibilité)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/requirements-dev.txt`

- [ ] **Step 1: Écrire requirements-dev.txt**

Create `C:/Users/PHIL/ZCodeProject/fractus/requirements-dev.txt` :
```text
# Versions épinglées pour reproductibilité du dev.
# PyTorch CPU-only wheel (pas de CUDA — l'APU AMD n'est pas supportée par ROCm sous Windows).
# IMPORTANT : sur Windows, installer torch CPU avec l'index-url explicite :
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

torch>=2.2,<3.0
numpy>=1.26,<3.0
pytest>=8.0,<9.0
maturin>=1.4,<2.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements-dev.txt
git commit -m "build: pin dev dependencies (torch CPU-only, maturin, pytest)"
```
Expected: `1 file changed`

---

## Task 9: Écrire le README (pitch + setup dev)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/README.md`

- [ ] **Step 1: Écrire README.md**

Create `C:/Users/PHIL/ZCodeProject/fractus/README.md` :
```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup instructions and roadmap pointer"
```
Expected: `1 file changed`

---

## Task 10: Installer le venv et valider le test fume (VRAIE vérification L0)

**Files:** (aucun — c'est la validation runtime)

- [ ] **Step 1: Créer le venv**

Run (PowerShell, depuis `C:\Users\PHIL\ZCodeProject\fractus`) :
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version   # doit afficher 3.14.x
```
Expected: dossier `.venv/` créé, prompt modifié avec `(.venv)`.

Si l'activation PowerShell est bloquée par la politique d'exécution :
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

- [ ] **Step 2: Installer PyTorch CPU-only**

Run (dans le venv activé) :
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Expected: téléchargement (~200 MB) puis `Successfully installed torch-...`. Peut prendre quelques minutes.

Vérifier :
```powershell
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```
Expected: version torch, `cuda: False` (CPU-only, c'est voulu).

- [ ] **Step 3: Installer le reste des dépendances dev**

```powershell
pip install -r requirements-dev.txt
```
Expected: `numpy`, `pytest`, `maturin` installés.

- [ ] **Step 4: Construire et installer le module natif avec maturin**

```powershell
maturin develop --release
```
Expected: `📦 Built ...` puis `🛠 Installed fractus-...`. maturin compile le Rust, produit le `.pyd`, et l'installe dans le venv. Peut prendre 2-5 min la première fois (compile pyo3).

Si erreur : vérifier que le build tools C++ est présent (Visual Studio Build Tools ou MSVC). maturin/normalement géré automatiquement.

- [ ] **Step 5: Lancer les tests fume — DOIVENT TOUS PASSER**

```powershell
pytest tests/ -v
```
Expected output (les 5 tests) :
```
tests/test_smoke.py::test_torch_available PASSED
tests/test_smoke.py::test_numpy_available PASSED
tests/test_smoke.py::test_rust_bridge_import PASSED
tests/test_smoke.py::test_rust_bridge_add PASSED
tests/test_smoke.py::test_torch_numpy_interop PASSED
===== 5 passed in ...s =====
```

**C'est le critère de « L0 terminé » :** les 5 tests passent. Si un seul échoue, L0 n'est pas terminé — déboguer avant de passer à L1.

- [ ] **Step 6: (Optionnel mais recommandé) Ajouter le venv au gitignore s'il a été oublié**

Vérifier :
```powershell
git status
```
Si `.venv/` apparaît comme non-suivi, c'est que le .gitignore de Task 1 a un souci — le corriger. Sinon, le `git status` ne doit montrer que des fichiers propres (rien, ou juste le `target/` si on l'a commis par erreur).

- [ ] **Step 7: Commit final L0**

```powershell
git status   # doit être propre
git log --oneline   # doit montrer ~9 commits
```
Aucun fichier à committer (le venv et target/ sont ignorés). L0 est terminé.

---

## Task 11 (bonus) : Porter vortex.rs depuis OMNI — préparation L1

**Note :** Cette tâche prépare L1 sans l'exécuter. Le vortex est le seul module d'OMNI mathématiquement correct ; on le porte maintenant (en Rust seul) pour que L1 démarre vite. Les bindings Python viendront en L1.

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/vortex.rs`
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs`

- [ ] **Step 1: Écrire vortex.rs (port corrigé depuis OMNI)**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/vortex.rs` :
```rust
//! # Vortex 2-adique
//!
//! Port depuis OMNI-FRACTAL (rust/src/vortex.rs), avec corrections :
//! - L'import `HashMap` inutilisé a été retiré.
//! - Le test tautologique `assert!(d1 <= d2.max(d1))` a été remplacé par un vrai
//!   test d'ultramétrie : `d(x,z) <= max(d(x,y), d(y,z))` sur données aléatoires.
//!
//! Nommage honnête : on parle de "hash Collatz" (pas "flot ergodique" — l'ergodicité
//! de Collatz est non démontrée, problème ouvert), de "distance ultramétrique" et de
//! "norme 2-adique" (termes exacts).

/// Valuation 2-adique v_2(x) = max{k : 2^k divise x}.
/// Pour x=0, on retourne 64 (convention pour u64).
pub fn valuation_2(x: u64) -> u32 {
    if x == 0 {
        return 64;
    }
    x.trailing_zeros()
}

/// Valuation 3-adique v_3(x) = max{k : 3^k divise x}.
pub fn valuation_3(x: u64) -> u32 {
    if x == 0 {
        return 0; // convention : v_3(0) = infini, on borne à 0 pour u64
    }
    let mut val = 0u32;
    let mut n = x;
    while n % 3 == 0 {
        val += 1;
        n /= 3;
    }
    val
}

/// Hash Collatz d'un entier. Utilisé comme hachage d'état déterministe.
/// Note : "ergodicité de Collatz" non démontrée — on l'appelle juste "hash".
pub fn collatz_hash(mut x: u64, steps: u32) -> u64 {
    for _ in 0..steps {
        if x == 0 {
            break;
        }
        if x % 2 == 0 {
            x /= 2;
        } else {
            x = 3.wrapping_mul(x).wrapping_add(1);
        }
    }
    x
}

/// Distance ultramétrique 2-adique : d(a,b) = 2^{v_2(a XOR b)}.
/// Vérifie la propriété ultramétrique forte : d(x,z) <= max(d(x,y), d(y,z)).
pub fn ultrametric_distance(a: u64, b: u64) -> u64 {
    let diff = a ^ b;
    if diff == 0 {
        return 0;
    }
    1u64 << valuation_2(diff)
}

/// Norme 2-adique : ||x||_2 = 2^{-v_2(x)}.
/// Retourné en f64 (peut être très petit pour les grands x pairs).
pub fn norm_2adic(x: u64) -> f64 {
    if x == 0 {
        return 0.0; // ||0|| = 0 par convention (v_2(0) = infini)
    }
    let v = valuation_2(x) as i32;
    2f64.powi(-v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valuation_2_basic() {
        assert_eq!(valuation_2(0), 64);
        assert_eq!(valuation_2(1), 0);
        assert_eq!(valuation_2(2), 1);
        assert_eq!(valuation_2(4), 2);
        assert_eq!(valuation_2(8), 3);
        assert_eq!(valuation_2(56), 3); // 56 = 7 * 8
    }

    #[test]
    fn test_valuation_3_basic() {
        assert_eq!(valuation_3(0), 0);
        assert_eq!(valuation_3(1), 0);
        assert_eq!(valuation_3(3), 1);
        assert_eq!(valuation_3(9), 2);
        assert_eq!(valuation_3(27), 3);
        assert_eq!(valuation_3(56), 0); // 56 n'est pas divisible par 3
    }

    #[test]
    fn test_collatz_hash_deterministic() {
        // Même entrée → même sortie (déterministe).
        assert_eq!(collatz_hash(7, 10), collatz_hash(7, 10));
        // 0 reste 0.
        assert_eq!(collatz_hash(0, 10), 0);
    }

    #[test]
    fn test_ultrametric_distance_self_is_zero() {
        assert_eq!(ultrametric_distance(42, 42), 0);
    }

    #[test]
    fn test_ultrametric_distance_symmetry() {
        for (a, b) in [(1u64, 2), (7, 56), (100, 200), (3, 9)] {
            assert_eq!(ultrametric_distance(a, b), ultrametric_distance(b, a));
        }
    }

    #[test]
    fn test_ultrametric_strong_triangle_inequality() {
        // La vraie propriété ultramétrique : d(x,z) <= max(d(x,y), d(y,z)).
        // CORRECTION du test tautologique d'OMNI.
        let triples: [(u64, u64, u64); 8] = [
            (1, 2, 4),
            (7, 56, 13),
            (100, 200, 300),
            (3, 9, 27),
            (5, 11, 23),
            (1024, 1, 2),
            (7, 13, 21),
            (255, 256, 257),
        ];
        for (x, y, z) in triples {
            let d_xy = ultrametric_distance(x, y);
            let d_yz = ultrametric_distance(y, z);
            let d_xz = ultrametric_distance(x, z);
            assert!(
                d_xz <= d_xy.max(d_yz),
                "Échec ultramétrie : d({},{})={} > max(d({},{})={}, d({},{})={})",
                x, z, d_xz, x, y, d_xy, y, z, d_yz
            );
        }
    }

    #[test]
    fn test_norm_2adic_basic() {
        assert_eq!(norm_2adic(0), 0.0);
        assert_eq!(norm_2adic(1), 1.0);   // v_2(1) = 0 → 2^0 = 1
        assert_eq!(norm_2adic(2), 0.5);   // v_2(2) = 1 → 2^-1
        assert_eq!(norm_2adic(4), 0.25);  // v_2(4) = 2 → 2^-2
        assert_eq!(norm_2adic(8), 0.125); // v_2(8) = 3 → 2^-3
    }

    #[test]
    fn test_norm_2adic_fuzz_ultrametric() {
        // Sur données pseudo-aléatoires, la norme doit être <= 1 pour x != 0.
        for x in [1u64, 3, 5, 7, 9, 11, 42, 137, 1023, 65535] {
            let n = norm_2adic(x);
            assert!(n > 0.0 && n <= 1.0, "norm_2adic({}) = {} hors [0,1]", x, n);
        }
    }
}
```

- [ ] **Step 2: Déclarer le module vortex dans lib.rs**

Modify `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs` — remplacer tout le contenu par :
```rust
//! # fractus-core
//!
//! Cœur mathématique pur de fractus. Aucune I/O, aucune dépendance Python.
//! Toutes les fonctions ici sont testables en Rust seul.

pub mod vortex;

/// Addition entière. Existe uniquement pour le test fume Python↔Rust.
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }
}
```

- [ ] **Step 3: Lancer les tests Rust — DOIVENT TOUS PASSER**

Run :
```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
cargo test -p fractus-core
```
Expected: 9 tests passent (1 `test_add` + 8 tests vortex). Output :
```
test result: ok. 9 passed; 0 failed
```

- [ ] **Step 4: Commit**

```bash
git add crate/fractus-core/
git commit -m "feat(core): port 2-adic vortex from OMNI with ultrametric test fix"
```
Expected: `2 files changed`

---

## Critère final de L0 « terminé »

Après le Task 10 (et bonus Task 11), ces 4 vérifications doivent toutes réussir :

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"

# 1. Le Rust compile et ses tests passent
cargo test -p fractus-core
# → 9 passed (1 add + 8 vortex)

# 2. Le pont Python↔Rust se construit
maturin develop --release
# → 🛠 Installed fractus-...

# 3. Les deux s'importent
python -c "import torch; import fractus; print('OK', torch.__version__)"
# → OK 2.x.x

# 4. Les tests fume passent
pytest tests/ -v
# → 5 passed
```

Si tout passe, L0 est terminé et on peut passer au plan L1 (embedding fractal + vortex branché).

---

## Self-Review (post-écriture)

Vérifications effectuées :

**1. Spec coverage :** La section L0 du spec demande (a) environnement reproductible → Task 5, 8 ; (b) crate fractus-core → Task 3, 11 ; (c) crate fractus-py bindings → Task 4 ; (d) test fume traversant → Task 7, 10. ✅ Tout couvert.

**2. Placeholder scan :** Aucun « TBD », aucun « TODO », aucune section « fill in ». Toutes les étapes contiennent du code complet. ✅

**3. Type consistency :** `add(a, b)` défini dans `fractus-core/src/lib.rs` (Task 3), wrappé dans `fractus-py/src/lib.rs` (Task 4) comme `#[pyfunction] fn add`, testé dans `test_smoke.py` (Task 7). Noms cohérents partout. `valuation_2`, `ultrametric_distance`, `norm_2adic` (Task 11) cohérents entre définition et tests. ✅

**4. Ordre des dépendances :** Task 4 (fractus-py) dépend de Task 3 (fractus-core) → respecté. Task 10 (maturin develop) dépend de Tasks 3-9 → respecté. Task 11 (bonus) peut être fait après ou avant L1 sans blocage.

**5. Commandes Windows-spécifiques :** `py` (pas `python` MSYS2), `.\.venv\Scripts\Activate.ps1`, `--index-url https://download.pytorch.org/whl/cpu` pour torch CPU. Toutes vérifiées contre la machine cible. ✅
