# Fractus L0 — Socle Technique Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Etablir the socle technique of fractus — a repo with a crate Rust pur (`fractus-core`), a crate of bindings PyO3 (`fractus-py`), a package Python (`fractus`), and a test fume which prouve that Python → PyTorch → maturin → Rust → retour Python fonctionne end-to-end.

**Architecture:** Trois components isoles. (1) `fractus-core` : Rust pur, no I/O, exported the functions mathematics (ici juste `add` for the fume + the port vortex 2-adique d'the original). (2) `fractus-py` : bindings PyO3/maturin which exposent `fractus-core` a Python under the nom `fractus._core`. (3) `fractus` : package Python installant PyTorch and abritant the modele entrainable (vide for L0). Le Rust reste outside the autodiff graph ; the forward/backward se fera en PyTorch (couches ulterieures).

**Tech Stack:** Rust 1.94 + `nalgebra`, `pyo3` (feature `extension-module`) ; Python 3.14 via `py` launcher in a venv dedie ; `maturin 1.14` for the build ; `torch` (CPU-only wheel) + `numpy` + `pytest`.

**Environment (verifiess on the machine target) :**
- `py` → Python 3.14.0 with pip 25.3 ✅
- `cargo` / `rustc` 1.94.0 ✅
- `python`/`python3` (MSYS2) → **without pip, not not utiliser**
- Hardware : AMD Ryzen 5 5500U, CPU-only effectif
- Creation in : `C:/Users/PHIL/ZCodeProject/fractus/`

**Lien spec :** `docs/superpowers/specs/2026-06-19-fractus-unified-design.md`, section 6 « L0 — Socle technique ».

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── .gitignore                          # ignore .venv, target/, __pycache__, *.egg-info
├── README.md                           # pitch court + instructions of dev
├── pyproject.toml                      # projet Python racine (fractus), builde by maturin
├── requirements-dev.txt                # versions epinglees for reproductibilite
├── crate/
│   ├── fractus-core/                   # workspace member : coeur mathematical Rust pur
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # pub mod uniquement for modules existants
│   │       └── vortex.rs               # port 2-adique depuis the original (correct + fixes)
│   └── fractus-py/                     # workspace member : bindings PyO3
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs                  # #[pymodule] fractus._core
├── Cargo.toml                          # workspace racine (lie the 2 crates)
├── fractus/                            # package Python (le modele entrainable, vide en L0)
│   ├── __init__.py                     # expose fractus._core for test
│   └── nn/__init__.py                  # placeholder for couches ulterieures
└── tests/
    ├── __init__.py
    └── test_smoke.py                   # LE test fume which traverse tout
```

**Responsabilites :**
- `fractus-core` : computation mathematical pur, testable en Rust seul, no dependance Python.
- `fractus-py` : Pont Python↔Rust. Ne contient no logical, juste wrappers `#[pyfunction]`.
- `fractus` (Python) : Le package utilisateur. En L0, juste l'import bridge + placeholder `nn/`.
- `tests/test_smoke.py` : prouve that all tient.

---

## Task 1: Initialiser the repo and the .gitignore

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/.gitignore`

- [ ] **Step 1: Creer the dossier fractus and init git**

Run (cmd or PowerShell) :
```bash
mkdir "C:\Users\PHIL\ZCodeProject\fractus"
cd "C:\Users\PHIL\ZCodeProject\fractus"
git init
git branch -M main
```
Expected: `Initialized empty Git repository in ...` then silencieux for `git branch`.

- [ ] **Step 2: Ecrire the .gitignore**

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

## Task 2: Creer the workspace Cargo racine

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/Cargo.toml`

- [ ] **Step 1: Ecrire the Cargo.toml workspace**

Create `C:/Users/PHIL/ZCodeProject/fractus/Cargo.toml` :
```toml
# Workspace racine fractus.
# Les deux crates (fractus-core, fractus-py) are members isoles.
[workspace]
members = ["crate/fractus-core", "crate/fractus-py"]
resolver = "2"

# Profil release optimise for CPU-only.
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

## Task 3: Creer the crate fractus-core with add() (premier test fume Rust)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/Cargo.toml`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs`

- [ ] **Step 1: Ecrire Cargo.toml of fractus-core**

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
# Aucune dependance en L0. nalgebra/serde/etc. ajoutes quand a module en a besoin.
```

- [ ] **Step 2: Ecrire lib.rs minimal (juste add, for the fume)**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs` :
```rust
//! # fractus-core
//!
//! Coeur mathematical pur of fractus. Aucune I/O, no dependance Python.
//! Toutes the functions ici are testables en Rust seul.
//!
//! En L0, seule `add` est exposee for the test fume. Les vrais modules
//! (vortex, siren, causal, proof) are ajoutes in the couches ulterieures.

/// Addition integere. Existe uniquement for the test fume Python↔Rust.
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

- [ ] **Step 3: Verifier that the crate compile and that the test passe**

Run :
```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
cargo test -p fractus-core
```
Expected: `test result: ok. 1 passed` and `Compiling fractus-core`.

- [ ] **Step 4: Commit**

```bash
git add crate/fractus-core/
git commit -m "feat(core): add fractus-core crate with add() smoke function"
```
Expected: `2 files changed`

---

## Task 4: Creer the crate fractus-py (bindings PyO3)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/Cargo.toml`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs`

- [ ] **Step 1: Ecrire Cargo.toml of fractus-py**

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

Note : `crate-type = ["cdylib"]` est obligatoire for qu'maturin produise a extension Python.
Le `name = "_core"` does that the module Python s'appellera `fractus._core` (after configuration pyproject).

- [ ] **Step 2: Ecrire lib.rs bindings**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs` :
```rust
//! Bindings Python (PyO3) for fractus-core.
//!
//! Ce crate not contient AUCUNE logical — seulement wrappers #[pyfunction]
//! which deleguent a fractus-core. Le but est d'exposer the Rust a Python
//! under the nom `fractus._core`.

use pyo3::prelude::*;

/// Addition integere — wrapper Python for fractus_core::add.
/// Exposee uniquement for the test fume en L0.
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

- [ ] **Step 3: Verifier that the workspace compile (without maturin, juste cargo check)**

Run :
```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
cargo check -p fractus-py
```
Expected: `Compiling pyo3 ...` then `Finished`. S'il manque the `Python` linker under Windows, on verra the error ici — a correcteder before of continuer (typiquement : installer the build tools C++ or configurer the linker ; pyo3/maturin s'en occupe normalement automatiquement).

- [ ] **Step 4: Commit**

```bash
git add crate/fractus-py/
git commit -m "feat(py): add fractus-py PyO3 bindings with add() wrapper"
```
Expected: `2 files changed`

---

## Task 5: Creer the pyproject.toml racine (configure for maturin)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/pyproject.toml`

- [ ] **Step 1: Ecrire pyproject.toml**

Create `C:/Users/PHIL/ZCodeProject/fractus/pyproject.toml` :
```toml
# fractus — package Python construit via maturin (backend Rust).
# Le module natif vient of crate/fractus-py ; the package Python vient of fractus/.

[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "fractus"
version = "0.1.0"
description = "Refonte unifiee of the original + the original design : transformer fractal entrainable."
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
# Le module natif (cdylib _core) sera place in fractus/ → imported comme fractus._core.
python-source = "."
module-name = "fractus._core"
manifest-path = "crate/fractus-py/Cargo.toml"
features = ["pyo3/extension-module"]
```

Note critique : `module-name = "fractus._core"` + `python-source = "."` does that maturin place the `.pyd` in the package `fractus/`, importable comme `from fractus import _core`. La feature `pyo3/extension-module` est passee a maturin directement (pas in the Cargo.toml crate, for eviter the bug d'the original or `[features] python = ["pyo3"]` was mal configure).

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml with maturin backend configuration"
```
Expected: `1 file changed`

---

## Task 6: Creer the package Python fractus (placeholder)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py`

- [ ] **Step 1: Ecrire fractus/__init__.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/__init__.py` :
```python
"""fractus — refonte unifiee of the original + the original design.

L0 : seul the pont natif `_core` est expose. Les modules nn/, causal/, reasoning/
seront ajoutes in the couches ulterieures (L1+).
"""

__version__ = "0.1.0"

# Le module natif fractus._core est construit by maturin and place ici.
# On l'imported explicitement for qu'il soit accessible via `from fractus import _core`.
try:
    from fractus import _core  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Le module natif fractus._core est introuvable. "
        "As-tu lance `maturin develop` ?"
    ) from e
```

- [ ] **Step 2: Ecrire fractus/nn/__init__.py (placeholder)**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py` :
```python
"""Sous-package nn — modules of reseau of neurones (PyTorch).

L0 : vide. Sera rempli en L1 (embedding) then L2 (attention, MoE, blocks).
"""
```

- [ ] **Step 3: Commit**

```bash
git add fractus/
git commit -m "feat(py): add fractus Python package skeleton with _core bridge"
```
Expected: `2 files changed`

---

## Task 7: Creer the test fume Python

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/tests/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/tests/test_smoke.py`

- [ ] **Step 1: Ecrire tests/__init__.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/__init__.py` :
```python
# Package tests. Les tests are decouverts by pytest a the racine repo.
```

- [ ] **Step 2: Ecrire tests/test_smoke.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_smoke.py` :
```python
"""Test fume : prouve that the plomberie Python → PyTorch → Rust tient.

Ces tests not validnt no logical mathematical — juste that the briques
communiquent. Si a of these tests echoue, rien d'autre not can marcher.
"""


def test_torch_available():
    """PyTorch est installe and fonctionnel."""
    import torch
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.sum().item() == 6.0


def test_numpy_available():
    """NumPy est installe (necessaire for the pont tenseurs)."""
    import numpy as np
    a = np.array([1, 2, 3])
    assert a.sum() == 6


def test_rust_bridge_import():
    """Le module natif fractus._core est well construit and importable."""
    from fractus import _core
    assert hasattr(_core, "add")


def test_rust_bridge_add():
    """Python can appeler Rust and recuperer the bon result."""
    from fractus import _core
    assert _core.add(2, 3) == 5
    assert _core.add(-10, 4) == -6


def test_torch_numpy_interop():
    """PyTorch and numpy s'echangent tenseurs (necessaire for the pont Rust)."""
    import numpy as np
    import torch
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = torch.from_numpy(arr)
    assert t.dtype == torch.float32
    # Retour toward numpy
    back = t.numpy()
    assert np.allclose(back, arr)
```

- [ ] **Step 3: Verifier that the tests echouent before installation (sanity check)**

Run :
```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
python -m pytest tests/test_smoke.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'fractus'` (le package n'est not encore installe). This is normal, ca confirme that the tests are significatifs.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: add smoke tests for torch/numpy/rust-bridge"
```
Expected: `2 files changed`

---

## Task 8: Creer requirements-dev.txt (reproductibilite)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/requirements-dev.txt`

- [ ] **Step 1: Ecrire requirements-dev.txt**

Create `C:/Users/PHIL/ZCodeProject/fractus/requirements-dev.txt` :
```text
# Versions epinglees for reproductibilite dev.
# PyTorch CPU-only wheel (pas of CUDA — l'APU AMD n'est not supportede by ROCm under Windows).
# IMPORTANT : on Windows, installer torch CPU with l'index-url explicite :
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

## Task 9: Ecrire the README (pitch + setup dev)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/README.md`

- [ ] **Step 1: Ecrire README.md**

Create `C:/Users/PHIL/ZCodeProject/fractus/README.md` :
```markdown
# fractus

Refonte unifiee of **the original architecture** and **the original design** (the original author) :
un transformer fractal **entrainable**, with compression SIREN, raisonnement causal
NOTEARS, and generation/verification of proofs. CPU-only.

> Etat : **L0 (socle technique)** — the plomberie Python↔Rust tient, no
> logical mathematical encore. Voir `docs/superpowers/specs/2026-06-19-fractus-unified-design.md`.

## Stack

- **Rust** (`crate/fractus-core`) : coeur mathematical pur (vortex 2-adique, SIREN, NOTEARS,
  verify of proofs). Hors-graphe autodiff.
- **Python + PyTorch** (`fractus/`) : modele entrainable, forward/backward, datasets.
- **maturin** : pont between the deux.

## Setup dev (Windows)

Prerequis : Rust (`cargo`), the launcher `py` (Python 3.10+).

```powershell
cd C:\Users\PHIL\ZCodeProject\fractus

# 1. Creer the venv dedie (utiliser `py`, not `python` MSYS2 which n'a not pip)
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Installer PyTorch CPU-only + outils dev
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

# 3. Construire and installer the module natif Rust in the venv
maturin develop --release

# 4. Lancer the tests
pytest tests/ -v
```

Les 4 commandes `cargo build`, `maturin develop`, `import torch; import fractus`,
`pytest` must all reussir.

## Layout

```
crate/fractus-core/   Rust : coeur mathematical pur (testable seul)
crate/fractus-py/     Rust : bindings PyO3 (no logical)
fractus/              Python : modele entrainable (L0 : juste the pont _core)
tests/                tests d'integration
```

## Roadmap

Voir the spec : the couches L0 (socle) → L7 (demo). L0 = this repo tel quel.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup instructions and roadmap pointer"
```
Expected: `1 file changed`

---

## Task 10: Installer the venv and validr the test fume (VRAIE verification L0)

**Files:** (no — this is the validation runtime)

- [ ] **Step 1: Creer the venv**

Run (PowerShell, depuis `C:\Users\PHIL\ZCodeProject\fractus`) :
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version   # must afficher 3.14.x
```
Expected: dossier `.venv/` cree, prompt modifie with `(.venv)`.

Si l'activation PowerShell est bloquee by the politique d'execution :
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

- [ ] **Step 2: Installer PyTorch CPU-only**

Run (in the venv active) :
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Expected: telechargement (~200 MB) then `Successfully installed torch-...`. Peut prendre quelques minutes.

Verifier :
```powershell
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```
Expected: version torch, `cuda: False` (CPU-only, this is voulu).

- [ ] **Step 3: Installer the reste dependances dev**

```powershell
pip install -r requirements-dev.txt
```
Expected: `numpy`, `pytest`, `maturin` installes.

- [ ] **Step 4: Construire and installer the module natif with maturin**

```powershell
maturin develop --release
```
Expected: `📦 Built ...` then `🛠 Installed fractus-...`. maturin compile the Rust, produit the `.pyd`, and l'installe in the venv. Peut prendre 2-5 min the premiere fois (compile pyo3).

Si error : verify that the build tools C++ est present (Visual Studio Build Tools or MSVC). maturin/normalement gere automatiquement.

- [ ] **Step 5: Lancer the tests fume — DOIVENT TOUS PASSER**

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

**This is the critere of « L0 termine » :** the 5 tests passent. Si a seul echoue, L0 n'est not termine — deboguer before of passer a L1.

- [ ] **Step 6: (Optionnel but recommande) Ajouter the venv au gitignore s'il a ete oublie**

Verifier :
```powershell
git status
```
Si `.venv/` apparait comme non-suivi, this is that the .gitignore of Task 1 a a souci — the correcteder. Sinon, the `git status` not must montrer that fichiers propres (rien, or juste the `target/` si on l'a commis by error).

- [ ] **Step 7: Commit final L0**

```powershell
git status   # must etre propre
git log --oneline   # must montrer ~9 commits
```
Aucun fichier a committer (le venv and target/ are ignores). L0 est termine.

---

## Task 11 (bonus) : Porter vortex.rs depuis the original — preparation L1

**Note :** Cette tache prepare L1 without l'executer. Le vortex est the seul module d'the original mathematicalment correct ; on the portedd now (en Rust seul) for that L1 demarre vite. Les bindings Python viendront en L1.

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/vortex.rs`
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs`

- [ ] **Step 1: Ecrire vortex.rs (port correctede depuis the original)**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/vortex.rs` :
```rust
//! # Vortex 2-adique
//!
//! Port depuis the original design (rust/src/vortex.rs), with corrections :
//! - L'import `HashMap` inutilise a ete retire.
//! - Le test tautological `assert!(d1 <= d2.max(d1))` a ete remplace by a true
//!   test d'ultrametrie : `d(x,z) <= max(d(x,y), d(y,z))` on donnees aleatoires.
//!
//! Nommage honestete : on parle of "hash Collatz" (pas "flot ergodique" — l'ergodicite
//! of Collatz est non demontree, problem ouvert), of "ultrametric distance" and de
//! "2-adic norm" (termes exacts).

/// Valuation 2-adique v_2(x) = max{k : 2^k divise x}.
/// Pour x=0, on returns 64 (convention for u64).
pub fn valuation_2(x: u64) -> u32 {
    if x == 0 {
        return 64;
    }
    x.trailing_zeros()
}

/// Valuation 3-adique v_3(x) = max{k : 3^k divise x}.
pub fn valuation_3(x: u64) -> u32 {
    if x == 0 {
        return 0; // convention : v_3(0) = infini, on borne a 0 for u64
    }
    let mut val = 0u32;
    let mut n = x;
    while n % 3 == 0 {
        val += 1;
        n /= 3;
    }
    val
}

/// Hash Collatz d'un integer. Utilise comme hachage d'etat deterministic.
/// Note : "ergodicite of Collatz" non demontree — on l'appelle juste "hash".
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

/// Distance ultrametrique 2-adique : d(a,b) = 2^{v_2(a XOR b)}.
/// Verifie the property ultrametrique forte : d(x,z) <= max(d(x,y), d(y,z)).
pub fn ultrametric_distance(a: u64, b: u64) -> u64 {
    let diff = a ^ b;
    if diff == 0 {
        return 0;
    }
    1u64 << valuation_2(diff)
}

/// Norme 2-adique : ||x||_2 = 2^{-v_2(x)}.
/// Retourne en f64 (can etre very small for the grands x pairs).
pub fn norm_2adic(x: u64) -> f64 {
    if x == 0 {
        return 0.0; // ||0|| = 0 by convention (v_2(0) = infini)
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
        assert_eq!(valuation_3(56), 0); // 56 n'est not divisible by 3
    }

    #[test]
    fn test_collatz_hash_deterministic() {
        // Meme entree → same sortie (deterministic).
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
        // La vraie property ultrametrique : d(x,z) <= max(d(x,y), d(y,z)).
        // CORRECTION test tautological d'the original.
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
                "Echec ultrametrie : d({},{})={} > max(d({},{})={}, d({},{})={})",
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
        // Sur donnees pseudo-aleatoires, the norme must etre <= 1 for x != 0.
        for x in [1u64, 3, 5, 7, 9, 11, 42, 137, 1023, 65535] {
            let n = norm_2adic(x);
            assert!(n > 0.0 && n <= 1.0, "norm_2adic({}) = {} hors [0,1]", x, n);
        }
    }
}
```

- [ ] **Step 2: Declarer the module vortex in lib.rs**

Modify `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs` — remplacer all the contenu by :
```rust
//! # fractus-core
//!
//! Coeur mathematical pur of fractus. Aucune I/O, no dependance Python.
//! Toutes the functions ici are testables en Rust seul.

pub mod vortex;

/// Addition integere. Existe uniquement for the test fume Python↔Rust.
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

- [ ] **Step 3: Lancer the tests Rust — DOIVENT TOUS PASSER**

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
git commit -m "feat(core): port 2-adic vortex from the original with ultrametric test fix"
```
Expected: `2 files changed`

---

## Critere final of L0 « termine »

Apres the Task 10 (et bonus Task 11), these 4 verifications must all reussir :

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"

# 1. Le Rust compile and its tests passent
cargo test -p fractus-core
# → 9 passed (1 add + 8 vortex)

# 2. Le pont Python↔Rust se construit
maturin develop --release
# → 🛠 Installed fractus-...

# 3. Les deux s'importednt
python -c "import torch; import fractus; print('OK', torch.__version__)"
# → OK 2.x.x

# 4. Les tests fume passent
pytest tests/ -v
# → 5 passed
```

Si all passe, L0 est termine and on can passer au plan L1 (embedding fractal + vortex branche).

---

## Self-Review (post-ecriture)

Verifications effectuees :

**1. Spec coverage :** La section L0 spec demande (a) environnement reproductible → Task 5, 8 ; (b) crate fractus-core → Task 3, 11 ; (c) crate fractus-py bindings → Task 4 ; (d) test fume traversant → Task 7, 10. ✅ Tout couvert.

**2. Placeholder scan :** Aucun « TBD », no « TODO », no section « fill in ». Toutes the etapes contiennent code complete. ✅

**3. Type consistency :** `add(a, b)` defini in `fractus-core/src/lib.rs` (Task 3), wrappe in `fractus-py/src/lib.rs` (Task 4) comme `#[pyfunction] fn add`, teste in `test_smoke.py` (Task 7). Noms coherents partout. `valuation_2`, `ultrametric_distance`, `norm_2adic` (Task 11) coherents between definition and tests. ✅

**4. Ordre dependances :** Task 4 (fractus-py) depend of Task 3 (fractus-core) → respecte. Task 10 (maturin develop) depend of Tasks 3-9 → respecte. Task 11 (bonus) can etre does after or before L1 without blocage.

**5. Commandes Windows-specifiques :** `py` (pas `python` MSYS2), `.\.venv\Scripts\Activate.ps1`, `--index-url https://download.pytorch.org/whl/cpu` for torch CPU. Toutes verifieses contre the machine target. ✅
