# Fractus L1 — Embedding fractal + vortex 2-adique branche Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Brancher le vortex 2-adique d'the original design sur le reseau de neurones for la premiere fois, en l'utilisant comme **conditionnement** d'un MLP entrainable PyTorch. Ajouter l'embedding de codepoint fractal (base de Fourier a decroissance Mandelbrot + 16 features morphologiques), tout en PyTorch pur for l'autodiff. Corriger deux errors des systems originaux : le vortex orphaned (never imported) et les « Mandelbrot frequencies » mal nommees.

**Architecture:** (1) `fractus-core` (Rust) expose `collatz_hash(token_id)` et `ultrametric_distance(a,b)` — computation exact, hors-graphe autodiff, appele depuis Python for precalculer le conditionnement. (2) `fractus/nn/embedding.py` (PyTorch) contient trois modules composes : `CharClassFeatures` (16 features morphologiques deterministes, portees de FNN), `MandelbrotFourierBasis` (base de Fourier a decroissance `(φ²)^{-k}`, nommee honnetement), et `FractalEmbedding` (assemblage final : les features morpho + la base de Fourier + les phases vortex-conditionnees sont projetees vers `d_model` par un `nn.Linear` entrainable). La forward est differentiable de bout en bout.

**Tech Stack:** Rust 1.94 + pyo3 0.29 (deja installe en L0) ; Python 3.14 + torch 2.12 CPU + numpy (deja installes en L0) ; pytest. Le module natif must etre reconstruit via `maturin develop` after modification du Rust.

**Lien spec :** `docs/superpowers/specs/2026-06-19-fractus-unified-design.md`, section « L1 — Embedding fractal + vortex 2-adique branche ». Decision cle validee : **option B** — le hash 2-adique exact (Rust, hors-graphe) conditionne un MLP entrainable (PyTorch, in le graphe).

**Prerequis :** L0 termine (repo `C:/Users/PHIL/ZCodeProject/fractus/` operationnel, venv `.venv` with torch + maturin installes, 9 tests Rust + 5 tests Python passent).

**Vocabulaire honnete applique in ce plan :**
- « Mandelbrot frequencies » → « Mandelbrot-decayed Fourier basis » (la decroissance `(φ²)^{-k}` est real et justifiee, but ce n'est pas l'ensemble de Mandelbrot).
- « Collatz ergodic flow » → « Collatz hash » (l'ergodicite de Collatz est non demontree, problem ouvert).
- On parle de « norme 2-adique » et « distance ultrametrique » (termes mathematics exacts).

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── crate/fractus-core/src/
│   ├── lib.rs                  # inchange (declare deja pub mod vortex)
│   └── vortex.rs               # MODIFY : expose aussi norm_2adic (deja defini, juste a binder)
├── crate/fractus-py/src/
│   └── lib.rs                  # MODIFY : ajoute wrappers #[pyfunction] for collatz_hash,
│                               #   ultrametric_distance, norm_2adic
├── fractus/nn/
│   ├── __init__.py             # MODIFY : exporte les classes publiques
│   ├── char_features.py        # CREATE : 16 features morphologiques (ported de FNN)
│   ├── fourier.py              # CREATE : base de Fourier a decroissance Mandelbrot
│   └── embedding.py            # CREATE : FractalEmbedding (assemblage + projection Linear)
└── tests/
    ├── test_vortex_bridge.py   # CREATE : tests du pont Python des functions 2-adiques
    └── test_embedding.py       # CREATE : tests de l'embedding fractal
```

**Responsabilites (un fichier = une responsabilite) :**
- `char_features.py` : uniquement les 16 features morphologiques (deterministe, without parameter).
- `fourier.py` : uniquement la base de Fourier a decroissance Mandelbrot (deterministe, without parameter).
- `embedding.py` : l'assemblage + la projection entrainable (le seul endroit with des `nn.Parameter`).
- `test_vortex_bridge.py` : verifies que les functions Rust sont bien appelables depuis Python et return les bonnes valeurs (pont operationnel).
- `test_embedding.py` : verifies formes, finitude, et surtout que `backward()` propage des gradients finis (le critere critique herite de FNN qui echouait la).

---

## Task 1: Exposer collatz_hash, ultrametric_distance, norm_2adic in les bindings Python

**Files:**
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs`

- [ ] **Step 1: Reecrire lib.rs with les nouveaux wrappers**

Replace the entire content of `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs` with :

```rust
//! Bindings Python (PyO3) for fractus-core.
//!
//! Ce crate ne contient AUCUNE logical — seulement des wrappers #[pyfunction]
//! qui deleguent a fractus-core. Le but est d'exposer le Rust a Python
//! under le nom `fractus._core`.

use pyo3::prelude::*;

/// Addition entiere — wrapper Python for fractus_core::add.
/// Exposee uniquement for le test fume.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    fractus_core::add(a, b)
}

/// Hash Collatz d'un token id. Wrapper for fractus_core::vortex::collatz_hash.
/// Utilise comme conditionnement deterministe (hors-graphe autodiff) for
/// l'embedding fractal (option B du spec L1).
#[pyfunction]
fn collatz_hash(x: u64, steps: u32) -> u64 {
    fractus_core::vortex::collatz_hash(x, steps)
}

/// Distance ultrametrique 2-adique : d(a,b) = 2^{-v_2(a ⊕ b)}.
/// Wrapper for fractus_core::vortex::ultrametric_distance. Dans (0, 1].
#[pyfunction]
fn ultrametric_distance(a: u64, b: u64) -> f64 {
    fractus_core::vortex::ultrametric_distance(a, b)
}

/// Norme 2-adique : ||x||_2 = 2^{-v_2(x)}. Wrapper for fractus_core::vortex::norm_2adic.
#[pyfunction]
fn norm_2adic(x: u64) -> f64 {
    fractus_core::vortex::norm_2adic(x)
}

/// Module Python `fractus._core`.
///
/// Signature pyo3 0.29 : le module est recu comme `&Bound<'_, PyModule>`.
/// Les methodes `.add_function(...)` viennent du trait `PyModuleMethods`
/// (re-exporte par `pyo3::prelude`).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(collatz_hash, m)?)?;
    m.add_function(wrap_pyfunction!(ultrametric_distance, m)?)?;
    m.add_function(wrap_pyfunction!(norm_2adic, m)?)?;
    Ok(())
}
```

- [ ] **Step 2: Reconstruire le module natif**

Run (depuis `C:/Users/PHIL/ZCodeProject/fractus`, venv active) :
```powershell
.venv\Scripts\python.exe -m maturin develop --release
```
Expected: `🛠 Installed fractus-0.1.0`. Peut prendre 30-60s (recompile pyo3 si le hash a change).

- [ ] **Step 3: Verifier vite que les nouvelles functions sont exposees**

```powershell
.venv\Scripts\python.exe -c "from fractus import _core; print('hash(7,5)=', _core.collatz_hash(7,5)); print('dist(1,2)=', _core.ultrametric_distance(1,2)); print('norm(8)=', _core.norm_2adic(8))"
```
Expected: trois valeurs numeriques without error (par ex. `hash(7,5)= 52`, `dist(1,2)= 0.5`, `norm(8)= 0.125`).

- [ ] **Step 4: Commit**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add crate/fractus-py/src/lib.rs
git commit -m "feat(py): expose collatz_hash, ultrametric_distance, norm_2adic to Python"
```
Expected: `1 file changed`.

---

## Task 2: Tests du pont Python des functions 2-adiques (TDD)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/tests/test_vortex_bridge.py`

- [ ] **Step 1: Ecrire les tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_vortex_bridge.py` :
```python
"""Tests du pont Python for les functions 2-adiques du vortex.

Verifie que les wrappers Rust sont bien exposes et return des valeurs
correctes. Ces tests ne font PAS de mathematical avancee (ca, c'est en Rust) —
ils valident juste le pont PyO3.
"""

import pytest


def test_collatz_hash_is_deterministic():
    """Meme entree → meme sortie (property requise for le conditionnement)."""
    from fractus import _core
    h1 = _core.collatz_hash(7, 10)
    h2 = _core.collatz_hash(7, 10)
    assert h1 == h2


def test_collatz_hash_zero_stays_zero():
    """Convention : 0 → 0."""
    from fractus import _core
    assert _core.collatz_hash(0, 100) == 0


def test_collatz_hash_returns_u64():
    """Le hash must etre un integer positif compatible with PyTorch indexing."""
    from fractus import _core
    h = _core.collatz_hash(42, 5)
    assert isinstance(h, int)
    assert h >= 0


def test_ultrametric_distance_self_is_zero():
    """d(a, a) = 0."""
    from fractus import _core
    assert _core.ultrametric_distance(42, 42) == 0.0


def test_ultrametric_distance_symmetric():
    """d(a, b) = d(b, a)."""
    from fractus import _core
    for a, b in [(1, 2), (7, 56), (100, 200)]:
        assert _core.ultrametric_distance(a, b) == _core.ultrametric_distance(b, a)


def test_ultrametric_distance_in_unit_interval():
    """Pour a != b, d(a,b) ∈ (0, 1] (norme p-adique)."""
    from fractus import _core
    for a, b in [(1, 2), (7, 56), (100, 200), (3, 9)]:
        d = _core.ultrametric_distance(a, b)
        assert 0.0 < d <= 1.0, f"d({a},{b}) = {d} hors (0, 1]"


def test_norm_2adic_basic():
    """||x||_2 = 2^{-v_2(x)}, verifies sur quelques valeurs connues."""
    from fractus import _core
    assert _core.norm_2adic(0) == 0.0
    assert _core.norm_2adic(1) == 1.0   # v_2(1)=0 → 2^0
    assert _core.norm_2adic(2) == 0.5   # v_2(2)=1 → 2^-1
    assert _core.norm_2adic(8) == 0.125  # v_2(8)=3 → 2^-3


def test_ultrametric_strong_triangle_in_python():
    """La property ultrametrique forte must tenir via le pont Python.
    C'est le test-pivot qui distingue 2^{-v} (correct) de 2^{+v} (bug OMNI)."""
    from fractus import _core
    # Le triplet (7, 56, 13) discrimine : passe with -v, echoue with +v.
    x, y, z = 7, 56, 13
    d_xy = _core.ultrametric_distance(x, y)
    d_yz = _core.ultrametric_distance(y, z)
    d_xz = _core.ultrametric_distance(x, z)
    assert d_xz <= max(d_xy, d_yz) + 1e-9
```

- [ ] **Step 2: Lancer les tests — DOIVENT TOUS PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_vortex_bridge.py -v
```
Expected: 8 passed. Si un test echoue, le pont PyO3 est casse — verify que Task 1 a bien reconstruit le module.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vortex_bridge.py
git commit -m "test(vortex): 8 tests du pont Python for collatz_hash/ultrametric/norm"
```
Expected: `1 file changed`.

---

## Task 3: Implementer les 16 features morphologiques (CharClassFeatures)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/char_features.py`

- [ ] **Step 1: Ecrire le test qui echoue**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_embedding.py` (initial content — sera etendu in les tâches suivantes) :
```python
"""Tests de l'embedding fractal : char features, base de Fourier, FractalEmbedding.

Le critere critique (herite de FNN qui echouait la) : la forward must etre
differentiable et backward() must propager des gradients finis partout.
"""

import torch
import pytest


# ---------------------------------------------------------------------------
# Task 3 : CharClassFeatures (16 features morphologiques)
# ---------------------------------------------------------------------------

def test_char_features_shape():
    """16 features for tout token id."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f.shape == (16,)


def test_char_features_vowel():
    """'a' est voyelle (feature 0 = 1)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f[0].item() == 1.0  # is_vowel


def test_char_features_digit_value():
    """'5' est un chiffre de valeur 5 (feature 11)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("5"))
    assert f[2].item() == 1.0   # is_digit
    assert f[11].item() == 5.0  # digit_value


def test_char_features_batch_consistency():
    """La meme lettre donne le meme vector de features."""
    from fractus.nn.char_features import CharClassFeatures
    f1 = CharClassFeatures.extract(ord("z"))
    f2 = CharClassFeatures.extract(ord("z"))
    assert torch.equal(f1, f2)
```

- [ ] **Step 2: Lancer le test for verify qu'il echoue**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'fractus.nn.char_features'`. C'est normal.

- [ ] **Step 3: Implementer CharClassFeatures**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/char_features.py` :
```python
"""16 features morphologiques deterministes par token.

Porte depuis the original architecture (src/embedding.rs, CharClassFeatures). Le token id est
interprete comme un codepoint Unicode ; for les ids < 128 ce sont des
caracteres ASCII, au-dela on derive les features de la valeur numerique.

Ces features n'ont AUCUN parameter entrainable — elles sont calculees
deterministiquement then concatenees a la base de Fourier in FractalEmbedding.
"""

import torch


class CharClassFeatures:
    """Extraction de 16 features morphologiques a partir d'un token id.

    Features (index : signification) :
        0  : is_vowel          (a, e, i, o, u)
        1  : is_consonant      (lettre non voyelle)
        2  : is_digit          (0-9)
        3  : is_space          (0x20)
        4  : is_uppercase
        5  : is_lowercase
        6  : is_punctuation    (!"#$%...)
        7  : is_alphabetic
        8  : is_numeric        (alias de is_digit ici)
        9  : is_whitespace     (espace, tab, newline)
        10 : is_control        (codepoint < 32 ou == 127)
        11 : digit_value       (0-9, ou 0 si pas un chiffre)
        12 : char_category     (categorie Unicode simplifiee comme float)
        13 : position_in_alphabet (0-25, ou -1 si pas une lettre ; on encode -1→0)
        14 : is_ascii          (codepoint < 128)
        15 : parity            (token id pair = 1, impair = 0)
    """

    N_FEATURES = 16

    VOWELS = frozenset(b"aeiouAEIOU")

    @staticmethod
    def extract(token_id: int) -> torch.Tensor:
        """Retourne un tenseur float32 de shape (16,)."""
        f = torch.zeros(CharClassFeatures.N_FEATURES, dtype=torch.float32)

        # On interprete l'octet de poids faible comme un caractere potentiel.
        as_byte = (token_id & 0xFF)

        # 0: is_vowel
        is_vowel = float(as_byte in CharClassFeatures.VOWELS)
        f[0] = is_vowel

        # 1: is_consonant (lettre alphabetique non voyelle)
        is_alpha = (
            (0x41 <= as_byte <= 0x5A) or  # A-Z
            (0x61 <= as_byte <= 0x7A)     # a-z
        )
        f[1] = float(is_alpha and is_vowel < 0.5)

        # 2: is_digit
        is_digit = 0x30 <= as_byte <= 0x39
        f[2] = float(is_digit)

        # 3: is_space (0x20)
        f[3] = float(as_byte == 0x20)

        # 4: is_uppercase
        f[4] = float(0x41 <= as_byte <= 0x5A)

        # 5: is_lowercase
        f[5] = float(0x61 <= as_byte <= 0x7A)

        # 6: is_punctuation (ASCII punctuation ranges)
        is_punct = (
            (0x21 <= as_byte <= 0x2F) or
            (0x3A <= as_byte <= 0x40) or
            (0x5B <= as_byte <= 0x60) or
            (0x7B <= as_byte <= 0x7E)
        )
        f[6] = float(is_punct)

        # 7: is_alphabetic
        f[7] = float(is_alpha)

        # 8: is_numeric (alias de is_digit ici)
        f[8] = float(is_digit)

        # 9: is_whitespace (espace, tab 0x09, newline 0x0A, CR 0x0D)
        f[9] = float(as_byte in (0x09, 0x0A, 0x0D, 0x20))

        # 10: is_control (codepoint < 32 ou == 127)
        f[10] = float(as_byte < 0x20 or as_byte == 0x7F)

        # 11: digit_value
        f[11] = float(as_byte - 0x30) if is_digit else 0.0

        # 12: char_category simplifie : 1.0 lettre, 2.0 chiffre, 3.0 ponctuation,
        #     4.0 espace, 5.0 controle, 0.0 autre.
        if is_alpha:
            f[12] = 1.0
        elif is_digit:
            f[12] = 2.0
        elif is_punct:
            f[12] = 3.0
        elif f[9] > 0.5:
            f[12] = 4.0
        elif f[10] > 0.5:
            f[12] = 5.0

        # 13: position_in_alphabet (0-25, ou 0 si pas une lettre)
        if 0x41 <= as_byte <= 0x5A:
            f[13] = float(as_byte - 0x41)
        elif 0x61 <= as_byte <= 0x7A:
            f[13] = float(as_byte - 0x61)

        # 14: is_ascii
        f[14] = float(token_id < 128)

        # 15: parity (token id pair)
        f[15] = float((token_id % 2) == 0)

        return f

    @staticmethod
    def extract_batch(token_ids: torch.Tensor) -> torch.Tensor:
        """Version vectorisee : token_ids de shape (N,) → features (N, 16).

        Comme le computation est deterministe et independent par token, on can
        precalculer une lookup table une fois for toute la taille du vocab.
        """
        ids_list = token_ids.tolist()
        rows = [CharClassFeatures.extract(int(i)) for i in ids_list]
        return torch.stack(rows, dim=0)
```

- [ ] **Step 4: Lancer les tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 4 passed (les 4 tests de char features).

- [ ] **Step 5: Commit**

```bash
git add fractus/nn/char_features.py tests/test_embedding.py
git commit -m "feat(nn): port CharClassFeatures (16 morphological features) from FNN"
```
Expected: `2 files changed`.

---

## Task 4: Implementer la base de Fourier a decroissance Mandelbrot

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/fourier.py`

- [ ] **Step 1: Ajouter les tests a tests/test_embedding.py**

Append to `C:/Users/PHIL/ZCodeProject/fractus/tests/test_embedding.py` (after the existing char-features tests) :
```python


# ---------------------------------------------------------------------------
# Task 4 : MandelbrotFourierBasis (base de Fourier a decroissance (φ²)^{-k})
# ---------------------------------------------------------------------------

def test_fourier_basis_shape():
    """Pour vocab 128 et 32 frequences : matrix (vocab, n_freq) en entree du computation."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=128, n_frequencies=32)
    M = basis.matrix()  # (vocab, n_freq)
    assert M.shape == (128, 32)


def test_fourier_basis_is_finite():
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=128, n_frequencies=16)
    M = basis.matrix()
    assert torch.isfinite(M).all()


def test_fourier_frequencies_decay():
    """Les frequences ω_k = (φ²)^{-k} must decroitre geometriquement."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=10, n_frequencies=4)
    # ω_0 = 1.0, ω_1 = 1/φ², ω_2 = 1/φ⁴, ...
    phi_sq = ((1 + 5 ** 0.5) / 2) ** 2
    expected = [phi_sq ** (-k) for k in range(4)]
    for k, exp in enumerate(expected):
        assert abs(basis.frequencies[k].item() - exp) < 1e-5, \
            f"freq[{k}] = {basis.frequencies[k].item()}, attendu {exp}"


def test_fourier_matrix_is_deterministic():
    """Deux appels donnent la meme matrix (pas d'alea)."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    b1 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    b2 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    assert torch.allclose(b1.matrix(), b2.matrix())
```

- [ ] **Step 2: Lancer for verify que les nouveaux tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 4 passed (char features), 4 failed/error (fourier — module absent).

- [ ] **Step 3: Implementer MandelbrotFourierBasis**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/fourier.py` :
```python
"""Base de Fourier a decroissance Mandelbrot for l'embedding fractal.

Inspire de the original architecture (src/math/mandelbrot.rs + src/embedding.rs) but renomme
honnetement : FNN appelait ca "Mandelbrot frequencies" en reference a l'ensemble
de Mandelbrot, alors qu'il s'agit juste d'une decroissance geometrique de base
φ² (le carre du number d'or). On appelle therefore ca "Mandelbrot-decayed Fourier
basis" — la decroissance est real et justifiee (separation d'echelles
multi-niveaux), but le lien a l'ensemble de Mandelbrot est nul.

Mathematique :
    φ = (1 + √5) / 2  ≈ 1.618
    φ² ≈ 2.618
    ω_k = (φ²)^{-k}    for k = 0, 1, ..., n_freq-1

La base de Fourier associe a each token id t et each frequence k la paire
(sin, cos) de ω_k · t :
    M[t, 2k]   = sin(ω_k · t)
    M[t, 2k+1] = cos(ω_k · t)

On stocke n_freq frequences ; la matrix produite a 2·n_freq colonnes
(sin+cos par frequence). Le caller (FractalEmbedding) gere la projection finale.

AUCUN parameter entrainable ici : tout est deterministe, precalcule une fois.
"""

import math
import torch


class MandelbrotFourierBasis:
    """Base de Fourier deterministe with decroissance (φ²)^{-k}.

    Attributs :
        vocab_size   : number de token ids couverts (0 .. vocab_size-1)
        n_frequencies : number de frequences ω_k
        frequencies  : tenseur (n_frequencies,) des ω_k, en float32
    """

    def __init__(self, vocab_size: int, n_frequencies: int):
        if vocab_size <= 0 or n_frequencies <= 0:
            raise ValueError("vocab_size et n_frequencies must etre > 0")
        self.vocab_size = vocab_size
        self.n_frequencies = n_frequencies

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        phi_sq = phi * phi  # ≈ 2.618
        ks = torch.arange(n_frequencies, dtype=torch.float32)
        # ω_k = (φ²)^{-k}
        self.frequencies = phi_sq ** (-ks)

        # Precalcul de la matrix (vocab_size, 2·n_frequencies).
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> torch.Tensor:
        """Construit la matrix M[t, :] = [sin(ω_k·t), cos(ω_k·t)] for tout k."""
        t = torch.arange(self.vocab_size, dtype=torch.float32).unsqueeze(1)  # (V, 1)
        omega = self.frequencies.unsqueeze(0)  # (1, K)
        phases = omega * t  # (V, K) broadcast
        sin_part = torch.sin(phases)  # (V, K)
        cos_part = torch.cos(phases)  # (V, K)
        # Interleave sin/cos : colonnes 0,2,4,... = sin ; 1,3,5,... = cos
        M = torch.empty(self.vocab_size, 2 * self.n_frequencies, dtype=torch.float32)
        M[:, 0::2] = sin_part
        M[:, 1::2] = cos_part
        return M

    def matrix(self) -> torch.Tensor:
        """Retourne la matrix precalculee (vocab_size, 2·n_frequencies)."""
        return self._matrix

    def dim_output(self) -> int:
        """Dimension de sortie (number de colonnes de la matrix)."""
        return 2 * self.n_frequencies
```

- [ ] **Step 4: Lancer all les tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 8 passed (4 char features + 4 fourier).

- [ ] **Step 5: Commit**

```bash
git add fractus/nn/fourier.py tests/test_embedding.py
git commit -m "feat(nn): add MandelbrotFourierBasis (honestly named Fourier basis)"
```
Expected: `2 files changed`.

---

## Task 5: Implementer FractalEmbedding (assemblage + projection entrainable)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/embedding.py`
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py`

- [ ] **Step 1: Ajouter les tests de FractalEmbedding**

Append to `C:/Users/PHIL/ZCodeProject/fractus/tests/test_embedding.py` :
```python


# ---------------------------------------------------------------------------
# Task 5 : FractalEmbedding (assemblage + projection entrainable)
# ---------------------------------------------------------------------------

def test_fractal_embedding_shape():
    """Sortie (N, d_model) for entree (N,) d'ids."""
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.tensor([0, 1, 2, 65, 97])  # mix
    out = emb(ids)
    assert out.shape == (5, 64)


def test_fractal_embedding_is_finite():
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.arange(128)
    out = emb(ids)
    assert torch.isfinite(out).all()


def test_fractal_embedding_backward_propagates():
    """CRITIQUE : backward() must propager des gradients finis a all les params.

    C'est exactement le test que the original architecture echouait (training.rs:399 utilisait du
    bruit aleatoire au lieu d'un gradient). Ici, la projection Linear est in
    le graphe autodiff, therefore les gradients must etre non-nuls et finis.
    """
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.tensor([0, 1, 2, 3, 4])
    out = emb(ids)
    loss = out.pow(2).sum()
    loss.backward()

    has_param_with_grad = False
    for name, p in emb.named_parameters():
        assert p.requires_grad, f"{name} should requires_grad=True"
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
            if p.grad.abs().sum().item() > 0:
                has_param_with_grad = True
    assert has_param_with_grad, "Aucun parameter n'a recu de gradient non-nul"


def test_fractal_embedding_respects_vocab_bounds():
    """Un id >= vocab_size must lever une error (pas de crash silencieux)."""
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=100, d_model=32, n_frequencies=8)
    with pytest.raises(IndexError):
        emb(torch.tensor([100]))  # hors borne
```

- [ ] **Step 2: Lancer for verify que les tests FractalEmbedding echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 8 passed (char + fourier), 4 failed/error (FractalEmbedding absent).

- [ ] **Step 3: Implementer FractalEmbedding**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/embedding.py` :
```python
"""FractalEmbedding : embedding de codepoint fractal entrainable.

Assemblage de trois sources de features for each token id t :

    (A) 16 features morphologiques deterministes (CharClassFeatures)
    (B) base de Fourier a decroissance Mandelbrot (MandelbrotFourierBasis)
    (C) conditionnement vortex : un hash 2-adique (Collatz, calcule en Rust,
        hors-graphe autodiff) est projete en phases via un MLP entrainable
        (PyTorch, in le graphe). C'est l'option B du spec L1 : le vortex
        2-adique influence l'apprentissage without pretendre etre differentiable.

La projection finale vers d_model est un nn.Linear entrainable. Toute la forward
est differentiable de bout en bout — les parties deterministes (A, B, et le hash
de C) sont precalculees en buffers hors-graphe ; seul le MLP de C et la
projection finale portent des parameters.

Corrections vs systems originaux :
- FNN n'apprenait pas (training.rs:399 = bruit) → ici backward() marche (test).
- OMNI : the 2-adic vortex was orphaned (never imported by Python) →
  ici il conditionne reellement l'embedding.
- OMNI : les « Mandelbrot frequencies » etaient mal nommees → ici on dit
  « Mandelbrot-decayed Fourier basis » (voir fourier.py).
"""

import torch
import torch.nn as nn

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis


class FractalEmbedding(nn.Module):
    """Embedding fractal entrainable.

    Args :
        vocab_size     : number de token ids couverts.
        d_model        : dimension de sortie.
        n_frequencies  : number de frequences ω_k for la base de Fourier.
        vortex_hidden  : width du MLP qui projette le hash Collatz en phases.
        collatz_steps  : number d'iterations Collatz for le hash (deterministe).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_frequencies: int = 16,
        vortex_hidden: int = 32,
        collatz_steps: int = 7,
    ):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size et d_model must etre > 0")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.collatz_steps = collatz_steps

        # (A) Features morphologiques : precalcul deterministe, hors-graphe.
        char_matrix = torch.stack(
            [CharClassFeatures.extract(t) for t in range(vocab_size)], dim=0
        )  # (vocab, 16)
        self.register_buffer("char_features", char_matrix)

        # (B) Base de Fourier Mandelbrot-decroissante : precalcul deterministe.
        self.fourier = MandelbrotFourierBasis(vocab_size, n_frequencies)
        fourier_matrix = self.fourier.matrix()  # (vocab, 2·n_freq)
        self.register_buffer("fourier_features", fourier_matrix)

        # (C) Conditionnement vortex : hash Collatz precalcule (hors-graphe),
        # then projete par un MLP entrainable (in le graphe).
        # On importe le hash depuis le module natif Rust.
        try:
            from fractus import _core
        except ImportError as e:
            raise ImportError(
                "fractus._core introuvable. Lance `maturin develop`."
            ) from e
        hashes = torch.tensor(
            [_core.collatz_hash(t, collatz_steps) for t in range(vocab_size)],
            dtype=torch.float32,
        )  # (vocab,)
        # Normalisation douce : on ramene in [0, 1) via / (max+1) for stabilite.
        max_h = hashes.max().item() + 1.0
        hashes_norm = hashes / max_h
        self.register_buffer("vortex_hashes", hashes_norm)  # (vocab,)

        # MLP entrainable : projette le scalar hash (1) vers un vector de
        # dimension vortex_phase_dim. C'est ici que le vortex « conditionne »
        # le reseau : le MLP apprend a interpreter le hash 2-adique.
        self.vortex_phase_dim = vortex_hidden
        self.vortex_mlp = nn.Sequential(
            nn.Linear(1, vortex_hidden),
            nn.Tanh(),
            nn.Linear(vortex_hidden, vortex_hidden),
        )

        # Projection finale entrainable vers d_model.
        # dim d'entree = 16 (char) + 2·n_freq (fourier) + vortex_hidden
        in_dim = 16 + fourier_matrix.shape[1] + vortex_hidden
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids : (N,) ou (N, L) d'integers in [0, vocab_size).

        Retourne (N, d_model) ou (N, L, d_model).
        """
        if token_ids.max() >= self.vocab_size or token_ids.min() < 0:
            raise IndexError(
                f"token_id hors [0, {self.vocab_size}) : "
                f"min={int(token_ids.min())}, max={int(token_ids.max())}"
            )

        original_shape = token_ids.shape
        flat = token_ids.reshape(-1)  # (M,)

        # (A) + (B) : lookup in les buffers precalcules (hors-graphe, but le
        # result alimente la projection entrainable, therefore le graphe traverse).
        char = self.char_features[flat]      # (M, 16)
        fourier = self.fourier_features[flat]  # (M, 2·n_freq)

        # (C) : hash precalcule → reshape (M, 1) → MLP entrainable (in le graphe).
        h = self.vortex_hashes[flat].unsqueeze(1)  # (M, 1)
        vortex_phases = self.vortex_mlp(h)         # (M, vortex_hidden)

        # Concat et projection.
        x = torch.cat([char, fourier, vortex_phases], dim=1)  # (M, in_dim)
        out = self.proj(x)  # (M, d_model)

        return out.reshape(*original_shape, self.d_model)
```

- [ ] **Step 4: Mettre a jour fractus/nn/__init__.py**

Replace the entire content of `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py` with :
```python
"""Sous-package nn — modules de reseau de neurones (PyTorch).

L1 : embedding fractal entrainable (FractalEmbedding).
"""

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis
from .embedding import FractalEmbedding

__all__ = ["CharClassFeatures", "MandelbrotFourierBasis", "FractalEmbedding"]
```

- [ ] **Step 5: Lancer all les tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 12 passed (4 char + 4 fourier + 4 FractalEmbedding). Le test `test_fractal_embedding_backward_propagates` est le critere critique : il prouve que l'embedding apprend vraiment.

- [ ] **Step 6: Commit**

```bash
git add fractus/nn/embedding.py fractus/nn/__init__.py tests/test_embedding.py
git commit -m "feat(nn): add FractalEmbedding with vortex 2-adic conditioning

- Assemblage : 16 char features + base Fourier Mandelbrot + conditionnement
  vortex (hash Collatz en Rust → MLP entrainable en PyTorch).
- Option B du spec L1 : le vortex 2-adique exact (hors-graphe) conditionne
  un MLP differentiable. Le vortex n'est PAS pretendu differentiable.
- Critere critique valide : backward() propage des gradients finis a all
  les parameters (le test que the original architecture echouait)."
```
Expected: `3 files changed`.

---

## Task 6: Demo interactive L1 — prouver que l'embedding apprend

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_embedding.py`

- [ ] **Step 1: Ecrire la demo**

Create `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_embedding.py` :
```python
"""Demo L1 : prouve que FractalEmbedding apprend vraiment.

Objectif : surfit un embedding cible aleatoire fixe en quelques steps Adam,
et montre que la loss baisse. C'est la preuve minimale que l'autodiff traverse
l'embedding fractal (ce que the original architecture ne savait pas faire).

Run :
    python scripts/demo_embedding.py
"""

import torch
from fractus.nn import FractalEmbedding


def main():
    torch.manual_seed(42)

    vocab = 64
    d_model = 32
    emb = FractalEmbedding(vocab_size=vocab, d_model=d_model, n_frequencies=12)
    print(f"Parametres entrainables : {sum(p.numel() for p in emb.parameters())}")

    # Cible aleatoire fixe : le but est de surfit cette cible.
    target = torch.randn(vocab, d_model)

    opt = torch.optim.Adam(emb.parameters(), lr=1e-2)

    initial_loss = None
    for step in range(200):
        opt.zero_grad()
        ids = torch.arange(vocab)
        out = emb(ids)
        loss = ((out - target) ** 2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        if step % 40 == 0 or step == 199:
            print(f"step {step:3d}  loss = {loss.item():.4f}")

    final_loss = loss.item()
    print()
    print(f"Loss initial : {initial_loss:.4f}")
    print(f"Loss finale   : {final_loss:.4f}")
    print(f"Baisse        : {(1 - final_loss / initial_loss) * 100:.1f}%")

    if final_loss < initial_loss * 0.5:
        print("\n✓ SUCCES : l'embedding fractal apprend (loss divisee par >2).")
    else:
        print("\n✗ ECHEC : la loss ne baisse pas assez — investiguer.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lancer la demo**

```powershell
.venv\Scripts\python.exe scripts\demo_embedding.py
```
Expected output (approximatif — les valeurs exactes dependent du seed) :
```
Parametres entrainables : ~3000
step   0  loss = 1.2xxx
step  40  loss = 0.6xxx
step  80  loss = 0.3xxx
step 120  loss = 0.15xxx
step 160  loss = 0.08xxx
step 199  loss = 0.05xxx

Loss initial : 1.2xxx
Loss finale   : 0.05xxx
Baisse        : ~95%

✓ SUCCES : l'embedding fractal apprend (loss divisee par >2).
```

Si la demo affiche ECHEC, le bug est serieux — deboguer before L2.

- [ ] **Step 3: Commit**

```bash
git add scripts/demo_embedding.py
git commit -m "demo(L1): prove FractalEmbedding learns (overfit target, loss drops)"
```
Expected: `1 file changed`.

---

## Critere final de L1 « termine »

Apres le Task 6, ces verifications must toutes reussir :

```powershell
cd "C:\Users\PHIL\ZCodeProject\fractus"

# 1. Le Rust compile et ses tests passent (inchange depuis L0)
cargo test -p fractus-core
# → 9 passed

# 2. Le module natif est a jour (inclut les nouveaux bindings)
.venv\Scripts\python.exe -m maturin develop --release
# → 🛠 Installed fractus-0.1.0

# 3. Tous les tests Python passent
.venv\Scripts\python.exe -m pytest tests/ -v
# → 5 (smoke) + 8 (vortex bridge) + 12 (embedding) = 25 passed

# 4. La demo prouve l'apprentissage
.venv\Scripts\python.exe scripts\demo_embedding.py
# → "✓ SUCCES : l'embedding fractal apprend"
```

Si tout passe, L1 est termine et on can passer au plan L2 (bloc transformer fractal : attention lineaire + Kuramoto + MoE Farey).

---

## Self-Review (post-ecriture)

**1. Spec coverage :**
- Spec L1 demande (a) CharClassFeatures ported → Task 3 ✅ ;
  (b) MandelbrotFourierBasis (renomme honnetement) → Task 4 ✅ ;
  (c) pont option B (hash conditionne MLP entrainable) → Task 5 ✅ ;
  (d) critere « backward() propage gradients finis » → test `test_fractal_embedding_backward_propagates` Task 5 ✅ ;
  (e) ultrametrie testee depuis Python → `test_ultrametric_strong_triangle_in_python` Task 2 ✅.
- Couverture complete.

**2. Placeholder scan :** no « TBD/TODO/fill in ». Chaque etape a du code complete ou des commandes exactes. ✅

**3. Type consistency :** `CharClassFeatures.extract(int) → Tensor(16,)`, `.extract_batch(Tensor) → Tensor(N,16)`. `MandelbrotFourierBasis.matrix() → Tensor(V, 2K)`, `.frequencies → Tensor(K,)`. `FractalEmbedding(vocab_size, d_model, n_frequencies, vortex_hidden, collatz_steps).forward(Tensor) → Tensor`. Coherent between definitions (Tasks 3/4/5) et tests. ✅

**4. Imports coherents :** `from fractus.nn import FractalEmbedding` fonctionne grâce a `fractus/nn/__init__.py` mis a jour (Task 5 Step 4). Le hash vient de `from fractus import _core` (Task 5 Step 3, in `__init__` du module). ✅

**5. Dependance sur L0 :** Le pont `_core` (L0) est requis for Task 1 (nouveaux wrappers) et Task 5 (hash Collatz). `cargo test -p fractus-core` inchange (le vortex.rs n'est pas modifie en L1, juste expose differemment). ✅

**6. YAGNI :** Pas de MoE, pas d'attention, pas de causal — juste l'embedding. Tout le reste vient en L2+. ✅
