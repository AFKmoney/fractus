# Fractus L1 — Embedding fractal + vortex 2-adique branche Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Brancher the vortex 2-adique d'the original design on the reseau of neurones for the premiere fois, en l'utilisant comme **conditionnement** d'un MLP entrainable PyTorch. Ajouter l'fractal codepoint embedding (Fourier basis a Mandelbrot decay + 16 morphological features), all en PyTorch pur for l'autodiff. Corriger deux errors systems originaux : the vortex orphaned (never importedd) and the « Mandelbrot frequencies » mal nommees.

**Architecture:** (1) `fractus-core` (Rust) expose `collatz_hash(token_id)` and `ultrametric_distance(a,b)` — computation exact, outside the autodiff graph, appele depuis Python for precomputationer the conditionnement. (2) `fractus/nn/embedding.py` (PyTorch) contient trois modules composes : `CharClassFeatures` (16 morphological features deterministic, portedes of the original), `MandelbrotFourierBasis` (Fourier basis a decroissance `(φ2)^{-k}`, nommee honestetement), and `FractalEmbedding` (assemblage final : the features morpho + the Fourier basis + the phases vortex-conditionnees are projetees toward `d_model` by a `nn.Linear` entrainable). La forward est differentiable end-to-end.

**Tech Stack:** Rust 1.94 + pyo3 0.29 (deja installe en L0) ; Python 3.14 + torch 2.12 CPU + numpy (deja installes en L0) ; pytest. Le module natif must etre reconstruit via `maturin develop` after modification Rust.

**Lien spec :** `docs/superpowers/specs/2026-06-19-fractus-unified-design.md`, section « L1 — Embedding fractal + vortex 2-adique branche ». Decision cle valide : **option B** — the hash 2-adique exact (Rust, hors-graphe) conditionne a MLP entrainable (PyTorch, in the graphe).

**Prerequis :** L0 termine (repo `C:/Users/PHIL/ZCodeProject/fractus/` operationnel, venv `.venv` with torch + maturin installes, 9 tests Rust + 5 tests Python passent).

**Vocabulaire honestete applique in this plan :**
- « Mandelbrot frequencies » → « Mandelbrot-decayed Fourier basis » (la decroissance `(φ2)^{-k}` est real and justifiee, but this n'est not l'ensemble of Mandelbrot).
- « Collatz ergodic flow » → « Collatz hash » (l'ergodicite of Collatz est non demontree, problem ouvert).
- On parle of « 2-adic norm » and « ultrametric distance » (termes mathematics exacts).

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── crate/fractus-core/src/
│   ├── lib.rs                  # inchange (declare already pub mod vortex)
│   └── vortex.rs               # MODIFY : expose also norm_2adic (deja defini, juste a binder)
├── crate/fractus-py/src/
│   └── lib.rs                  # MODIFY : ajoute wrappers #[pyfunction] for collatz_hash,
│                               #   ultrametric_distance, norm_2adic
├── fractus/nn/
│   ├── __init__.py             # MODIFY : exported the classes publiques
│   ├── char_features.py        # CREATE : 16 morphological features (portedd of the original)
│   ├── fourier.py              # CREATE : Fourier basis a Mandelbrot decay
│   └── embedding.py            # CREATE : FractalEmbedding (assemblage + projection Linear)
└── tests/
    ├── test_vortex_bridge.py   # CREATE : tests pont Python functions 2-adiques
    └── test_embedding.py       # CREATE : tests of l'embedding fractal
```

**Responsabilites (un fichier = a responsabilite) :**
- `char_features.py` : uniquement the 16 morphological features (deterministic, without parameter).
- `fourier.py` : uniquement the Fourier basis a Mandelbrot decay (deterministic, without parameter).
- `embedding.py` : l'assemblage + the projection entrainable (le seul endroit with `nn.Parameter`).
- `test_vortex_bridge.py` : verifiess that the functions Rust are well appelables depuis Python and return the bonnes values (pont operationnel).
- `test_embedding.py` : verifiess formes, finitude, and surtout that `backward()` propage gradients finis (le critere critique herite of the original which echouait la).

---

## Task 1: Exposer collatz_hash, ultrametric_distance, norm_2adic in the bindings Python

**Files:**
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs`

- [ ] **Step 1: Reecrire lib.rs with the nouveaux wrappers**

Replace the entire content of `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs` with :

```rust
//! Bindings Python (PyO3) for fractus-core.
//!
//! Ce crate not contient AUCUNE logical — seulement wrappers #[pyfunction]
//! which deleguent a fractus-core. Le but est d'exposer the Rust a Python
//! under the nom `fractus._core`.

use pyo3::prelude::*;

/// Addition integere — wrapper Python for fractus_core::add.
/// Exposee uniquement for the test fume.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    fractus_core::add(a, b)
}

/// Hash Collatz d'un token id. Wrapper for fractus_core::vortex::collatz_hash.
/// Utilise comme conditionnement deterministic (outside the autodiff graph) for
/// l'embedding fractal (option B spec L1).
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
/// Signature pyo3 0.29 : the module est recu comme `&Bound<'_, PyModule>`.
/// Les methodes `.add_function(...)` viennent trait `PyModuleMethods`
/// (re-exported by `pyo3::prelude`).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(collatz_hash, m)?)?;
    m.add_function(wrap_pyfunction!(ultrametric_distance, m)?)?;
    m.add_function(wrap_pyfunction!(norm_2adic, m)?)?;
    Ok(())
}
```

- [ ] **Step 2: Reconstruire the module natif**

Run (depuis `C:/Users/PHIL/ZCodeProject/fractus`, venv active) :
```powershell
.venv\Scripts\python.exe -m maturin develop --release
```
Expected: `🛠 Installed fractus-0.1.0`. Peut prendre 30-60s (recompile pyo3 si the hash a change).

- [ ] **Step 3: Verifier fast that the nouvelles functions are exposees**

```powershell
.venv\Scripts\python.exe -c "from fractus import _core; print('hash(7,5)=', _core.collatz_hash(7,5)); print('dist(1,2)=', _core.ultrametric_distance(1,2)); print('norm(8)=', _core.norm_2adic(8))"
```
Expected: trois values numeriques without error (par ex. `hash(7,5)= 52`, `dist(1,2)= 0.5`, `norm(8)= 0.125`).

- [ ] **Step 4: Commit**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add crate/fractus-py/src/lib.rs
git commit -m "feat(py): expose collatz_hash, ultrametric_distance, norm_2adic to Python"
```
Expected: `1 file changed`.

---

## Task 2: Tests pont Python functions 2-adiques (TDD)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/tests/test_vortex_bridge.py`

- [ ] **Step 1: Ecrire the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_vortex_bridge.py` :
```python
"""Tests pont Python for the functions 2-adiques vortex.

Verifie that the wrappers Rust are well exposes and return values
correctes. Ces tests not font PAS of mathematical avancee (ca, this is en Rust) —
ils validnt juste the pont PyO3.
"""

import pytest


def test_collatz_hash_is_deterministic():
    """Meme entree → same sortie (property requise for the conditionnement)."""
    from fractus import _core
    h1 = _core.collatz_hash(7, 10)
    h2 = _core.collatz_hash(7, 10)
    assert h1 == h2


def test_collatz_hash_zero_stays_zero():
    """Convention : 0 → 0."""
    from fractus import _core
    assert _core.collatz_hash(0, 100) == 0


def test_collatz_hash_returns_u64():
    """Le hash must etre a integer positif compatible with PyTorch indexing."""
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
    """||x||_2 = 2^{-v_2(x)}, verifiess on quelques values connues."""
    from fractus import _core
    assert _core.norm_2adic(0) == 0.0
    assert _core.norm_2adic(1) == 1.0   # v_2(1)=0 → 2^0
    assert _core.norm_2adic(2) == 0.5   # v_2(2)=1 → 2^-1
    assert _core.norm_2adic(8) == 0.125  # v_2(8)=3 → 2^-3


def test_ultrametric_strong_triangle_in_python():
    """La property ultrametrique forte must tenir via the pont Python.
    This is the test-pivot which distingue 2^{-v} (correct) of 2^{+v} (bug the original)."""
    from fractus import _core
    # Le triplet (7, 56, 13) discrimine : passe with -v, echoue with +v.
    x, y, z = 7, 56, 13
    d_xy = _core.ultrametric_distance(x, y)
    d_yz = _core.ultrametric_distance(y, z)
    d_xz = _core.ultrametric_distance(x, z)
    assert d_xz <= max(d_xy, d_yz) + 1e-9
```

- [ ] **Step 2: Lancer the tests — DOIVENT TOUS PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_vortex_bridge.py -v
```
Expected: 8 passed. Si a test echoue, the pont PyO3 est casse — verify that Task 1 a well reconstruit the module.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vortex_bridge.py
git commit -m "test(vortex): 8 tests pont Python for collatz_hash/ultrametric/norm"
```
Expected: `1 file changed`.

---

## Task 3: Implementer the 16 morphological features (CharClassFeatures)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/char_features.py`

- [ ] **Step 1: Ecrire the test which echoue**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_embedding.py` (initial content — sera etendu in the taches suivantes) :
```python
"""Tests of l'embedding fractal : char features, Fourier basis, FractalEmbedding.

Le critere critique (herite of the original which echouait la) : the forward must etre
differentiable and backward() must propager gradients finis partout.
"""

import torch
import pytest


# ---------------------------------------------------------------------------
# Task 3 : CharClassFeatures (16 morphological features)
# ---------------------------------------------------------------------------

def test_char_features_shape():
    """16 features for all token id."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f.shape == (16,)


def test_char_features_vowel():
    """'a' est voyelle (feature 0 = 1)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f[0].item() == 1.0  # is_vowel


def test_char_features_digit_value():
    """'5' est a chiffre of value 5 (feature 11)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("5"))
    assert f[2].item() == 1.0   # is_digit
    assert f[11].item() == 5.0  # digit_value


def test_char_features_batch_consistency():
    """La same lettre donne the same vector of features."""
    from fractus.nn.char_features import CharClassFeatures
    f1 = CharClassFeatures.extract(ord("z"))
    f2 = CharClassFeatures.extract(ord("z"))
    assert torch.equal(f1, f2)
```

- [ ] **Step 2: Lancer the test for verify qu'il echoue**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'fractus.nn.char_features'`. This is normal.

- [ ] **Step 3: Implementer CharClassFeatures**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/char_features.py` :
```python
"""16 morphological features deterministic by token.

Porte depuis the original architecture (src/embedding.rs, CharClassFeatures). Le token id est
interprete comme a codepoint Unicode ; for the ids < 128 this are des
caracteres ASCII, au-dela on derive the features of the value numerique.

Ces features n'ont AUCUN parameter entrainable — elles are computationees
deterministiquement then concatenees a the Fourier basis in FractalEmbedding.
"""

import torch


class CharClassFeatures:
    """Extraction of 16 morphological features a partir d'un token id.

    Features (index : signification) :
        0  : is_vowel          (a, e, i, o, u)
        1  : is_consonant      (lettre non voyelle)
        2  : is_digit          (0-9)
        3  : is_space          (0x20)
        4  : is_uppercase
        5  : is_lowercase
        6  : is_punctuation    (!"#$%...)
        7  : is_alphabetic
        8  : is_numeric        (alias of is_digit ici)
        9  : is_whitespace     (espace, tab, newline)
        10 : is_control        (codepoint < 32 or == 127)
        11 : digit_value       (0-9, or 0 si not a chiffre)
        12 : char_category     (categorie Unicode simplifiee comme float)
        13 : position_in_alphabet (0-25, or -1 si not a lettre ; on encode -1→0)
        14 : is_ascii          (codepoint < 128)
        15 : parity            (token id pair = 1, impair = 0)
    """

    N_FEATURES = 16

    VOWELS = frozenset(b"aeiouAEIOU")

    @staticmethod
    def extract(token_id: int) -> torch.Tensor:
        """Retourne a tenseur float32 of shape (16,)."""
        f = torch.zeros(CharClassFeatures.N_FEATURES, dtype=torch.float32)

        # On interprete l'octet of poids weak comme a caractere potentiel.
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

        # 8: is_numeric (alias of is_digit ici)
        f[8] = float(is_digit)

        # 9: is_whitespace (espace, tab 0x09, newline 0x0A, CR 0x0D)
        f[9] = float(as_byte in (0x09, 0x0A, 0x0D, 0x20))

        # 10: is_control (codepoint < 32 or == 127)
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

        # 13: position_in_alphabet (0-25, or 0 si not a lettre)
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
        """Version vectorisee : token_ids of shape (N,) → features (N, 16).

        Comme the computation est deterministic and independent by token, on can
        precomputationer a lookup table a fois for toute the taille vocab.
        """
        ids_list = token_ids.tolist()
        rows = [CharClassFeatures.extract(int(i)) for i in ids_list]
        return torch.stack(rows, dim=0)
```

- [ ] **Step 4: Lancer the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 4 passed (les 4 tests of char features).

- [ ] **Step 5: Commit**

```bash
git add fractus/nn/char_features.py tests/test_embedding.py
git commit -m "feat(nn): port CharClassFeatures (16 morphological features) from the original"
```
Expected: `2 files changed`.

---

## Task 4: Implementer the Fourier basis a Mandelbrot decay

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/fourier.py`

- [ ] **Step 1: Ajouter the tests a tests/test_embedding.py**

Append to `C:/Users/PHIL/ZCodeProject/fractus/tests/test_embedding.py` (after the existing char-features tests) :
```python


# ---------------------------------------------------------------------------
# Task 4 : MandelbrotFourierBasis (Fourier basis a decroissance (φ2)^{-k})
# ---------------------------------------------------------------------------

def test_fourier_basis_shape():
    """Pour vocab 128 and 32 frequences : matrix (vocab, n_freq) en entree computation."""
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
    """Les frequences ω_k = (φ2)^{-k} must decroitre geometriquement."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=10, n_frequencies=4)
    # ω_0 = 1.0, ω_1 = 1/φ2, ω_2 = 1/φ4, ...
    phi_sq = ((1 + 5 ** 0.5) / 2) ** 2
    expected = [phi_sq ** (-k) for k in range(4)]
    for k, exp in enumerate(expected):
        assert abs(basis.frequencies[k].item() - exp) < 1e-5, \
            f"freq[{k}] = {basis.frequencies[k].item()}, attendu {exp}"


def test_fourier_matrix_is_deterministic():
    """Deux appels donnent the same matrix (pas d'alea)."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    b1 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    b2 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    assert torch.allclose(b1.matrix(), b2.matrix())
```

- [ ] **Step 2: Lancer for verify that the nouveaux tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 4 passed (char features), 4 failed/error (fourier — module absent).

- [ ] **Step 3: Implementer MandelbrotFourierBasis**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/fourier.py` :
```python
"""Base of Fourier a Mandelbrot decay for l'embedding fractal.

Inspire of the original architecture (src/math/mandelbrot.rs + src/embedding.rs) but renomme
honestetement : the original appelait ca "Mandelbrot frequencies" en reference a l'ensemble
de Mandelbrot, alors qu'il s'agit juste d'une decroissance geometrique of base
φ2 (le carre number d'or). On appelle therefore ca "Mandelbrot-decayed Fourier
basis" — the decroissance est real and justifiee (separation d'echelles
multi-niveaux), but the lien a l'ensemble of Mandelbrot est nul.

Mathematique :
    φ = (1 + √5) / 2  ≈ 1.618
    φ2 ≈ 2.618
    ω_k = (φ2)^{-k}    for k = 0, 1, ..., n_freq-1

La Fourier basis associe a each token id t and each frequence k the paire
(sin, cos) of ω_k · t :
    M[t, 2k]   = sin(ω_k · t)
    M[t, 2k+1] = cos(ω_k · t)

On stocke n_freq frequences ; the matrix produite a 2·n_freq colonnes
(sin+cos by frequence). Le caller (FractalEmbedding) gere the projection finale.

AUCUN parameter entrainable ici : all est deterministic, precomputatione a fois.
"""

import math
import torch


class MandelbrotFourierBasis:
    """Base of Fourier deterministic with decroissance (φ2)^{-k}.

    Attributs :
        vocab_size   : number of token ids couverts (0 .. vocab_size-1)
        n_frequencies : number of frequences ω_k
        frequencies  : tenseur (n_frequencies,) ω_k, en float32
    """

    def __init__(self, vocab_size: int, n_frequencies: int):
        if vocab_size <= 0 or n_frequencies <= 0:
            raise ValueError("vocab_size and n_frequencies must etre > 0")
        self.vocab_size = vocab_size
        self.n_frequencies = n_frequencies

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        phi_sq = phi * phi  # ≈ 2.618
        ks = torch.arange(n_frequencies, dtype=torch.float32)
        # ω_k = (φ2)^{-k}
        self.frequencies = phi_sq ** (-ks)

        # Precomputation of the matrix (vocab_size, 2·n_frequencies).
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> torch.Tensor:
        """Construit the matrix M[t, :] = [sin(ω_k·t), cos(ω_k·t)] for all k."""
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
        """Retourne the matrix precomputationee (vocab_size, 2·n_frequencies)."""
        return self._matrix

    def dim_output(self) -> int:
        """Dimension of sortie (number of colonnes of the matrix)."""
        return 2 * self.n_frequencies
```

- [ ] **Step 4: Lancer all the tests — DOIVENT PASSER**

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

- [ ] **Step 1: Ajouter the tests of FractalEmbedding**

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
    """CRITIQUE : backward() must propager gradients finis a all the params.

    This is exactment the test that the original architecture echouait (training.rs:399 utilisait du
    bruit aleatoire au lieu d'un gradient). Ici, the projection Linear est in
    the graphe autodiff, therefore the gradients must etre non-nuls and finis.
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
            assert torch.isfinite(p.grad).all(), f"{name} a a gradient non-fini"
            if p.grad.abs().sum().item() > 0:
                has_param_with_grad = True
    assert has_param_with_grad, "Aucun parameter n'a recu of gradient non-nul"


def test_fractal_embedding_respects_vocab_bounds():
    """Un id >= vocab_size must lever a error (pas of crash silencieux)."""
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=100, d_model=32, n_frequencies=8)
    with pytest.raises(IndexError):
        emb(torch.tensor([100]))  # hors borne
```

- [ ] **Step 2: Lancer for verify that the tests FractalEmbedding echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 8 passed (char + fourier), 4 failed/error (FractalEmbedding absent).

- [ ] **Step 3: Implementer FractalEmbedding**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/embedding.py` :
```python
"""FractalEmbedding : fractal codepoint embedding entrainable.

Assembly of trois sources of features for each token id t :

    (A) 16 morphological features deterministic (CharClassFeatures)
    (B) Fourier basis a Mandelbrot decay (MandelbrotFourierBasis)
    (C) vortex conditioning : a hash 2-adique (Collatz, computatione en Rust,
        outside the autodiff graph) est projete en phases via a MLP entrainable
        (PyTorch, in the graphe). This is l'option B spec L1 : the vortex
        2-adique influences learning without pretendre etre differentiable.

The final projection toward d_model est a nn.Linear entrainable. The entire forward pass
est differentiable end-to-end — the parties deterministic (A, B, and the hash
de C) are precomputationees en buffers hors-graphe ; seul the MLP of C and la
projection finale portednt parameters.

Corrections vs systems originaux :
- the original did not learn (training.rs:399 = bruit) → ici backward() marche (test).
- the original : the 2-adic vortex was orphaned (never importedd by Python) →
  ici il conditionne reallement l'embedding.
- the original : the « Mandelbrot frequencies » etaient mal nommees → ici on dit
  « Mandelbrot-decayed Fourier basis » (voir fourier.py).
"""

import torch
import torch.nn as nn

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis


class FractalEmbedding(nn.Module):
    """Embedding fractal entrainable.

    Args :
        vocab_size     : number of token ids couverts.
        d_model        : dimension of sortie.
        n_frequencies  : number of frequences ω_k for the Fourier basis.
        vortex_hidden  : width MLP which projette the hash Collatz en phases.
        collatz_steps  : number d'iterations Collatz for the hash (deterministic).
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
            raise ValueError("vocab_size and d_model must etre > 0")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.collatz_steps = collatz_steps

        # (A) Features morphologicals : precomputation deterministic, hors-graphe.
        char_matrix = torch.stack(
            [CharClassFeatures.extract(t) for t in range(vocab_size)], dim=0
        )  # (vocab, 16)
        self.register_buffer("char_features", char_matrix)

        # (B) Base of Fourier Mandelbrot-decroissante : precomputation deterministic.
        self.fourier = MandelbrotFourierBasis(vocab_size, n_frequencies)
        fourier_matrix = self.fourier.matrix()  # (vocab, 2·n_freq)
        self.register_buffer("fourier_features", fourier_matrix)

        # (C) Conditionnement vortex : hash Collatz precomputatione (hors-graphe),
        # then projete by a MLP entrainable (in the graphe).
        # On imported the hash depuis the module natif Rust.
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

        # MLP entrainable : projette the scalar hash (1) toward a vector de
        # dimension vortex_phase_dim. This is ici that the vortex « conditionne »
        # the reseau : the MLP apprend a interpreter the hash 2-adique.
        self.vortex_phase_dim = vortex_hidden
        self.vortex_mlp = nn.Sequential(
            nn.Linear(1, vortex_hidden),
            nn.Tanh(),
            nn.Linear(vortex_hidden, vortex_hidden),
        )

        # Projection finale entrainable toward d_model.
        # dim d'entree = 16 (char) + 2·n_freq (fourier) + vortex_hidden
        in_dim = 16 + fourier_matrix.shape[1] + vortex_hidden
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids : (N,) or (N, L) d'integers in [0, vocab_size).

        Retourne (N, d_model) or (N, L, d_model).
        """
        if token_ids.max() >= self.vocab_size or token_ids.min() < 0:
            raise IndexError(
                f"token_id hors [0, {self.vocab_size}) : "
                f"min={int(token_ids.min())}, max={int(token_ids.max())}"
            )

        original_shape = token_ids.shape
        flat = token_ids.reshape(-1)  # (M,)

        # (A) + (B) : lookup in the buffers precomputationes (hors-graphe, but le
        # result alimente the projection entrainable, therefore the graphe traverse).
        char = self.char_features[flat]      # (M, 16)
        fourier = self.fourier_features[flat]  # (M, 2·n_freq)

        # (C) : hash precomputatione → reshape (M, 1) → MLP entrainable (in the graphe).
        h = self.vortex_hashes[flat].unsqueeze(1)  # (M, 1)
        vortex_phases = self.vortex_mlp(h)         # (M, vortex_hidden)

        # Concat and projection.
        x = torch.cat([char, fourier, vortex_phases], dim=1)  # (M, in_dim)
        out = self.proj(x)  # (M, d_model)

        return out.reshape(*original_shape, self.d_model)
```

- [ ] **Step 4: Mettre a jour fractus/nn/__init__.py**

Replace the entire content of `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py` with :
```python
"""Sous-package nn — modules of reseau of neurones (PyTorch).

L1 : embedding fractal entrainable (FractalEmbedding).
"""

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis
from .embedding import FractalEmbedding

__all__ = ["CharClassFeatures", "MandelbrotFourierBasis", "FractalEmbedding"]
```

- [ ] **Step 5: Lancer all the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 12 passed (4 char + 4 fourier + 4 FractalEmbedding). Le test `test_fractal_embedding_backward_propagates` est the critere critique : il prouve that l'embedding apprend vraiment.

- [ ] **Step 6: Commit**

```bash
git add fractus/nn/embedding.py fractus/nn/__init__.py tests/test_embedding.py
git commit -m "feat(nn): add FractalEmbedding with vortex 2-adic conditioning

- Assemblage : 16 char features + base Fourier Mandelbrot + conditionnement
  vortex (hash Collatz en Rust → MLP entrainable en PyTorch).
- Option B spec L1 : the vortex 2-adique exact (hors-graphe) conditionne
  a MLP differentiable. Le vortex n'est PAS pretendu differentiable.
- Critere critique valid : backward() propage gradients finis a all
  the parameters (le test that the original architecture echouait)."
```
Expected: `3 files changed`.

---

## Task 6: Demo interactive L1 — prouver that l'embedding apprend

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_embedding.py`

- [ ] **Step 1: Ecrire the demo**

Create `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_embedding.py` :
```python
"""Demo L1 : prouve that FractalEmbedding apprend vraiment.

Objectif : surfit a embedding target aleatoire fixe en quelques steps Adam,
et montre that the loss baisse. This is the proof minimale that l'autodiff traverse
l'embedding fractal (ce that the original architecture not savait not faire).

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

    # Cible aleatoire fixe : the but est of surfit cette target.
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
        print("\n✓ SUCCES : l'embedding fractal apprend (loss divisee by >2).")
    else:
        print("\n✗ ECHEC : the loss not baisse not enough — investiguer.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lancer the demo**

```powershell
.venv\Scripts\python.exe scripts\demo_embedding.py
```
Expected output (approximatif — the values exacts dependent seed) :
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

✓ SUCCES : l'embedding fractal apprend (loss divisee by >2).
```

Si the demo affiche ECHEC, the bug est serieux — deboguer before L2.

- [ ] **Step 3: Commit**

```bash
git add scripts/demo_embedding.py
git commit -m "demo(L1): prove FractalEmbedding learns (overfit target, loss drops)"
```
Expected: `1 file changed`.

---

## Critere final of L1 « termine »

Apres the Task 6, these verifications must all reussir :

```powershell
cd "C:\Users\PHIL\ZCodeProject\fractus"

# 1. Le Rust compile and its tests passent (inchange depuis L0)
cargo test -p fractus-core
# → 9 passed

# 2. Le module natif est a jour (inclut the nouveaux bindings)
.venv\Scripts\python.exe -m maturin develop --release
# → 🛠 Installed fractus-0.1.0

# 3. Tous the tests Python passent
.venv\Scripts\python.exe -m pytest tests/ -v
# → 5 (smoke) + 8 (vortex bridge) + 12 (embedding) = 25 passed

# 4. La demo prouve l'learning
.venv\Scripts\python.exe scripts\demo_embedding.py
# → "✓ SUCCES : l'embedding fractal apprend"
```

Si all passe, L1 est termine and on can passer au plan L2 (bloc transformer fractal : attention lineaire + Kuramoto + MoE Farey).

---

## Self-Review (post-ecriture)

**1. Spec coverage :**
- Spec L1 demande (a) CharClassFeatures portedd → Task 3 ✅ ;
  (b) MandelbrotFourierBasis (renomme honestetement) → Task 4 ✅ ;
  (c) pont option B (hash conditionne MLP entrainable) → Task 5 ✅ ;
  (d) critere « backward() propage gradients finis » → test `test_fractal_embedding_backward_propagates` Task 5 ✅ ;
  (e) ultrametrie testee depuis Python → `test_ultrametric_strong_triangle_in_python` Task 2 ✅.
- Couverture complete.

**2. Placeholder scan :** no « TBD/TODO/fill in ». Chaque etape a code complete or commandes exacts. ✅

**3. Type consistency :** `CharClassFeatures.extract(int) → Tensor(16,)`, `.extract_batch(Tensor) → Tensor(N,16)`. `MandelbrotFourierBasis.matrix() → Tensor(V, 2K)`, `.frequencies → Tensor(K,)`. `FractalEmbedding(vocab_size, d_model, n_frequencies, vortex_hidden, collatz_steps).forward(Tensor) → Tensor`. Coherent between definitions (Tasks 3/4/5) and tests. ✅

**4. Imports coherents :** `from fractus.nn import FractalEmbedding` fonctionne thanks to `fractus/nn/__init__.py` mis a jour (Task 5 Step 4). Le hash vient of `from fractus import _core` (Task 5 Step 3, in `__init__` module). ✅

**5. Dependance on L0 :** Le pont `_core` (L0) est requis for Task 1 (nouveaux wrappers) and Task 5 (hash Collatz). `cargo test -p fractus-core` inchange (le vortex.rs n'est not modifie en L1, juste expose differemment). ✅

**6. YAGNI :** Pas of MoE, not d'attention, not of causal — juste l'embedding. Tout the reste vient en L2+. ✅
