# Fractus L1 — Embedding fractal + vortex 2-adique branché Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Brancher le vortex 2-adique d'OMNI-FRACTAL sur le réseau de neurones pour la première fois, en l'utilisant comme **conditionnement** d'un MLP entraînable PyTorch. Ajouter l'embedding de codepoint fractal (base de Fourier à décroissance Mandelbrot + 16 features morphologiques), tout en PyTorch pur pour l'autodiff. Corriger deux erreurs des systèmes originaux : le vortex orphelin (jamais importé) et les « Mandelbrot frequencies » mal nommées.

**Architecture:** (1) `fractus-core` (Rust) expose `collatz_hash(token_id)` et `ultrametric_distance(a,b)` — calcul exact, hors-graphe autodiff, appelé depuis Python pour précalculer le conditionnement. (2) `fractus/nn/embedding.py` (PyTorch) contient trois modules composés : `CharClassFeatures` (16 features morphologiques déterministes, portées de FNN), `MandelbrotFourierBasis` (base de Fourier à décroissance `(φ²)^{-k}`, nommée honnêtement), et `FractalEmbedding` (assemblage final : les features morpho + la base de Fourier + les phases vortex-conditionnées sont projetées vers `d_model` par un `nn.Linear` entraînable). La forward est différentiable de bout en bout.

**Tech Stack:** Rust 1.94 + pyo3 0.29 (déjà installé en L0) ; Python 3.14 + torch 2.12 CPU + numpy (déjà installés en L0) ; pytest. Le module natif doit être reconstruit via `maturin develop` après modification du Rust.

**Lien spec :** `docs/superpowers/specs/2026-06-19-fractus-unified-design.md`, section « L1 — Embedding fractal + vortex 2-adique branché ». Décision clé validée : **option B** — le hash 2-adique exact (Rust, hors-graphe) conditionne un MLP entraînable (PyTorch, dans le graphe).

**Prérequis :** L0 terminé (repo `C:/Users/PHIL/ZCodeProject/fractus/` opérationnel, venv `.venv` avec torch + maturin installés, 9 tests Rust + 5 tests Python passent).

**Vocabulaire honnête appliqué dans ce plan :**
- « Mandelbrot frequencies » → « Mandelbrot-decayed Fourier basis » (la décroissance `(φ²)^{-k}` est réelle et justifiée, mais ce n'est pas l'ensemble de Mandelbrot).
- « Collatz ergodic flow » → « Collatz hash » (l'ergodicité de Collatz est non démontrée, problème ouvert).
- On parle de « norme 2-adique » et « distance ultramétrique » (termes mathématiques exacts).

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── crate/fractus-core/src/
│   ├── lib.rs                  # inchangé (déclare déjà pub mod vortex)
│   └── vortex.rs               # MODIFY : expose aussi norm_2adic (déjà défini, juste à binder)
├── crate/fractus-py/src/
│   └── lib.rs                  # MODIFY : ajoute wrappers #[pyfunction] pour collatz_hash,
│                               #   ultrametric_distance, norm_2adic
├── fractus/nn/
│   ├── __init__.py             # MODIFY : exporte les classes publiques
│   ├── char_features.py        # CREATE : 16 features morphologiques (porté de FNN)
│   ├── fourier.py              # CREATE : base de Fourier à décroissance Mandelbrot
│   └── embedding.py            # CREATE : FractalEmbedding (assemblage + projection Linear)
└── tests/
    ├── test_vortex_bridge.py   # CREATE : tests du pont Python des fonctions 2-adiques
    └── test_embedding.py       # CREATE : tests de l'embedding fractal
```

**Responsabilités (un fichier = une responsabilité) :**
- `char_features.py` : uniquement les 16 features morphologiques (déterministe, sans paramètre).
- `fourier.py` : uniquement la base de Fourier à décroissance Mandelbrot (déterministe, sans paramètre).
- `embedding.py` : l'assemblage + la projection entraînable (le seul endroit avec des `nn.Parameter`).
- `test_vortex_bridge.py` : vérifie que les fonctions Rust sont bien appelables depuis Python et retournent les bonnes valeurs (pont opérationnel).
- `test_embedding.py` : vérifie formes, finitude, et surtout que `backward()` propage des gradients finis (le critère critique hérité de FNN qui échouait là).

---

## Task 1: Exposer collatz_hash, ultrametric_distance, norm_2adic dans les bindings Python

**Files:**
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs`

- [ ] **Step 1: Réécrire lib.rs avec les nouveaux wrappers**

Replace the entire content of `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs` with :

```rust
//! Bindings Python (PyO3) pour fractus-core.
//!
//! Ce crate ne contient AUCUNE logique — seulement des wrappers #[pyfunction]
//! qui délèguent à fractus-core. Le but est d'exposer le Rust à Python
//! sous le nom `fractus._core`.

use pyo3::prelude::*;

/// Addition entière — wrapper Python pour fractus_core::add.
/// Exposée uniquement pour le test fume.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    fractus_core::add(a, b)
}

/// Hash Collatz d'un token id. Wrapper pour fractus_core::vortex::collatz_hash.
/// Utilisé comme conditionnement déterministe (hors-graphe autodiff) pour
/// l'embedding fractal (option B du spec L1).
#[pyfunction]
fn collatz_hash(x: u64, steps: u32) -> u64 {
    fractus_core::vortex::collatz_hash(x, steps)
}

/// Distance ultramétrique 2-adique : d(a,b) = 2^{-v_2(a ⊕ b)}.
/// Wrapper pour fractus_core::vortex::ultrametric_distance. Dans (0, 1].
#[pyfunction]
fn ultrametric_distance(a: u64, b: u64) -> f64 {
    fractus_core::vortex::ultrametric_distance(a, b)
}

/// Norme 2-adique : ||x||_2 = 2^{-v_2(x)}. Wrapper pour fractus_core::vortex::norm_2adic.
#[pyfunction]
fn norm_2adic(x: u64) -> f64 {
    fractus_core::vortex::norm_2adic(x)
}

/// Module Python `fractus._core`.
///
/// Signature pyo3 0.29 : le module est reçu comme `&Bound<'_, PyModule>`.
/// Les méthodes `.add_function(...)` viennent du trait `PyModuleMethods`
/// (ré-exporté par `pyo3::prelude`).
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

Run (depuis `C:/Users/PHIL/ZCodeProject/fractus`, venv activé) :
```powershell
.venv\Scripts\python.exe -m maturin develop --release
```
Expected: `🛠 Installed fractus-0.1.0`. Peut prendre 30-60s (recompile pyo3 si le hash a changé).

- [ ] **Step 3: Vérifier vite que les nouvelles fonctions sont exposées**

```powershell
.venv\Scripts\python.exe -c "from fractus import _core; print('hash(7,5)=', _core.collatz_hash(7,5)); print('dist(1,2)=', _core.ultrametric_distance(1,2)); print('norm(8)=', _core.norm_2adic(8))"
```
Expected: trois valeurs numériques sans erreur (par ex. `hash(7,5)= 52`, `dist(1,2)= 0.5`, `norm(8)= 0.125`).

- [ ] **Step 4: Commit**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add crate/fractus-py/src/lib.rs
git commit -m "feat(py): expose collatz_hash, ultrametric_distance, norm_2adic to Python"
```
Expected: `1 file changed`.

---

## Task 2: Tests du pont Python des fonctions 2-adiques (TDD)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/tests/test_vortex_bridge.py`

- [ ] **Step 1: Écrire les tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_vortex_bridge.py` :
```python
"""Tests du pont Python pour les fonctions 2-adiques du vortex.

Vérifie que les wrappers Rust sont bien exposés et retournent des valeurs
correctes. Ces tests ne font PAS de mathématique avancée (ça, c'est en Rust) —
ils valident juste le pont PyO3.
"""

import pytest


def test_collatz_hash_is_deterministic():
    """Même entrée → même sortie (propriété requise pour le conditionnement)."""
    from fractus import _core
    h1 = _core.collatz_hash(7, 10)
    h2 = _core.collatz_hash(7, 10)
    assert h1 == h2


def test_collatz_hash_zero_stays_zero():
    """Convention : 0 → 0."""
    from fractus import _core
    assert _core.collatz_hash(0, 100) == 0


def test_collatz_hash_returns_u64():
    """Le hash doit être un entier positif compatible avec PyTorch indexing."""
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
    """||x||_2 = 2^{-v_2(x)}, vérifié sur quelques valeurs connues."""
    from fractus import _core
    assert _core.norm_2adic(0) == 0.0
    assert _core.norm_2adic(1) == 1.0   # v_2(1)=0 → 2^0
    assert _core.norm_2adic(2) == 0.5   # v_2(2)=1 → 2^-1
    assert _core.norm_2adic(8) == 0.125  # v_2(8)=3 → 2^-3


def test_ultrametric_strong_triangle_in_python():
    """La propriété ultramétrique forte doit tenir via le pont Python.
    C'est le test-pivot qui distingue 2^{-v} (correct) de 2^{+v} (bug OMNI)."""
    from fractus import _core
    # Le triplet (7, 56, 13) discrimine : passe avec -v, échoue avec +v.
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
Expected: 8 passed. Si un test échoue, le pont PyO3 est cassé — vérifier que Task 1 a bien reconstruit le module.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vortex_bridge.py
git commit -m "test(vortex): 8 tests du pont Python pour collatz_hash/ultrametric/norm"
```
Expected: `1 file changed`.

---

## Task 3: Implémenter les 16 features morphologiques (CharClassFeatures)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/char_features.py`

- [ ] **Step 1: Écrire le test qui échoue**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_embedding.py` (initial content — sera étendu dans les tâches suivantes) :
```python
"""Tests de l'embedding fractal : char features, base de Fourier, FractalEmbedding.

Le critère critique (hérité de FNN qui échouait là) : la forward doit être
différentiable et backward() doit propager des gradients finis partout.
"""

import torch
import pytest


# ---------------------------------------------------------------------------
# Task 3 : CharClassFeatures (16 features morphologiques)
# ---------------------------------------------------------------------------

def test_char_features_shape():
    """16 features pour tout token id."""
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
    """La même lettre donne le même vecteur de features."""
    from fractus.nn.char_features import CharClassFeatures
    f1 = CharClassFeatures.extract(ord("z"))
    f2 = CharClassFeatures.extract(ord("z"))
    assert torch.equal(f1, f2)
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'fractus.nn.char_features'`. C'est normal.

- [ ] **Step 3: Implémenter CharClassFeatures**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/char_features.py` :
```python
"""16 features morphologiques déterministes par token.

Porté depuis FNN v5.0 (src/embedding.rs, CharClassFeatures). Le token id est
interprété comme un codepoint Unicode ; pour les ids < 128 ce sont des
caractères ASCII, au-delà on dérive les features de la valeur numérique.

Ces features n'ont AUCUN paramètre entraînable — elles sont calculées
déterministiquement puis concaténées à la base de Fourier dans FractalEmbedding.
"""

import torch


class CharClassFeatures:
    """Extraction de 16 features morphologiques à partir d'un token id.

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
        12 : char_category     (catégorie Unicode simplifiée comme float)
        13 : position_in_alphabet (0-25, ou -1 si pas une lettre ; on encode -1→0)
        14 : is_ascii          (codepoint < 128)
        15 : parity            (token id pair = 1, impair = 0)
    """

    N_FEATURES = 16

    VOWELS = frozenset(b"aeiouAEIOU")

    @staticmethod
    def extract(token_id: int) -> torch.Tensor:
        """Retourne un tenseur float32 de forme (16,)."""
        f = torch.zeros(CharClassFeatures.N_FEATURES, dtype=torch.float32)

        # On interprète l'octet de poids faible comme un caractère potentiel.
        as_byte = (token_id & 0xFF)

        # 0: is_vowel
        is_vowel = float(as_byte in CharClassFeatures.VOWELS)
        f[0] = is_vowel

        # 1: is_consonant (lettre alphabétique non voyelle)
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

        # 12: char_category simplifié : 1.0 lettre, 2.0 chiffre, 3.0 ponctuation,
        #     4.0 espace, 5.0 contrôle, 0.0 autre.
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
        """Version vectorisée : token_ids de forme (N,) → features (N, 16).

        Comme le calcul est déterministe et indépendant par token, on peut
        précalculer une lookup table une fois pour toute la taille du vocab.
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

## Task 4: Implémenter la base de Fourier à décroissance Mandelbrot

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/fourier.py`

- [ ] **Step 1: Ajouter les tests à tests/test_embedding.py**

Append to `C:/Users/PHIL/ZCodeProject/fractus/tests/test_embedding.py` (after the existing char-features tests) :
```python


# ---------------------------------------------------------------------------
# Task 4 : MandelbrotFourierBasis (base de Fourier à décroissance (φ²)^{-k})
# ---------------------------------------------------------------------------

def test_fourier_basis_shape():
    """Pour vocab 128 et 32 fréquences : matrice (vocab, n_freq) en entrée du calcul."""
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
    """Les fréquences ω_k = (φ²)^{-k} doivent décroître géométriquement."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=10, n_frequencies=4)
    # ω_0 = 1.0, ω_1 = 1/φ², ω_2 = 1/φ⁴, ...
    phi_sq = ((1 + 5 ** 0.5) / 2) ** 2
    expected = [phi_sq ** (-k) for k in range(4)]
    for k, exp in enumerate(expected):
        assert abs(basis.frequencies[k].item() - exp) < 1e-5, \
            f"freq[{k}] = {basis.frequencies[k].item()}, attendu {exp}"


def test_fourier_matrix_is_deterministic():
    """Deux appels donnent la même matrice (pas d'aléa)."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    b1 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    b2 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    assert torch.allclose(b1.matrix(), b2.matrix())
```

- [ ] **Step 2: Lancer pour vérifier que les nouveaux tests échouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 4 passed (char features), 4 failed/error (fourier — module absent).

- [ ] **Step 3: Implémenter MandelbrotFourierBasis**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/fourier.py` :
```python
"""Base de Fourier à décroissance Mandelbrot pour l'embedding fractal.

Inspiré de FNN v5.0 (src/math/mandelbrot.rs + src/embedding.rs) mais renommé
honnêtement : FNN appelait ça "Mandelbrot frequencies" en référence à l'ensemble
de Mandelbrot, alors qu'il s'agit juste d'une décroissance géométrique de base
φ² (le carré du nombre d'or). On appelle donc ça "Mandelbrot-decayed Fourier
basis" — la décroissance est réelle et justifiée (séparation d'échelles
multi-niveaux), mais le lien à l'ensemble de Mandelbrot est nul.

Mathématique :
    φ = (1 + √5) / 2  ≈ 1.618
    φ² ≈ 2.618
    ω_k = (φ²)^{-k}    pour k = 0, 1, ..., n_freq-1

La base de Fourier associe à chaque token id t et chaque fréquence k la paire
(sin, cos) de ω_k · t :
    M[t, 2k]   = sin(ω_k · t)
    M[t, 2k+1] = cos(ω_k · t)

On stocke n_freq fréquences ; la matrice produite a 2·n_freq colonnes
(sin+cos par fréquence). Le caller (FractalEmbedding) gère la projection finale.

AUCUN paramètre entraînable ici : tout est déterministe, précalculé une fois.
"""

import math
import torch


class MandelbrotFourierBasis:
    """Base de Fourier déterministe avec décroissance (φ²)^{-k}.

    Attributs :
        vocab_size   : nombre de token ids couverts (0 .. vocab_size-1)
        n_frequencies : nombre de fréquences ω_k
        frequencies  : tenseur (n_frequencies,) des ω_k, en float32
    """

    def __init__(self, vocab_size: int, n_frequencies: int):
        if vocab_size <= 0 or n_frequencies <= 0:
            raise ValueError("vocab_size et n_frequencies doivent être > 0")
        self.vocab_size = vocab_size
        self.n_frequencies = n_frequencies

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        phi_sq = phi * phi  # ≈ 2.618
        ks = torch.arange(n_frequencies, dtype=torch.float32)
        # ω_k = (φ²)^{-k}
        self.frequencies = phi_sq ** (-ks)

        # Précalcul de la matrice (vocab_size, 2·n_frequencies).
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> torch.Tensor:
        """Construit la matrice M[t, :] = [sin(ω_k·t), cos(ω_k·t)] pour tout k."""
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
        """Retourne la matrice précalculée (vocab_size, 2·n_frequencies)."""
        return self._matrix

    def dim_output(self) -> int:
        """Dimension de sortie (nombre de colonnes de la matrice)."""
        return 2 * self.n_frequencies
```

- [ ] **Step 4: Lancer tous les tests — DOIVENT PASSER**

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

## Task 5: Implémenter FractalEmbedding (assemblage + projection entraînable)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/embedding.py`
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py`

- [ ] **Step 1: Ajouter les tests de FractalEmbedding**

Append to `C:/Users/PHIL/ZCodeProject/fractus/tests/test_embedding.py` :
```python


# ---------------------------------------------------------------------------
# Task 5 : FractalEmbedding (assemblage + projection entraînable)
# ---------------------------------------------------------------------------

def test_fractal_embedding_shape():
    """Sortie (N, d_model) pour entrée (N,) d'ids."""
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
    """CRITIQUE : backward() doit propager des gradients finis à tous les params.

    C'est exactement le test que FNN v5.0 échouait (training.rs:399 utilisait du
    bruit aléatoire au lieu d'un gradient). Ici, la projection Linear est dans
    le graphe autodiff, donc les gradients doivent être non-nuls et finis.
    """
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.tensor([0, 1, 2, 3, 4])
    out = emb(ids)
    loss = out.pow(2).sum()
    loss.backward()

    has_param_with_grad = False
    for name, p in emb.named_parameters():
        assert p.requires_grad, f"{name} devrait requires_grad=True"
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
            if p.grad.abs().sum().item() > 0:
                has_param_with_grad = True
    assert has_param_with_grad, "Aucun paramètre n'a reçu de gradient non-nul"


def test_fractal_embedding_respects_vocab_bounds():
    """Un id >= vocab_size doit lever une erreur (pas de crash silencieux)."""
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=100, d_model=32, n_frequencies=8)
    with pytest.raises(IndexError):
        emb(torch.tensor([100]))  # hors borne
```

- [ ] **Step 2: Lancer pour vérifier que les tests FractalEmbedding échouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 8 passed (char + fourier), 4 failed/error (FractalEmbedding absent).

- [ ] **Step 3: Implémenter FractalEmbedding**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/embedding.py` :
```python
"""FractalEmbedding : embedding de codepoint fractal entraînable.

Assemblage de trois sources de features pour chaque token id t :

    (A) 16 features morphologiques déterministes (CharClassFeatures)
    (B) base de Fourier à décroissance Mandelbrot (MandelbrotFourierBasis)
    (C) conditionnement vortex : un hash 2-adique (Collatz, calculé en Rust,
        hors-graphe autodiff) est projeté en phases via un MLP entraînable
        (PyTorch, dans le graphe). C'est l'option B du spec L1 : le vortex
        2-adique influence l'apprentissage sans prétendre être différentiable.

La projection finale vers d_model est un nn.Linear entraînable. Toute la forward
est différentiable de bout en bout — les parties déterministes (A, B, et le hash
de C) sont précalculées en buffers hors-graphe ; seul le MLP de C et la
projection finale portent des paramètres.

Corrections vs systèmes originaux :
- FNN n'apprenait pas (training.rs:399 = bruit) → ici backward() marche (test).
- OMNI : le vortex 2-adique était orphelin (jamais importé par le Python) →
  ici il conditionne réellement l'embedding.
- OMNI : les « Mandelbrot frequencies » étaient mal nommées → ici on dit
  « Mandelbrot-decayed Fourier basis » (voir fourier.py).
"""

import torch
import torch.nn as nn

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis


class FractalEmbedding(nn.Module):
    """Embedding fractal entraînable.

    Args :
        vocab_size     : nombre de token ids couverts.
        d_model        : dimension de sortie.
        n_frequencies  : nombre de fréquences ω_k pour la base de Fourier.
        vortex_hidden  : largeur du MLP qui projette le hash Collatz en phases.
        collatz_steps  : nombre d'itérations Collatz pour le hash (déterministe).
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
            raise ValueError("vocab_size et d_model doivent être > 0")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.collatz_steps = collatz_steps

        # (A) Features morphologiques : précalcul déterministe, hors-graphe.
        char_matrix = torch.stack(
            [CharClassFeatures.extract(t) for t in range(vocab_size)], dim=0
        )  # (vocab, 16)
        self.register_buffer("char_features", char_matrix)

        # (B) Base de Fourier Mandelbrot-décroissante : précalcul déterministe.
        self.fourier = MandelbrotFourierBasis(vocab_size, n_frequencies)
        fourier_matrix = self.fourier.matrix()  # (vocab, 2·n_freq)
        self.register_buffer("fourier_features", fourier_matrix)

        # (C) Conditionnement vortex : hash Collatz précalculé (hors-graphe),
        # puis projeté par un MLP entraînable (dans le graphe).
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
        # Normalisation douce : on ramène dans [0, 1) via / (max+1) pour stabilité.
        max_h = hashes.max().item() + 1.0
        hashes_norm = hashes / max_h
        self.register_buffer("vortex_hashes", hashes_norm)  # (vocab,)

        # MLP entraînable : projette le scalaire hash (1) vers un vecteur de
        # dimension vortex_phase_dim. C'est ici que le vortex « conditionne »
        # le réseau : le MLP apprend à interpréter le hash 2-adique.
        self.vortex_phase_dim = vortex_hidden
        self.vortex_mlp = nn.Sequential(
            nn.Linear(1, vortex_hidden),
            nn.Tanh(),
            nn.Linear(vortex_hidden, vortex_hidden),
        )

        # Projection finale entraînable vers d_model.
        # dim d'entrée = 16 (char) + 2·n_freq (fourier) + vortex_hidden
        in_dim = 16 + fourier_matrix.shape[1] + vortex_hidden
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids : (N,) ou (N, L) d'entiers dans [0, vocab_size).

        Retourne (N, d_model) ou (N, L, d_model).
        """
        if token_ids.max() >= self.vocab_size or token_ids.min() < 0:
            raise IndexError(
                f"token_id hors [0, {self.vocab_size}) : "
                f"min={int(token_ids.min())}, max={int(token_ids.max())}"
            )

        original_shape = token_ids.shape
        flat = token_ids.reshape(-1)  # (M,)

        # (A) + (B) : lookup dans les buffers précalculés (hors-graphe, mais le
        # résultat alimente la projection entraînable, donc le graphe traverse).
        char = self.char_features[flat]      # (M, 16)
        fourier = self.fourier_features[flat]  # (M, 2·n_freq)

        # (C) : hash précalculé → reshape (M, 1) → MLP entraînable (dans le graphe).
        h = self.vortex_hashes[flat].unsqueeze(1)  # (M, 1)
        vortex_phases = self.vortex_mlp(h)         # (M, vortex_hidden)

        # Concat et projection.
        x = torch.cat([char, fourier, vortex_phases], dim=1)  # (M, in_dim)
        out = self.proj(x)  # (M, d_model)

        return out.reshape(*original_shape, self.d_model)
```

- [ ] **Step 4: Mettre à jour fractus/nn/__init__.py**

Replace the entire content of `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py` with :
```python
"""Sous-package nn — modules de réseau de neurones (PyTorch).

L1 : embedding fractal entraînable (FractalEmbedding).
"""

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis
from .embedding import FractalEmbedding

__all__ = ["CharClassFeatures", "MandelbrotFourierBasis", "FractalEmbedding"]
```

- [ ] **Step 5: Lancer tous les tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 12 passed (4 char + 4 fourier + 4 FractalEmbedding). Le test `test_fractal_embedding_backward_propagates` est le critère critique : il prouve que l'embedding apprend vraiment.

- [ ] **Step 6: Commit**

```bash
git add fractus/nn/embedding.py fractus/nn/__init__.py tests/test_embedding.py
git commit -m "feat(nn): add FractalEmbedding with vortex 2-adic conditioning

- Assemblage : 16 char features + base Fourier Mandelbrot + conditionnement
  vortex (hash Collatz en Rust → MLP entraînable en PyTorch).
- Option B du spec L1 : le vortex 2-adique exact (hors-graphe) conditionne
  un MLP différentiable. Le vortex n'est PAS prétendu différentiable.
- Critère critique validé : backward() propage des gradients finis à tous
  les paramètres (le test que FNN v5.0 échouait)."
```
Expected: `3 files changed`.

---

## Task 6: Démo interactive L1 — prouver que l'embedding apprend

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_embedding.py`

- [ ] **Step 1: Écrire la démo**

Create `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_embedding.py` :
```python
"""Démo L1 : prouve que FractalEmbedding apprend vraiment.

Objectif : surfit un embedding cible aléatoire fixe en quelques steps Adam,
et montre que la loss baisse. C'est la preuve minimale que l'autodiff traverse
l'embedding fractal (ce que FNN v5.0 ne savait pas faire).

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
    print(f"Paramètres entraînables : {sum(p.numel() for p in emb.parameters())}")

    # Cible aléatoire fixe : le but est de surfit cette cible.
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
    print(f"Loss initiale : {initial_loss:.4f}")
    print(f"Loss finale   : {final_loss:.4f}")
    print(f"Baisse        : {(1 - final_loss / initial_loss) * 100:.1f}%")

    if final_loss < initial_loss * 0.5:
        print("\n✓ SUCCÈS : l'embedding fractal apprend (loss divisée par >2).")
    else:
        print("\n✗ ÉCHEC : la loss ne baisse pas assez — investiguer.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lancer la démo**

```powershell
.venv\Scripts\python.exe scripts\demo_embedding.py
```
Expected output (approximatif — les valeurs exactes dépendent du seed) :
```
Paramètres entraînables : ~3000
step   0  loss = 1.2xxx
step  40  loss = 0.6xxx
step  80  loss = 0.3xxx
step 120  loss = 0.15xxx
step 160  loss = 0.08xxx
step 199  loss = 0.05xxx

Loss initiale : 1.2xxx
Loss finale   : 0.05xxx
Baisse        : ~95%

✓ SUCCÈS : l'embedding fractal apprend (loss divisée par >2).
```

Si la démo affiche ÉCHEC, le bug est sérieux — déboguer avant L2.

- [ ] **Step 3: Commit**

```bash
git add scripts/demo_embedding.py
git commit -m "demo(L1): prove FractalEmbedding learns (overfit target, loss drops)"
```
Expected: `1 file changed`.

---

## Critère final de L1 « terminé »

Après le Task 6, ces vérifications doivent toutes réussir :

```powershell
cd "C:\Users\PHIL\ZCodeProject\fractus"

# 1. Le Rust compile et ses tests passent (inchangé depuis L0)
cargo test -p fractus-core
# → 9 passed

# 2. Le module natif est à jour (inclut les nouveaux bindings)
.venv\Scripts\python.exe -m maturin develop --release
# → 🛠 Installed fractus-0.1.0

# 3. Tous les tests Python passent
.venv\Scripts\python.exe -m pytest tests/ -v
# → 5 (smoke) + 8 (vortex bridge) + 12 (embedding) = 25 passed

# 4. La démo prouve l'apprentissage
.venv\Scripts\python.exe scripts\demo_embedding.py
# → "✓ SUCCÈS : l'embedding fractal apprend"
```

Si tout passe, L1 est terminé et on peut passer au plan L2 (bloc transformer fractal : attention linéaire + Kuramoto + MoE Farey).

---

## Self-Review (post-écriture)

**1. Spec coverage :**
- Spec L1 demande (a) CharClassFeatures porté → Task 3 ✅ ;
  (b) MandelbrotFourierBasis (renommé honnêtement) → Task 4 ✅ ;
  (c) pont option B (hash conditionne MLP entraînable) → Task 5 ✅ ;
  (d) critère « backward() propage gradients finis » → test `test_fractal_embedding_backward_propagates` Task 5 ✅ ;
  (e) ultramétrie testée depuis Python → `test_ultrametric_strong_triangle_in_python` Task 2 ✅.
- Couverture complète.

**2. Placeholder scan :** aucun « TBD/TODO/fill in ». Chaque étape a du code complet ou des commandes exactes. ✅

**3. Type consistency :** `CharClassFeatures.extract(int) → Tensor(16,)`, `.extract_batch(Tensor) → Tensor(N,16)`. `MandelbrotFourierBasis.matrix() → Tensor(V, 2K)`, `.frequencies → Tensor(K,)`. `FractalEmbedding(vocab_size, d_model, n_frequencies, vortex_hidden, collatz_steps).forward(Tensor) → Tensor`. Cohérent entre définitions (Tasks 3/4/5) et tests. ✅

**4. Imports cohérents :** `from fractus.nn import FractalEmbedding` fonctionne grâce à `fractus/nn/__init__.py` mis à jour (Task 5 Step 4). Le hash vient de `from fractus import _core` (Task 5 Step 3, dans `__init__` du module). ✅

**5. Dépendance sur L0 :** Le pont `_core` (L0) est requis pour Task 1 (nouveaux wrappers) et Task 5 (hash Collatz). `cargo test -p fractus-core` inchangé (le vortex.rs n'est pas modifié en L1, juste exposé différemment). ✅

**6. YAGNI :** Pas de MoE, pas d'attention, pas de causal — juste l'embedding. Tout le reste vient en L2+. ✅
