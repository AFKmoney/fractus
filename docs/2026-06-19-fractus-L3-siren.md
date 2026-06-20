# Fractus L3 — Vraie SIREN + compressiwe measuree honestetement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corriger the falsehood the more visible d'the original design : the fausse SIREN (utilisait `nn.SiLU` instead of `sin(ω0·)`) and the ratio of compression hardcode a 20.4×. On implemente a **vraie SIREN** (sinusoides comme non-linearite, ω0=30 justifie by Sitzmann 2020), on l'applique aux matrixs of poids transformer fractus, and on **mesure honestetement** the ratio of compression obtenu.

**Position scientifique honestete (decision valide) :** Une SIREN represente well functions **lisses** (images, champs scalars). Or the poids d'un reseau entraine are essentiellement **bruit structure dense**. On s'attend therefore a a ratio of compression **faible** (~1× a 3×), **pas** 20.4×. La documentation L3 dira franchement pourquoi, and the demo mesurera the verite. This is exactment l'inverse falsehood d'the original.

**Architecture :** (1) `fractus/nn/siren.py` — `TorusSirenWeight` : vraie SIREN `sin(ω0·(Wx+b))` on the tore T2 = [0,1)2, which regenerated a matrix W[h,w] depuis a grille of coordonnees. (2) `SirenLinear` : a `nn.Module` which se comported comme `nn.Linear` but dont the matrix of poids est produite by a SIREN (parameters SIREN entrainables). (3) `fractus/metrics/compression.py` : `measure_compression_ratio(model)` mesure reallement the ratio (taille dense equivalente / params SIREN). (4) Demo : on remplace the projections d'attention by `SirenLinear`, on entraine, we measure.

**Tech Stack:** PyTorch 2.12 CPU, pytest.

**Lien spec :** `docs/superpowers/specs/2026-06-19-fractus-unified-design.md`, section « L3 — Compression SIREN vraie + mesure honestete ».

**Prerequis :** L2 termine (62 tests passent, FractalBlockFull fonctionne).

**Maths of reference (SIREN, Sitzmann and al. 2020) :**
- Non-linearite : `sin(ω0 · (Wx + b))` (PAS SiLU, PAS ReLU).
- ω0 = 30.0 (value empirique papier SIREN — **pas** 56, which n'est not justifie).
- Init speciale couches : premiere couche `U(-1/ω0, 1/ω0)` ; couches suivantes `U(-√(6/ω02·fan_in), √(6/ω02·fan_in))`.
- Pour the compression of poids : on evalue the SIREN on a grille (h,w) → matrix W[h,w].

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── fractus/nn/
│   ├── __init__.py             # MODIFY : exported TorusSirenWeight, SirenLinear
│   ├── siren.py                # CREATE : TorusSirenWeight (vraie SIREN sin(ω0·))
│   └── siren_linear.py         # CREATE : SirenLinear (Linear dont W vient d'une SIREN)
├── fractus/metrics/
│   ├── __init__.py             # CREATE
│   └── compression.py          # CREATE : measure_compression_ratio (mesure honestete)
└── tests/
    ├── test_siren.py           # CREATE : tests SIREN (sin present, not SiLU, backward)
    ├── test_siren_linear.py    # CREATE : tests SirenLinear (shape, backward)
    └── test_compression.py     # CREATE : test measure_compression_ratio (pas of hardcode)
```

**Responsabilites :**
- `siren.py` : the SIREN pure (represente a champ scalar on T2).
- `siren_linear.py` : adaptateur `nn.Module` which produit a `nn.Linear`-like dont W = SIREN(grid).
- `metrics/compression.py` : mesure real ratio, AUCUN litteral hardcode.

---

## Task 1: TorusSirenWeight (vraie SIREN sin(ω0·))

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren.py`

- [ ] **Step 1: Ecrire the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_siren.py` :
```python
"""Tests of TorusSirenWeight : vraie SIREN sin(ω0·), not SiLU."""

import inspect
import torch


def test_siren_uses_sin_not_silu():
    """CRITERE L3 : the SIREN must utiliser torch.sin, PAS nn.SiLU.
    This is exactment the falsehood d'the original (torus_siren.py:15,17 utilisait SiLU)."""
    from fractus.nn import siren
    src = inspect.getsource(siren)
    assert 'torch.sin' in src or 'sin(' in src, "La SIREN must utiliser sin(ω0·)"
    assert 'SiLU' not in src and 'silu' not in src.lower(), \
        "Plus of SiLU (le falsehood d'the original)"


def test_siren_omega0_is_30_not_56():
    """ω0 = 30 (justifie by Sitzmann 2020), PAS 56 (non justifie, heritage the original)."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    assert abs(s.omega0 - 30.0) < 1e-6, f"ω0 should etre 30.0, eu {s.omega0}"


def test_siren_output_shape():
    """La SIREN evaluee on the grille produit a matrix (out_h, out_w)."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    assert W.shape == (16, 16)


def test_siren_is_finite():
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    assert torch.isfinite(W).all()


def test_siren_backward_propagates():
    """CRITERE L3 : backward propage a gradient fini ET non-nul a CHAQUE parameter."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    loss = W.pow(2).sum()
    loss.backward()

    params = list(s.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a a gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a recu a gradient nul"


def test_siren_fewer_params_than_dense():
    """La SIREN must avoir MOINS of parameters that the matrix dense equivalente
    (sinon il n'y a not of compression). Pour (32,32) with hidden=16 :
    dense = 1024 params, SIREN ≈ 2·16 + 16·16 + 16·1 + biases ≈ 300-400.
    Donc ratio > 2 attendu AU NIVEAU DES PARAMETRES. (Mais the quality de
    reconstruction est a other question — voir demo.)"""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=32, out_w=32, hidden=16)
    n_siren = sum(p.numel() for p in s.parameters())
    n_dense = 32 * 32
    assert n_siren < n_dense, \
        f"SIREN ({n_siren} params) should etre < dense ({n_dense})"
```

- [ ] **Step 2: Lancer for verify that the tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_siren.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer TorusSirenWeight**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren.py` :
```python
"""TorusSirenWeight : vraie SIREN sin(ω0·) for representer a matrix of poids.

CORRECTION DU MENSONGE D'the original design :
- the original utilisait nn.SiLU (torus_siren.py:15,17) → ici on utilise torch.sin(ω0·(Wx+b)),
  the VRAIE non-linearite SIREN (Sitzmann and al. 2020).
- the original utilisait ω0=56 non justifie → ici ω0=30.0 (value empirique papier SIREN,
  which montre that ω0≈30 est optimal for the representation of functions continues).
- the original commentait "Simple reconstruction: sum of harmonics (real implementation uses
  Fourier)" (torus_siren.py:39) → ici the reconstruction est REELLE (forward SIREN
  on grille 2D).

POSITION SCIENTIFIQUE HONNETE :
Une SIREN represente well functions lisses (images, champs scalars).
Les poids d'un reseau entraine are essentiellement bruit structure dense.
On s'attend therefore a a ratio of compression FAIBLE (~1× a 3×), PAS 20.4×.
The ratio is MEASURED (metrics/compression.py), never hardcoded.

Math (Sitzmann 2020) :
    Non-linearite : sin(ω0 · (Wx + b)) for each couche cachee.
    Couche of sortie : lineaire (pas of sin).
    Init : premiere couche U(-1/ω0, 1/ω0) ; suivantes U(-√(6/(ω02·fan_in)), ...).

La SIREN prend en entree coords (u,v) ∈ [0,1)2 on the tore T2 and produit
un scalar W[u,v]. Evaluee on a grille h×w, elle regenerated the matrix W.
"""

import math
import torch
import torch.nn as nn


class TorusSirenWeight(nn.Module):
    """SIREN which represente a matrix of poids W[out_h, out_w] comme a champ
    scalar on the tore T2 = [0,1)2.

    Args:
        out_h, out_w : dimensions of the matrix a regenerate.
        hidden       : width couches cachees of the SIREN.
        omega0       : frequence fondamentale (30.0 by defaut, Sitzmann 2020).
    """

    def __init__(
        self,
        out_h: int,
        out_w: int,
        hidden: int = 32,
        omega0: float = 30.0,
    ):
        super().__init__()
        if out_h < 1 or out_w < 1 or hidden < 1:
            raise ValueError("out_h, out_w, hidden must etre >= 1")
        self.out_h = out_h
        self.out_w = out_w
        self.hidden = hidden
        self.omega0 = omega0

        # Couches : Linear(2 → hidden) → sin → Linear(hidden → hidden) → sin → Linear(hidden → 1).
        # Trois couches au total (comme SIREN papier for champs scalars).
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self._init_siren_weights()

        # Grille precomputationee (hors-graphe because constant).
        grid = self._build_grid(out_h, out_w)  # (out_h·out_w, 2)
        self.register_buffer("grid", grid)

    def _init_siren_weights(self):
        """Init SIREN specifique (Sitzmann 2020, section 3.2)."""
        # Premiere couche : U(-1/ω0, 1/ω0).
        with torch.no_grad():
            nn.init.uniform_(self.fc1.weight, -1.0 / self.omega0, 1.0 / self.omega0)
            nn.init.zeros_(self.fc1.bias)
            # Couches suivantes : U(-√(6/(ω02·fan_in)), √(6/(ω02·fan_in))).
            for layer in [self.fc2, self.fc3]:
                fan_in = layer.weight.shape[1]
                bound = math.sqrt(6.0 / (self.omega0 ** 2 * fan_in))
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _build_grid(h: int, w: int) -> torch.Tensor:
        """Grille of coords (u,v) ∈ [0,1)2 on the tore, shape (h·w, 2)."""
        u = torch.linspace(0, 1, h, dtype=torch.float32)
        v = torch.linspace(0, 1, w, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1)  # (h, w, 2)
        return grid.reshape(-1, 2)  # (h·w, 2)

    def forward(self) -> torch.Tensor:
        """Evalue the SIREN on the grille → matrix W[out_h, out_w].

        This is the 'decompression' : on regenerated W depuis the params SIREN.
        """
        x = self.grid  # (h·w, 2)
        # Couche 1 + sin(ω0·).
        x = torch.sin(self.omega0 * self.fc1(x))
        # Couche 2 + sin(ω0·).
        x = torch.sin(self.omega0 * self.fc2(x))
        # Couche of sortie : lineaire (pas of sin).
        x = self.fc3(x)  # (h·w, 1)
        return x.squeeze(-1).reshape(self.out_h, self.out_w)
```

- [ ] **Step 4: Lancer the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_siren.py -v
```
Expected: 6 passed. Le test `test_siren_uses_sin_not_silu` est the critere L3 critique.

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add fractus/nn/siren.py tests/test_siren.py
git commit -m "feat(nn): add TorusSirenWeight (real sin(ω0·) SIREN, ω0=30, Sitzmann init)"
```

---

## Task 2: SirenLinear (Linear dont W vient d'une SIREN)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren_linear.py`

- [ ] **Step 1: Ecrire the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_siren_linear.py` :
```python
"""Tests of SirenLinear : se comported comme nn.Linear but W = SIREN(grid)."""

import torch


def test_siren_linear_shape():
    """SirenLinear(in, out) se comported comme nn.Linear : (B, in) → (B, out)."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 16)


def test_siren_linear_is_finite():
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    assert torch.isfinite(layer(x)).all()


def test_siren_linear_backward_propagates():
    """CRITERE L3 : backward propage a gradient fini ET non-nul a CHAQUE parameter
    of the SIREN (qui EST the matrix of poids, in the graphe)."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    y = layer(x)
    loss = y.pow(2).sum()
    loss.backward()

    params = list(layer.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum().item() > 0, f"{name} a recu a gradient nul"


def test_siren_linear_has_no_dense_weight():
    """SirenLinear not must PAS avoir of nn.Parameter of poids dense separe —
    the matrix vient integerement of the SIREN."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    # Les seuls params must etre ceux of the SIREN + bias.
    param_names = [n for n, _ in layer.named_parameters()]
    assert not any("dense" in n or "weight" in n.lower() and "siren" not in n.lower()
                   for n in param_names), \
        f"SirenLinear not should not avoir of poids dense separe : {param_names}"
```

- [ ] **Step 2: Lancer for verify that the tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_siren_linear.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer SirenLinear**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren_linear.py` :
```python
"""SirenLinear : couche nn.Linear-like dont the matrix of poids est produite
par a SIREN.

CORRECTION vs the original : in the original, the matrix decompressee W was computationee then
JETEE (training_loop.py:30-37 appliquait mirror a W then tournait on l'entree
brute). Ici, the SIREN EST the matrix : on evalue the SIREN a each forward for
obtenir W, then on does y = x @ W + b. Tout est in the graphe autodiff.

Usage : remplacer certaines nn.Linear by SirenLinear for compresser leurs
poids via SIREN. Le trade-off : less of params (compression) but a forward
plus cher (evaluation SIREN a each appel) and a expressivite potentiellement
reduite (les poids SIREN are lisses, not denses — voir demo L3).
"""

import torch
import torch.nn as nn

from .siren import TorusSirenWeight


class SirenLinear(nn.Module):
    """Couche lineaire dont the matrix W = SIREN(grid).

    Args:
        in_features, out_features : dimensions (comme nn.Linear).
        hidden : width of the SIREN which produit W.
        omega0 : frequence SIREN.
        bias   : si True, ajoute a biais entrainable (comme nn.Linear).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: int = 32,
        omega0: float = 30.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # La matrix of poids vient d'une SIREN evaluee on a grille
        # (in_features, out_features).
        self.siren = TorusSirenWeight(
            out_h=in_features, out_w=out_features, hidden=hidden, omega0=omega0
        )
        # Biais entrainable separe (pas compresse — this is a vector, not a matrix).
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., in_features) → (..., out_features).

        W = self.siren() : (in_features, out_features), in the graphe autodiff.
        y = x @ W + bias.
        """
        W = self.siren()  # (in_features, out_features), differentiable
        y = x @ W
        if self.bias is not None:
            y = y + self.bias
        return y
```

- [ ] **Step 4: Lancer the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_siren_linear.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add fractus/nn/siren_linear.py tests/test_siren_linear.py
git commit -m "feat(nn): add SirenLinear (Linear whose W comes from a SIREN, in-graph)"
```

---

## Task 3: Mesure honestete ratio of compression

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/compression.py`

- [ ] **Step 1: Ecrire the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_compression.py` :
```python
"""Tests of measure_compression_ratio : mesure REELLE, not of hardcode."""

import inspect
import torch


def test_compression_no_hardcoded_204():
    """CRITERE L3 : the code of mesure not must PAS contenir the litteral 20.4
    (le falsehood hardcode d'the original training_loop.py:52)."""
    from fractus.metrics import compression
    src = inspect.getsource(compression)
    assert "20.4" not in src, "Le litteral 20.4 est interdit (falsehood the original)"


def test_compression_pure_dense_returns_one():
    """Un modele 100% dense (pas of SirenLinear) → ratio 1.0."""
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(16, 16)
    ratio = measure_compression_ratio(m)
    assert abs(ratio - 1.0) < 1e-6


def test_compression_with_siren_gt_one():
    """Un modele with SirenLinear → ratio > 1 (moins of params that l'equivalent dense)."""
    from fractus.nn.siren_linear import SirenLinear
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Sequential(
        SirenLinear(32, 32, hidden=16),  # SIREN instead of Linear(32,32)
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),  # dense classique
    )
    ratio = measure_compression_ratio(m)
    # Le ratio must etre > 1 (les SIREN have less of params that the matrix dense
    # equivalente). La value exact depend of hidden, but > 1 est guaranteed.
    assert ratio > 1.0, f"Ratio attendu > 1, eu {ratio}"


def test_compression_returns_finite():
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(8, 8)
    r = measure_compression_ratio(m)
    assert isinstance(r, float)
    assert r > 0
```

- [ ] **Step 2: Lancer for verify that the tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_compression.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer metrics/compression.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/__init__.py` :
```python
"""Sous-package metrics : mesures honestetes (compression, causal, perplexite).

L3 : compression (mesure real, not of hardcode).
"""
```

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/compression.py` :
```python
"""Mesure HONNETE ratio of compression d'un modele.

CORRECTION DU MENSONGE D'the original design :
- the original hardcodait "compression_ratio": 20.4 in training_loop.py:52.
- Ici, the ratio est MESURE : on compte the parameters reallement utilises et
  on the compare a the taille qu'auraient the matrixs si elles etaient denses.

Definition ratio :
    ratio = (somme tailles denses equivalentes SirenLinear) /
            (somme params SIREN + params denses restants)

Pour a SirenLinear(in, out, hidden=h) :
    - taille dense equivalente = in·out (la matrix qu'elle remplace)
    - params SIREN = 2·h + h·h + h·1 + biases ≈ h2 + 3h
    Le ratio of CETTE couche = in·out / params_SIREN.

Pour a modele mixte (SirenLinear + nn.Linear), the ratio global est :
    (Σ tailles denses equivalentes) / (Σ params totaux).

On not pretend PAS 20.4×. On mesure. La demo L3 montrera the true chiffre.
"""

import torch
import torch.nn as nn

from ..nn.siren_linear import SirenLinear


def _count_params(module: nn.Module) -> int:
    """Nombre total of parameters entrainables d'un module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def measure_compression_ratio(model: nn.Module) -> float:
    """Mesure REELLEMENT the ratio of compression d'un modele.

    Args:
        model : a nn.Module pouvant contenir SirenLinear et/ou nn.Linear.
    Returns:
        ratio > 0. Ratio = 1.0 si the modele est 100% dense.
        Ratio > 1 si the modele contient SirenLinear (compression effective).
        Ratio < 1 est possible but rare (SIREN more grosse that the matrix).
    """
    total_dense_equivalent = 0  # taille qu'auraient the matrixs si elles etaient denses
    total_actual_params = 0     # params reallement utilises

    for module in model.modules():
        if isinstance(module, SirenLinear):
            # Taille dense equivalente : in·out (la matrix W qu'elle remplace).
            dense_eq = module.in_features * module.out_features
            # Parametres reals : params SIREN + bias.
            actual = _count_params(module)
            total_dense_equivalent += dense_eq
            total_actual_params += actual
        elif isinstance(module, nn.Linear) and not isinstance(module, SirenLinear):
            # nn.Linear classique : dense_eq == actual (pas of compression).
            dense_eq = module.in_features * module.out_features
            actual = _count_params(module)
            total_dense_equivalent += dense_eq
            total_actual_params += actual
        elif isinstance(module, (nn.LayerNorm, nn.Embedding, nn.Parameter)):
            # Autres modules : not of compression (on the compte a their taille real).
            actual = _count_params(module)
            total_dense_equivalent += actual
            total_actual_params += actual

    if total_actual_params == 0:
        return 1.0
    return total_dense_equivalent / total_actual_params
```

- [ ] **Step 4: Mettre a jour fractus/nn/__init__.py**

Modify `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py` — ajouter :
```python
from .siren import TorusSirenWeight
from .siren_linear import SirenLinear
```
Et etendre `__all__` with `"TorusSirenWeight"`, `"SirenLinear"`.

- [ ] **Step 5: Lancer the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_compression.py tests/test_siren.py tests/test_siren_linear.py -v
```
Expected: 4 + 6 + 4 = 14 passed.

- [ ] **Step 6: Commit**

```bash
git add fractus/metrics/ fractus/nn/__init__.py tests/test_compression.py
git commit -m "feat(metrics): add measure_compression_ratio (honest, no 20.4 hardcode)"
```

---

## Task 4: Demo L3 — mesurer the vraie compression on the transformer

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_siren_compression.py`

- [ ] **Step 1: Ecrire the demo**

Create `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_siren_compression.py` :
```python
"""Demo L3 : mesurer the VRAIE compression SIREN on a modele.

On construit deux variantes d'un mini-MLP :
    (A) 100% dense (nn.Linear)
    (B) Couches cachees en SirenLinear, derniere couche dense.

On mesure :
    - Le ratio of compression (params SIREN vs dense equivalent).
    - La capacite d'learning : can-on surfit a target with (B) also bien
      qu'with (A) ?

POSITION SCIENTIFIQUE HONNETE :
On s'attend a a ratio MODESTE (~2× a 5×) and a a loss of quality d'learning
(les poids SIREN are lisses, this which limite the capacite a exprimer functions
arbitraires). This is the verite — a comparer au falsehood 20.4× d'the original.

Run :
    python scripts/demo_siren_compression.py
"""

import torch
import torch.nn as nn
from fractus.nn import SirenLinear
from fractus.metrics.compression import measure_compression_ratio


def make_dense_model(d_in, d_hidden, d_out):
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


def make_siren_model(d_in, d_hidden, d_out, siren_hidden=16):
    return nn.Sequential(
        SirenLinear(d_in, d_hidden, hidden=siren_hidden), nn.ReLU(),
        SirenLinear(d_hidden, d_hidden, hidden=siren_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),  # derniere couche dense
    )


def train_and_eval(model, X, Y, n_steps=300, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    initial = None
    for step in range(n_steps):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        if initial is None:
            initial = loss.item()
        loss.backward()
        opt.step()
    final = loss.item()
    return initial, final


def main():
    torch.manual_seed(42)
    d_in, d_hidden, d_out = 16, 32, 8
    n_samples = 64

    # Cible : function non-triviale (sinus a frequence non alignee).
    X = torch.randn(n_samples, d_in)
    Y = torch.sin(X[:, :d_out] * 1.3) + 0.5 * torch.cos(X[:, :d_out] * 0.7)

    dense = make_dense_model(d_in, d_hidden, d_out)
    siren = make_siren_model(d_in, d_hidden, d_out, siren_hidden=16)

    n_dense = sum(p.numel() for p in dense.parameters())
    n_siren = sum(p.numel() for p in siren.parameters())
    ratio_dense = measure_compression_ratio(dense)
    ratio_siren = measure_compression_ratio(siren)

    print("=== Compressiwe measuree ===")
    print(f"Modele dense  : {n_dense} params, ratio = {ratio_dense:.2f}×")
    print(f"Modele SIREN  : {n_siren} params, ratio = {ratio_siren:.2f}×")
    print(f"Economie      : {(1 - n_siren/n_dense)*100:.1f}% of params en moins")
    print()

    print("=== Capacite d'learning (surfit target sinus) ===")
    i_d, f_d = train_and_eval(dense, X, Y)
    i_s, f_s = train_and_eval(siren, X, Y)
    print(f"Dense  : loss {i_d:.4f} → {f_d:.4f}  (baisse {(1-f_d/i_d)*100:.1f}%)")
    print(f"SIREN  : loss {i_s:.4f} → {f_s:.4f}  (baisse {(1-f_s/i_s)*100:.1f}%)")
    print()

    print("=== Verdict honestete ===")
    print(f"Ratio of compression real : {ratio_siren:.2f}×")
    print(f"  (a comparer au '20.4×' hardcode d'the original design, which was false)")
    print(f"Perte of quality learning : {(f_s - f_d):.4f} (SIREN - Dense)")
    if ratio_siren > 1.5 and f_s < i_s * 0.5:
        print("\n✓ La SIREN comprime (>1.5×) ET apprend — honestete and utile.")
    elif ratio_siren > 1.5:
        print("\n~ La SIREN comprime but apprend less well — trade-off a documenter.")
    else:
        print("\n~ Compression weak (<1.5×) — the SIREN n'est not adaptee a these poids.")
    print("\nConclusion : the these '20.4× without loss' d'the original n'est not reproduite.")
    print("La SIREN est utile for functions lisses, not for poids denses.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lancer the demo**

```powershell
.venv\Scripts\python.exe scripts\demo_siren_compression.py
```
Expected: ratio mesure between 1.5× and 5× (PAS 20.4×). Qualite d'learning SIREN <= Dense.

- [ ] **Step 3: Commit**

```bash
git add scripts/demo_siren_compression.py
git commit -m "demo(L3): measure honest SIREN compression ratio (not 20.4x)"
```

---

## Critere final of L3 « termine »

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
# → 62 (L0+L1+L2) + 6 (siren) + 4 (siren_linear) + 4 (compression) = 76 passed

.venv\Scripts\python.exe scripts\demo_siren_compression.py
# → ratio mesure (pas 20.4×), verdict honestete affiche
```

L3 termine → on a a vraie SIREN, a mesure honestete, and the documentation franche of pourquoi 20.4× was a falsehood. On passe then a L4 (NOTEARS causal).

---

## Self-Review

**1. Spec coverage :** (a) TorusSirenWeight vraie sin(ω0·) → Task 1 ✅ ; (b) SirenLinear (W in the graphe) → Task 2 ✅ ; (c) measure_compression_ratio honestete → Task 3 ✅ ; (d) demo mesurant the true ratio → Task 4 ✅ ; (e) tests critiques (sin not SiLU, ω0=30 not 56, not of 20.4 hardcode) → Task 1 Step 1 + Task 3 Step 1 ✅.

**2. Placeholder scan :** no TBD. ✅

**3. Honnetete mathematical :** ω0=30 justifie (Sitzmann 2020), sin instead of SiLU, init SIREN correcte, ratio mesure not hardcode, demo documente explicitement the position scientifique (SIREN on poids denses = compression faible). ✅

**4. Type consistency :** `TorusSirenWeight(out_h, out_w, hidden, omega0).forward() → Tensor(out_h, out_w)`. `SirenLinear(in, out, hidden, omega0).forward((..., in)) → (..., out)`. `measure_compression_ratio(nn.Module) → float`. Coherent. ✅
