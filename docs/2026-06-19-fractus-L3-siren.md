# Fractus L3 — Vraie SIREN + compression mesuree honnetement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corriger le falsehood le plus visible d'the original design : la fausse SIREN (utilisait `nn.SiLU` au lieu de `sin(ω₀·)`) et le ratio de compression hardcode a 20.4×. On implemente une **vraie SIREN** (sinusoides comme non-linearite, ω₀=30 justifie par Sitzmann 2020), on l'applique aux matrices de poids du transformer fractus, et on **mesure honnetement** le ratio de compression obtenu.

**Position scientifique honnete (decision validee) :** Une SIREN represente bien des functions **lisses** (images, champs scalaires). Or les poids d'un reseau entraine sont essentiellement du **bruit structure dense**. On s'attend therefore a un ratio de compression **faible** (~1× a 3×), **pas** 20.4×. La documentation L3 dira franchement pourquoi, et la demo mesurera la verite. C'est exactement l'inverse du falsehood d'OMNI.

**Architecture :** (1) `fractus/nn/siren.py` — `TorusSirenWeight` : vraie SIREN `sin(ω₀·(Wx+b))` sur le tore T² = [0,1)², qui regenere une matrix W[h,w] depuis une grille de coordonnees. (2) `SirenLinear` : un `nn.Module` qui se comporte comme `nn.Linear` but dont la matrix de poids est produite par une SIREN (parameters SIREN entrainables). (3) `fractus/metrics/compression.py` : `measure_compression_ratio(model)` mesure reellement le ratio (taille dense equivalente / params SIREN). (4) Demo : on remplace les projections d'attention par des `SirenLinear`, on entraine, on mesure.

**Tech Stack:** PyTorch 2.12 CPU, pytest.

**Lien spec :** `docs/superpowers/specs/2026-06-19-fractus-unified-design.md`, section « L3 — Compression SIREN vraie + mesure honnete ».

**Prerequis :** L2 termine (62 tests passent, FractalBlockFull fonctionne).

**Maths de reference (SIREN, Sitzmann et al. 2020) :**
- Non-linearite : `sin(ω₀ · (Wx + b))` (PAS SiLU, PAS ReLU).
- ω₀ = 30.0 (valeur empirique du papier SIREN — **pas** 56, qui n'est pas justifie).
- Init speciale des couches : premiere couche `U(-1/ω₀, 1/ω₀)` ; couches suivantes `U(-√(6/ω₀²·fan_in), √(6/ω₀²·fan_in))`.
- Pour la compression de poids : on evalue la SIREN sur une grille (h,w) → matrix W[h,w].

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── fractus/nn/
│   ├── __init__.py             # MODIFY : exporte TorusSirenWeight, SirenLinear
│   ├── siren.py                # CREATE : TorusSirenWeight (vraie SIREN sin(ω₀·))
│   └── siren_linear.py         # CREATE : SirenLinear (Linear dont W vient d'une SIREN)
├── fractus/metrics/
│   ├── __init__.py             # CREATE
│   └── compression.py          # CREATE : measure_compression_ratio (mesure honnete)
└── tests/
    ├── test_siren.py           # CREATE : tests SIREN (sin present, pas SiLU, backward)
    ├── test_siren_linear.py    # CREATE : tests SirenLinear (shape, backward)
    └── test_compression.py     # CREATE : test measure_compression_ratio (pas de hardcode)
```

**Responsabilites :**
- `siren.py` : la SIREN pure (represente un champ scalar sur T²).
- `siren_linear.py` : adaptateur `nn.Module` qui produit une `nn.Linear`-like dont W = SIREN(grid).
- `metrics/compression.py` : mesure real du ratio, AUCUN litteral hardcode.

---

## Task 1: TorusSirenWeight (vraie SIREN sin(ω₀·))

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren.py`

- [ ] **Step 1: Ecrire les tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_siren.py` :
```python
"""Tests de TorusSirenWeight : vraie SIREN sin(ω₀·), pas SiLU."""

import inspect
import torch


def test_siren_uses_sin_not_silu():
    """CRITERE L3 : la SIREN must utiliser torch.sin, PAS nn.SiLU.
    C'est exactement le falsehood d'OMNI (torus_siren.py:15,17 utilisait SiLU)."""
    from fractus.nn import siren
    src = inspect.getsource(siren)
    assert 'torch.sin' in src or 'sin(' in src, "La SIREN must utiliser sin(ω₀·)"
    assert 'SiLU' not in src and 'silu' not in src.lower(), \
        "Plus de SiLU (le falsehood d'OMNI)"


def test_siren_omega0_is_30_not_56():
    """ω₀ = 30 (justifie par Sitzmann 2020), PAS 56 (non justifie, heritage OMNI)."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    assert abs(s.omega0 - 30.0) < 1e-6, f"ω₀ should etre 30.0, eu {s.omega0}"


def test_siren_output_shape():
    """La SIREN evaluee sur la grille produit une matrix (out_h, out_w)."""
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
    """CRITERE L3 : backward propage un gradient fini ET non-nul a CHAQUE parameter."""
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
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a recu un gradient nul"


def test_siren_fewer_params_than_dense():
    """La SIREN must avoir MOINS de parameters que la matrix dense equivalente
    (sinon il n'y a pas de compression). Pour (32,32) with hidden=16 :
    dense = 1024 params, SIREN ≈ 2·16 + 16·16 + 16·1 + biases ≈ 300-400.
    Donc ratio > 2 attendu AU NIVEAU DES PARAMETRES. (Mais la quality de
    reconstruction est une autre question — voir demo.)"""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=32, out_w=32, hidden=16)
    n_siren = sum(p.numel() for p in s.parameters())
    n_dense = 32 * 32
    assert n_siren < n_dense, \
        f"SIREN ({n_siren} params) should etre < dense ({n_dense})"
```

- [ ] **Step 2: Lancer for verify que les tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_siren.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer TorusSirenWeight**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren.py` :
```python
"""TorusSirenWeight : vraie SIREN sin(ω₀·) for representer une matrix de poids.

CORRECTION DU MENSONGE D'the original design :
- OMNI utilisait nn.SiLU (torus_siren.py:15,17) → ici on utilise torch.sin(ω₀·(Wx+b)),
  la VRAIE non-linearite SIREN (Sitzmann et al. 2020).
- OMNI utilisait ω₀=56 non justifie → ici ω₀=30.0 (valeur empirique du papier SIREN,
  qui montre que ω₀≈30 est optimal for la representation de functions continues).
- OMNI commentait "Simple reconstruction: sum of harmonics (real implementation uses
  Fourier)" (torus_siren.py:39) → ici la reconstruction est REELLE (forward SIREN
  sur grille 2D).

POSITION SCIENTIFIQUE HONNETE :
Une SIREN represente bien des functions lisses (images, champs scalaires).
Les poids d'un reseau entraine sont essentiellement du bruit structure dense.
On s'attend therefore a un ratio de compression FAIBLE (~1× a 3×), PAS 20.4×.
The ratio is MEASURED (metrics/compression.py), never hardcoded.

Math (Sitzmann 2020) :
    Non-linearite : sin(ω₀ · (Wx + b)) for each couche cachee.
    Couche de sortie : lineaire (pas de sin).
    Init : premiere couche U(-1/ω₀, 1/ω₀) ; suivantes U(-√(6/(ω₀²·fan_in)), ...).

La SIREN prend en entree des coords (u,v) ∈ [0,1)² sur le tore T² et produit
un scalar W[u,v]. Evaluee sur une grille h×w, elle regenere la matrix W.
"""

import math
import torch
import torch.nn as nn


class TorusSirenWeight(nn.Module):
    """SIREN qui represente une matrix de poids W[out_h, out_w] comme un champ
    scalar sur le tore T² = [0,1)².

    Args:
        out_h, out_w : dimensions de la matrix a regenerer.
        hidden       : width des couches cachees de la SIREN.
        omega0       : frequence fondamentale (30.0 par defaut, Sitzmann 2020).
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
        # Trois couches au total (comme SIREN papier for champs scalaires).
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self._init_siren_weights()

        # Grille precalculee (hors-graphe because constant).
        grid = self._build_grid(out_h, out_w)  # (out_h·out_w, 2)
        self.register_buffer("grid", grid)

    def _init_siren_weights(self):
        """Init SIREN specifique (Sitzmann 2020, section 3.2)."""
        # Premiere couche : U(-1/ω₀, 1/ω₀).
        with torch.no_grad():
            nn.init.uniform_(self.fc1.weight, -1.0 / self.omega0, 1.0 / self.omega0)
            nn.init.zeros_(self.fc1.bias)
            # Couches suivantes : U(-√(6/(ω₀²·fan_in)), √(6/(ω₀²·fan_in))).
            for layer in [self.fc2, self.fc3]:
                fan_in = layer.weight.shape[1]
                bound = math.sqrt(6.0 / (self.omega0 ** 2 * fan_in))
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _build_grid(h: int, w: int) -> torch.Tensor:
        """Grille de coords (u,v) ∈ [0,1)² sur le tore, shape (h·w, 2)."""
        u = torch.linspace(0, 1, h, dtype=torch.float32)
        v = torch.linspace(0, 1, w, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1)  # (h, w, 2)
        return grid.reshape(-1, 2)  # (h·w, 2)

    def forward(self) -> torch.Tensor:
        """Evalue la SIREN sur la grille → matrix W[out_h, out_w].

        C'est la 'decompression' : on regenere W depuis les params SIREN.
        """
        x = self.grid  # (h·w, 2)
        # Couche 1 + sin(ω₀·).
        x = torch.sin(self.omega0 * self.fc1(x))
        # Couche 2 + sin(ω₀·).
        x = torch.sin(self.omega0 * self.fc2(x))
        # Couche de sortie : lineaire (pas de sin).
        x = self.fc3(x)  # (h·w, 1)
        return x.squeeze(-1).reshape(self.out_h, self.out_w)
```

- [ ] **Step 4: Lancer les tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_siren.py -v
```
Expected: 6 passed. Le test `test_siren_uses_sin_not_silu` est le critere L3 critique.

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add fractus/nn/siren.py tests/test_siren.py
git commit -m "feat(nn): add TorusSirenWeight (real sin(ω₀·) SIREN, ω₀=30, Sitzmann init)"
```

---

## Task 2: SirenLinear (Linear dont W vient d'une SIREN)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren_linear.py`

- [ ] **Step 1: Ecrire les tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_siren_linear.py` :
```python
"""Tests de SirenLinear : se comporte comme nn.Linear but W = SIREN(grid)."""

import torch


def test_siren_linear_shape():
    """SirenLinear(in, out) se comporte comme nn.Linear : (B, in) → (B, out)."""
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
    """CRITERE L3 : backward propage un gradient fini ET non-nul a CHAQUE parameter
    de la SIREN (qui EST la matrix de poids, in le graphe)."""
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
        assert p.grad.abs().sum().item() > 0, f"{name} a recu un gradient nul"


def test_siren_linear_has_no_dense_weight():
    """SirenLinear ne must PAS avoir de nn.Parameter de poids dense separe —
    la matrix vient entierement de la SIREN."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    # Les seuls params must etre ceux de la SIREN + bias.
    param_names = [n for n, _ in layer.named_parameters()]
    assert not any("dense" in n or "weight" in n.lower() and "siren" not in n.lower()
                   for n in param_names), \
        f"SirenLinear ne should pas avoir de poids dense separe : {param_names}"
```

- [ ] **Step 2: Lancer for verify que les tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_siren_linear.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer SirenLinear**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren_linear.py` :
```python
"""SirenLinear : couche nn.Linear-like dont la matrix de poids est produite
par une SIREN.

CORRECTION vs OMNI : in OMNI, la matrix decompressee W was calculee then
JETEE (training_loop.py:30-37 appliquait mirror a W then tournait sur l'entree
brute). Ici, la SIREN EST la matrix : on evalue la SIREN a each forward for
obtenir W, then on fait y = x @ W + b. Tout est in le graphe autodiff.

Usage : remplacer certaines nn.Linear par SirenLinear for compresser leurs
poids via SIREN. Le trade-off : moins de params (compression) but un forward
plus cher (evaluation SIREN a each appel) et une expressivite potentiellement
reduite (les poids SIREN sont lisses, pas denses — voir demo L3).
"""

import torch
import torch.nn as nn

from .siren import TorusSirenWeight


class SirenLinear(nn.Module):
    """Couche lineaire dont la matrix W = SIREN(grid).

    Args:
        in_features, out_features : dimensions (comme nn.Linear).
        hidden : width de la SIREN qui produit W.
        omega0 : frequence SIREN.
        bias   : si True, ajoute un biais entrainable (comme nn.Linear).
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
        # La matrix de poids vient d'une SIREN evaluee sur une grille
        # (in_features, out_features).
        self.siren = TorusSirenWeight(
            out_h=in_features, out_w=out_features, hidden=hidden, omega0=omega0
        )
        # Biais entrainable separe (pas compresse — c'est un vector, pas une matrix).
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., in_features) → (..., out_features).

        W = self.siren() : (in_features, out_features), in le graphe autodiff.
        y = x @ W + bias.
        """
        W = self.siren()  # (in_features, out_features), differentiable
        y = x @ W
        if self.bias is not None:
            y = y + self.bias
        return y
```

- [ ] **Step 4: Lancer les tests — DOIVENT PASSER**

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

## Task 3: Mesure honnete du ratio de compression

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/compression.py`

- [ ] **Step 1: Ecrire les tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_compression.py` :
```python
"""Tests de measure_compression_ratio : mesure REELLE, pas de hardcode."""

import inspect
import torch


def test_compression_no_hardcoded_204():
    """CRITERE L3 : le code de mesure ne must PAS contenir le litteral 20.4
    (le falsehood hardcode d'OMNI training_loop.py:52)."""
    from fractus.metrics import compression
    src = inspect.getsource(compression)
    assert "20.4" not in src, "Le litteral 20.4 est interdit (falsehood OMNI)"


def test_compression_pure_dense_returns_one():
    """Un modele 100% dense (pas de SirenLinear) → ratio 1.0."""
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(16, 16)
    ratio = measure_compression_ratio(m)
    assert abs(ratio - 1.0) < 1e-6


def test_compression_with_siren_gt_one():
    """Un modele with SirenLinear → ratio > 1 (moins de params que l'equivalent dense)."""
    from fractus.nn.siren_linear import SirenLinear
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Sequential(
        SirenLinear(32, 32, hidden=16),  # SIREN au lieu de Linear(32,32)
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),  # dense classique
    )
    ratio = measure_compression_ratio(m)
    # Le ratio must etre > 1 (les SIREN ont moins de params que la matrix dense
    # equivalente). La valeur exact depend de hidden, but > 1 est garanti.
    assert ratio > 1.0, f"Ratio attendu > 1, eu {ratio}"


def test_compression_returns_finite():
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(8, 8)
    r = measure_compression_ratio(m)
    assert isinstance(r, float)
    assert r > 0
```

- [ ] **Step 2: Lancer for verify que les tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_compression.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer metrics/compression.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/__init__.py` :
```python
"""Sous-package metrics : mesures honnetes (compression, causal, perplexite).

L3 : compression (mesure real, pas de hardcode).
"""
```

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/compression.py` :
```python
"""Mesure HONNETE du ratio de compression d'un modele.

CORRECTION DU MENSONGE D'the original design :
- OMNI hardcodait "compression_ratio": 20.4 in training_loop.py:52.
- Ici, le ratio est MESURE : on compte les parameters reellement utilises et
  on les compare a la taille qu'auraient les matrices si elles etaient denses.

Definition du ratio :
    ratio = (somme des tailles denses equivalentes des SirenLinear) /
            (somme des params SIREN + params denses restants)

Pour une SirenLinear(in, out, hidden=h) :
    - taille dense equivalente = in·out (la matrix qu'elle remplace)
    - params SIREN = 2·h + h·h + h·1 + biases ≈ h² + 3h
    Le ratio de CETTE couche = in·out / params_SIREN.

Pour un modele mixte (SirenLinear + nn.Linear), le ratio global est :
    (Σ tailles denses equivalentes) / (Σ params totaux).

On ne pretend PAS 20.4×. On mesure. La demo L3 montrera le true chiffre.
"""

import torch
import torch.nn as nn

from ..nn.siren_linear import SirenLinear


def _count_params(module: nn.Module) -> int:
    """Nombre total de parameters entrainables d'un module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def measure_compression_ratio(model: nn.Module) -> float:
    """Mesure REELLEMENT le ratio de compression d'un modele.

    Args:
        model : un nn.Module pouvant contenir des SirenLinear et/ou des nn.Linear.
    Returns:
        ratio > 0. Ratio = 1.0 si le modele est 100% dense.
        Ratio > 1 si le modele contient des SirenLinear (compression effective).
        Ratio < 1 est possible but rare (SIREN plus grosse que la matrix).
    """
    total_dense_equivalent = 0  # taille qu'auraient les matrices si elles etaient denses
    total_actual_params = 0     # params reellement utilises

    for module in model.modules():
        if isinstance(module, SirenLinear):
            # Taille dense equivalente : in·out (la matrix W qu'elle remplace).
            dense_eq = module.in_features * module.out_features
            # Parametres reels : params SIREN + bias.
            actual = _count_params(module)
            total_dense_equivalent += dense_eq
            total_actual_params += actual
        elif isinstance(module, nn.Linear) and not isinstance(module, SirenLinear):
            # nn.Linear classique : dense_eq == actual (pas de compression).
            dense_eq = module.in_features * module.out_features
            actual = _count_params(module)
            total_dense_equivalent += dense_eq
            total_actual_params += actual
        elif isinstance(module, (nn.LayerNorm, nn.Embedding, nn.Parameter)):
            # Autres modules : pas de compression (on les compte a leur taille real).
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

- [ ] **Step 5: Lancer les tests — DOIVENT PASSER**

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

## Task 4: Demo L3 — mesurer la vraie compression sur le transformer

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_siren_compression.py`

- [ ] **Step 1: Ecrire la demo**

Create `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_siren_compression.py` :
```python
"""Demo L3 : mesurer la VRAIE compression SIREN sur un modele.

On construit deux variantes d'un mini-MLP :
    (A) 100% dense (nn.Linear)
    (B) Couches cachees en SirenLinear, derniere couche dense.

On mesure :
    - Le ratio de compression (params SIREN vs dense equivalent).
    - La capacite d'apprentissage : can-on surfit une cible with (B) aussi bien
      qu'with (A) ?

POSITION SCIENTIFIQUE HONNETE :
On s'attend a un ratio MODESTE (~2× a 5×) et a une loss de quality d'apprentissage
(les poids SIREN sont lisses, ce qui limite la capacite a exprimer des functions
arbitraires). C'est la verite — a comparer au falsehood 20.4× d'OMNI.

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

    print("=== Compression mesuree ===")
    print(f"Modele dense  : {n_dense} params, ratio = {ratio_dense:.2f}×")
    print(f"Modele SIREN  : {n_siren} params, ratio = {ratio_siren:.2f}×")
    print(f"Economie      : {(1 - n_siren/n_dense)*100:.1f}% de params en moins")
    print()

    print("=== Capacite d'apprentissage (surfit cible sinus) ===")
    i_d, f_d = train_and_eval(dense, X, Y)
    i_s, f_s = train_and_eval(siren, X, Y)
    print(f"Dense  : loss {i_d:.4f} → {f_d:.4f}  (baisse {(1-f_d/i_d)*100:.1f}%)")
    print(f"SIREN  : loss {i_s:.4f} → {f_s:.4f}  (baisse {(1-f_s/i_s)*100:.1f}%)")
    print()

    print("=== Verdict honnete ===")
    print(f"Ratio de compression real : {ratio_siren:.2f}×")
    print(f"  (a comparer au '20.4×' hardcode d'the original design, qui was false)")
    print(f"Perte de quality apprentissage : {(f_s - f_d):.4f} (SIREN - Dense)")
    if ratio_siren > 1.5 and f_s < i_s * 0.5:
        print("\n✓ La SIREN comprime (>1.5×) ET apprend — honnete et utile.")
    elif ratio_siren > 1.5:
        print("\n~ La SIREN comprime but apprend moins bien — trade-off a documenter.")
    else:
        print("\n~ Compression faible (<1.5×) — la SIREN n'est pas adaptee a ces poids.")
    print("\nConclusion : la these '20.4× without loss' d'OMNI n'est pas reproduite.")
    print("La SIREN est utile for des functions lisses, pas for des poids denses.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lancer la demo**

```powershell
.venv\Scripts\python.exe scripts\demo_siren_compression.py
```
Expected: ratio mesure between 1.5× et 5× (PAS 20.4×). Qualite d'apprentissage SIREN <= Dense.

- [ ] **Step 3: Commit**

```bash
git add scripts/demo_siren_compression.py
git commit -m "demo(L3): measure honest SIREN compression ratio (not 20.4x)"
```

---

## Critere final de L3 « termine »

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
# → 62 (L0+L1+L2) + 6 (siren) + 4 (siren_linear) + 4 (compression) = 76 passed

.venv\Scripts\python.exe scripts\demo_siren_compression.py
# → ratio mesure (pas 20.4×), verdict honnete affiche
```

L3 termine → on a une vraie SIREN, une mesure honnete, et la documentation franche de pourquoi 20.4× was un falsehood. On passe then a L4 (NOTEARS causal).

---

## Self-Review

**1. Spec coverage :** (a) TorusSirenWeight vraie sin(ω₀·) → Task 1 ✅ ; (b) SirenLinear (W in le graphe) → Task 2 ✅ ; (c) measure_compression_ratio honnete → Task 3 ✅ ; (d) demo mesurant le true ratio → Task 4 ✅ ; (e) tests critiques (sin pas SiLU, ω₀=30 pas 56, pas de 20.4 hardcode) → Task 1 Step 1 + Task 3 Step 1 ✅.

**2. Placeholder scan :** no TBD. ✅

**3. Honnetete mathematical :** ω₀=30 justifie (Sitzmann 2020), sin au lieu de SiLU, init SIREN correcte, ratio mesure pas hardcode, demo documente explicitement la position scientifique (SIREN sur poids denses = compression faible). ✅

**4. Type consistency :** `TorusSirenWeight(out_h, out_w, hidden, omega0).forward() → Tensor(out_h, out_w)`. `SirenLinear(in, out, hidden, omega0).forward((..., in)) → (..., out)`. `measure_compression_ratio(nn.Module) → float`. Coherent. ✅
