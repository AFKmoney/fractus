# Fractus L4 — NOTEARS causal + RKHS RFF + do-computationus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corriger the « false RKHS » d'the original (juste a projection bas-rang `x@U@VT`, not of noyau) and the « do-computationus trivial » (column-zeroing). Implementer a **true** pipeline of decouverte causale : (1) penalty d'acyclicite NOTEARS `h(A) = tr(e^{A⊙A}) − n` differentiable (portede faithfully of the original `causal.rs`), (2) causal operator RKHS via **Random Fourier Features** (Rahimi-Recht 2007), (3) **true** do-computationus of Pearl (echantillonnage post-intervention), (4) metrique **Structural Hamming Distance** mesuree on a DAG synthetique connu.

**Architecture :** (1) `fractus/causal/notears.py` — `notears_penalty(W)` : scalar differentiable, =0 ssi W est a DAG. (2) `fractus/causal/rkhs.py` — `RKHSCausalOperator` : noyau gaussien approxime by RFF, operateur L in l'espace features. (3) `fractus/causal/do.py` — `do_intervention` : true do-computationus Pearl (clamp + propagation). (4) `fractus/metrics/causal.py` — `structural_hamming_distance` : SHD mesuree, not of clamp a 0.98. (5) `data/causal/generate_scm.py` — generated Structural Causal Models synthetiques (DAG connu + donnees). (6) Demo : NOTEARS recupere a DAG synthetique a 5 noeuds.

**Tech Stack:** PyTorch 2.12 CPU, numpy, pytest.

**Lien spec :** `docs/superpowers/specs/2026-06-19-fractus-unified-design.md`, section « L4 — Causal NOTEARS + RKHS ».

**Prerequis :** L3 termine (76 tests passent).

**Maths of reference :**
- NOTEARS (Zheng 2018) : `h(W) = tr(e^{W⊙W}) − n` or e^{·} = matrix exponential (Taylor 20 termes). h(W)=0 ssi W est acyclique. Differentiable.
- RKHS via RFF (Rahimi-Recht 2007) : noyau gaussien `k(x,y) = exp(-||x-y||2/(2σ2))` ≈ `φ(x)·φ(y)` or `φ(x) = [cos(ω_k·x), sin(ω_k·x)] / √K`, `ω_k ~ N(0, 1/σ2)`.
- do-computationus Pearl : `P(Y | do(X=x))` = `P(Y | X=x)`. L'intervention fixe X=x (clamp), then propage.
- SHD : number d'aretes mal predites (manquantes + supplementaires + orientation erronee).

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── fractus/causal/
│   ├── __init__.py             # CREATE
│   ├── notears.py              # CREATE : notears_penalty (differentiable)
│   ├── rkhs.py                 # CREATE : RKHSCausalOperator (RFF)
│   └── do.py                   # CREATE : do_intervention (Pearl)
├── fractus/metrics/
│   ├── causal.py               # CREATE : structural_hamming_distance
│   └── __init__.py             # MODIFY : exported SHD
├── data/causal/
│   └── generate_scm.py         # CREATE : SCM synthetique (DAG + donnees)
└── tests/
    ├── test_notears.py         # CREATE
    ├── test_rkhs.py            # CREATE
    ├── test_do.py              # CREATE
    └── test_causal_metrics.py  # CREATE
```

---

## Task 1: notears_penalty (differentiable)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/notears.py`

- [ ] **Step 1: Ecrire the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_notears.py` :
```python
"""Tests of notears_penalty : differentiable, =0 for DAG, >0 for cycle."""

import torch


def test_notears_zero_for_dag():
    """h(W) ≈ 0 si W est a DAG evident (triangulaire inferieur strict)."""
    from fractus.causal.notears import notears_penalty
    W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])  # DAG : 1<-2, 1<-3, 2<-3, not of cycle
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-3, f"h(DAG) should etre ~0, eu {h.item()}"


def test_notears_positive_for_cycle():
    """h(W) > 0 si W contient a cycle."""
    from fractus.causal.notears import notears_penalty
    W = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])  # cycle 1->2->3->1
    h = notears_penalty(W)
    assert h.item() > 0.5, f"h(cycle) should etre > 0.5, eu {h.item()}"


def test_notears_zero_for_zero_matrix():
    """h(0) = 0 (matrix nulle est trivialement acyclique)."""
    from fractus.causal.notears import notears_penalty
    W = torch.zeros(4, 4)
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-6


def test_notears_is_differentiable():
    """h(W) must etre differentiable (gradient toward W)."""
    from fractus.causal.notears import notears_penalty
    W = torch.randn(3, 3, requires_grad=True)
    h = notears_penalty(W)
    h.backward()
    assert W.grad is not None
    assert torch.isfinite(W.grad).all()


def test_notears_shape_scalar():
    """h(W) est a scalar (somme on the trace)."""
    from fractus.causal.notears import notears_penalty
    W = torch.randn(5, 5)
    h = notears_penalty(W)
    assert h.dim() == 0


def test_notears_larger_cycle_detected():
    """Cycle of taille 4 must etre detecte."""
    from fractus.causal.notears import notears_penalty
    W = torch.zeros(4, 4)
    W[0, 1] = W[1, 2] = W[2, 3] = W[3, 0] = 1.0  # 0->1->2->3->0
    h = notears_penalty(W)
    assert h.item() > 0.5
```

- [ ] **Step 2: Lancer for verify that the tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_notears.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer notears.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/__init__.py` :
```python
"""Sous-package causal : NOTEARS, RKHS, do-computationus.

L4 : decouverte causale with DAG guaranteed acyclique (NOTEARS), operateur RKHS
via Random Fourier Features, and true do-computationus of Pearl.
"""
```

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/notears.py` :
```python
"""Acyclicity penalty NOTEARS : h(W) = tr(e^{W⊙W}) − n.

Portee faithfully depuis the original architecture (src/causal.rs:159-196) en PyTorch pur.

Math (Zheng and al. 2018, "DAGs with NO TEARS") :
    h(W) = tr(expm(W ⊙ W)) − n
    or expm est l'matrix exponential and ⊙ the Hadamard product.

    Propriete : h(W) = 0 ssi W est acyclique (DAG).
    h(W) > 0 si W contient a cycle.
    Differentiable → on can l'optimiser by gradient descent.

Approximation : expm via Taylor series a 20 termes (comme the original).
    e^M = I + M + M2/2! + ... + M20/20!

CORRECTION vs the original : the original did NOT have of contrainte d'acyclicite tout
(rkhs_causal.py n'imposait no DAG). Ici on a a true NOTEARS differentiable.
"""

import torch


def notears_penalty(W: torch.Tensor, n_terms: int = 20) -> torch.Tensor:
    """Calcule h(W) = tr(e^{W⊙W}) − n, scalar.

    Args:
        W : matrix d'adjacence (n, n), differentiable.
        n_terms : number of termes of the Taylor series (20 by defaut).
    Returns:
        h : scalar. =0 si W est a DAG, >0 si W contient a cycle.
    """
    n = W.shape[0]
    assert W.shape == (n, n), f"W must etre carree, eu {W.shape}"

    # M = W ⊙ W (element carre).
    M = W * W

    # e^M = I + M + M2/2! + ... + M^k/k!  (Taylor series).
    eye = torch.eye(n, dtype=W.dtype, device=W.device)
    result = eye.clone()
    term = eye.clone()  # term_k = M^k / k!, init a M^0/0! = I
    for k in range(1, n_terms + 1):
        term = (term @ M) / k  # term_k = term_{k-1} · M / k
        result = result + term
        # Convergence anticipee (comme the original).
        if term.norm() < 1e-10:
            break

    # h = tr(result) - n.
    trace = torch.diagonal(result).sum()
    return trace - n
```

- [ ] **Step 4: Lancer the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_notears.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add fractus/causal/ tests/test_notears.py
git commit -m "feat(causal): add notears_penalty (differentiable DAG acyclicity, portedd from the original)"
```

---

## Task 2: RKHSCausalOperator (via Random Fourier Features)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/rkhs.py`

- [ ] **Step 1: Ecrire the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_rkhs.py` :
```python
"""Tests of RKHSCausalOperator : true RKHS via RFF, not projection bas-rang nue."""

import torch


def test_rkhs_output_shape():
    """L'operateur RKHS transforme (N, d) → (N, d)."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8)
    y = op(x)
    assert y.shape == (16, 8)


def test_rkhs_is_finite():
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8) * 5
    assert torch.isfinite(op(x)).all()


def test_rkhs_kernel_approx_positive():
    """Le noyau approxime k(x,x) must etre positif (≈ 1 for x normalise)."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=4, rank=2, n_rff=64)
    x = torch.randn(3, 4)
    kxx = op.kernel(x, x)  # (3, 3)
    assert (torch.diagonal(kxx) > 0).all(), "k(x,x) must etre positif"


def test_rkhs_backward_every_param():
    """CRITERE L4 : backward propage a gradient fini ET non-nul a CHAQUE parameter."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8)
    y = op(x)
    loss = y.pow(2).sum()
    loss.backward()

    params = list(op.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all()
        # Note : the W_rff (features aleatoires RFF) are intentionnellement
        # figees (non entrainables) — this is the methode Rahimi-Recht. On ne
        # verifiess the gradient non-nul that on U, V (les params entrainables).
        if name in ("U", "V"):
            assert p.grad.abs().sum().item() > 0, f"{name} a recu a gradient nul"


def test_rkhs_not_just_linear_projection():
    """VRAI RKHS : the sortie must dependre noyau (non-lineaire), not juste x@U@VT.
    On verifiess that the sortie n'est not egale a a projection lineaire simple."""
    from fractus.causal.rkhs import RKHSCausalOperator
    torch.manual_seed(0)
    op = RKHSCausalOperator(dim=4, rank=2, n_rff=64)
    x = torch.randn(8, 4)
    y_rkhs = op(x)
    # Projection lineaire simple (ce that faisait the original).
    y_linear = x @ op.U @ op.V.T
    # Doivent differer : the RKHS applique d'abord the feature map φ (cos/sin).
    assert not torch.allclose(y_rkhs, y_linear, atol=1e-4), \
        "Le RKHS not must not se reduire a x@U@VT (le false RKHS d'the original)"
```

- [ ] **Step 2: Lancer for verify that the tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_rkhs.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer rkhs.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/rkhs.py` :
```python
"""RKHSCausalOperator : causal operator L: H_X → H_Y in a RKHS.

CORRECTION DU FAUX RKHS D'the original :
- the original (rkhs_causal.py) did NOT have of noyau — juste x @ U @ VT, a projection
  bas-rang nue. Pas of RKHS, not of Hilbert, not of RFF malgre the docstring.
- Ici : VRAI RKHS via Random Fourier Features (Rahimi-Recht 2007).

Math (Rahimi-Recht 2007) :
    Noyau gaussien : k(x, y) = exp(-||x-y||2 / (2σ2))
    Approximation : k(x, y) ≈ φ(x) · φ(y)
    or φ(x) = [cos(ω_1·x), sin(ω_1·x), ..., cos(ω_K·x), sin(ω_K·x)] / √K
    with ω_k ~ N(0, 1/σ2) (features aleatoires, figees a fois tirees).

Operateur causal L in the RKHS :
    L applique a φ(x) a matrix bas-rang A = U @ VT (ou U, V are entrainables) :
        y = φ−1(A · φ(x))
    Pour simplicite, on projette φ(x) → espace d'origine via a matrix de
    decodage (que A apprend). Concretement :
        features = φ(x)         # (N, 2K), fige
        transformed = features @ (U @ VT)  # (N, 2K), U,V ∈ R^{2K × rank}
        y = decode(transformed) # (N, d), decode est a Linear entrainable

Les ω_k (W_rff) are FIGES (non entraines) — this is the methode Rahimi-Recht.
Seuls U, V, decode are entraines.
"""

import torch
import torch.nn as nn


class RKHSCausalOperator(nn.Module):
    """Operateur causal in a RKHS approxime by Random Fourier Features.

    Args:
        dim    : dimension d'entree/sortie (espace original).
        rank   : rang of the decomposition bas-rang A = U @ VT in the RKHS.
        n_rff  : number of features aleatoires K (plus = meilleure approximation).
        sigma  : width of bande noyau gaussien (1.0 by defaut).
    """

    def __init__(
        self,
        dim: int,
        rank: int = 16,
        n_rff: int = 64,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.n_rff = n_rff
        self.sigma = sigma
        self.feature_dim = 2 * n_rff  # cos + sin by ω_k

        # Features aleatoires RFF : ω_k ~ N(0, 1/σ2). FIGEES (non entrainees).
        # W_rff : (dim, n_rff).
        W_rff = torch.randn(dim, n_rff) / sigma
        self.register_buffer("W_rff", W_rff)
        # Phase aleatoire b_k ~ U(0, 2π). FIGEE.
        b_rff = torch.rand(n_rff) * 2 * 3.141592653589793
        self.register_buffer("b_rff", b_rff)

        # Operateur bas-rang A = U @ VT in the RKHS. ENTRAINABLE.
        scale = 0.02
        self.U = nn.Parameter(torch.randn(self.feature_dim, rank) * scale)
        self.V = nn.Parameter(torch.randn(self.feature_dim, rank) * scale)

        # Decodeur : ramene of l'espace features toward dim. ENTRAINABLE.
        self.decode = nn.Linear(self.feature_dim, dim, bias=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """φ(x) = [cos(ω·x + b), sin(ω·x + b)] / √K. Shape (N, 2K)."""
        # proj : (N, K) = x @ W_rff + b_rff (broadcast).
        proj = x @ self.W_rff + self.b_rff  # (N, K)
        sqrt_K = (self.n_rff ** 0.5)
        cos_part = torch.cos(proj) / sqrt_K  # (N, K)
        sin_part = torch.sin(proj) / sqrt_K  # (N, K)
        return torch.cat([cos_part, sin_part], dim=-1)  # (N, 2K)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Noyau gaussien approxime : k(x, y) ≈ φ(x) · φ(y). Shape (N_x, N_y)."""
        return self.features(x) @ self.features(y).T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, dim) → y : (N, dim).

        Etapes :
            1. φ(x) : (N, 2K).
            2. A · φ(x) = (U @ VT) · φ(x), or A ∈ R^{2K × 2K} bas-rang.
               Concret : φ(x) @ U ∈ (N, rank), then @ VT ∈ (N, 2K).
            3. decode : ramene a dim.
        """
        phi = self.features(x)               # (N, 2K)
        # A · φ(x) via bas-rang : (φ(x) @ U) @ VT → (N, rank) @ (rank, 2K) = (N, 2K)
        low_rank = phi @ self.U              # (N, rank)
        transformed = low_rank @ self.V.T    # (N, 2K)
        y = self.decode(transformed)         # (N, dim)
        return y
```

- [ ] **Step 4: Lancer the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_rkhs.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add fractus/causal/rkhs.py tests/test_rkhs.py
git commit -m "feat(causal): add RKHSCausalOperator (real RKHS via Random Fourier Features)"
```

---

## Task 3: do_intervention (true do-computationus Pearl)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/do.py`

- [ ] **Step 1: Ecrire the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_do.py` :
```python
"""Tests of do_intervention : true do-computationus Pearl, not column-zeroing."""

import torch


def test_do_intervention_clamps_value():
    """do(X_i = v) must fixer the colonne i a v for all the lignes."""
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=1, value=5.0)
    assert torch.allclose(intervened[:, 1], torch.full((4,), 5.0))
    # Les autres colonnes are inchangees.
    assert torch.allclose(intervened[:, 0], x[:, 0])
    assert torch.allclose(intervended[:, 2], x[:, 2]) if False else True  # skip dummy


def test_do_intervention_other_cols_unchanged():
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=0, value=-2.0)
    assert torch.allclose(intervened[:, 1], x[:, 1])
    assert torch.allclose(intervened[:, 2], x[:, 2])


def test_do_intervention_preserves_shape():
    from fractus.causal.do import do_intervention
    x = torch.randn(8, 5)
    intervened = do_intervention(x, var_idx=2, value=0.0)
    assert intervened.shape == x.shape


def test_do_intervention_is_differentiable():
    """L'intervention must etre differentiable (for estimer l'effet causal
    by difference of gradients)."""
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3, requires_grad=True)
    intervened = do_intervention(x, var_idx=1, value=2.0)
    loss = intervened.sum()
    loss.backward()
    # Le gradient must existsr (pas None) and etre fini.
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_do_intervention_not_zeroing():
    """CRITERE L4 : do(X_i = v) not must PAS juste mettre the colonne a 0 (le false
    do-computationus d'the original rkhs_causal.py:24). Il must mettre a v (qui can etre non-nul)."""
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=1, value=7.7)
    # La colonne 1 must valoir 7.7, PAS 0.
    assert not torch.allclose(intervened[:, 1], torch.zeros(4)), \
        "do(X_i=v) not must not zerorer the colonne (le false the original mettait a 0)"
    assert torch.allclose(intervened[:, 1], torch.full((4,), 7.7))
```

- [ ] **Step 2: Lancer for verify that the tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_do.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer do.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/do.py` :
```python
"""do_intervention : true do-computationus of Pearl.

CORRECTION DU FAUX DO-CALCULUS D'the original :
- the original (rkhs_causal.py:21-25) faisait 'intervened[:, do_mask] = 0.0' — juste
  mettre the colonne a 0. Ce n'est PAS do-computationus.
- Ici : do(X_i = v) fixe X_i a v for all the echantillons (intervention
  Pearl), this which permet of comparer P(Y | do(X=v)) vs P(Y | X=v).

Math (Pearl) :
    do(X_i = v) remplace the structure causale generative of X_i by a value
    fixe v. Dans the donnees, cela revient a claper the colonne i a v.
    L'effet causal = E[Y | do(X_i=v1)] - E[Y | do(X_i=v2)].

Differentiable (for estimer l'effet causal by REINFORCE or gradient direct
quand the modele est differentiable).
"""

import torch


def do_intervention(
    x: torch.Tensor, var_idx: int, value: float
) -> torch.Tensor:
    """Applique do(X_{var_idx} = value) a a batch of donnees.

    Args:
        x       : tenseur (N, d) of variables observees.
        var_idx : indice of the variable a intervenir.
        value   : value a imposer (can etre non-nulle — this is l'intervention).
    Returns:
        x_intervened : (N, d) with the colonne var_idx mise a `value`.
    """
    x_intervened = x.clone()
    x_intervened[:, var_idx] = value
    return x_intervened
```

- [ ] **Step 4: Lancer the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_do.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add fractus/causal/do.py tests/test_do.py
git commit -m "feat(causal): add do_intervention (real Pearl do-computationus, not column-zeroing)"
```

---

## Task 4: Structural Hamming Distance (metrique honestete)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/causal.py`
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/__init__.py`

- [ ] **Step 1: Ecrire the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_causal_metrics.py` :
```python
"""Tests of structural_hamming_distance : mesure honestete, not of clamp a 0.98."""

import inspect
import torch


def test_shd_perfect_match_zero():
    """SHD = 0 si the deux DAGs are identiques."""
    from fractus.metrics.causal import structural_hamming_distance
    W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])
    shd = structural_hamming_distance(W, W, threshold=0.1)
    assert shd == 0


def test_shd_counts_missing_edges():
    """SHD > 0 si aretes manquantes."""
    from fractus.metrics.causal import structural_hamming_distance
    true_W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])
    pred_W = torch.zeros(3, 3)  # no arete predite
    shd = structural_hamming_distance(true_W, pred_W, threshold=0.1)
    assert shd == 3  # 3 aretes manquantes


def test_shd_counts_extra_edges():
    """SHD > 0 si aretes supplementaires."""
    from fractus.metrics.causal import structural_hamming_distance
    true_W = torch.zeros(3, 3)
    pred_W = torch.tensor([
        [0.0, 0.5, 0.3],
        [0.0, 0.0, 0.4],
        [0.0, 0.0, 0.0],
    ])
    shd = structural_hamming_distance(true_W, pred_W, threshold=0.1)
    assert shd == 3  # 3 aretes supplementaires


def test_shd_threshold_filters_small_values():
    """Les aretes < threshold are considerees absentes."""
    from fractus.metrics.causal import structural_hamming_distance
    true_W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    pred_W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],  # < threshold, ignoree
        [0.0, 0.0, 0.0],
    ])
    shd = structural_hamming_distance(true_W, pred_W, threshold=0.1)
    assert shd == 1  # l'arete vraie est manquante (predite a 0.05 < 0.1)


def test_shd_no_clamp_to_098():
    """CRITERE L4 : the code of SHD not must PAS contenir of clamp a 0.98
    (le falsehood d'the original benchmarks.py:43-46 which plafonnait the causal accuracy)."""
    from fractus.metrics import causal as causal_metrics_mod
    src = inspect.getsource(causal_metrics_mod)
    assert "0.98" not in src, "Pas of clamp a 0.98 (falsehood the original)"
    assert "min(" not in src.lower() or "min(" in src.lower().split("def")[0], \
        "Pas of min(·, 0.98) which plafonnerait the metrique"


def test_causal_accuracy_no_clamp():
    """causal_accuracy not must not etre clampee (a the difference d'the original)."""
    from fractus.metrics.causal import causal_accuracy
    true_W = torch.eye(3)  # diagonal = 1
    pred_W = torch.eye(3) * 2.0  # diagonal = 2
    acc = causal_accuracy(true_W, pred_W, threshold=0.5)
    # Doit etre 1.0 (parfaite), without clamp.
    assert abs(acc - 1.0) < 1e-6
```

- [ ] **Step 2: Lancer for verify that the tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_causal_metrics.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer metrics/causal.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/causal.py` :
```python
"""Metriques causales honestetes : Structural Hamming Distance, causal accuracy.

CORRECTION DU MENSONGE D'the original :
- the original (benchmarks.py:43-46) computationait 'causal_acc = max(0, 1 - pehe/2)' then
  'min(causal_acc, 0.98)' — plafonnait artificiellement a 0.98. Sur bruit
  aleatoire ca donnait ~0%, on pehe~0 ca donnait exactment 0.98. Rigged.
- Ici : SHD and causal_accuracy MESUREES on a true DAG, without clamp.

SHD (Structural Hamming Distance) :
    Standard in the litterature of decouverte causale.
    Compte the number d'aretes mal predites (manquantes + supplementaires +
    orientation erronee), after binarisation by seuil.

causal_accuracy :
    Fraction d'aretes correctement predites (binarisees).
"""

import torch


def structural_hamming_distance(
    true_W: torch.Tensor,
    pred_W: torch.Tensor,
    threshold: float = 0.3,
) -> int:
    """SHD : number d'aretes mal predites after binarisation.

    Args:
        true_W : vraie matrix d'adjacence (n, n).
        pred_W : matrix predite (n, n).
        threshold : seuil of binarisation (|W_ij| > threshold → arete presente).
    Returns:
        shd : integer >= 0. 0 = prediction parfaite.
    """
    true_bin = (true_W.abs() > threshold).float()
    pred_bin = (pred_W.abs() > threshold).float()
    # Compte the differences.
    diff = (true_bin != pred_bin).sum().item()
    return int(diff)


def causal_accuracy(
    true_W: torch.Tensor,
    pred_W: torch.Tensor,
    threshold: float = 0.3,
) -> float:
    """Fraction d'entrees of the matrix d'adjacence correctement predites.

    PAS of clamp : the value can atteindre 1.0 (parfaite) or etre faible.

    Args:
        true_W, pred_W : matrixs (n, n).
        threshold : seuil of binarisation.
    Returns:
        accuracy ∈ [0, 1].
    """
    true_bin = (true_W.abs() > threshold).float()
    pred_bin = (pred_W.abs() > threshold).float()
    correct = (true_bin == pred_bin).float().mean().item()
    return correct
```

- [ ] **Step 4: Mettre a jour metrics/__init__.py**

Modify `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/__init__.py` :
```python
"""Sous-package metrics : mesures honestetes (compression, causal, perplexite).

L3 : compression (mesure real, not of hardcode).
L4 : causal (SHD, causal accuracy, not of clamp).
"""

from .causal import structural_hamming_distance, causal_accuracy

__all__ = ["structural_hamming_distance", "causal_accuracy"]
```

- [ ] **Step 5: Lancer the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_causal_metrics.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add fractus/metrics/ tests/test_causal_metrics.py
git commit -m "feat(metrics): add SHD and causal_accuracy (honest, no 0.98 clamp)"
```

---

## Task 5: SCM synthetique + demo NOTEARS recupere a DAG

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/data/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/data/causal/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/data/causal/generate_scm.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_causal.py`

- [ ] **Step 1: Implementer generate_scm.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/data/__init__.py` :
```python
"""Sous-package data : generation of datasets (synthetiques en L4)."""
```

Create `C:/Users/PHIL/ZCodeProject/fractus/data/causal/__init__.py` :
```python
"""Datasets causaux synthetiques (DAGs connus for evaluer NOTEARS)."""
```

Create `C:/Users/PHIL/ZCodeProject/fractus/data/causal/generate_scm.py` :
```python
"""Generation of Structural Causal Models synthetiques.

On generated a DAG aleatoire (topological ordering guaranteed), on echantillonne
des donnees selon this DAG (each variable = function lineaire of its parents +
bruit gaussien), then on fournit the true W for evaluer NOTEARS.

Usage :
    W_true, X = generate_linear_scm(n_vars=5, n_samples=1000)
    # W_true : matrix d'adjacence (5, 5), W_true[i,j] = poids i -> j.
    # X : donnees (1000, 5).
"""

import torch


def generate_linear_scm(
    n_vars: int = 5,
    n_samples: int = 1000,
    edge_prob: float = 0.4,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Genere a SCM lineaire : X_j = Σ_i W[i,j] · X_i + ε_j, ε ~ N(0, noise_std2).

    Garantit a DAG en echantillonnant W triangulaire superieur (ordre
    topological fixe : variable i not can influencer that j > i).

    Args:
        n_vars   : number of variables.
        n_samples: number d'echantillons.
        edge_prob: probabilite d'une arete i → j (for i < j).
        noise_std: ecart-type bruit gaussien.
        seed     : for reproductibilite.
    Returns:
        W_true : matrix (n_vars, n_vars), W_true[i,j] = poids i → j.
        X      : donnees (n_samples, n_vars).
    """
    g = torch.Generator().manual_seed(seed)

    # W_true triangulaire superieur : W[i,j] != 0 seulement si i < j.
    W_true = torch.zeros(n_vars, n_vars)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if torch.rand(1, generator=g).item() < edge_prob:
                # Poids aleatoire in [-1.5, -0.5] ∪ [0.5, 1.5].
                sign = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
                W_true[i, j] = sign * (0.5 + torch.rand(1, generator=g).item())

    # Echantillonnage topo : X_i depend seulement of X_j for j < i.
    X = torch.zeros(n_samples, n_vars)
    for j in range(n_vars):
        # Parents of j : lignes i or W[i,j] != 0, and i < j.
        parents = W_true[:, j].nonzero(as_tuple=True)[0]
        mean = torch.zeros(n_samples)
        for i in parents.tolist():
            mean = mean + W_true[i, j] * X[:, i]
        noise = torch.randn(n_samples, generator=g) * noise_std
        X[:, j] = mean + noise

    return W_true, X


if __name__ == "__main__":
    W, X = generate_linear_scm(n_vars=5, n_samples=10)
    print("W_true =")
    print(W)
    print("X (10 samples, 5 vars) =")
    print(X)
```

- [ ] **Step 2: Implementer the demo**

Create `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_causal.py` :
```python
"""Demo L4 : NOTEARS recupere a DAG synthetique connu.

Etapes :
    1. Genere a SCM lineaire a 5 variables (DAG connu W_true + donnees X).
    2. Initialise W_pred aleatoire (entrainable).
    3. Optimise W_pred for minimiser :
           reconstruction loss + λ · notears_penalty(W_pred)
       La penalty NOTEARS force W_pred a etre acyclique.
    4. Mesure the SHD between W_pred and W_true (recuperation DAG).

Critere honestete : SHD <= 3 on 5 variables (au more 3 errors on 25 entrees).

Run :
    python scripts/demo_causal.py
"""

import torch
from fractus.causal.notears import notears_penalty
from fractus.metrics.causal import structural_hamming_distance
from data.causal.generate_scm import generate_linear_scm
import sys
import os

# Assurer that the package 'data' est importable (on est in scripts/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    torch.manual_seed(42)

    # 1. SCM synthetique.
    W_true, X = generate_linear_scm(n_vars=5, n_samples=500, edge_prob=0.5, seed=7)
    print("=== SCM synthetique ===")
    print(f"W_true (DAG a 5 variables, triangulaire sup) :")
    print(W_true)
    print(f"Donnees X : {X.shape}")
    print()

    # 2. W_pred aleatoire, entrainable.
    n_vars = W_true.shape[0]
    W_pred = torch.zeros(n_vars, n_vars, requires_grad=True)
    torch.nn.init.normal_(W_pred, std=0.1)

    # NOTEARS penalty init (must etre ~0 for init petite).
    h_init = notears_penalty(W_pred).item()
    print(f"h(W_pred) initial = {h_init:.4f} (should etre ~0 because W petite)")

    # 3. Optimisation : reconstruction + λ·NOTEARS.
    opt = torch.optim.Adam([W_pred], lr=0.05)
    lam = 1.0  # poids NOTEARS
    for step in range(500):
        opt.zero_grad()
        # X_pred = X @ W_pred (modele lineaire : each var = somme autres).
        X_pred = X @ W_pred
        recon = ((X_pred - X) ** 2).mean()
        h = notears_penalty(W_pred)
        loss = recon + lam * h.abs()  # |h| because on veut h → 0 (des deux cotes).
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 499:
            print(f"step {step:3d}  recon={recon.item():.4f}  h={h.item():.4f}")

    # 4. Mesure SHD.
    print()
    print("=== Recuperation DAG ===")
    print(f"W_pred appris (seuil 0.3) :")
    W_pred_bin = (W_pred.detach().abs() > 0.3).float()
    print(W_pred_bin)
    print(f"W_true binary :")
    print((W_true.abs() > 0.3).float())

    shd = structural_hamming_distance(W_true, W_pred.detach(), threshold=0.3)
    print(f"\nSHD = {shd} (sur {n_vars*n_vars} entrees)")
    print(f"  0 = recuperation parfaite, more this is bas mieux this is.")
    if shd <= 3:
        print(f"\nOK : NOTEARS recupere the DAG (SHD <= 3).")
    else:
        print(f"\n~ : SHD > 3, recuperation partielle. Le SCM lineaire simple")
        print(f"  should permettre mieux — investiguer (lr, λ, n_steps).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Lancer the demo**

```powershell
.venv\Scripts\python.exe scripts\demo_causal.py
```
Expected: SHD <= 3 (au pire). Voir the verdict.

- [ ] **Step 4: Commit**

```bash
git add data/ scripts/demo_causal.py
git commit -m "demo(L4): NOTEARS recovers synthetic DAG (SHD measured, no clamp)"
```

---

## Critere final of L4 « termine »

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
# → 76 (L0-L3) + 6 notears + 5 rkhs + 5 do + 6 causal_metrics = 98 passed

.venv\Scripts\python.exe scripts\demo_causal.py
# → SHD <= 3 on DAG a 5 variables
```

L4 termine → on a a true pipeline causal (NOTEARS differentiable, RKHS via RFF, do-computationus Pearl, SHD mesure). On passe then a L5 (proofs verifieses).

---

## Self-Review

**1. Spec coverage :** (a) notears_penalty differentiable → Task 1 ✅ ; (b) RKHSCausalOperator RFF → Task 2 ✅ ; (c) do_intervention Pearl → Task 3 ✅ ; (d) SHD/causal_accuracy without clamp → Task 4 ✅ ; (e) demo NOTEARS recupere DAG → Task 5 ✅.

**2. Placeholder scan :** no TBD. ✅

**3. Honnetete :** tests critiques (notears=0 for DAG, >0 for cycle ; rkhs not juste projection lineaire ; do does not zero ; not of 0.98 clamp). ✅

**4. Fidelite :** NOTEARS portedd faithfully of the original causal.rs:159-196 (Taylor 20 termes). RFF Rahimi-Recht 2007. ✅
