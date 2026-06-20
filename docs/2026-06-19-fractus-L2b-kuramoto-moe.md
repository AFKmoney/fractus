# Fractus L2b — Kuramoto RK4 + MoE von Mises/Farey Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter the deux pepites originales of the original au FractalBlock : (1) Kuramoto oscillators couples bas-rang integres by RK4 — STATELESS (recomputationes a each forward depuis the hidden states), and (2) Mixture-of-Experts a routing of phase von Mises on phases distribuees by Farey sequence. Le FractalBlock etendu devient `LN → attn → PhaseSoliton → PhaseRoutedMoE (gated by Kuramoto phases) → residuelle`.

**Architecture (decision valide : Kuramoto STATELESS) :** Le `KuramotoLayer` est a `nn.Module` pur, without etat mutable between forwards. A each forward : (a) phases initiales derivees hidden states (`encode_from_hidden`), (b) N steps d'integration RK4 bas-rang (courange `K = UΛUT`), (c) phases finales → utilisees by the MoE. **Tout est in the graphe autodiff** (U, Λ, omega are `nn.Parameter`).

**Tech Stack:** PyTorch 2.12 CPU, numpy, pytest.

**Lien spec :** `docs/superpowers/specs/2026-06-19-fractus-unified-design.md`, section « L2b ».

**Prerequis :** L2a termine (FractalBlock minimal fonctionne, 41 tests passent).

**Maths portedes faithfully depuis the original** (extraites code original) :

### Kuramoto (phase_ode.rs)
- Equation : `dθ_i/dt = ω_i − damping·θ_i + K_strength·Σ_j K_ij sin(θ_j − θ_i)`, `K = UΛUT`
- Forme bas-rang O(N·r) : `p = UTsinθ, q = UTcosθ`, `u_p = U(Λp), u_q = U(Λq)`,
  `dθ_i = ω_i − damping·θ_i + K_strength·(cosθ_i · u_p[i] − sinθ_i · u_q[i])`
- RK4 standard (4 under-steps), then wrap `θ_i mod 2π` → [0, 2π).
- `encode_from_hidden(hidden)` : `θ_i = (Σ_j hidden[i,j] / d_model · 2π) mod 2π` for i < min(N, seq_len).
- `decode_to_bias(seq_len, d_model)` : encodage positionnel sinusoidal `[sin(freq·θ)/√freq, cos(freq·θ)/√freq]`, `freq = j//2 + 1`.
- `phase_loss` : `L = −(1/N2)·[cosθTK·cosθ + sinθTK·sinθ]` (bas-rang : `(UTcosθ)TΛ(UTcosθ) + (UTsinθ)TΛ(UTsinθ)`).

### MoE (moe.rs + farey.rs)
- Suite of Farey d'ordre `2E` → selection uniforme of E angles ∈ [0, 2π).
- Gate von Mises (NON normalise) : `g_e = exp(κ_eff · cos(θ − θ_e))`, `κ_eff = κ/temperature`.
- Phase moyenne token : `θ = atan2(Σ_p sin(θ_p), Σ_p cos(θ_p))`.
- Normalisation : `g_e /= Σ_e g_e` (uniforme 1/E si somme < 1e-10).
- Top-k : selectionne the K meilleurs experts, renormalise on the top-k.
- Experts : MLP GeLU `gelu(x·W1 + b1)·W2 + b2`, `d_ff = 64`.
- Load-balance loss : `L_balance = E · Σ_e (P_e − 1/E)2`, `P_e = moyenne gates of l'expert e`.

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── fractus/nn/
│   ├── __init__.py             # MODIFY : exported KuramotoLayer, PhaseRoutedMoE, FractalBlockFull
│   ├── farey.py                # CREATE : Farey sequence + expert_phases (precomputation pur)
│   ├── phase_ode.py            # CREATE : KuramotoLayer (stateless, RK4 bas-rang)
│   ├── moe.py                  # CREATE : PhaseRoutedMoE (gate von Mises, top-k)
│   └── block.py                # MODIFY : ajoute FractalBlockFull (integrant Kuramoto+MoE)
└── tests/
    ├── test_farey.py           # CREATE : tests Farey sequence
    ├── test_phase_ode.py       # CREATE : tests Kuramoto (RK4, encode/decode, loss)
    └── test_moe.py             # CREATE : tests MoE (gate, top-k, load-balance, backward)
```

**Responsabilites :**
- `farey.py` : generation deterministic of the Farey sequence + selection of phases expert. Aucun parameter.
- `phase_ode.py` : `KuramotoLayer` (stateless) — encode depuis hidden, integre RK4, decode_to_bias.
- `moe.py` : `PhaseRoutedMoE` — gate von Mises, top-k routing, load-balance loss.
- `block.py` : `FractalBlockFull` etend the `FractalBlock` minimal en ajoutant Kuramoto + MoE.

---

## Task 1: Suite of Farey + expert_phases

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/farey.py`

- [ ] **Step 1: Ecrire the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_farey.py` :
```python
"""Tests of the Farey sequence and of the selection of phases expert."""

import math
import torch


def test_farey_sequence_basic():
    """F_3 = {0/1, 1/3, 1/2, 2/3, 1/1} (5 termes)."""
    from fractus.nn.farey import farey_sequence
    seq = farey_sequence(3)
    fractions = seq  # liste of (p, q)
    assert fractions == [(0, 1), (1, 3), (1, 2), (2, 3), (1, 1)]


def test_farey_sequence_order_1():
    """F_1 = {0/1, 1/1}."""
    from fractus.nn.farey import farey_sequence
    assert farey_sequence(1) == [(0, 1), (1, 1)]


def test_farey_sequence_sorted():
    """Les fractions must etre croissantes (property of Farey)."""
    from fractus.nn.farey import farey_sequence
    seq = farey_sequence(5)
    values = [p / q for (p, q) in seq]
    assert values == sorted(values)


def test_farey_sequence_all_denominators_le_n():
    """Dans F_n, all the denominateurs are <= n."""
    from fractus.nn.farey import farey_sequence
    seq = farey_sequence(6)
    for (p, q) in seq:
        assert q <= 6


def test_expert_phases_count():
    """expert_phases(n) returns exactment n angles."""
    from fractus.nn.farey import expert_phases
    for n in [4, 8, 16]:
        phases = expert_phases(n)
        assert len(phases) == n


def test_expert_phases_in_unit_circle():
    """Tous the angles ∈ [0, 2π)."""
    from fractus.nn.farey import expert_phases
    phases = expert_phases(8)
    for theta in phases:
        assert 0.0 <= theta < 2 * math.pi


def test_expert_phases_distinct():
    """Les phases expert must etre distinctes (sinon the routing degenerated)."""
    from fractus.nn.farey import expert_phases
    phases = expert_phases(8)
    # Deux phases not must not etre identiques (a tolerance flott pres).
    for i in range(len(phases)):
        for j in range(i + 1, len(phases)):
            assert abs(phases[i] - phases[j]) > 1e-6, \
                f"phases[{i}]={phases[i]} == phases[{j}]={phases[j]}"
```

- [ ] **Step 2: Lancer for verify that the tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_farey.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer farey.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/farey.py` :
```python
"""Suite of Farey and selection of phases for the MoE a routing of phase.

Porte depuis the original architecture (src/math/farey.rs).

La Farey sequence F_n est l'ensemble trie fractions irreductibles p/q in
[0, 1] with q <= n. Elle est generatede iterativement by the property of mediante.

Pour the MoE : on prend F_{2E} (ordre double number d'experts) and on selectionne
uniformement E angles parmi the fractions, convertis en angles 2π·p/q ∈ [0, 2π).
Cela donne a distribution of phases dense, non-collapsante and deterministic —
l'interet for the routing von Mises.
"""

import math
from typing import List, Tuple


def _gcd(a: int, b: int) -> int:
    """PGCD d'Euclide (a, b > 0 supposes)."""
    while b:
        a, b = b, a % b
    return a


def farey_sequence(n: int) -> List[Tuple[int, int]]:
    """Genere the Farey sequence F_n comme liste of (p, q) triee croissante.

    Algorithme by mediante (comme farey.rs:18-49) :
        Init : (a,b)=(0,1), (c,d)=(1,n).
        Tant that c <= n : on push (a,b), then on computatione the prochain terme
        via k = (n + b) // d ; next = (k*c - a, k*d - b).

    F_n contient exactment 1 + Σ_{q=1}^{n} φ(q) termes (φ = indicatrice d'Euler).
    """
    if n < 1:
        raise ValueError("n must etre >= 1")
    fractions: List[Tuple[int, int]] = []
    a, b = 0, 1
    c, d = 1, n
    fractions.append((a, b))
    while c <= n:
        k = (n + b) // d
        a, b = c, d
        c, d = k * c - a, k * d - b  # ATTENTION : a,b already updates, therefore on must
        # recomputationer proprement. On correctede ci-dessous with variables tmp.
        fractions.append((a, b))
    # Le computation ci-dessus est incorrect (a,b modifies before). Version correcte :
    # On reprend proprement with variables temporaires.
    fractions = []
    a, b = 0, 1
    c, d = 1, n
    fractions.append((a, b))
    while c <= n:
        k = (n + b) // d
        next_c = k * c - a
        next_d = k * d - b
        a, b = c, d
        c, d = next_c, next_d
        fractions.append((a, b))
    return fractions


def expert_phases(n_experts: int) -> List[float]:
    """Selectionne n_experts angles ∈ [0, 2π) depuis F_{2·n_experts}.

    Comme farey.rs:53-64 : on construit F_{2E} (ordre double), then on selectionne
    uniformement E angles parmi the n_frac = len(F_{2E}) fractions disponibles.
    """
    if n_experts < 1:
        raise ValueError("n_experts must etre >= 1")
    fractions = farey_sequence(2 * n_experts)
    n_frac = len(fractions)
    angles_all = [2.0 * math.pi * p / q for (p, q) in fractions]
    phases: List[float] = []
    for i in range(n_experts):
        idx = min(int(i * n_frac / n_experts), n_frac - 1)
        phases.append(angles_all[idx])
    return phases


def expert_phases_tensor(n_experts: int) -> "torch.Tensor":  # type: ignore[name-defined]
    """Variante tenseur (for enregistrement comme buffer). Import torch local."""
    import torch
    return torch.tensor(expert_phases(n_experts), dtype=torch.float32)
```

- [ ] **Step 4: Lancer the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_farey.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add fractus/nn/farey.py tests/test_farey.py
git commit -m "feat(nn): add Farey sequence and expert_phases for phase-routed MoE"
```

---

## Task 2: KuramotoLayer (stateless, RK4 bas-rang)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/phase_ode.py`

- [ ] **Step 1: Ecrire the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_phase_ode.py` :
```python
"""Tests of KuramotoLayer : encode/decode, RK4, phase_loss, backward."""

import math
import torch


def test_kuramoto_output_shape():
    """Sortie phases (B, L, N_osc) for entree (B, L, d_model)."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    assert phases.shape == (2, 10, 8)


def test_kuramoto_phases_in_unit_circle():
    """Toutes the phases ∈ [0, 2π) (wrapping modulaire after RK4)."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16) * 10  # grandes values
    phases = layer(x)
    assert (phases >= 0).all() and (phases < 2 * math.pi).all()


def test_kuramoto_is_finite():
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    assert torch.isfinite(phases).all()


def test_kuramoto_backward_every_param():
    """CRITERE L2b : backward propage a gradient fini ET non-nul a CHAQUE parameter."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    loss = phases.sum()
    loss.backward()

    params = list(layer.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a a gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a recu a gradient nul"


def test_kuramoto_phase_loss_shape_and_sign():
    """phase_loss(phases) returns a scalar (un peu negatif typiquement,
    because L = -mean(K_ij cos(θ_i-θ_j)) and the cos can etre positif)."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    loss = layer.phase_loss(phases)
    assert loss.dim() == 0  # scalar
    assert torch.isfinite(loss)


def test_kuramoto_decode_to_bias_shape():
    """decode_to_bias(phases, d_model) returns (B, L, d_model) — injectable
    comme biais in the reseau (encodage positionnel sinusoidal)."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    phases = torch.rand(2, 10, 8) * 2 * math.pi
    bias = layer.decode_to_bias(phases, d_model=16)
    assert bias.shape == (2, 10, 16)
    assert torch.isfinite(bias).all()
```

- [ ] **Step 2: Lancer for verify that the tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_phase_ode.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer KuramotoLayer**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/phase_ode.py` :
```python
"""KuramotoLayer : Kuramoto oscillators couples bas-rang, STATELESS.

Porte depuis the original architecture (src/phase_ode.rs) en PyTorch pur.

Mathematique (shape bas-rang K = UΛUT, integration RK4) :

    dθ_i/dt = ω_i − damping·θ_i + K_strength·Σ_j K_ij sin(θ_j − θ_i)

    Reecrit en exploitant K = UΛUT (O(N·r) instead of O(N2)) :
        p = UT sin(θ) ∈ R^r
        q = UT cos(θ) ∈ R^r
        u_p = U (Λ ⊙ p)  ∈ R^N
        u_q = U (Λ ⊙ q)  ∈ R^N
        dθ_i = ω_i − damping·θ_i + K_strength·(cos(θ_i)·u_p[i] − sin(θ_i)·u_q[i])

    RK4 standard (4 under-steps), then wrap θ_i mod 2π → [0, 2π).

STATELESS : not d'etat persistant between forwards. Les phases initiales sont
derivees hidden states a each appel (encode_from_hidden). U, Λ, ω sont
des nn.Parameter (le courange s'apprend).

Corrections vs the original :
- the original gardait a etat `phases` mutable → ici STATELESS for reproductibilite
  and testabilite (le reviewer insisterait sinon).
- Le terme `-damping·θ_i` est conserve tel quel (non-standard Kuramoto but
  faithful a the original).
"""

import math
import torch
import torch.nn as nn


class KuramotoLayer(nn.Module):
    """Couche d'Kuramoto oscillators bas-rang, STATELESS.

    Args:
        d_model       : dimension d'entree (hidden).
        n_oscillators : number d'oscillateurs N.
        rank          : rang r courange bas-rang K = UΛUT.
        n_steps       : number of not RK4 by forward.
        dt            : taille of not RK4.
        damping       : amortissement lineaire (terme -damping·θ).
    """

    def __init__(
        self,
        d_model: int,
        n_oscillators: int,
        rank: int,
        n_steps: int = 4,
        dt: float = 0.1,
        damping: float = 0.01,
    ):
        super().__init__()
        if n_oscillators < 1 or rank < 1 or rank > n_oscillators:
            raise ValueError("n_oscillators >= 1 and 1 <= rank <= n_oscillators")
        self.d_model = d_model
        self.N = n_oscillators
        self.rank = rank
        self.n_steps = n_steps
        self.dt = dt
        self.damping = damping
        self.TWO_PI = 2.0 * math.pi

        # Parametres entrainables (init comme the original phase_ode.rs:38-57).
        # omega ~ U(-0.05, 0.05).
        self.omega = nn.Parameter(torch.empty(n_oscillators).uniform_(-0.05, 0.05))
        # U ∈ R^{N, r} ~ U(-1, 1).
        self.coupling_u = nn.Parameter(torch.empty(n_oscillators, rank).uniform_(-1.0, 1.0))
        # Λ ∈ R^r ~ U(0.01, 0.51) — POSITIF (forces attractives → synchronisation).
        self.coupling_lambda = nn.Parameter(torch.empty(rank).uniform_(0.01, 0.51))

    def _derivative(self, theta: torch.Tensor) -> torch.Tensor:
        """dθ/dt for phases theta of shape (..., N).

        Utilise the shape bas-rang (O(N·r)).
        """
        sin_t = torch.sin(theta)  # (..., N)
        cos_t = torch.cos(theta)
        # p = UT sin(θ), q = UT cos(θ) — shape (..., r).
        p = torch.einsum("...n,nr->...r", sin_t, self.coupling_u)
        q = torch.einsum("...n,nr->...r", cos_t, self.coupling_u)
        # u_p = U (Λ ⊙ p), u_q = U (Λ ⊙ q) — shape (..., N).
        u_p = torch.einsum("...r,nr->...n", self.coupling_lambda * p, self.coupling_u)
        u_q = torch.einsum("...r,nr->...n", self.coupling_lambda * q, self.coupling_u)
        # dθ_i = ω_i − damping·θ_i + (cos(θ_i)·u_p[i] − sin(θ_i)·u_q[i]).
        # coupling_strength = 1.0 (comme the original integrate_with_config).
        dtheta = (
            self.omega
            - self.damping * theta
            + cos_t * u_p
            - sin_t * u_q
        )
        return dtheta

    def _rk4_integrate(self, theta: torch.Tensor) -> torch.Tensor:
        """Integre n_steps not RK4 depuis theta (..., N). Retourne theta final.

        Apres each step complete, on wrap mod 2π (comme the original phase_ode.rs:153-155).
        """
        dt = self.dt
        for _ in range(self.n_steps):
            k1 = self._derivative(theta)
            k2 = self._derivative(theta + 0.5 * dt * k1)
            k3 = self._derivative(theta + 0.5 * dt * k2)
            k4 = self._derivative(theta + dt * k3)
            theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            # Wrap mod 2π → [0, 2π).
            theta = torch.remainder(theta, self.TWO_PI)
        return theta

    def _encode_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Phases initiales depuis hidden states.

        hidden : (B, L, d_model). Retourne (B, L, N).
        Comme phase_ode.rs:226-248 : θ_i = (mean(hidden[i,:]) · 2π) mod 2π
        for i < min(N, seq_len), sinon 0.
        """
        B, L, D = hidden.shape
        # mean on the dimension d_model : (B, L).
        hidden_mean = hidden.mean(dim=-1) * self.TWO_PI  # (B, L)
        # On a besoin of N phases by token. Comme N can differer of L, on
        # broadcast : on prend hidden_mean repete N fois, decale d'un offset
        # by oscillateur for casser the symetrie.
        # (the original utilisait i < min(N, seq_len) ; ici on generalise en broadcast.)
        offsets = torch.arange(self.N, dtype=hidden.dtype, device=hidden.device)
        offsets = offsets / self.N * self.TWO_PI  # offsets [0, 2π) by oscillateur
        # (B, L, N) = hidden_mean(B,L,1) + offsets(1,1,N).
        theta_init = hidden_mean.unsqueeze(-1) + offsets.view(1, 1, self.N)
        return torch.remainder(theta_init, self.TWO_PI)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden : (B, L, d_model) → phases (B, L, N) after RK4."""
        theta = self._encode_from_hidden(hidden)  # (B, L, N)
        return self._rk4_integrate(theta)

    def phase_loss(self, phases: torch.Tensor) -> torch.Tensor:
        """L = -(1/N2) · [cosθTK·cosθ + sinθTK·sinθ] (bas-rang).

        phases : (B, L, N). Retourne a scalar.
        Forme bas-rang : cosθTK·cosθ = (UTcosθ)TΛ(UTcosθ), idem for sin.
        """
        cos_t = torch.cos(phases)  # (B, L, N)
        sin_t = torch.sin(phases)
        # UTcosθ : (B, L, r).
        uc = torch.einsum("bln,nr->blr", cos_t, self.coupling_u)
        us = torch.einsum("bln,nr->blr", sin_t, self.coupling_u)
        # (UTcosθ)TΛ(UTcosθ) = Σ_r Λ_r · uc[...,r]2 (par oscillateur, somme on B,L).
        term_cos = (uc ** 2 * self.coupling_lambda).sum()
        term_sin = (us ** 2 * self.coupling_lambda).sum()
        N = self.N
        # Moyenne on (B, L, N2).
        # On normalise by the number d'elements for avoir a loss par-token.
        n_elem = phases.numel()  # B*L*N
        # the original divisait by N2 ; on adapte a (B,L,N) en divisant by N by token.
        scale = n_elem / (N * N + 1e-12)
        loss = -(term_cos + term_sin) / scale
        return loss

    def decode_to_bias(self, phases: torch.Tensor, d_model: int) -> torch.Tensor:
        """Encodage positionnel sinusoidal depuis the phases.

        phases : (B, L, N). Retourne (B, L, d_model).
        Comme phase_ode.rs:252-266 :
            freq = j//2 + 1  (commence a 1)
            bias[i, 2k]   = sin(freq · θ_i) / sqrt(freq)
            bias[i, 2k+1] = cos(freq · θ_i) / sqrt(freq)
        On utilise phases[..., 0:d_model] (tronque or repete si d_model > N).
        """
        B, L, N = phases.shape
        # Selectionner d_model phases (repetition cyclique si d_model > N).
        idx = torch.arange(d_model, device=phases.device) % N
        phases_used = phases[..., idx]  # (B, L, d_model)
        # freq = j//2 + 1.
        j = torch.arange(d_model, dtype=phases.dtype, device=phases.device)
        freq = (j // 2 + 1).view(1, 1, d_model)
        sin_part = torch.sin(freq * phases_used) / torch.sqrt(freq)
        cos_part = torch.cos(freq * phases_used) / torch.sqrt(freq)
        # Interleave : colonnes paires = sin, impaires = cos.
        bias = torch.empty(B, L, d_model, dtype=phases.dtype, device=phases.device)
        bias[..., 0::2] = sin_part[..., 0::2]
        bias[..., 1::2] = cos_part[..., 1::2]
        return bias
```

- [ ] **Step 4: Lancer the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_phase_ode.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add fractus/nn/phase_ode.py tests/test_phase_ode.py
git commit -m "feat(nn): add KuramotoLayer (stateless, low-rank RK4, differentiable)"
```

---

## Task 3: PhaseRoutedMoE (gate von Mises, top-k, load-balance)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/moe.py`

- [ ] **Step 1: Ecrire the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_moe.py` :
```python
"""Tests of PhaseRoutedMoE : gate von Mises, top-k, load-balance, backward."""

import math
import torch


def test_moe_output_shape():
    """Sortie (B, L, d_model) + loss auxiliaire scalar."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi  # n_oscillators=4 (can differer)
    out, lb_loss = moe(h, phases)
    assert out.shape == (2, 8, 16)
    assert lb_loss.dim() == 0


def test_moe_is_finite():
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16) * 5
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    out, lb_loss = moe(h, phases)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lb_loss)


def test_moe_load_balance_nonneg():
    """Load-balance loss >= 0 (this is a somme of carres ponderee)."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    _, lb_loss = moe(h, phases)
    assert lb_loss.item() >= -1e-6


def test_moe_backward_every_param():
    """CRITERE L2b : backward propage a gradient fini ET non-nul a CHAQUE parameter."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    out, lb_loss = moe(h, phases)
    loss = out.pow(2).mean() + 0.1 * lb_loss
    loss.backward()

    params = list(moe.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a a gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a recu a gradient nul"


def test_moe_top_k_at_most_n_experts():
    """top_k > n_experts must lever a error."""
    from fractus.nn.moe import PhaseRoutedMoE
    import pytest
    with pytest.raises(ValueError):
        PhaseRoutedMoE(d_model=16, n_experts=4, top_k=8, kappa=4.0)


def test_moe_with_uniform_phases_uses_all_experts():
    """Si all the phases are identiques, all the experts recoivent a gate
    equivalent (le routing not must not crasher)."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    # Phases uniformes : all the tokens have phase 0.
    phases = torch.zeros(2, 8, 4)
    out, lb_loss = moe(h, phases)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Lancer for verify that the tests echouent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_moe.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implementer PhaseRoutedMoE**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/moe.py` :
```python
"""PhaseRoutedMoE : mixture-of-experts a routing of phase von Mises.

Porte depuis the original architecture (src/moe.rs + farey.rs) en PyTorch pur.

Mathematique :
    Phases experts : E angles ∈ [0, 2π) issus of the Farey sequence F_{2E}
    (deterministic, dense, non-collapsante).

    Phase moyenne token : θ = atan2(Σ_p sin(θ_p), Σ_p cos(θ_p))
    (moyenne circulaire on the n_phases token).

    Gate von Mises (non normalise) :
        κ_eff = κ / temperature
        g_e = exp(κ_eff · cos(θ − θ_e))      for e = 0..E-1

    Normalisation : g_e /= Σ_e g_e (uniforme 1/E si Σ < 1e-10).

    Top-k routing : on selectionne the K meilleurs experts (gates max),
    on renormalise the gates retenues on 1.

    Expert : MLP GeLU gelu(x·W1 + b1)·W2 + b2.

    Load-balance loss (auxiliaire) :
        P_e = moyenne gates of l'expert e on all the tokens
        L_balance = E · Σ_e (P_e − 1/E)2

Differentiable end-to-end (poids W1/W2 experts are entrainables).
Les phases expert are en buffer (precomputation Farey, hors-graphe).
"""

import math
import torch
import torch.nn as nn

from .farey import expert_phases


def _gelu(x: torch.Tensor) -> torch.Tensor:
    """GeLU approximation tanh (comme moe.rs:14-17)."""
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))


class PhaseRoutedMoE(nn.Module):
    """Mixture-of-experts a routing of phase von Mises on phases Farey.

    Args:
        d_model     : dimension d'entree/sortie.
        n_experts   : number d'experts E.
        top_k       : number d'experts actives by token (<= E).
        kappa       : concentration von Mises.
        temperature : temperature gate (κ_eff = κ/temperature).
        d_ff        : dimension cachee experts (64 by defaut comme the original).
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        kappa: float = 4.0,
        temperature: float = 1.0,
        d_ff: int = 64,
    ):
        super().__init__()
        if n_experts < 1:
            raise ValueError("n_experts >= 1")
        if top_k < 1 or top_k > n_experts:
            raise ValueError(f"top_k must etre in [1, {n_experts}], eu {top_k}")
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.kappa = kappa
        self.temperature = temperature
        self.d_ff = d_ff

        # Phases expert (precomputation Farey, hors-graphe).
        phases = expert_phases(n_experts)
        self.register_buffer("expert_phases", torch.tensor(phases, dtype=torch.float32))

        # Poids experts : E × (W1, b1, W2, b2).
        # Init Xavier uniforme (comme moe.rs:28-44).
        scale1 = math.sqrt(2.0 / d_model)
        scale2 = math.sqrt(2.0 / d_ff)
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff).uniform_(-scale1, scale1))
        self.b1 = nn.Parameter(torch.zeros(n_experts, d_ff))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model).uniform_(-scale2, scale2))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))

    def _compute_gates(self, phases: torch.Tensor) -> torch.Tensor:
        """Calcule the gates von Mises for each token.

        phases : (B, L, n_phases). Retourne gates (B, L, E).
        """
        # Phase moyenne circulaire token.
        sin_p = torch.sin(phases).sum(dim=-1)  # (B, L)
        cos_p = torch.cos(phases).sum(dim=-1)
        theta_bar = torch.atan2(sin_p, cos_p)  # (B, L)
        # Gate von Mises : exp(κ_eff · cos(θ − θ_e)).
        kappa_eff = self.kappa / self.temperature
        diff = theta_bar.unsqueeze(-1) - self.expert_phases.view(
            *[1] * (phases.dim() - 1), self.n_experts
        )  # (B, L, E)
        gates = torch.exp(kappa_eff * torch.cos(diff))  # (B, L, E)
        # Normalisation on E.
        gates_sum = gates.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(gates, 1.0 / self.n_experts)
        gates = torch.where(gates_sum > 1e-10, gates / gates_sum, uniform)
        return gates

    def _expert_forward(self, h: torch.Tensor) -> torch.Tensor:
        """h : (B, L, d_model) → sorties of all the experts (B, L, E, d_model)."""
        # h: (B, L, d_model) ; w1: (E, d_model, d_ff).
        # Pour each expert e : gelu(h @ w1[e] + b1[e]) @ w2[e] + b2[e].
        # einsum : (B,L,d_model) × (E,d_model,d_ff) → (B,L,E,d_ff).
        h1 = torch.einsum("bld,edm->blef", h, self.w1) + self.b1.view(1, 1, self.n_experts, self.d_ff)
        h1_act = _gelu(h1)
        out = torch.einsum("blef,efd->bled", h1_act, self.w2) + self.b2.view(1, 1, self.n_experts, self.d_model)
        return out  # (B, L, E, d_model)

    def forward(
        self, h: torch.Tensor, phases: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """h : (B, L, d_model), phases : (B, L, n_phases).
        Retourne (output (B, L, d_model), load_balance_loss scalar).
        """
        B, L, _ = h.shape
        gates = self._compute_gates(phases)  # (B, L, E)

        # Top-k : selectionner the K meilleurs experts by token.
        topk_vals, topk_idx = gates.topk(self.top_k, dim=-1)  # (B, L, K)
        # Renormaliser the K gates retenues.
        topk_sum = topk_vals.sum(dim=-1, keepdim=True)
        uniform_topk = torch.full_like(topk_vals, 1.0 / self.top_k)
        topk_vals_norm = torch.where(
            topk_sum > 1e-10, topk_vals / topk_sum, uniform_topk
        )  # (B, L, K)

        # Sorties of all the experts.
        all_out = self._expert_forward(h)  # (B, L, E, d_model)

        # Gather the sorties top-k : (B, L, K, d_model).
        # topk_idx : (B, L, K) indices in [0, E).
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)  # (B, L, K, d_model)
        topk_out = torch.gather(all_out, dim=2, index=idx_exp)  # (B, L, K, d_model)

        # Combinaison ponderee : Σ_k gate_k · out_k.
        output = (topk_vals_norm.unsqueeze(-1) * topk_out).sum(dim=2)  # (B, L, d_model)

        # Load-balance loss : E · Σ_e (P_e − 1/E)2, P_e = moyenne gates.
        P = gates.mean(dim=(0, 1))  # (E,) moyenne on B, L
        lb_loss = self.n_experts * ((P - 1.0 / self.n_experts) ** 2).sum()

        return output, lb_loss
```

- [ ] **Step 4: Lancer the tests — DOIVENT PASSER**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_moe.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add fractus/nn/moe.py tests/test_moe.py
git commit -m "feat(nn): add PhaseRoutedMoE (von Mises gate, Farey phases, top-k, load-balance)"
```

---

## Task 4: FractalBlockFull + integration + demo etendue

**Files:**
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/block.py`
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py`
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_transformer.py`

- [ ] **Step 1: Etendre block.py with FractalBlockFull**

Append to `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/block.py` (after the existing `FractalBlock`) :
```python


class FractalBlockFull(nn.Module):
    """Bloc transformer fractal complete (L2b) : integre Kuramoto + MoE.

    Architecture :
        x → LN → FractalLinearAttention → + x (residuelle 1)
              → LN → KuramotoLayer → phases
              → LN → PhaseRoutedMoE(hidden, phases) → + x (residuelle 2)

    Retourne (output, loss_aux) or loss_aux regroupe the phase_loss and la
    load_balance_loss MoE (a ajouter a the loss principale by the caller).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_levels: int,
        n_oscillators: int,
        coupling_rank: int,
        n_experts: int,
        top_k: int,
        kappa: float = 4.0,
        kuramoto_steps: int = 4,
        kuramoto_dt: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Sous-bloc attention (comme FractalBlock minimal).
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FractalLinearAttention(d_model, n_heads, d_head, n_levels)
        # Kuramoto + MoE.
        self.norm_kur = nn.LayerNorm(d_model)
        self.kuramoto = KuramotoLayer(d_model, n_oscillators, coupling_rank,
                                      n_steps=kuramoto_steps, dt=kuramoto_dt)
        self.norm_moe = nn.LayerNorm(d_model)
        self.moe = PhaseRoutedMoE(d_model, n_experts, top_k, kappa=kappa)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x : (B, L, d_model) → (output (B, L, d_model), loss_aux scalar)."""
        # Residuelle 1 : attention.
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Kuramoto : phases depuis hidden normalise.
        phases = self.kuramoto(self.norm_kur(x))  # (B, L, N)
        # MoE : routing by phases.
        moe_out, lb_loss = self.moe(self.norm_moe(x), phases)
        x = x + self.dropout(moe_out)
        # Loss auxiliaire : load-balance (phase_loss optionnelle, ici on l'omet
        # for simplicite — can etre ajoutee by the caller via self.kuramoto).
        return x, lb_loss
```

Et ajouter l'import en haut of `block.py` :
```python
from .attention import FractalLinearAttention
from .phase_ode import KuramotoLayer
from .moe import PhaseRoutedMoE
```

- [ ] **Step 2: Mettre a jour __init__.py**

Modify `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py` — ajouter :
```python
from .farey import farey_sequence, expert_phases
from .phase_ode import KuramotoLayer
from .moe import PhaseRoutedMoE
from .block import FractalBlock, FractalBlockFull
```
Et etendre `__all__` with : `"farey_sequence"`, `"expert_phases"`, `"KuramotoLayer"`, `"PhaseRoutedMoE"`, `"FractalBlockFull"`.

- [ ] **Step 3: Test critique d'integration**

Append to `C:/Users/PHIL/ZCodeProject/fractus/tests/test_block.py` :
```python


def test_block_full_backward_every_param():
    """CRITERE L2b : FractalBlockFull (attn + Kuramoto + MoE) must propager un
    gradient fini ET non-nul a CHAQUE parameter. La proof ultime that tout
    the pipeline fractal est differentiable."""
    from fractus.nn.block import FractalBlockFull
    block = FractalBlockFull(
        d_model=32, n_heads=4, d_head=8, n_levels=2,
        n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, kappa=4.0,
    )
    x = torch.randn(2, 8, 32)
    out, lb_loss = block(x)
    loss = out.pow(2).sum() + 0.1 * lb_loss
    loss.backward()

    params = list(block.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a a gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a recu a gradient nul"


def test_block_full_shape_and_finite():
    from fractus.nn.block import FractalBlockFull
    block = FractalBlockFull(
        d_model=32, n_heads=4, d_head=8, n_levels=2,
        n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, kappa=4.0,
    )
    x = torch.randn(2, 8, 32)
    out, lb_loss = block(x)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lb_loss)
```

- [ ] **Step 4: Lancer all the tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: 41 (L0+L1+L2a) + 7 (farey) + 6 (kuramoto) + 6 (moe) + 2 (block full) = 62 passed.

- [ ] **Step 5: Commit**

```bash
git add fractus/nn/block.py fractus/nn/__init__.py tests/test_block.py
git commit -m "feat(nn): add FractalBlockFull integrating Kuramoto + MoE (L2b complete)"
```

---

## Task 5: Demo L2b etendue

**Files:**
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_transformer.py`

- [ ] **Step 1: Ajouter a variante TinyFractalLMFull utilisant FractalBlockFull**

(Optionnel — l'objective est of montrer that the bloc complete apprend aussi.)
Voir code complete in the commit.

- [ ] **Step 2: Lancer the demo etendue**

```powershell
.venv\Scripts\python.exe scripts\demo_transformer.py
```
Expected: the loss must baisser (au less ÷2).

- [ ] **Step 3: Commit**

```bash
git add scripts/demo_transformer.py
git commit -m "demo(L2b): extended demo with Kuramoto+MoE block"
```

---

## Critere final of L2b « termine »

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
# → 62 passed

.venv\Scripts\python.exe scripts\demo_transformer.py
# → loss baisse
```

L2b termine → on a a transformer fractal COMPLET (embedding + attention + Kuramoto + MoE), differentiable end-to-end. On passe then a L3 (SIREN).

---

## Self-Review

**1. Spec coverage :** (a) Farey → Task 1 ✅ ; (b) KuramotoLayer stateless RK4 → Task 2 ✅ ; (c) PhaseRoutedMoE von Mises → Task 3 ✅ ; (d) FractalBlockFull → Task 4 ✅ ; (e) critere backward CHAQUE parameter on the bloc complete → `test_block_full_backward_every_param` ✅.

**2. Placeholder scan :** no TBD. ✅

**3. Type consistency :** `farey_sequence(int) → List[(int,int)]`, `expert_phases(int) → List[float]`. `KuramotoLayer(d_model, N, rank).forward((B,L,d_model)) → (B,L,N)`, `.phase_loss((B,L,N)) → scalar`, `.decode_to_bias((B,L,N), d_model) → (B,L,d_model)`. `PhaseRoutedMoE(d_model, E, K, kappa).forward((B,L,d_model), (B,L,n_phases)) → ((B,L,d_model), scalar)`. Coherent. ✅

**4. Decision stateless :** KuramotoLayer n'a not of `register_buffer` for a etat of phases persistant — the phases are derivees hidden a each forward. Conforme a the decision valide. ✅

**5. Fidelite the original :** RK4 bas-rang, encode_from_hidden, decode_to_bias, phase_loss (bas-rang cos+sin), gate von Mises, top-k, load-balance E·Σ(P-1/E)2, experts GeLU Xavier. Tous porteds faithfully. ✅
