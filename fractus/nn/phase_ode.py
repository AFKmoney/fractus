"""KuramotoLayer: low-rank coupled Kuramoto oscillators, STATELESS.

Ported from the original architecture (src/phase_ode.rs) in pure PyTorch.

Math (low-rank form K = U*Lambda*U^T, RK4 integration):
    d_theta_i/dt = omega_i - damping*theta_i + sum_j K_ij sin(theta_j - theta_i)
    Standard RK4 (4 sub-steps), then wrap theta_i mod 2*pi after each step.

STATELESS: no persistent state between forwards. Initial phases are derived from
hidden states at each call. U, Lambda, omega are nn.Parameter (coupling is learned).
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
            raise ValueError("n_oscillators >= 1 et 1 <= rank <= n_oscillators")
        self.d_model = d_model
        self.N = n_oscillators
        self.rank = rank
        self.n_steps = n_steps
        self.dt = dt
        self.damping = damping
        self.TWO_PI = 2.0 * math.pi

        # Parametres entrainables (init comme the original phase_ode.rs:38-57).
        self.omega = nn.Parameter(torch.empty(n_oscillators).uniform_(-0.05, 0.05))
        self.coupling_u = nn.Parameter(torch.empty(n_oscillators, rank).uniform_(-1.0, 1.0))
        self.coupling_lambda = nn.Parameter(torch.empty(rank).uniform_(0.01, 0.51))

    def _derivative(self, theta: torch.Tensor) -> torch.Tensor:
        """dθ/dt for phases theta of shape (..., N). Forme bas-rang O(N·r)."""
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        p = torch.einsum("...n,nr->...r", sin_t, self.coupling_u)
        q = torch.einsum("...n,nr->...r", cos_t, self.coupling_u)
        u_p = torch.einsum("...r,nr->...n", self.coupling_lambda * p, self.coupling_u)
        u_q = torch.einsum("...r,nr->...n", self.coupling_lambda * q, self.coupling_u)
        dtheta = (
            self.omega
            - self.damping * theta
            + cos_t * u_p
            - sin_t * u_q
        )
        return dtheta

    def _rk4_integrate(self, theta: torch.Tensor) -> torch.Tensor:
        """Integre n_steps not RK4 depuis theta (..., N). Wrap mod 2π after each step."""
        dt = self.dt
        for _ in range(self.n_steps):
            k1 = self._derivative(theta)
            k2 = self._derivative(theta + 0.5 * dt * k1)
            k3 = self._derivative(theta + 0.5 * dt * k2)
            k4 = self._derivative(theta + dt * k3)
            theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            theta = torch.remainder(theta, self.TWO_PI)
        return theta

    def _encode_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Phases initiales depuis hidden states (B, L, d_model) → (B, L, N)."""
        hidden_mean = hidden.mean(dim=-1) * self.TWO_PI  # (B, L)
        offsets = torch.arange(self.N, dtype=hidden.dtype, device=hidden.device)
        offsets = offsets / self.N * self.TWO_PI
        theta_init = hidden_mean.unsqueeze(-1) + offsets.view(1, 1, self.N)
        return torch.remainder(theta_init, self.TWO_PI)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden : (B, L, d_model) → phases (B, L, N) after RK4."""
        theta = self._encode_from_hidden(hidden)
        return self._rk4_integrate(theta)

    def phase_loss(self, phases: torch.Tensor) -> torch.Tensor:
        """L = -(1/N2)·[cosθTK·cosθ + sinθTK·sinθ] (bas-rang), moyennee by token.

        Normalisation : on divise by (B·L·N/N) = B·L (un scalar by token,
        not the formula 1/N2 stricte of the original which supposait a seul token). This is
        coherent with l'usage batche.
        """
        cos_t = torch.cos(phases)
        sin_t = torch.sin(phases)
        uc = torch.einsum("bln,nr->blr", cos_t, self.coupling_u)
        us = torch.einsum("bln,nr->blr", sin_t, self.coupling_u)
        term_cos = (uc ** 2 * self.coupling_lambda).sum()
        term_sin = (us ** 2 * self.coupling_lambda).sum()
        N = self.N
        scale = phases.numel() / (N * N + 1e-12)
        return -(term_cos + term_sin) / scale

    def decode_to_bias(self, phases: torch.Tensor, d_model: int) -> torch.Tensor:
        """Encodage positionnel sinusoidal depuis the phases. (B,L,N) → (B,L,d_model).

        Non cablee in FractalBlockFull.forward (L2b) — methode utilitaire
        exposee for usage futur (ex. injecter a biais positionnel Kuramoto
        in a couche donnee). Testee separement.
        """
        B, L, N = phases.shape
        idx = torch.arange(d_model, device=phases.device) % N
        phases_used = phases[..., idx]
        j = torch.arange(d_model, dtype=phases.dtype, device=phases.device)
        freq = (j // 2 + 1).view(1, 1, d_model)
        sin_part = torch.sin(freq * phases_used) / torch.sqrt(freq)
        cos_part = torch.cos(freq * phases_used) / torch.sqrt(freq)
        bias = torch.empty(B, L, d_model, dtype=phases.dtype, device=phases.device)
        bias[..., 0::2] = sin_part[..., 0::2]
        bias[..., 1::2] = cos_part[..., 1::2]
        return bias
