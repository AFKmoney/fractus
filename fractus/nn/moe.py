"""PhaseRoutedMoE : mixture-of-experts à routing de phase von Mises.

Porté depuis FNN v5.0 (src/moe.rs + farey.rs) en PyTorch pur.

Mathématique :
    Phases des experts : E angles ∈ [0, 2π) issus de la suite de Farey F_{2E}.

    Phase moyenne du token : θ̄ = atan2(Σ_p sin(θ_p), Σ_p cos(θ_p))
    (moyenne circulaire sur les n_phases du token).

    Gate von Mises (non normalisé) :
        κ_eff = κ / temperature
        g_e = exp(κ_eff · cos(θ̄ − θ_e))      pour e = 0..E-1

    Normalisation : g_e /= Σ_e g_e (uniforme 1/E si Σ < 1e-10).

    Top-k routing : on sélectionne les K meilleurs experts (gates max),
    on renormalise les gates retenues sur 1.

    Expert : MLP GeLU gelu(x·W1 + b1)·W2 + b2.

    Load-balance loss (auxiliaire) :
        P_e = moyenne des gates de l'expert e sur tous les tokens
        L_balance = E · Σ_e (P_e − 1/E)²

Différentiable de bout en bout (poids W1/W2 des experts sont entraînables).
Les phases expert sont en buffer (précalcul Farey, hors-graphe).
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
    """Mixture-of-experts à routing de phase von Mises sur phases Farey.

    Args:
        d_model     : dimension d'entrée/sortie.
        n_experts   : nombre d'experts E.
        top_k       : nombre d'experts activés par token (<= E).
        kappa       : concentration von Mises.
        temperature : température du gate (κ_eff = κ/temperature).
        d_ff        : dimension cachée des experts (64 par défaut comme FNN).
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
            raise ValueError(f"top_k doit être dans [1, {n_experts}], eu {top_k}")
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.kappa = kappa
        self.temperature = temperature
        self.d_ff = d_ff

        # Phases expert (précalcul Farey, hors-graphe).
        phases = expert_phases(n_experts)
        self.register_buffer("expert_phases", torch.tensor(phases, dtype=torch.float32))

        # Poids des experts : E × (W1, b1, W2, b2). Init Xavier uniforme.
        scale1 = math.sqrt(2.0 / d_model)
        scale2 = math.sqrt(2.0 / d_ff)
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff).uniform_(-scale1, scale1))
        self.b1 = nn.Parameter(torch.zeros(n_experts, d_ff))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model).uniform_(-scale2, scale2))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))

    def _compute_gates(self, phases: torch.Tensor) -> torch.Tensor:
        """Calcule les gates von Mises pour chaque token.

        phases : (B, L, n_phases). Retourne gates (B, L, E).
        """
        sin_p = torch.sin(phases).sum(dim=-1)  # (B, L)
        cos_p = torch.cos(phases).sum(dim=-1)
        theta_bar = torch.atan2(sin_p, cos_p)  # (B, L)
        kappa_eff = self.kappa / self.temperature
        diff = theta_bar.unsqueeze(-1) - self.expert_phases.view(
            *[1] * (phases.dim() - 1), self.n_experts
        )  # (B, L, E)
        gates = torch.exp(kappa_eff * torch.cos(diff))  # (B, L, E)
        gates_sum = gates.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(gates, 1.0 / self.n_experts)
        gates = torch.where(gates_sum > 1e-10, gates / gates_sum, uniform)
        return gates

    def _expert_forward(self, h: torch.Tensor) -> torch.Tensor:
        """h : (B, L, d_model) → sorties de tous les experts (B, L, E, d_model).

        Pour chaque expert e : gelu(h @ w1[e] + b1[e]) @ w2[e] + b2[e].
        """
        B, L, D = h.shape
        # h: (B, L, D=d), w1: (E, D=d, F=f) → pour chaque (b,l,e,f) : Σ_d h[b,l,d]·w1[e,d,f].
        h1 = torch.einsum("bld,edf->blef", h, self.w1) + self.b1.view(1, 1, self.n_experts, self.d_ff)
        h1_act = _gelu(h1)
        # h1_act: (B,L,E,F=f), w2: (E, F=f, D=d) → out: (B,L,E,D=d).
        out = torch.einsum("blef,efd->bled", h1_act, self.w2) + self.b2.view(1, 1, self.n_experts, self.d_model)
        return out

    def forward(
        self, h: torch.Tensor, phases: torch.Tensor
    ):
        """h : (B, L, d_model), phases : (B, L, n_phases).
        Retourne (output (B, L, d_model), load_balance_loss scalaire).
        """
        gates = self._compute_gates(phases)  # (B, L, E)

        topk_vals, topk_idx = gates.topk(self.top_k, dim=-1)  # (B, L, K)
        topk_sum = topk_vals.sum(dim=-1, keepdim=True)
        uniform_topk = torch.full_like(topk_vals, 1.0 / self.top_k)
        topk_vals_norm = torch.where(
            topk_sum > 1e-10, topk_vals / topk_sum, uniform_topk
        )

        all_out = self._expert_forward(h)  # (B, L, E, d_model)
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
        topk_out = torch.gather(all_out, dim=2, index=idx_exp)  # (B, L, K, d_model)
        output = (topk_vals_norm.unsqueeze(-1) * topk_out).sum(dim=2)  # (B, L, d_model)

        P = gates.mean(dim=(0, 1))  # (E,)
        lb_loss = self.n_experts * ((P - 1.0 / self.n_experts) ** 2).sum()

        return output, lb_loss
