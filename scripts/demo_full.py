"""Démo L7 : intégration complète des 3 tâches du spec fractus.

Le spec disait (L7) : 3 démos démontrables — texte, raisonnement mathématique,
inférence causale. Cette démo orchestre les 3 en un seul script, et utilise
les métriques honnêtes de fractus.metrics.

Ce qui marche (déjà validé dans les démos L2b, L4) :
    - Texte : TinyFractalLM apprend 'hello world', loss 4.73 → 0.65.
    - Causal : NOTEARS récupère un DAG synthétique, SHD = 0.
Ce qui ne marche pas (découvert en L5) :
    - Preuves : REINFORCE pur n'apprend pas (erreur stagne). On documente
      honnêtement au lieu de prétendre.

Run :
    python scripts/demo_full.py
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.nn import FractalEmbedding, FractalBlockFull
from fractus.causal.notears import notears_penalty
from fractus.metrics.causal import structural_hamming_distance
from fractus.metrics.perplexity import honest_perplexity
from fractus.metrics.compression import measure_compression_ratio
from data.causal.generate_scm import generate_linear_scm


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Tâche 1 : Génération de texte (FractalBlockFull)
# ---------------------------------------------------------------------------

class TinyFractalLM(nn.Module):
    def __init__(self, vocab, d_model, n_blocks=2):
        super().__init__()
        self.embed = FractalEmbedding(vocab, d_model, n_frequencies=8)
        self.blocks = nn.ModuleList([
            FractalBlockFull(
                d_model=d_model, n_heads=4, d_head=8, n_levels=2,
                n_oscillators=8, coupling_rank=4,
                n_experts=4, top_k=2, kappa=4.0,
            )
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        aux = torch.tensor(0.0)
        for b in self.blocks:
            x, lb = b(x)
            aux = aux + lb
        x = self.norm(x)
        return self.head(x), aux


def demo_text():
    section("Tâche 1 : Génération de texte (FractalBlockFull complet)")
    torch.manual_seed(42)
    text = "hello world " * 8
    vocab = 128
    ids = torch.tensor([ord(c) - 32 for c in text if 0 <= ord(c) - 32 < vocab])
    seq_len = 16
    n_seqs = len(ids) // seq_len
    ids = ids[:n_seqs * seq_len].view(n_seqs, seq_len)

    model = TinyFractalLM(vocab=vocab, d_model=32, n_blocks=2)
    print(f"Paramètres : {sum(p.numel() for p in model.parameters())}")
    print(f"Ratio compression (mesuré) : {measure_compression_ratio(model):.2f}x")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    initial_loss = None
    for epoch in range(40):
        opt.zero_grad()
        logits, aux = model(ids)
        ce = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, vocab), ids[:, 1:].reshape(-1))
        loss = ce + 0.1 * aux
        if initial_loss is None:
            initial_loss = ce.item()
        loss.backward()
        opt.step()
    final_loss = ce.item()

    # Perplexité honnête (pas proxy).
    ppl = honest_perplexity(model, ids[:, :-1], ids[:, 1:])

    print(f"\nCE Loss : {initial_loss:.4f} -> {final_loss:.4f} ({(1-final_loss/initial_loss)*100:.0f}% baisse)")
    print(f"Perplexité honnête : {ppl:.2f}  (= exp(CE))")

    # Génération.
    model.eval()
    with torch.no_grad():
        ctx = torch.tensor([[ord(c) - 32 for c in "hello"]])
        for _ in range(15):
            logits, _ = model(ctx)
            nxt = max(0, min(vocab - 1, int(logits[0, -1].argmax().item())))
            ctx = torch.cat([ctx, torch.tensor([[nxt]])], dim=1)
        gen = "".join(chr(int(i) + 32) for i in ctx[0].tolist())
    print(f"Génération : '{gen}'")


# ---------------------------------------------------------------------------
# Tâche 2 : Raisonnement mathématique (ProofGenerator + Verifier)
# ---------------------------------------------------------------------------

def demo_proofs():
    section("Tâche 2 : Raisonnement mathématique (vérificateur exact)")
    from fractus.reasoning.proof import ProofGenerator, ProofVerifier, ProofReward
    from fractus.reasoning.conjecture import ConjectureDiscoveryLoop

    # 2a. Le vérificateur exact est SOUND : tout ce qu'il accepte est vrai.
    verifier = ProofVerifier()
    examples = [
        ("7 est premier", lambda: verifier.verify_primality(7, True)),
        ("8 n'est pas premier", lambda: verifier.verify_primality(8, False)),
        ("Fermat 2^6 mod 7 = 1", lambda: verifier.verify_fermat(2, 7)),
        ("Wilson 6! mod 7 = 6", lambda: verifier.verify_wilson(7)),
        ("gcd(12,18)=6, lcm=36, 6*36=216=12*18", lambda: verifier.verify_gcd(12, 18)),
    ]
    print("Vérifications exactes (soundness garantie) :")
    for desc, fn in examples:
        result = fn()
        print(f"  [{'OK' if result else 'KO'}] {desc}")

    # 2b. Découverte de conjectures (Popper).
    print("\nDécouverte de conjectures (falsification popperienne) :")
    loop = ConjectureDiscoveryLoop(state_dim=32, n_trials=50, seed=42)
    for _ in range(30):
        loop.discover_step()
    print(f"  Conjectures en mémoire : {len(loop.memory.discovered)}")
    print(f"  Découvertes (survécu + novelty) : {loop.n_discoveries}")
    print(f"  Taux de découverte : {loop.discovery_rate():.1%}")

    # 2c. VERDICT HONNÊTE sur REINFORCE.
    print("\nVERDICT HONNÊTE (générateur de preuves) :")
    print("  Le ProofGenerator + REINFORCE pur n'apprend PAS la tâche de preuve")
    print("  (erreur stagne, voir demo_proof_reinforce.py). Le vérificateur est")
    print("  sound, mais le générateur ne suffit pas à l'exploiter. Future work.")
    print("  → Les conjectures, elles, sont testées par falsification exacte (marche).")


# ---------------------------------------------------------------------------
# Tâche 3 : Inférence causale (NOTEARS + do-calculus)
# ---------------------------------------------------------------------------

def demo_causal():
    section("Tâche 3 : Inférence causale (NOTEARS + do-calculus)")
    torch.manual_seed(42)
    W_true, X = generate_linear_scm(n_vars=5, n_samples=500, edge_prob=0.5, seed=7)
    print(f"SCM synthétique : {X.shape[0]} échantillons, {X.shape[1]} variables")
    print(f"Vrai DAG ({int((W_true != 0).sum())} arêtes) :")
    print((W_true != 0).int())

    n_vars = W_true.shape[0]
    W_pred = torch.zeros(n_vars, n_vars, requires_grad=True)
    torch.nn.init.normal_(W_pred, std=0.1)
    opt = torch.optim.Adam([W_pred], lr=0.05)
    for _ in range(500):
        opt.zero_grad()
        X_pred = X @ W_pred
        recon = ((X_pred - X) ** 2).mean()
        h = notears_penalty(W_pred)
        loss = recon + h.abs()
        loss.backward()
        opt.step()

    shd = structural_hamming_distance(W_true, W_pred.detach(), threshold=0.3)
    print(f"\nDAG appris (seuil 0.3) :")
    print((W_pred.detach().abs() > 0.3).int())
    print(f"\nSHD = {shd}  (0 = parfait)")
    print(f"  (cas jouet linéaire + triangulaire — prouve que le pipeline tourne,")
    print(f"   pas la compétence sur données réelles.)")

    # do-calculus : effet de do(X_0 = v) sur X_4.
    from fractus.causal.do import do_intervention
    X_do = do_intervention(X, var_idx=0, value=2.0)
    print(f"\ndo(X_0 = 2.0) appliqué : colonne 0 fixée à 2.0")
    print(f"  Effet moyen sur X_4 : observationnel {X[:, 4].mean():.3f} → "
          f"interventionnel {X_do[:, 4].mean():.3f}")


def main():
    print("FRACTUS — Démo L7 complète (3 tâches intégrées)")
    print("Réfonte honnête de FNN v5.0 + OMNI-FRACTAL")
    demo_text()
    demo_proofs()
    demo_causal()
    section("FIN")
    print("Tout ce qui précède utilise des MESURES HONNÊTES (pas de hardcode).")
    print("Voir docs/superpowers/specs/2026-06-19-fractus-unified-design.md.")


if __name__ == "__main__":
    main()
