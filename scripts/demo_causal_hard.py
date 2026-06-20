"""Démo L4+ : NOTEARS sur SCM non-linéaire avec ordre topologique INCONNU.

VALIDATION SÉRIEUSE au-delà du cas jouet L4 (linéaire + triangulaire).
Si NOTEARS récupère le DAG ici, on a une vraie preuve de compétence.

Run :
    python scripts/demo_causal_hard.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.causal.notears import notears_penalty
from fractus.metrics.causal import structural_hamming_distance
from data.causal.generate_scm_hard import generate_nonlinear_scm


def main():
    torch.manual_seed(42)

    print("=== SCM non-linéaire avec ordre topologique INCONNU ===")
    W_true, X = generate_nonlinear_scm(n_vars=5, n_samples=2000, edge_prob=0.5, seed=11)
    n_edges = int((W_true != 0).sum())
    print(f"Variables : {W_true.shape[0]}, échantillons : {X.shape[0]}")
    print(f"Arêtes vraies : {n_edges}")
    print(f"W_true (NON triangulaire — ordre topo caché) :")
    print((W_true != 0).int())

    print()
    print("=== NOTEARS linéaire sur données non-linéaires ===")
    n_vars = W_true.shape[0]
    W_pred = torch.zeros(n_vars, n_vars, requires_grad=True)
    torch.nn.init.normal_(W_pred, std=0.1)
    opt = torch.optim.Adam([W_pred], lr=0.05)
    for step in range(1000):
        opt.zero_grad()
        X_pred = X @ W_pred
        recon = ((X_pred - X) ** 2).mean()
        h = notears_penalty(W_pred)
        loss = recon + 1.0 * h.abs()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"  step {step:4d}  recon={recon.item():.4f}  h={h.item():.4f}")

    shd = structural_hamming_distance(W_true, W_pred.detach(), threshold=0.3)
    n_pred = int((W_pred.detach().abs() > 0.3).sum())
    n_correct = int(((W_true.abs() > 0.3) & (W_pred.detach().abs() > 0.3)).sum())

    print()
    print(f"SHD = {shd} (sur {n_vars*n_vars} entrées)")
    print(f"Arêtes : vraies={n_edges}, prédites={n_pred}, correctes={n_correct}")
    print(f"W_pred appris :")
    print((W_pred.detach().abs() > 0.3).int())

    print()
    if shd <= 2:
        print(f"OK : NOTEARS récupère le DAG non-linéaire à ordre inconnu (SHD <= 2).")
        print(f"  Validation au-delà du cas jouet L4 : NOTEARS est compétent.")
        print(f"  Note : NOTEARS linéaire est robuste à la non-linéarité modérée")
        print(f"  (tanh ≈ identité pour petites entrées). Pour une non-linéarité")
        print(f"  forte, il faudrait NOTEARS non-linéaire (future work).")
    else:
        print(f"~ : SHD = {shd} > 2. NOTEARS linéaire a du mal sur ces données.")
        print(f"  Considérer : plus d'échantillons, NOTEARS non-linéaire.")


if __name__ == "__main__":
    main()
