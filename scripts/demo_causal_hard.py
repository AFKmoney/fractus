"""Demo L4+ : NOTEARS on SCM non-lineaire with topological order INCONNU.

VALIDATION SERIEUSE au-dela cas jouet L4 (lineaire + triangulaire).
Si NOTEARS recupere the DAG ici, on a a vraie proof of competence.

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

    print("=== SCM non-lineaire with ordre topologique INCONNU ===")
    W_true, X = generate_nonlinear_scm(n_vars=5, n_samples=2000, edge_prob=0.5, seed=11)
    n_edges = int((W_true != 0).sum())
    print(f"Variables : {W_true.shape[0]}, echantillons : {X.shape[0]}")
    print(f"Aretes vraies : {n_edges}")
    print(f"W_true (NON triangulaire — ordre topo cache) :")
    print((W_true != 0).int())

    print()
    print("=== NOTEARS lineaire sur donnees non-lineaires ===")
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
    print(f"SHD = {shd} (sur {n_vars*n_vars} inputs)")
    print(f"Aretes : vraies={n_edges}, predites={n_pred}, correctes={n_correct}")
    print(f"W_pred appris :")
    print((W_pred.detach().abs() > 0.3).int())

    print()
    if shd <= 2:
        print(f"OK : NOTEARS recupere le DAG non-lineaire a ordre inconnu (SHD <= 2).")
        print(f"  Validation au-dela du cas jouet L4 : NOTEARS est competent.")
        print(f"  Note : NOTEARS lineaire est robuste a la non-linearite moderee")
        print(f"  (tanh ≈ identite for petites inputs). Pour une non-linearite")
        print(f"  forte, il faudrait NOTEARS non-lineaire (future work).")
    else:
        print(f"~ : SHD = {shd} > 2. NOTEARS lineaire a du mal sur ces donnees.")
        print(f"  Considerer : plus d'echantillons, NOTEARS non-lineaire.")


if __name__ == "__main__":
    main()
