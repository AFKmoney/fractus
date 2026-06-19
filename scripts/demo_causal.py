"""Démo L4 : NOTEARS récupère un DAG synthétique connu.

Étapes :
    1. Génère un SCM linéaire à 5 variables (DAG connu W_true + données X).
    2. Initialise W_pred aléatoire (entraînable).
    3. Optimise W_pred pour minimiser :
           reconstruction loss + λ · |notears_penalty(W_pred)|
       La pénalité NOTEARS force W_pred à être acyclique.
    4. Mesure le SHD entre W_pred et W_true.

Critère honnête : SHD <= 3 sur 5 variables (au plus 3 erreurs sur 25 entrées).

Run :
    python scripts/demo_causal.py
"""

import sys
import os
import torch

# Assurer que le package 'data' est importable depuis scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.causal.notears import notears_penalty
from fractus.metrics.causal import structural_hamming_distance
from data.causal.generate_scm import generate_linear_scm


def main():
    torch.manual_seed(42)

    # 1. SCM synthétique.
    W_true, X = generate_linear_scm(n_vars=5, n_samples=500, edge_prob=0.5, seed=7)
    print("=== SCM synthétique ===")
    print("W_true (DAG à 5 variables, triangulaire sup) :")
    print(W_true)
    print(f"Données X : {X.shape}")
    print()

    # 2. W_pred aléatoire, entraînable.
    n_vars = W_true.shape[0]
    W_pred = torch.zeros(n_vars, n_vars, requires_grad=True)
    torch.nn.init.normal_(W_pred, std=0.1)

    h_init = notears_penalty(W_pred).item()
    print(f"h(W_pred) initial = {h_init:.4f} (devrait être ~0 car W petite)")

    # 3. Optimisation : reconstruction + λ·|NOTEARS|.
    opt = torch.optim.Adam([W_pred], lr=0.05)
    lam = 1.0
    for step in range(500):
        opt.zero_grad()
        X_pred = X @ W_pred
        recon = ((X_pred - X) ** 2).mean()
        h = notears_penalty(W_pred)
        loss = recon + lam * h.abs()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 499:
            print(f"step {step:3d}  recon={recon.item():.4f}  h={h.item():.4f}")

    # 4. Mesure SHD.
    print()
    print("=== Récupération du DAG ===")
    print("W_pred appris (seuil 0.3) :")
    W_pred_bin = (W_pred.detach().abs() > 0.3).float()
    print(W_pred_bin)
    print("W_true binaire :")
    print((W_true.abs() > 0.3).float())

    shd = structural_hamming_distance(W_true, W_pred.detach(), threshold=0.3)
    print(f"\nSHD = {shd} (sur {n_vars*n_vars} entrées)")
    print(f"  0 = récupération parfaite, plus c'est bas mieux c'est.")
    if shd <= 3:
        print(f"\nOK : NOTEARS récupère le DAG (SHD <= 3).")
    else:
        print(f"\n~ : SHD > 3, récupération partielle.")


if __name__ == "__main__":
    main()
