"""Demo L4 : NOTEARS recupere a DAG synthetique connu.

AVERTISSEMENT D'HONNETETE : cette demo utilise a SCM LINEAIRE and triangulaire
superieur (topological order trivial). This is the cas-jouet ideal for NOTEARS —
il a ete concu exactment for this reglage. SHD=0 ici prouve that the PIPELINE
tourne (les modules communiquent, NOTEARS s'optimise, the penalty fonctionne),
PAS that NOTEARS est competent on donnees realles. Un SCM non-lineaire with
topological order inconnu serait nettement more dur (future work).

Etapes :
    1. Genere a SCM lineaire a 5 variables (DAG connu W_true + donnees X).
    2. Initialise W_pred aleatoire (entrainable).
    3. Optimise W_pred for minimiser :
           reconstruction loss + λ · |notears_penalty(W_pred)|
       La penalty NOTEARS force W_pred a etre acyclique.
    4. Mesure the SHD between W_pred and W_true.

Critere : SHD <= 3 on 5 variables (cas jouet ideal — must passer).

Run :
    python scripts/demo_causal.py
"""

import sys
import os
import torch

# Assurer that the package 'data' est importable depuis scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.causal.notears import notears_penalty
from fractus.metrics.causal import structural_hamming_distance
from data.causal.generate_scm import generate_linear_scm


def main():
    torch.manual_seed(42)

    # 1. SCM synthetique.
    W_true, X = generate_linear_scm(n_vars=5, n_samples=500, edge_prob=0.5, seed=7)
    print("=== SCM synthetique ===")
    print("W_true (DAG a 5 variables, triangulaire sup) :")
    print(W_true)
    print(f"Donnees X : {X.shape}")
    print()

    # 2. W_pred aleatoire, entrainable.
    n_vars = W_true.shape[0]
    W_pred = torch.zeros(n_vars, n_vars, requires_grad=True)
    torch.nn.init.normal_(W_pred, std=0.1)

    h_init = notears_penalty(W_pred).item()
    print(f"h(W_pred) initial = {h_init:.4f} (should etre ~0 because W petite)")

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
    print("=== Recuperation du DAG ===")
    print("W_pred appris (seuil 0.3) :")
    W_pred_bin = (W_pred.detach().abs() > 0.3).float()
    print(W_pred_bin)
    print("W_true binary :")
    print((W_true.abs() > 0.3).float())

    shd = structural_hamming_distance(W_true, W_pred.detach(), threshold=0.3)
    print(f"\nSHD = {shd} (sur {n_vars*n_vars} entrees)")
    print(f"  0 = recuperation parfaite, plus c'est bas mieux c'est.")
    print(f"  (Note : cas-jouet ideal — SCM lineaire + triangulaire. Ne prouve")
    print(f"   pas la competence sur donnees reelles, juste que le pipeline tourne.)")
    if shd <= 3:
        print(f"\nOK : le pipeline causal tourne (SHD <= 3 sur cas jouet).")
    else:
        print(f"\n~ : SHD > 3, le pipeline a un souci meme sur cas jouet.")


if __name__ == "__main__":
    main()
