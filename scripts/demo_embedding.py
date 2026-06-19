"""Démo L1 : prouve que FractalEmbedding apprend vraiment.

Objectif : surfit une cible à des fréquences NON présentes dans la base Fourier
interne de l'embedding. Cela force l'embedding à réellement combiner ses trois
sources (char features + Fourier + conditionnement vortex) via sa projection
entraînable, au lieu de simplement recopier une combinaison de ses colonnes
d'entrée (ce qu'une Linear fait trivialement).

Les fréquences de la base interne sont ω_k = (φ²)^{-k} pour k=0..n_freq-1,
soit décroissantes à partir de 1.0. On choisit donc des fréquences cibles
indépendantes : croissantes (1.3, 0.7) qui ne sont PAS dans la base.

C'est la preuve minimale honnête que l'autodiff traverse l'embedding fractal
(ce que FNN v5.0 ne savait pas faire — training.rs:399 utilisait du bruit).

Run :
    python scripts/demo_embedding.py
"""

import torch
from fractus.nn import FractalEmbedding


def main():
    torch.manual_seed(42)

    vocab = 64
    d_model = 32
    emb = FractalEmbedding(vocab_size=vocab, d_model=d_model, n_frequencies=12)
    print(f"Paramètres entraînables : {sum(p.numel() for p in emb.parameters())}")

    # Cible indépendante de l'espace d'entrée : sinus à des fréquences qui
    # ne sont PAS dans la base (φ²)^{-k}. Les fréquences de la base sont
    # décroissantes à partir de 1.0 ; on choisit des fréquences positives
    # variées (0.7, 1.3, 2.1) qui demandent une approximation non-triviale.
    ids = torch.arange(vocab, dtype=torch.float32).unsqueeze(1)  # (V, 1)
    freqs_cible = torch.linspace(0.3, 2.1, d_model)  # (d_model,)
    phases = freqs_cible * ids  # (V, d_model)
    # Cible = mélange non-linéaire (sin + cos à phases différentes) pour
    # forcer une régression réelle, pas une simple copie.
    target = torch.sin(phases) * 0.6 + torch.cos(phases * 1.7 + 0.3) * 0.4

    opt = torch.optim.Adam(emb.parameters(), lr=3e-3)
    # Cosine schedule : aide la convergence fine en fin d'entraînement.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)

    initial_loss = None
    for step in range(600):
        opt.zero_grad()
        token_ids = torch.arange(vocab)
        out = emb(token_ids)
        loss = ((out - target) ** 2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 599:
            print(f"step {step:3d}  loss = {loss.item():.6f}  lr = {sched.get_last_lr()[0]:.5f}")

    final_loss = loss.item()
    print()
    print(f"Loss initiale : {initial_loss:.6f}")
    print(f"Loss finale   : {final_loss:.6f}")
    print(f"Baisse        : {(1 - final_loss / initial_loss) * 100:.1f}%")

    # Critère honnête : la cible n'est PAS trivialement dans l'espace d'entrée,
    # donc une baisse de ÷3 (67%) prouve déjà que l'autodiff traverse le
    # pipeline (char + Fourier + vortex → MLP → proj). On ne vise pas ÷10000.
    if final_loss < initial_loss / 3.0:
        print("\n✓ SUCCÈS : l'embedding fractal apprend (loss divisée par >3 sur "
              "une cible hors de l'espace d'entrée — preuve honnête d'autodiff).")
    elif final_loss < initial_loss / 2.0:
        print("\n✓ PARTIEL : loss divisée par >2 — l'autodiff marche mais la "
              "capacité d'approximation est limitée (acceptable pour la démo L1).")
    else:
        print("\n✗ ÉCHEC : la loss ne baisse pas assez — investiguer.")


if __name__ == "__main__":
    main()


