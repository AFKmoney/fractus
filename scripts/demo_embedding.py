"""Demo L1 : prouve that FractalEmbedding apprend vraiment.

Objectif : surfit a target a frequences NON presentes in the base Fourier
interne of l'embedding. Cela force l'embedding a reallement combiner its trois
sources (char features + Fourier + vortex conditioning) via sa projection
entrainable, instead of simplement recopier a combinaison of its colonnes
d'entree (ce qu'une Linear does trivialement).

Les frequences of the base interne are ω_k = (φ2)^{-k} for k=0..n_freq-1,
soit decroissantes a partir of 1.0. On choisit therefore frequences targets
independantes : croissantes (1.3, 0.7) which not are PAS in the base.

This is the proof minimale honestete that l'autodiff traverse l'embedding fractal
(ce that the original architecture not savait not faire — training.rs:399 utilisait bruit).

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
    print(f"Parametres entrainables : {sum(p.numel() for p in emb.parameters())}")

    # Cible independante of l'espace d'entree : sinus a frequences qui
    # not are PAS in the base (φ2)^{-k}. Les frequences of the base sont
    # decroissantes a partir of 1.0 ; on choisit frequences positives
    # variees (0.7, 1.3, 2.1) which demandent a approximation non-triviale.
    ids = torch.arange(vocab, dtype=torch.float32).unsqueeze(1)  # (V, 1)
    freqs_cible = torch.linspace(0.3, 2.1, d_model)  # (d_model,)
    phases = freqs_cible * ids  # (V, d_model)
    # Cible = melange non-lineaire (sin + cos a phases differentes) for
    # forcer a regression real, not a simple copie.
    target = torch.sin(phases) * 0.6 + torch.cos(phases * 1.7 + 0.3) * 0.4

    opt = torch.optim.Adam(emb.parameters(), lr=3e-3)
    # Cosine schedule : aide the convergesnce fine en fin d'training.
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
    print(f"Loss initial : {initial_loss:.6f}")
    print(f"Loss finale   : {final_loss:.6f}")
    print(f"Baisse        : {(1 - final_loss / initial_loss) * 100:.1f}%")

    # Critere honestete : the target n'est PAS trivialement in l'espace d'entree,
    # therefore a baisse of ÷3 (67%) prouve already that l'autodiff traverse le
    # pipeline (char + Fourier + vortex → MLP → proj). On not vise not ÷10000.
    if final_loss < initial_loss / 3.0:
        print("\n✓ SUCCES : l'embedding fractal apprend (loss divisee par >3 sur "
              "une cible hors de l'espace d'entree — preuve honnete d'autodiff).")
    elif final_loss < initial_loss / 2.0:
        print("\n✓ PARTIEL : loss divisee par >2 — l'autodiff marche but la "
              "capacite d'approximation est limitee (acceptable for la demo L1).")
    else:
        print("\n✗ ECHEC : la loss ne baisse pas assez — investiguer.")


if __name__ == "__main__":
    main()


