"""Démo L1 : prouve que FractalEmbedding apprend vraiment.

Objectif : surfit une cible structurée (lisse en fonction du token id) en
quelques steps Adam. On n'utilise PAS une cible purement aléatoire : un embedding
de 3456 paramètres ne peut pas reproduire 2048 valeurs de bruit indépendant
(ce serait marginalement impossible). On choisit donc une cible qui exploite
la structure de l'embedding (caractères + Fourier + vortex) : une combinaison
lisse de sinus/cosinus des token ids.

C'est la preuve minimale que l'autodiff traverse l'embedding fractal (ce que
FNN v5.0 ne savait pas faire).

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

    # Cible structurée ET exprimable : combinaison lisse de sinus/cosinus
    # aux fréquences de la base Fourier elle-même (ω_k = (φ²)^{-k}).
    # Comme l'embedding contient exactement ces fréquences, surfit parfait
    # est théoriquement atteignable. On mélange deux fréquences basses pour
    # avoir une cible non-triviale mais dans la base.
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=vocab, n_frequencies=d_model)
    M = basis.matrix()  # (vocab, 2*d_model) : colonnes sin/cos
    # Cible = moitié sinus (colonnes paires) - moitié cosinus (colonnes impaires).
    target = M[:, 0::2][:, :d_model] * 0.7 - M[:, 1::2][:, :d_model] * 0.3  # (V, d_model)

    opt = torch.optim.Adam(emb.parameters(), lr=1e-2)

    initial_loss = None
    for step in range(300):
        opt.zero_grad()
        token_ids = torch.arange(vocab)
        out = emb(token_ids)
        loss = ((out - target) ** 2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 299:
            print(f"step {step:3d}  loss = {loss.item():.6f}")

    final_loss = loss.item()
    print()
    print(f"Loss initiale : {initial_loss:.6f}")
    print(f"Loss finale   : {final_loss:.6f}")
    print(f"Baisse        : {(1 - final_loss / initial_loss) * 100:.1f}%")

    if final_loss < initial_loss * 0.1:
        print("\n✓ SUCCÈS : l'embedding fractal apprend (loss divisée par >10).")
    elif final_loss < initial_loss * 0.5:
        print("\n✓ PARTIEL : la loss baisse de >50% — l'autodiff marche, "
              "mais la convergence n'est pas parfaite (acceptable pour la démo).")
    else:
        print("\n✗ ÉCHEC : la loss ne baisse pas assez — investiguer.")


if __name__ == "__main__":
    main()

