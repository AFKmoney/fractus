"""Démo L3 : mesurer la VRAIE compression SIREN sur un modèle.

On construit deux variantes d'un mini-MLP :
    (A) 100% dense (nn.Linear)
    (B) Couches cachées en SirenLinear, dernière couche dense.

On mesure :
    - Le ratio de compression (params SIREN vs dense équivalent).
    - La capacité d'apprentissage : peut-on surfit une cible avec (B) aussi bien
      qu'avec (A) ?

POSITION SCIENTIFIQUE HONNÊTE :
On s'attend à un ratio MODESTE (~2× à 5×) et à une perte de qualité d'apprentissage
(les poids SIREN sont lisses, ce qui limite la capacité à exprimer des fonctions
arbitraires). C'est la vérité — à comparer au mensonge 20.4× d'OMNI.

Run :
    python scripts/demo_siren_compression.py
"""

import torch
import torch.nn as nn
from fractus.nn import SirenLinear
from fractus.metrics.compression import measure_compression_ratio


def make_dense_model(d_in, d_hidden, d_out):
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


def make_siren_model(d_in, d_hidden, d_out, siren_hidden=16):
    return nn.Sequential(
        SirenLinear(d_in, d_hidden, hidden=siren_hidden), nn.ReLU(),
        SirenLinear(d_hidden, d_hidden, hidden=siren_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),  # dernière couche dense
    )


def train_and_eval(model, X, Y, n_steps=300, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    initial = None
    for step in range(n_steps):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        if initial is None:
            initial = loss.item()
        loss.backward()
        opt.step()
    final = loss.item()
    return initial, final


def main():
    torch.manual_seed(42)
    d_in, d_hidden, d_out = 16, 32, 8
    n_samples = 64

    # Cible : fonction non-triviale (sinus à fréquence non alignée).
    X = torch.randn(n_samples, d_in)
    Y = torch.sin(X[:, :d_out] * 1.3) + 0.5 * torch.cos(X[:, :d_out] * 0.7)

    dense = make_dense_model(d_in, d_hidden, d_out)
    siren = make_siren_model(d_in, d_hidden, d_out, siren_hidden=16)

    n_dense = sum(p.numel() for p in dense.parameters())
    n_siren = sum(p.numel() for p in siren.parameters())
    ratio_dense = measure_compression_ratio(dense)
    ratio_siren = measure_compression_ratio(siren)

    print("=== Compression mesurée ===")
    print(f"Modèle dense  : {n_dense} params, ratio = {ratio_dense:.2f}x")
    print(f"Modèle SIREN  : {n_siren} params, ratio = {ratio_siren:.2f}x")
    print(f"Économie      : {(1 - n_siren/n_dense)*100:.1f}% de params en moins")
    print()

    print("=== Capacité d'apprentissage (surfit cible sinus) ===")
    i_d, f_d = train_and_eval(dense, X, Y)
    i_s, f_s = train_and_eval(siren, X, Y)
    print(f"Dense  : loss {i_d:.4f} -> {f_d:.4f}  (baisse {(1-f_d/i_d)*100:.1f}%)")
    print(f"SIREN  : loss {i_s:.4f} -> {f_s:.4f}  (baisse {(1-f_s/i_s)*100:.1f}%)")
    print()

    print("=== Verdict honnête ===")
    print(f"Ratio de compression réel : {ratio_siren:.2f}x")
    print(f"  (à comparer au '20.4x' hardcodé d'OMNI-FRACTAL, qui était faux)")
    print(f"Perte de qualité apprentissage : {(f_s - f_d):.4f} (SIREN - Dense)")
    if ratio_siren > 1.5 and f_s < i_s * 0.5:
        print("\nOK : la SIREN comprime (>1.5x) ET apprend — honnête et utile.")
    elif ratio_siren > 1.5:
        print("\n~ : la SIREN comprime mais apprend moins bien — trade-off à documenter.")
    else:
        print("\n~ : compression faible (<1.5x) — la SIREN n'est pas adaptée à ces poids.")
    print("\nConclusion : la thèse '20.4x sans perte' d'OMNI n'est pas reproduite.")
    print("La SIREN est utile pour des fonctions lisses, pas pour des poids denses.")


if __name__ == "__main__":
    main()
