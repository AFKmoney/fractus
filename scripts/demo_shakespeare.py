"""Démo Scaling : transformer fractal sur tinyshakespeare (entraînement court).

Entraîne TinyFractalLM sur un SOUS-ENSEMBLE de tinyshakespeare (200 batches),
mesure la perplexité honnête, et génère du texte.

HONNÊTETÉ SUR LA LIMITE CPU : l'entraînement complet (1 epoch = ~900 batches)
prend ~10 min sur le Ryzen 5 à cause de Kuramoto RK4 (4 sous-steps) et du
masque triangulaire de l'attention. Cette démo fait un entraînement COURT
(200 batches, ~3 min) qui prouve que le modèle APPREND sur du vrai texte,
mais sans convergence complète. Pour un entraînement complet : GPU ou
vectorisation approfondie du Kuramoto (future work).

Setup (CPU-only) :
    vocab = 65 (tinyshakespeare)
    d_model = 48
    n_blocks = 2
    seq_len = 32
    ~40k params

Run :
    python scripts/demo_shakespeare.py
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.nn import FractalEmbedding, FractalBlockFull
from fractus.metrics.perplexity import honest_perplexity
from data.text.tinyshakespeare import TinyShakespeareDataset


class ShakespeareFractalLM(nn.Module):
    def __init__(self, vocab, d_model, n_blocks):
        super().__init__()
        self.embed = FractalEmbedding(vocab, d_model, n_frequencies=12)
        self.blocks = nn.ModuleList([
            FractalBlockFull(
                d_model=d_model, n_heads=4, d_head=d_model // 4, n_levels=2,
                n_oscillators=8, coupling_rank=4,
                n_experts=4, top_k=2, kappa=4.0,
            )
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        aux = torch.tensor(0.0, device=x.device)
        for b in self.blocks:
            x, lb = b(x)
            aux = aux + lb
        x = self.norm(x)
        return self.head(x), aux


def main():
    torch.manual_seed(42)
    seq_len = 32
    dataset = TinyShakespeareDataset(seq_len=seq_len)
    print(f"Dataset : {len(dataset)} séquences de {seq_len} tokens, vocab={dataset.vocab_size}")

    # Split train/val (90/10).
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = ShakespeareFractalLM(vocab=dataset.vocab_size, d_model=48, n_blocks=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle : {n_params} paramètres ({n_params/1000:.0f}k)")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Évaluer perplexité initiale.
    model.eval()
    with torch.no_grad():
        inp, tgt = next(iter(val_loader))
        initial_ppl = honest_perplexity(model, inp, tgt)
    print(f"Perplexité initiale : {initial_ppl:.2f}  (= vocab ≈ {dataset.vocab_size})")

    # Entraînement COURT : 200 batches (sous-ensemble, ~3 min sur CPU).
    n_batches = 200
    print(f"\nEntraînement court : {n_batches} batches (sous-ensemble du dataset)...")
    t0 = time.time()
    model.train()
    losses = []
    batch_iter = iter(train_loader)
    for batch_idx in range(n_batches):
        try:
            inp, tgt = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            inp, tgt = next(batch_iter)
        opt.zero_grad()
        logits, aux = model(inp)
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, dataset.vocab_size), tgt.reshape(-1)
        )
        loss = ce + 0.1 * aux
        loss.backward()
        opt.step()
        losses.append(ce.item())
        if batch_idx % 50 == 0 or batch_idx == n_batches - 1:
            avg_recent = sum(losses[-50:]) / max(len(losses[-50:]), 1)
            print(f"  batch {batch_idx:4d}/{n_batches}  ce={ce.item():.4f}  "
                  f"(moy recent={avg_recent:.4f}, ppl={torch.exp(torch.tensor(avg_recent)):.2f})")

    elapsed = time.time() - t0
    print(f"\nTemps : {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Évaluer perplexité finale sur validation.
    model.eval()
    val_ces = []
    with torch.no_grad():
        for inp, tgt in val_loader:
            logits, _ = model(inp)
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, dataset.vocab_size), tgt.reshape(-1)
            )
            val_ces.append(ce.item())
    final_ppl = torch.exp(torch.tensor(sum(val_ces) / len(val_ces))).item()
    print(f"Perplexité finale (val) : {final_ppl:.2f}")
    print(f"Baisse : {(1 - final_ppl/initial_ppl)*100:.1f}%")

    # Génération.
    print("\n=== Génération (greedy, 150 chars) ===")
    prompt = "ROMEO:\n"
    ctx = torch.tensor([[dataset.char_to_id[c] for c in prompt]])
    with torch.no_grad():
        for _ in range(150):
            if ctx.shape[1] > seq_len:
                ctx = ctx[:, -seq_len:]
            logits, _ = model(ctx)
            nxt = int(logits[0, -1].argmax().item())
            ctx = torch.cat([ctx, torch.tensor([[nxt]])], dim=1)
    generated = dataset.decode(ctx[0])
    print(generated)

    if final_ppl < initial_ppl * 0.7:
        print(f"\nOK : le transformer fractal apprend sur tinyshakespeare réel "
              f"(ppl {initial_ppl:.1f} -> {final_ppl:.1f}, ÷{initial_ppl/final_ppl:.1f}).")
        print(f"  Note : entraînement court (200 batches). Convergence complète")
        print(f"  nécessiterait GPU ou vectorisation Kuramoto (future work).")
    else:
        print(f"\n~ : ppl baisse peu. Plus de batches / modèle plus gros aiderait.")


if __name__ == "__main__":
    main()
