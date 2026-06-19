"""Démo L2a : premier transformer fractal entraînable.

Assembler FractalEmbedding + N×FractalBlock + projection logit, et l'entraîner
sur une toy séquence de texte (prédiction du prochain token). C'est la première
démonstration end-to-end : le modèle apprend vraiment, la loss baisse.

On utilise un tout petit setup (CPU-only) :
    vocab  = 128 (ASCII imprimable : ord(c)-32 ∈ [0,95] ⊂ [0,128))
    d_model = 32
    n_blocks = 2
    seq_len  = 16

Corrige l'erreur centrale de FNN v5.0 (training.rs:399 = bruit) : ici Adam
reçoit de vrais gradients et la loss baisse.

Run :
    python scripts/demo_transformer.py
"""

import torch
import torch.nn as nn
from fractus.nn import FractalEmbedding, FractalBlock


class TinyFractalLM(nn.Module):
    """Embedding + blocs + projection logit (prédiction prochain token)."""

    def __init__(self, vocab, d_model, n_heads, d_head, n_levels, n_blocks):
        super().__init__()
        self.embed = FractalEmbedding(vocab, d_model, n_frequencies=8)
        self.blocks = nn.ModuleList([
            FractalBlock(d_model, n_heads, d_head, n_levels)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)  # (B, L, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)  # (B, L, vocab) logits


def main():
    torch.manual_seed(42)

    # Toy "texte" : une séquence répétitive que le modèle peut apprendre.
    # vocab=128 couvre tout l'ASCII imprimable (ord-32 ∈ [0,95] ⊂ [0,128)).
    text = "hello world " * 8
    vocab = 128  # ASCII 32..159 (couvre toutes les lettres minuscules)
    ids = torch.tensor([ord(c) - 32 for c in text if 0 <= ord(c) - 32 < vocab])
    print(f"Séquence : {len(ids)} tokens, vocab={vocab}")
    print(f"Extrait : {''.join(chr(int(i)+32) for i in ids[:24])!r}")

    # Découper en batchs de séquences.
    seq_len = 16
    n_seqs = len(ids) // seq_len
    ids = ids[:n_seqs * seq_len].view(n_seqs, seq_len)
    print(f"Batchs : {n_seqs} séquences de longueur {seq_len}")

    # Modèle minimal.
    model = TinyFractalLM(
        vocab=vocab, d_model=32, n_heads=4, d_head=8, n_levels=2, n_blocks=2
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres : {n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Cible : prédire le token SUIVANT (décalage de 1).
    initial_loss = None
    for epoch in range(40):
        opt.zero_grad()
        logits = model(ids)  # (n_seqs, seq_len, vocab)
        # Shift : prédire token t+1 à partir de token t.
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab),
            ids[:, 1:].reshape(-1),
        )
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        if epoch % 8 == 0 or epoch == 39:
            print(f"epoch {epoch:2d}  loss = {loss.item():.4f}")

    final_loss = loss.item()
    print()
    print(f"Loss initiale : {initial_loss:.4f}  (= log({vocab}) ≈ {torch.log(torch.tensor(float(vocab))).item():.3f})")
    print(f"Loss finale   : {final_loss:.4f}")
    print(f"Baisse        : {(1 - final_loss / initial_loss) * 100:.1f}%")

    # Générer un peu de texte pour visualiser.
    print()
    print("Génération (greedy) :")
    model.eval()
    with torch.no_grad():
        # Prompt initial : 'hello' — maintenant valide car vocab=128 couvre les
        # minuscules (ord('h')-32=72 < 128).
        context = torch.tensor([[ord(c) - 32 for c in "hello"]])
        for _ in range(20):
            logits = model(context)
            next_id = logits[0, -1].argmax().item()
            # Clamp défensif : borne dans [0, vocab).
            next_id = max(0, min(vocab - 1, next_id))
            context = torch.cat([context, torch.tensor([[next_id]])], dim=1)
        generated = "".join(chr(int(i) + 32) for i in context[0].tolist())
    print(f"  '{generated}'")

    if final_loss < initial_loss * 0.5:
        print("\n✓ SUCCÈS : le transformer fractal apprend (loss divisée par >2).")
    else:
        print("\n✗ ÉCHEC : loss ne baisse pas assez.")


if __name__ == "__main__":
    main()
