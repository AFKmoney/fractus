"""Generation of Structural Causal Models synthetiques.

On generated a DAG aleatoire (topological ordering guaranteed), on echantillonne
des donnees selon this DAG (each variable = function lineaire of its parents +
bruit galsoen), then on fournit the true W for evaluer NOTEARS.
"""

import torch


def generate_linear_scm(
    n_vars: int = 5,
    n_samples: int = 1000,
    edge_prob: float = 0.4,
    noise_std: float = 0.5,
    seed: int = 42,
):
    """Genere a SCM lineaire : X_j = Σ_i W[i,j] · X_i + ε_j.

    Garantit a DAG en echantillonnant W triangulaire superieur.
    """
    g = torch.Generator().manual_seed(seed)
    W_true = torch.zeros(n_vars, n_vars)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if torch.rand(1, generator=g).item() < edge_prob:
                sign = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
                W_true[i, j] = sign * (0.5 + torch.rand(1, generator=g).item())

    X = torch.zeros(n_samples, n_vars)
    for j in range(n_vars):
        parents = W_true[:, j].nonzero(as_tuple=True)[0]
        mean = torch.zeros(n_samples)
        for i in parents.tolist():
            mean = mean + W_true[i, j] * X[:, i]
        noise = torch.randn(n_samples, generator=g) * noise_std
        X[:, j] = mean + noise

    return W_true, X


if __name__ == "__main__":
    W, X = generate_linear_scm(n_vars=5, n_samples=10)
    print("W_true =")
    print(W)
    print("X (10 samples, 5 vars) =")
    print(X)
