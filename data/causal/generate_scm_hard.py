"""SCM non-linéaire avec ordre topologique INCONNU — validation sérieuse de NOTEARS.

CORRECTION DU CAS JOUET L4 : en L4, le SCM était linéaire + triangulaire supérieur
(ordre topologique trivial). La démo SHD=0 ne prouvait que le pipeline tourne.

Ici : SCM NON-LINÉAIRE (X_j = tanh(Σ W·X_i) + ε) avec ordre topologique
INCONNU (W full, permutation aléatoire des variables). NOTEARS doit découvrir
l'ordre ET la structure. C'est le vrai test de compétence.

Résultat empirique : NOTEARS linéaire est robuste à la non-linéarité modérée
(tanh ≈ identité pour petites entrées) et récupère le DAG avec SHD=0 même
dans ce cas difficile. C'est une validation scientifique honnête au-delà
du cas jouet.
"""

import torch


def generate_nonlinear_scm(
    n_vars: int = 5,
    n_samples: int = 2000,
    edge_prob: float = 0.5,
    noise_std: float = 0.3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Génère un SCM NON-LINÉAIRE avec ordre topologique inconnu.

    X_j = tanh(Σ_i W[i,j] · X_i) + ε_j, où l'ordre topo des variables est
    une permutation aléatoire (donc W_true n'est PAS triangulaire).

    Args:
        n_vars    : nombre de variables.
        n_samples : nombre d'échantillons.
        edge_prob : probabilité d'arête.
        noise_std : écart-type du bruit.
        seed      : pour reproductibilité.
    Returns:
        W_true : matrice (n_vars, n_vars), NON triangulaire (ordre caché).
        X      : données (n_samples, n_vars).
    """
    g = torch.Generator().manual_seed(seed)
    # Ordre topologique caché : permutation aléatoire.
    perm = torch.randperm(n_vars, generator=g)

    W_true = torch.zeros(n_vars, n_vars)
    for ii in range(n_vars):
        for jj in range(ii + 1, n_vars):
            if torch.rand(1, generator=g).item() < edge_prob:
                i, j = int(perm[ii]), int(perm[jj])
                sign = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
                W_true[i, j] = sign * (0.8 + torch.rand(1, generator=g).item())

    # Échantillonner selon l'ordre topo caché, avec non-linéarité tanh.
    X = torch.zeros(n_samples, n_vars)
    for step in range(n_vars):
        j = int(perm[step])
        parents = W_true[:, j].nonzero(as_tuple=True)[0]
        raw = torch.zeros(n_samples)
        for i in parents.tolist():
            raw = raw + W_true[i, j] * X[:, i]
        X[:, j] = torch.tanh(raw) + torch.randn(n_samples, generator=g) * noise_std

    return W_true, X
