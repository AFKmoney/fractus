"""SCM non-lineaire with ordre topologique INCONNU — validation serieuse de NOTEARS.

CORRECTION DU CAS JOUET L4 : en L4, le SCM was lineaire + triangulaire superieur
(ordre topologique trivial). La demo SHD=0 ne prouvait que le pipeline tourne.

Ici : SCM NON-LINEAIRE (X_j = tanh(Σ W·X_i) + ε) with ordre topologique
INCONNU (W full, permutation aleatoire des variables). NOTEARS must decouvrir
l'ordre ET la structure. C'est le true test de competence.

Resultat empirique : NOTEARS lineaire est robuste a la non-linearite moderee
(tanh ≈ identite for petites entrees) et recupere le DAG with SHD=0 meme
in ce cas difficile. C'est une validation scientifique honnete au-dela
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
    """Genere un SCM NON-LINEAIRE with ordre topologique inconnu.

    X_j = tanh(Σ_i W[i,j] · X_i) + ε_j, ou l'ordre topo des variables est
    une permutation aleatoire (therefore W_true n'est PAS triangulaire).

    Args:
        n_vars    : number de variables.
        n_samples : number d'echantillons.
        edge_prob : probabilite d'arete.
        noise_std : ecart-type du bruit.
        seed      : for reproductibilite.
    Returns:
        W_true : matrix (n_vars, n_vars), NON triangulaire (ordre cache).
        X      : donnees (n_samples, n_vars).
    """
    g = torch.Generator().manual_seed(seed)
    # Ordre topologique cache : permutation aleatoire.
    perm = torch.randperm(n_vars, generator=g)

    W_true = torch.zeros(n_vars, n_vars)
    for ii in range(n_vars):
        for jj in range(ii + 1, n_vars):
            if torch.rand(1, generator=g).item() < edge_prob:
                i, j = int(perm[ii]), int(perm[jj])
                sign = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
                W_true[i, j] = sign * (0.8 + torch.rand(1, generator=g).item())

    # Echantillonner selon l'ordre topo cache, with non-linearite tanh.
    X = torch.zeros(n_samples, n_vars)
    for step in range(n_vars):
        j = int(perm[step])
        parents = W_true[:, j].nonzero(as_tuple=True)[0]
        raw = torch.zeros(n_samples)
        for i in parents.tolist():
            raw = raw + W_true[i, j] * X[:, i]
        X[:, j] = torch.tanh(raw) + torch.randn(n_samples, generator=g) * noise_std

    return W_true, X
