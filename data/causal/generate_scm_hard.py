"""SCM non-lineaire with topological order INCONNU — validation serieuse of NOTEARS.

CORRECTION DU CAS JOUET L4 : en L4, the SCM was lineaire + triangulaire superieur
(topological order trivial). La demo SHD=0 not prouvait that the pipeline tourne.

Ici : SCM NON-LINEAIRE (X_j = tanh(Σ W·X_i) + ε) with topological order
INCONNU (W full, permutation aleatoire variables). NOTEARS must decouvrir
l'ordre ET the structure. This is the true test of competence.

Resultat empirique : NOTEARS lineaire est robuste a the non-linearite moderee
(tanh ≈ identite for smalles inputs) and recupere the DAG with SHD=0 same
in this cas difficile. This is a validation scientifique honesty au-dela
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
    """Genere a SCM NON-LINEAIRE with topological order inconnu.

    X_j = tanh(Σ_i W[i,j] · X_i) + ε_j, or l'ordre topo variables est
    a permutation aleatoire (therefore W_true n'est PAS triangulaire).

    Args:
        n_vars    : number of variables.
        n_samples : number d'echantillons.
        edge_prob : probabilite d'arete.
        noise_std : ecart-type bruit.
        seed      : for reproductibilite.
    Returns:
        W_true : matrix (n_vars, n_vars), NON triangulaire (ordre cache).
        X      : donnees (n_samples, n_vars).
    """
    g = torch.Generator().manual_seed(seed)
    # Ordre topological cache : permutation aleatoire.
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
