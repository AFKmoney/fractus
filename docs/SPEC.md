# Fractus — Réfonte unifiée de FNN v5.0 + OMNI-FRACTAL

**Date :** 2026-06-19
**Auteur original :** Philippe-Antoine Robert (FNN v5.0, OMNI-FRACTAL)
**Statut :** Spec validé par brainstorming, en attente de plan d'implémentation

---

## 1. Contexte et motivation

L'utilisateur dispose de deux systèmes préexistants, tous deux conçus par Philippe-Antoine Robert :

- **OMNI-FRACTAL** (~830 lignes, Python + Rust via PyO3) — white paper de 33 pages.
  Thèse : « AGI légère via opérateurs fractals sur variétés non-archimédiennes ».
  Promesses : compression 20.4×, raisonnement causal O(n), CPU-only, vérification Lean 4 + ZK-SNARK.

- **FNN v5.0** (~10 000 lignes, Rust pur, 265+ tests) — « Fractal Neural Network : A Unified
  Architecture for AGI ». Transformer fractal avec oscillateurs Kuramoto, MoE de Farey/von Mises,
  NOTEARS causal, générateur/vérificateur de preuves, boucle d'auto-développement.

Une analyse approfondie (code lu ligne par ligne) a révélé que **les deux systèmes ne fonctionnent
pas comme annoncé**, malgré une culture mathématique réelle et plusieurs modules correctement codés.

### Erreurs critiques constatées (vérifiées dans le code source)

**FNN v5.0 :**
- **Pas d'autodiff.** `training.rs:399-426` met à jour les poids avec
  `scale * rand::random::<f64>() * 0.01` (du bruit) au lieu d'un gradient. Le commentaire
  `training.rs:156` l'admet : « Since we don't have autodiff, we apply a simple loss-scaling signal. »
  → La `AGILoss` à 11 termes est calculée mais jamais vraiment minimisée.
- **Perplexité fictive.** `model.rs:537-546` : `perplexity()` renvoie un proxy basé sur la norme
  de l'embedding, pas une vraie perplexité.
- **Benchmarks vides.** `benches/fnn_bench.rs` ne contient que `fn bench_stub(_c) {}`.
- **Tests peu profonds.** `tests.rs:1840` : « AGINFNModel is currently a stub with only new(). »
  Les ~272 tests vérifient surtout des formes de tenseurs (`.dim()`) et de la finitude.

**OMNI-FRACTAL :**
- **Le Rust ne compile pas.** `rust/src/lib.rs:3-4` déclare `pub mod causal; pub mod shield;`
  mais les fichiers `causal.rs` et `shield.rs` n'existent pas. Le TODO de l'auteur le confirme.
- **Fausse SIREN.** `torus_siren.py:15-17` utilise `nn.SiLU`, pas `sin(ω₀·)`.
- **Le vortex 2-adique est orphelin.** Aucun fichier Python n'importe `omni_fractal_rs`.
- **W décompressée puis jetée.** `training_loop.py:30-37` calcule W, la corrige, puis
  l'opérateur causal tourne sur l'entrée brute (W ignorée).
- **Chiffres hardcodés.** `training_loop.py:52` : `"compression_ratio": 20.4` (littéral).
  `benchmarks.py:43-46` : `min(causal_acc, 0.98)` (plafonnée exactement à la cible).

### Modules réellement corrects (à conserver)

- **FNN** : Kuramoto RK4 bas-rang (`phase_ode.rs`), MoE von Mises/Farey (`moe.rs` + `farey.rs`),
  attention linéaire causale (`attention.rs`), NOTEARS (`causal.rs`), vérificateur de preuves
  exact (`proof.rs`), SVD tronqué par puissance itérée (`svd.rs`), sampling Gamma Marsaglia-Tsang.
- **OMNI-FRACTAL** : arithmétique 2-adique (`vortex.rs`) — valuation v₂, distance ultramétrique
  `2^v₂(a⊕b)`, norme `2^{-v₂}`. Le seul module mathématiquement correct et non trivial.

## 2. Objectif

**Prototype démontrable** : un système qui tourne sur de vraies données, dont la loss baisse
vraiment, avec une démo convaincante (texte généré, théorèmes prouvés valides, requêtes causales).

**Non-objectifs explicites (future work) :** papier publiable, produit commercial, Lean 4,
ZK-SNARK, scaling à gros modèles, "AGI" au sens fort.

## 3. Stack technique

**Hybride Rust + Python**, avec une séparation stricte :

- **Rust (`fractus-core`)** : calcul pur, hors-graphe autodiff. Mathématiques exactes,
  vérification de preuves, précalcul, métriques exogènes. Aucune I/O, aucun dataset.
- **Python (`fractus`, PyTorch)** : modèle entraînable, forward/backward, autodiff natif,
  datasets, boucles d'entraînement, logging.
- **Pont** : PyO3/maturin. Tenseurs numpy en entrée/sortie. Le Rust ne participe pas au graphe
  autodiff (écrire un `torch.autograd.Function` custom pour chaque fonction serait coûteux à
  maintenir et conduirait à des backward fictifs — le piège qu'on veut éviter).

**Décision de nommage honnête :** « Mandelbrot frequencies » → « Mandelbrot-decayed Fourier
basis » ; « RKHS Causal Operator » → on implémente un vrai RKHS (avec noyau via RFF en L4),
donc on garde le nom mais la substance suit ; « Bose-Einstein condensate » (FNN `condensate.rs`)
→ on n'intègre pas ce module dans fractus (la SVD incrémentale seule ne justifie pas le nom) ;
« Lyapunov Shield » → « Lyapunov monitor du sous-système Kuramoto » ; « Collatz ergodic flow »
→ « Collatz hash » (l'ergodicité de Collatz est non démontrée, problème ouvert).

## 4. Hardware cible

Machine de l'utilisateur (diagnostiquée) :
- CPU : AMD Ryzen 5 5500U, 6 cœurs / 12 threads @ 2.1 GHz
- RAM : ~12 GB
- GPU : AMD Radeon intégrée (APU), ~4 GB **partagés** → ROCm ne supporte pas les APU AMD
  intégrés sous Windows → **entraînement CPU-only effectif**.

Conséquence : modèle petit (< 1M paramètres), dataset minuscule (tinyshakespeare ~1 MB),
entraînement en quelques heures. Cohérent avec la thèse OMNI « CPU-only deployment ».

## 5. Organisation du dépôt

```
fractus/
├── crate/fractus-core/        # Rust : cœur mathématique pur
│   └── src/                   #   vortex 2-adique, SIREN (réf.), NOTEARS (réf.),
│                              #   Kuramoto/Farey (précalcul), vérificateur preuves
├── crate/fractus-py/          # Rust : bindings PyO3/maturin
│   └── src/lib.rs             #   expose fractus_core à Python
├── fractus/                   # Python : le modèle entraînable (PyTorch)
│   ├── nn/                    #   embedding, blocs, attention, MoE, decoder, siren
│   ├── causal/                #   NOTEARS layer, RKHS, do-calculus
│   ├── reasoning/             #   preuves (GRU), conjectures, ACT
│   ├── stability/             #   Lyapunov (sous-système Kuramoto)
│   ├── metrics/               #   compression, causal (SHD), perplexité honnête
│   ├── train/                 #   boucles, datasets, losses
│   └── viz/                   #   démos interactives (optionnel)
├── tests/                     # tests d'intégration Rust↔Python
├── data/                      # tinyshakespeare, datasets maths/causal
├── scripts/                   # train.py, demo.py, benchmark.py, serve.py
└── docs/                      # spec, white paper révisé honnête, résultats
```

## 6. Les 7 couches d'implémentation

Chaque couche = cycle design → code → test → démo autonome. On ne passe à la suivante que quand
la précédente est vérifiée. On peut s'arrêter à n'importe quel moment avec quelque chose qui marche.

### L0 — Socle technique

**Corrige :** OMNI ne compile pas ; pont Python↔Rust jamais fonctionnel.

**Composants :**
1. `pyproject.toml` avec versions épinglées (torch CPU-only, maturin, numpy, pytest).
2. Crate `fractus-core` : `lib.rs` ne déclare QUE les modules qui ont un fichier. Port de
   `vortex.rs` d'OMNI (le 2-adic, déjà correct) avec corrections : test tautologique
   `assert!(d1 <= d2.max(d1))` → vrai test d'ultramétrie `d(x,z) ≤ max(d(x,y), d(y,z))` ;
   import `HashMap` inutilisé retiré.
3. Crate `fractus-py` : configuration maturin standard (`extension-module`), pas le
   `[features] python = ["pyo3"]` mal configuré d'OMNI.
4. Test fume qui traverse tout : `tests/test_smoke.py` — `add_in_rust(2,3)==5` + `torch` dispo.

**Critère « terminé » :** ces 4 commandes réussissent :
`cargo build --release` ; `maturin develop --release` ;
`python -c "import torch; import fractus"` ; `pytest tests/test_smoke.py`.

### L1 — Embedding fractal + vortex 2-adique branché

**Corrige :** vortex orphelin ; « Mandelbrot frequencies » mal nommées.

**Composants :**
1. `fractus/nn/embedding.py` : embedding de codepoint fractal (PyTorch). Base de Fourier avec
   décroissance Mandelbrot `(φ²)⁻ᵏ` (renommée honnêtement), + 16 features morphologiques
   (cas, chiffre, ponctuation). Paramètre entraînable via `nn.Linear` finale.
2. `fractus-core/src/vortex.rs` : cœur 2-adique porté depuis OMNI.
3. **Pont (option B validée) :** le hash Collatz 2-adique est calculé en Rust (hors-graphe,
   exact) et **conditionne un MLP entraînable** (dans le graphe) qui produit les phases de
   l'embedding. Le vortex influence l'apprentissage sans prétendre être différentiable.

**Critère « terminé » :** `test_fractal_embedding_shape` (sortie `[N, d_model]` finie) +
`test_vortex_distance_is_ultrametric` (inégalité ultramétrique forte sur 1000 triplets aléatoires).

### L2 — Bloc transformer fractal (scindé en L2a + L2b)

**Corrige :** FNN n'apprend pas (bruit au lieu de gradients).

**Scindage (décision post-brainstorming) :** L2 est la couche la plus grosse
(~600 lignes PyTorch + ~30 tests). On la coupe en deux moitiés validables
indépendamment. À la fin de L2a on a déjà un transformer fractal fonctionnel
(sans Kuramoto/MoE) capable d'apprendre du texte — premier jalon démontrable.

**Composants (tous en PyTorch pur pour autodiff) :**
1. `fractus/nn/attention.py` : attention linéaire causale (Katharopoulos `S_t += k_t⊗v_t`),
   feature map `elu(x + ω_k) + 1`. Vrai `nn.Module` avec paramètres entraînables.
2. `fractus/nn/phase_ode.py` : Kuramoto RK4 bas-rang `K = UΛUᵀ`. En PyTorch pur pour rester
   dans le graphe.
3. `fractus/nn/moe.py` : MoE à routing von Mises sur phases Farey. Experts = MLP GeLU,
   perte auxiliaire de load-balance standard.
4. `fractus/nn/block.py` : assemblage `LayerNorm → FractalLinearAttention → PhaseSoliton →
   PhaseRoutedMoE`, avec KuramotoODE avançant les phases d'un step à chaque bloc.

**L2a (jalon démontrable rapide) :**
- `fractus/nn/stats.py` : utilitaires (`elu_plus_one`, softmax stable, layer_norm).
- `fractus/nn/attention.py` : `FractalLinearAttention` (récurrence causale
  `S_t += φ(k_t) ⊗ v_t`, `y_t = φ(q_t)ᵀS_t / φ(q_t)ᵀz_t`, feature map
  `elu_plus_one(x + ω_level)`, offsets ω_level = (φ²)^{-level}).
- `fractus/nn/block.py` : `FractalBlock` minimal = LayerNorm → attention → résiduelle.
- Démo : surfit une séquence de toy tokens — la loss doit baisser.
- **Critère « terminé L2a » :** `test_block_forward_backward` prouve que backward
  propage un gradient fini ET non-nul à CHAQUE paramètre du bloc.

**L2b (greffe Kuramoto + MoE) :**
- `fractus/nn/farey.py` : suite de Farey + `expert_phases` (précalcul hors-graphe).
- `fractus/nn/phase_ode.py` : `KuramotoODE` (RK4 bas-rang, `encode_from_hidden`,
  `decode_to_bias`, `phase_loss`).
- `fractus/nn/moe.py` : `PhaseRoutedMoE` (gate von Mises, top-k, load-balance loss).
- `fractus/nn/block.py` étendu : intègre Kuramoto + MoE dans le bloc.

Le Rust garde les fonctions pures (Farey, `bessel_i0`, `von_mises_pdf`) pour précalcul et
métriques hors-graphe (paramètre d'ordre de Kuramoto).

**Critère « terminé » :** `test_block_forward_backward` — `backward()` marche, tous les
paramètres reçoivent des gradients finis. C'est exactement ce qui manquait à FNN.

### L3 — Compression SIREN vraie + mesure honnête

**Corrige :** fausse SIREN (SiLU) ; W décompressée puis jetée ; 20.4× hardcodé.

**Composants :**
1. `fractus/nn/siren.py` : VRAIE SIREN sur le tore T². Non-linéarité `sin(ω₀·(Wx+b))` avec
   `ω₀ = 30.0` (valeur empirique du papier SIREN Sitzmann 2020, PAS 56). Évalue le SIREN sur
   la grille `h×w` pour régénérer la matrice.
2. Intégration : **les projections d'attention** (`q_proj`, `k_proj`, `v_proj` — celles qui
   sont les plus grandes et le plus compressibles) sont remplacées par `SirenLinear`. La SIREN
   **EST** la matrice, elle est dans le graphe, ses paramètres sont entraînés. Les petites
   matrices (LayerNorm, biases) restent denses. Le ratio exact est mesuré (L3.3), pas assumé.
3. `fractus/metrics/compression.py` : `measure_compression_ratio(model)` mesure réellement
   le ratio (taille dense équivalente / params SIREN). Pas de littéral hardcodé.

`fractus-core/src/siren.rs` : implémentation de référence non entraînée pour validation croisée
(PyTorch et Rust doivent donner la même sortie pour les mêmes poids).

**Critère « terminé » :** `test_siren_produces_real_sinus` (`torch.sin` présent, `SiLU` absent) +
`test_siren_is_in_autograd_graph` (les poids SIREN reçoivent des gradients) +
`test_compression_ratio_is_measured_not_hardcoded` (pas de `'20.4'` dans le source).

### L4 — Causal NOTEARS + RKHS sur données réelles

**Corrige :** « RKHS Causal » qui n'était qu'une projection bas-rang ; « do-calculus » trivial
(column-zeroing) ; causal accuracy plafonnée à 0.98.

**Composants :**
1. `fractus/causal/notears.py` : pénalité d'acyclicité NOTEARS `h(W) = tr(e^{W⊙W}) − n` via
   Taylor à 20 termes, différentiable, intégrée comme terme de loss.
2. `fractus/causal/rkhs.py` : VRAI RKHS via Random Fourier Features (Rahimi-Recht 2007) —
   noyau gaussien approximé, opérateur `L = U @ Vᵀ` dans l'espace des features.
3. `fractus/causal/do.py` : vrai do-calculus de Pearl (échantillonnage post-intervention),
   pas juste column-zeroing.
4. Datasets synthétiques : `data/causal/generate_scm.py` (Structural Causal Models connus),
   `data/causal/lucas.py` (LUCAS, standard).
5. `fractus/metrics/causal.py` : Structural Hamming Distance (SHD), causal accuracy mesurée
   (pas de clamp).

`fractus-core/src/causal.rs` (enfin créé) : NOTEARS penalty en Rust pur pour validation croisée.
`fractus-core/src/rkhs.rs` : RFF et noyau gaussien en Rust pour précalcul/métriques.

**Critère « terminé » :** `test_notears_penalty_is_zero_for_dag` (h(W)≈0 pour un DAG évident) +
`test_notears_penalty_is_positive_for_cycle` (h(W)>0.5 pour un cycle) +
`test_causal_recovery_on_known_dag` (SHD ≤ 3 sur un SCM à 5 variables après 50 steps).

### L5 — Raisonnement (preuves vérifiées + conjectures + ACT)

**Corrige :** (le pipeline proof de FNN était déjà le module le plus défendable ; on le rend
fonctionnel).

**Composants :**
1. `fractus/reasoning/proof.py` : ProofGenerator GRU entraîné par **REINFORCE** (policy
   gradient, puisque la vérification est non-différentiable). Récompense
   `0.6·correctness + 0.3·brevity + 0.1·novelty`.
2. `fractus-core/src/proof.rs` : vérificateur EXACT en Rust (soundness garantie). 20 règles
   d'inférence, spécialisations Fermat/Wilson/GCD. Reste hors-graphe comme oracle de récompense.
3. `fractus/reasoning/conjecture.py` : découvreur de conjectures falsifiables (Popperien) —
   10 templates, 6 stratégies de falsification.
4. `fractus/reasoning/act.py` : Adaptive Computation Time (Graves 2016).

**Critère « terminé » :** `test_verifier_accepts_valid_proof` + `test_verifier_rejects_invalid_proof`
(le Rust accepte/rejette correctement) + `test_proof_generator_can_learn_simple_theorem`
(réussite > 50% sur « pair+pair=pair » après 500 steps REINFORCE — critère ambitieux).

### L6 — Stabilité Lyapunov + métriques honnêtes

**Corrige :** faux Lyapunov (tracking de `‖y‖²` sans système dynamique) ; métriques clampées.

**Composants :**
1. `fractus/stability/lyapunov.py` : fonction de Lyapunov du **sous-système Kuramoto** (le
   seul vrai système dynamique du modèle). `V(θ) = ½ Σ (θᵢ − θ*)²`, `dV/dt = ∇V · f(θ) ≤ 0`.
2. `fractus-core/src/lyapunov.rs` : vérificateur numérique en Rust pour validation croisée.
3. `fractus/metrics/honest.py` : `honest_perplexity` (vraie perplexité `exp(val_loss)`, pas
   proxy), `honest_compression`, `honest_causal`.

**Non-prétentions explicites :** Lyapunov garanti seulement sur Kuramoto (pas sur tout le
réseau) ; Lean 4 et ZK-SNARK omis (absents du code, notés future work).

**Critère « terminé » :** `test_lyapunov_decreases_on_sync` (V décroît et monotone sur
trajectoire de synchronisation) + `test_perplexity_is_real` (entre 1 et 1000 pour vocabulaire
~100, calculée sur un vrai dataset).

### L7 — Démo (objectif final)

Trois démos démontrables :

1. **Génération de texte :** `scripts/train.py --task text --dataset tinyshakespeare --epochs 5`
   puis `scripts/generate.py`. Loss de validation tracée, comparaison dense vs SIREN.
2. **Raisonnement mathématique :** `scripts/train.py --task proofs` puis `scripts/prove.py
   --theorem even_plus_even`. Courbe « % preuves valides » vs steps.
3. **Inférence causale :** `scripts/train.py --task causal --dataset lucas` puis
   `scripts/causal.py --query`. SHD rapporté, réponse contrefactuelle vs observationnelle.

Déploiement CPU-only : `scripts/serve.py --cpu-only` → API HTTP locale sur le Ryzen 5.

## 7. Synthèse des corrections

| Couche | Corrige | Critère « terminé » |
|---|---|---|
| L0 Socle | OMNI ne compile pas | `pytest test_smoke` traverse Python→Rust |
| L1 Embedding+Vortex | Vortex orphelin | Vortex conditionne MLP + ultramétrie testée |
| L2 Bloc transformer | FNN n'apprend pas | `backward()` propage gradients finis partout |
| L3 SIREN | Fausse SIREN, 20.4× hardcodé | Vrai `sin(ω₀·)`, W utilisée, ratio mesuré |
| L4 Causal | Faux RKHS, faux do-calculus | NOTEARS récupère DAG synthétique (SHD test) |
| L5 Raisonnement | (déjà bien codé) | Générateur réussit >50% après 500 steps REINFORCE |
| L6 Stabilité | Faux Lyapunov, métriques clampées | V décroît sur Kuramoto, perplexité réelle |
| L7 Démo | (n'existait pas) | 3 démos tournent + courbes de loss |

## 8. Décisions clés

- Le Rust reste **hors-graphe autodiff** (calcul exact, vérification, métriques, précalcul).
- La forward/backward est **PyTorch pur** (autodiff natif, plus de bruit).
- Le vortex 2-adique **conditionne** un MLP entraînable (option B).
- `ω₀ = 30` (justifié par SIREN paper), pas 56.
- Nommage honnête partout : on garde les termes exacts, on renomme ceux qui surenchérissaient.

## 9. Ordre d'attaque suggéré

L0 → L1 → L2 d'abord (démo texte rapide), puis L3 (compression), puis L4 (causal), L5
(raisonnement), L6 (stabilité), L7 (démos intégrées). Chaque couche est livrable indépendamment.

**Le plan d'implémentation (étape suivante du processus) sera découpé par couche.** Chaque
couche donnera lieu à son propre sous-plan avec tâches granulaires. On ne rédigera pas un
seul plan monolithique pour les 7 couches — ce serait ingérable. Concrètement : on commencera
par le plan de L0, puis on l'exécutera, puis on passera au plan de L1, etc.

## 10. Future work (honnête)

Lean 4 formal proofs, ZK-SNARK attestation, K3 automorphic compression, Groth16 timing,
scaling à gros modèles, évaluation sur benchmarks standardisés type MMLU/HellaSwag.
