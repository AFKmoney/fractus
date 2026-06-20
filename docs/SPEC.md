# Fractus — Refonte unifiee de the original architecture + the original design

**Date :** 2026-06-19
**Auteur original :** the original author (the original architecture, the original design)
**Statut :** Spec valide par brainstorming, en attente de plan d'implementation

---

## 1. Contexte et motivation

L'utilisateur dispose de deux systems preexistants, all deux concus par the original author :

- **the original design** (~830 lignes, Python + Rust via PyO3) — white paper de 33 pages.
  These : « AGI legere via operateurs fractals sur varietes non-archimediennes ».
  Promesses : compression 20.4×, raisonnement causal O(n), CPU-only, verification Lean 4 + ZK-SNARK.

- **the original architecture** (~10 000 lignes, Rust pur, 265+ tests) — « Fractal Neural Network : A Unified
  Architecture for AGI ». Transformer fractal with oscillateurs Kuramoto, MoE de Farey/von Mises,
  NOTEARS causal, generateur/verify de preuves, boucle d'auto-developpement.

Une analyse approfondie (code lu ligne par ligne) a revele que **les deux systems ne fonctionnent
pas comme annonce**, malgre une culture mathematical real et plusieurs modules correctement codes.

### Erreurs critiques constatees (verifiees in le code source)

**the original architecture :**
- **Pas d'autodiff.** `training.rs:399-426` met a jour les poids with
  `scale * rand::random::<f64>() * 0.01` (du bruit) au lieu d'un gradient. Le commentaire
  `training.rs:156` l'admet : « Since we don't have autodiff, we apply a simple loss-scaling signal. »
  → La `AGILoss` a 11 termes est calculee but jamais vraiment minimisee.
- **Perplexite fictive.** `model.rs:537-546` : `perplexity()` renvoie un proxy base sur la norme
  de l'embedding, pas une vraie perplexite.
- **Benchmarks vides.** `benches/fnn_bench.rs` ne contient que `fn bench_stub(_c) {}`.
- **Tests peu profonds.** `tests.rs:1840` : « AGINFNModel is currently a stub with only new(). »
  Les ~272 tests verifient surtout des formes de tenseurs (`.dim()`) et de la finitude.

**the original design :**
- **Le Rust ne compile pas.** `rust/src/lib.rs:3-4` declare `pub mod causal; pub mod shield;`
  but les fichiers `causal.rs` et `shield.rs` n'existent pas. Le TODO de l'auteur le confirme.
- **Fausse SIREN.** `torus_siren.py:15-17` utilise `nn.SiLU`, pas `sin(ω₀·)`.
- **Le vortex 2-adique est orphaned.** Aucun fichier Python n'importe `omni_fractal_rs`.
- **W decompressee then jetee.** `training_loop.py:30-37` calcule W, la corrige, then
  l'operateur causal tourne sur l'entree brute (W ignoree).
- **Chiffres hardcodes.** `training_loop.py:52` : `"compression_ratio": 20.4` (litteral).
  `benchmarks.py:43-46` : `min(causal_acc, 0.98)` (plafonnee exactement a la cible).

### Modules reellement corrects (a conserver)

- **FNN** : Kuramoto RK4 bas-rang (`phase_ode.rs`), MoE von Mises/Farey (`moe.rs` + `farey.rs`),
  attention lineaire causale (`attention.rs`), NOTEARS (`causal.rs`), verify de preuves
  exact (`proof.rs`), SVD tronque par puissance iteree (`svd.rs`), sampling Gamma Marsaglia-Tsang.
- **the original design** : arithmetique 2-adique (`vortex.rs`) — valuation v₂, distance ultrametrique
  `2^v₂(a⊕b)`, norme `2^{-v₂}`. Le seul module mathematiquement correct et non trivial.

## 2. Objectif

**Prototype demontrable** : un system qui tourne sur de vraies donnees, dont la loss baisse
vraiment, with une demo convaincante (texte generated, theorems prouves valides, requetes causales).

**Non-objectifs explicites (future work) :** papier publiable, produit commercial, Lean 4,
ZK-SNARK, scaling a gros modeles, "AGI" au sens fort.

## 3. Stack technique

**Hybride Rust + Python**, with une separation stricte :

- **Rust (`fractus-core`)** : computation pur, hors-graphe autodiff. Mathematiques exactes,
  verification de preuves, precalcul, metriques exogenes. Aucune I/O, no dataset.
- **Python (`fractus`, PyTorch)** : modele entrainable, forward/backward, autodiff natif,
  datasets, boucles d'entrainement, logging.
- **Pont** : PyO3/maturin. Tenseurs numpy en entree/sortie. Le Rust ne participe pas au graphe
  autodiff (ecrire un `torch.autograd.Function` custom for each function serait couteux a
  maintenir et conduirait a des backward fictifs — le piege qu'on veut eviter).

**Decision de nommage honnete :** « Mandelbrot frequencies » → « Mandelbrot-decayed Fourier
basis » ; « RKHS Causal Operator » → on implemente un true RKHS (with noyau via RFF en L4),
therefore on garde le nom but la substance suit ; « Bose-Einstein condensate » (FNN `condensate.rs`)
→ on n'integre pas ce module in fractus (la SVD incrementale seule ne justifie pas le nom) ;
« Lyapunov Shield » → « Lyapunov monitor du under-system Kuramoto » ; « Collatz ergodic flow »
→ « Collatz hash » (l'ergodicite de Collatz est non demontree, problem ouvert).

## 4. Hardware cible

Machine de l'utilisateur (diagnostiquee) :
- CPU : AMD Ryzen 5 5500U, 6 coeurs / 12 threads @ 2.1 GHz
- RAM : ~12 GB
- GPU : AMD Radeon integree (APU), ~4 GB **partages** → ROCm ne supporte pas les APU AMD
  integres under Windows → **entrainement CPU-only effectif**.

Consequence : modele petit (< 1M parameters), dataset minuscule (tinyshakespeare ~1 MB),
entrainement en quelques heures. Coherent with la these OMNI « CPU-only deployment ».

## 5. Organisation du depot

```
fractus/
├── crate/fractus-core/        # Rust : coeur mathematical pur
│   └── src/                   #   vortex 2-adique, SIREN (ref.), NOTEARS (ref.),
│                              #   Kuramoto/Farey (precalcul), verify preuves
├── crate/fractus-py/          # Rust : bindings PyO3/maturin
│   └── src/lib.rs             #   expose fractus_core a Python
├── fractus/                   # Python : le modele entrainable (PyTorch)
│   ├── nn/                    #   embedding, blocs, attention, MoE, decoder, siren
│   ├── causal/                #   NOTEARS layer, RKHS, do-calculus
│   ├── reasoning/             #   preuves (GRU), conjectures, ACT
│   ├── stability/             #   Lyapunov (under-system Kuramoto)
│   ├── metrics/               #   compression, causal (SHD), perplexite honnete
│   ├── train/                 #   boucles, datasets, losses
│   └── viz/                   #   demos interactives (optionnel)
├── tests/                     # tests d'integration Rust↔Python
├── data/                      # tinyshakespeare, datasets maths/causal
├── scripts/                   # train.py, demo.py, benchmark.py, serve.py
└── docs/                      # spec, white paper revise honnete, results
```

## 6. Les 7 couches d'implementation

Chaque couche = cycle design → code → test → demo autonome. On ne passe a la suivante que quand
la precedente est verifiee. On can s'arreter a n'importe quel moment with quelque chose qui marche.

### L0 — Socle technique

**Corrige :** OMNI ne compile pas ; pont Python↔Rust jamais fonctionnel.

**Composants :**
1. `pyproject.toml` with versions epinglees (torch CPU-only, maturin, numpy, pytest).
2. Crate `fractus-core` : `lib.rs` ne declare QUE les modules qui ont un fichier. Port de
   `vortex.rs` d'OMNI (le 2-adic, deja correct) with corrections : test tautologique
   `assert!(d1 <= d2.max(d1))` → true test d'ultrametrie `d(x,z) ≤ max(d(x,y), d(y,z))` ;
   import `HashMap` inutilise retire.
3. Crate `fractus-py` : configuration maturin standard (`extension-module`), pas le
   `[features] python = ["pyo3"]` mal configure d'OMNI.
4. Test fume qui traverse tout : `tests/test_smoke.py` — `add_in_rust(2,3)==5` + `torch` dispo.

**Critere « termine » :** ces 4 commandes reussissent :
`cargo build --release` ; `maturin develop --release` ;
`python -c "import torch; import fractus"` ; `pytest tests/test_smoke.py`.

### L1 — Embedding fractal + vortex 2-adique branche

**Corrige :** vortex orphaned ; « Mandelbrot frequencies » mal nommees.

**Composants :**
1. `fractus/nn/embedding.py` : embedding de codepoint fractal (PyTorch). Base de Fourier with
   decroissance Mandelbrot `(φ²)⁻ᵏ` (renommee honnetement), + 16 features morphologiques
   (cas, chiffre, ponctuation). Parametre entrainable via `nn.Linear` finale.
2. `fractus-core/src/vortex.rs` : coeur 2-adique ported depuis OMNI.
3. **Pont (option B validee) :** le hash Collatz 2-adique est calcule en Rust (hors-graphe,
   exact) et **conditionne un MLP entrainable** (in le graphe) qui produit les phases de
   l'embedding. Le vortex influence l'apprentissage without pretendre etre differentiable.

**Critere « termine » :** `test_fractal_embedding_shape` (sortie `[N, d_model]` finie) +
`test_vortex_distance_is_ultrametric` (inegalite ultrametrique forte sur 1000 triplets aleatoires).

### L2 — Bloc transformer fractal (scinde en L2a + L2b)

**Corrige :** FNN n'apprend pas (bruit au lieu de gradients).

**Scindage (decision post-brainstorming) :** L2 est la couche la plus grosse
(~600 lignes PyTorch + ~30 tests). On la coupe en deux moities validables
independamment. A la fin de L2a on a deja un transformer fractal fonctionnel
(without Kuramoto/MoE) capable d'apprendre du texte — premier jalon demontrable.

**Composants (all en PyTorch pur for autodiff) :**
1. `fractus/nn/attention.py` : attention lineaire causale (Katharopoulos `S_t += k_t⊗v_t`),
   feature map `elu(x + ω_k) + 1`. Vrai `nn.Module` with parameters entrainables.
2. `fractus/nn/phase_ode.py` : Kuramoto RK4 bas-rang `K = UΛUᵀ`. En PyTorch pur for rester
   in le graphe.
3. `fractus/nn/moe.py` : MoE a routing von Mises sur phases Farey. Experts = MLP GeLU,
   loss auxiliaire de load-balance standard.
4. `fractus/nn/block.py` : assemblage `LayerNorm → FractalLinearAttention → PhaseSoliton →
   PhaseRoutedMoE`, with KuramotoODE avancant les phases d'un step a each bloc.

**L2a (jalon demontrable rapide) :**
- `fractus/nn/stats.py` : utilitaires (`elu_plus_one`, softmax stable, layer_norm).
- `fractus/nn/attention.py` : `FractalLinearAttention` (recurrence causale
  `S_t += φ(k_t) ⊗ v_t`, `y_t = φ(q_t)ᵀS_t / φ(q_t)ᵀz_t`, feature map
  `elu_plus_one(x + ω_level)`, offsets ω_level = (φ²)^{-level}).
- `fractus/nn/block.py` : `FractalBlock` minimal = LayerNorm → attention → residuelle.
- Demo : surfit une sequence de toy tokens — la loss must baisser.
- **Critere « termine L2a » :** `test_block_forward_backward` prouve que backward
  propage un gradient fini ET non-nul a CHAQUE parameter du bloc.

**L2b (greffe Kuramoto + MoE) :**
- `fractus/nn/farey.py` : suite de Farey + `expert_phases` (precalcul hors-graphe).
- `fractus/nn/phase_ode.py` : `KuramotoODE` (RK4 bas-rang, `encode_from_hidden`,
  `decode_to_bias`, `phase_loss`).
- `fractus/nn/moe.py` : `PhaseRoutedMoE` (gate von Mises, top-k, load-balance loss).
- `fractus/nn/block.py` etendu : integre Kuramoto + MoE in le bloc.

Le Rust garde les functions pures (Farey, `bessel_i0`, `von_mises_pdf`) for precalcul et
metriques hors-graphe (parameter d'ordre de Kuramoto).

**Critere « termine » :** `test_block_forward_backward` — `backward()` marche, all les
parameters recoivent des gradients finis. C'est exactement ce qui manquait a FNN.

### L3 — Compression SIREN vraie + mesure honnete

**Corrige :** fausse SIREN (SiLU) ; W decompressee then jetee ; 20.4× hardcode.

**Composants :**
1. `fractus/nn/siren.py` : VRAIE SIREN sur le tore T². Non-linearite `sin(ω₀·(Wx+b))` with
   `ω₀ = 30.0` (valeur empirique du papier SIREN Sitzmann 2020, PAS 56). Evalue le SIREN sur
   la grille `h×w` for regenerer la matrix.
2. Integration : **les projections d'attention** (`q_proj`, `k_proj`, `v_proj` — celles qui
   sont les plus grandes et le plus compressibles) sont remplacees par `SirenLinear`. La SIREN
   **EST** la matrix, elle est in le graphe, ses parameters sont entraines. Les petites
   matrices (LayerNorm, biases) restent denses. Le ratio exact est mesure (L3.3), pas assume.
3. `fractus/metrics/compression.py` : `measure_compression_ratio(model)` mesure reellement
   le ratio (taille dense equivalente / params SIREN). Pas de litteral hardcode.

`fractus-core/src/siren.rs` : implementation de reference non entrainee for validation croisee
(PyTorch et Rust must donner la meme sortie for les memes poids).

**Critere « termine » :** `test_siren_produces_real_sinus` (`torch.sin` present, `SiLU` absent) +
`test_siren_is_in_autograd_graph` (les poids SIREN recoivent des gradients) +
`test_compression_ratio_is_measured_not_hardcoded` (pas de `'20.4'` in le source).

### L4 — Causal NOTEARS + RKHS sur donnees reelles

**Corrige :** « RKHS Causal » qui n'was qu'une projection bas-rang ; « do-calculus » trivial
(column-zeroing) ; causal accuracy plafonnee a 0.98.

**Composants :**
1. `fractus/causal/notears.py` : penalty d'acyclicite NOTEARS `h(W) = tr(e^{W⊙W}) − n` via
   Taylor a 20 termes, differentiable, integree comme terme de loss.
2. `fractus/causal/rkhs.py` : VRAI RKHS via Random Fourier Features (Rahimi-Recht 2007) —
   noyau gaussien approxime, operateur `L = U @ Vᵀ` in l'espace des features.
3. `fractus/causal/do.py` : true do-calculus de Pearl (echantillonnage post-intervention),
   pas juste column-zeroing.
4. Datasets synthetiques : `data/causal/generate_scm.py` (Structural Causal Models connus),
   `data/causal/lucas.py` (LUCAS, standard).
5. `fractus/metrics/causal.py` : Structural Hamming Distance (SHD), causal accuracy mesuree
   (pas de clamp).

`fractus-core/src/causal.rs` (finally cree) : NOTEARS penalty en Rust pur for validation croisee.
`fractus-core/src/rkhs.rs` : RFF et noyau gaussien en Rust for precalcul/metriques.

**Critere « termine » :** `test_notears_penalty_is_zero_for_dag` (h(W)≈0 for un DAG evident) +
`test_notears_penalty_is_positive_for_cycle` (h(W)>0.5 for un cycle) +
`test_causal_recovery_on_known_dag` (SHD ≤ 3 sur un SCM a 5 variables after 50 steps).

### L5 — Raisonnement (preuves verifiees + conjectures + ACT)

**Corrige :** (le pipeline proof de FNN was deja le module le plus defendable ; on le rend
fonctionnel).

**Composants :**
1. `fractus/reasoning/proof.py` : ProofGenerator GRU entraine par **REINFORCE** (policy
   gradient, puisque la verification est non-differentiable). Recompense
   `0.6·correctness + 0.3·brevity + 0.1·novelty`.
2. `fractus-core/src/proof.rs` : verify EXACT en Rust (soundness garantie). 20 regles
   d'inference, specialisations Fermat/Wilson/GCD. Reste hors-graphe comme oracle de recompense.
3. `fractus/reasoning/conjecture.py` : decouvreur de conjectures falsifiables (Popperien) —
   10 templates, 6 strategies de falsification.
4. `fractus/reasoning/act.py` : Adaptive Computation Time (Graves 2016).

**Critere « termine » :** `test_verify_accepts_valid_proof` + `test_verify_rejects_invalid_proof`
(le Rust accepte/rejette correctement) + `test_proof_generator_can_learn_simple_theorem`
(reussite > 50% sur « pair+pair=pair » after 500 steps REINFORCE — critere ambitieux).

### L6 — Stabilite Lyapunov + metriques honnetes

**Corrige :** false Lyapunov (tracking de `‖y‖²` without system dynamique) ; metriques clampees.

**Composants :**
1. `fractus/stability/lyapunov.py` : function de Lyapunov du **under-system Kuramoto** (le
   seul true system dynamique du modele). `V(θ) = ½ Σ (θᵢ − θ*)²`, `dV/dt = ∇V · f(θ) ≤ 0`.
2. `fractus-core/src/lyapunov.rs` : verify numerique en Rust for validation croisee.
3. `fractus/metrics/honest.py` : `honest_perplexity` (vraie perplexite `exp(val_loss)`, pas
   proxy), `honest_compression`, `honest_causal`.

**Non-pretentions explicites :** Lyapunov garanti seulement sur Kuramoto (pas sur tout le
reseau) ; Lean 4 et ZK-SNARK omis (absents du code, notes future work).

**Critere « termine » :** `test_lyapunov_decreases_on_sync` (V decroit et monotone sur
trajectoire de synchronisation) + `test_perplexity_is_real` (between 1 et 1000 for vocabulaire
~100, calculee sur un true dataset).

### L7 — Demo (objective final)

Trois demos demontrables :

1. **Generation de texte :** `scripts/train.py --task text --dataset tinyshakespeare --epochs 5`
   then `scripts/generate.py`. Loss de validation tracee, comparaison dense vs SIREN.
2. **Raisonnement mathematical :** `scripts/train.py --task proofs` then `scripts/prove.py
   --theorem even_plus_even`. Courbe « % preuves valides » vs steps.
3. **Inference causale :** `scripts/train.py --task causal --dataset lucas` then
   `scripts/causal.py --query`. SHD rapporte, reponse contrefactuelle vs observationnelle.

Deploiement CPU-only : `scripts/serve.py --cpu-only` → API HTTP locale sur le Ryzen 5.

## 7. Synthese des corrections

| Couche | Corrige | Critere « termine » |
|---|---|---|
| L0 Socle | OMNI ne compile pas | `pytest test_smoke` traverse Python→Rust |
| L1 Embedding+Vortex | Vortex orphaned | Vortex conditionne MLP + ultrametrie testee |
| L2 Bloc transformer | FNN n'apprend pas | `backward()` propage gradients finis partout |
| L3 SIREN | Fausse SIREN, 20.4× hardcode | Vrai `sin(ω₀·)`, W utilisee, ratio mesure |
| L4 Causal | Faux RKHS, false do-calculus | NOTEARS recupere DAG synthetique (SHD test) |
| L5 Raisonnement | (deja bien code) | Generateur reussit >50% after 500 steps REINFORCE |
| L6 Stabilite | Faux Lyapunov, metriques clampees | V decroit sur Kuramoto, perplexite real |
| L7 Demo | (n'existait pas) | 3 demos tournent + courbes de loss |

## 8. Decisions cles

- Le Rust reste **hors-graphe autodiff** (computation exact, verification, metriques, precalcul).
- La forward/backward est **PyTorch pur** (autodiff natif, plus de bruit).
- Le vortex 2-adique **conditionne** un MLP entrainable (option B).
- `ω₀ = 30` (justifie par SIREN paper), pas 56.
- Nommage honnete partout : on garde les termes exacts, on renomme ceux qui surencherissaient.

## 9. Ordre d'attaque suggere

L0 → L1 → L2 d'abord (demo texte rapide), then L3 (compression), then L4 (causal), L5
(raisonnement), L6 (stabilite), L7 (demos integrees). Chaque couche est livrable independamment.

**Le plan d'implementation (etape suivante du process) sera decoupe par couche.** Chaque
couche donnera lieu a son propre under-plan with tâches granulaires. On ne redigera pas un
seul plan monolithique for les 7 couches — ce serait ingerable. Concretement : on commencera
par le plan de L0, then on l'executera, then on passera au plan de L1, etc.

## 10. Future work (honnete)

Lean 4 formal proofs, ZK-SNARK attestation, K3 automorphic compression, Groth16 timing,
scaling a gros modeles, evaluation sur benchmarks standardises type MMLU/HellaSwag.
