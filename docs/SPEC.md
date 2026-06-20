# Fractus — Refonte unifiee of the original architecture + the original design

**Date :** 2026-06-19
**Auteur original :** the original author (the original architecture, the original design)
**Statut :** Spec valid by brainstorming, en attente of plan d'implementation

---

## 1. Contexte and motivation

L'utilisateur dispose of deux systems preexistants, all deux concus by the original author :

- **the original design** (~830 lignes, Python + Rust via PyO3) — white paper of 33 pages.
  These : « AGI legere via operateurs fractals on varietes non-archimediennes ».
  Promesses : compression 20.4×, raisonnement causal O(n), CPU-only, verification Lean 4 + ZK-SNARK.

- **the original architecture** (~10 000 lignes, Rust pur, 265+ tests) — « Fractal Neural Network : A Unified
  Architecture for AGI ». Transformer fractal with oscillateurs Kuramoto, MoE of Farey/von Mises,
  NOTEARS causal, generateur/verify of proofs, boucle d'auto-developpement.

Une analyse approfondie (code lu ligne by ligne) a revele that **les deux systems not fonctionnent
pas comme annonce**, malgre a culture mathematical real and plusieurs modules correctement codes.

### Erreurs critiques constatees (verifieses in the code source)

**the original architecture :**
- **Pas d'autodiff.** `training.rs:399-426` met a jour the poids with
  `scale * rand::random::<f64>() * 0.01` (du bruit) au lieu d'un gradient. Le commentaire
  `training.rs:156` l'admet : « Since we don't have autodiff, we apply a simple loss-scaling signal. »
  → La `AGILoss` a 11 termes est computationee but never vraiment minimisee.
- **Perplexite fictive.** `model.rs:537-546` : `perplexity()` renvoie a proxy base on the norme
  of l'embedding, not a vraie perplexite.
- **Benchmarks vides.** `benches/fnn_bench.rs` not contient that `fn bench_stub(_c) {}`.
- **Tests peu profonds.** `tests.rs:1840` : « AGINFNModel is currently a stub with only new(). »
  Les ~272 tests verifiesnt surtout formes of tenseurs (`.dim()`) and of the finitude.

**the original design :**
- **Le Rust not compile pas.** `rust/src/lib.rs:3-4` declare `pub mod causal; pub mod shield;`
  but the fichiers `causal.rs` and `shield.rs` n'existsnt pas. Le TODO of l'auteur the confirme.
- **Fausse SIREN.** `torus_siren.py:15-17` utilise `nn.SiLU`, not `sin(ω0·)`.
- **Le vortex 2-adique est orphaned.** Aucun fichier Python n'imported `omni_fractal_rs`.
- **W decompressee then jetee.** `training_loop.py:30-37` computatione W, the correctede, then
  l'causal operator tourne on l'entree brute (W ignoree).
- **Chiffres hardcodes.** `training_loop.py:52` : `"compression_ratio": 20.4` (litteral).
  `benchmarks.py:43-46` : `min(causal_acc, 0.98)` (plafonnee exactment a the target).

### Modules reallement corrects (a conserver)

- **the original** : Kuramoto RK4 bas-rang (`phase_ode.rs`), MoE von Mises/Farey (`moe.rs` + `farey.rs`),
  attention lineaire causale (`attention.rs`), NOTEARS (`causal.rs`), verify of proofs
  exact (`proof.rs`), SVD tronque by puissance iteree (`svd.rs`), sampling Gamma Marsaglia-Tsang.
- **the original design** : arithmetique 2-adique (`vortex.rs`) — valuation v2, ultrametric distance
  `2^v2(a⊕b)`, norme `2^{-v2}`. Le seul module mathematicalment correct and non trivial.

## 2. Objectif

**Prototype demontrable** : a system which tourne on of vraies donnees, dont the loss baisse
vraiment, with a demo convaincante (texte generated, theorems prouves valids, requetes causales).

**Non-objectifs explicites (future work) :** papier publiable, produit commercial, Lean 4,
ZK-SNARK, scaling a gros modeles, "AGI" au sens fort.

## 3. Stack technique

**Hybride Rust + Python**, with a separation stricte :

- **Rust (`fractus-core`)** : computation pur, outside the autodiff graph. Mathematiques exacts,
  verification of proofs, precomputation, metriques exogenes. Aucune I/O, no dataset.
- **Python (`fractus`, PyTorch)** : modele entrainable, forward/backward, autodiff natif,
  datasets, boucles d'training, logging.
- **Pont** : PyO3/maturin. Tenseurs numpy en entree/sortie. Le Rust not participe not au graphe
  autodiff (ecrire a `torch.autograd.Function` custom for each function serait couteux a
  maintenir and conduirait a backward fictifs — the piege qu'on veut eviter).

**Decision of nommage honestete :** « Mandelbrot frequencies » → « Mandelbrot-decayed Fourier
basis » ; « RKHS Causal Operator » → on implemente a true RKHS (with noyau via RFF en L4),
therefore on garde the nom but the substance suit ; « Bose-Einstein condensate » (the original `condensate.rs`)
→ on n'integre not this module in fractus (la SVD incrementale seule not justifie not the nom) ;
« Lyapunov Shield » → « Lyapunov monitor under-system Kuramoto » ; « Collatz ergodic flow »
→ « Collatz hash » (l'ergodicite of Collatz est non demontree, problem ouvert).

## 4. Hardware target

Machine of l'utilisateur (diagnostiquee) :
- CPU : AMD Ryzen 5 5500U, 6 coeurs / 12 threads @ 2.1 GHz
- RAM : ~12 GB
- GPU : AMD Radeon integree (APU), ~4 GB **partages** → ROCm not supported not the APU AMD
  integres under Windows → **training CPU-only effectif**.

Consequence : modele small (< 1M parameters), dataset minuscule (tinyshakespeare ~1 MB),
training en quelques heures. Coherent with the these the original « CPU-only deployment ».

## 5. Organisation depot

```
fractus/
├── crate/fractus-core/        # Rust : coeur mathematical pur
│   └── src/                   #   vortex 2-adique, SIREN (ref.), NOTEARS (ref.),
│                              #   Kuramoto/Farey (precomputation), verify proofs
├── crate/fractus-py/          # Rust : bindings PyO3/maturin
│   └── src/lib.rs             #   expose fractus_core a Python
├── fractus/                   # Python : the modele entrainable (PyTorch)
│   ├── nn/                    #   embedding, blocs, attention, MoE, decoder, siren
│   ├── causal/                #   NOTEARS layer, RKHS, do-computationus
│   ├── reasoning/             #   proofs (GRU), conjectures, ACT
│   ├── stability/             #   Lyapunov (under-system Kuramoto)
│   ├── metrics/               #   compression, causal (SHD), perplexite honestete
│   ├── train/                 #   boucles, datasets, losses
│   └── viz/                   #   demos interactives (optionnel)
├── tests/                     # tests d'integration Rust↔Python
├── data/                      # tinyshakespeare, datasets maths/causal
├── scripts/                   # train.py, demo.py, benchmark.py, serve.py
└── docs/                      # spec, white paper revise honestete, results
```

## 6. Les 7 couches d'implementation

Chaque couche = cycle design → code → test → demo autonome. On not passe a the suivante that quand
la precedente est verifiese. On can s'arreter a n'imported quel moment with quelque chose which marche.

### L0 — Socle technique

**Corrige :** the original not compile not ; pont Python↔Rust never fonctionnel.

**Composants :**
1. `pyproject.toml` with versions epinglees (torch CPU-only, maturin, numpy, pytest).
2. Crate `fractus-core` : `lib.rs` not declare QUE the modules which have a fichier. Port de
   `vortex.rs` d'the original (le 2-adic, already correct) with corrections : test tautological
   `assert!(d1 <= d2.max(d1))` → true test d'ultrametrie `d(x,z) ≤ max(d(x,y), d(y,z))` ;
   import `HashMap` inutilise retire.
3. Crate `fractus-py` : configuration maturin standard (`extension-module`), not le
   `[features] python = ["pyo3"]` mal configure d'the original.
4. Test fume which traverse all : `tests/test_smoke.py` — `add_in_rust(2,3)==5` + `torch` dispo.

**Critere « termine » :** these 4 commandes reussissent :
`cargo build --release` ; `maturin develop --release` ;
`python -c "import torch; import fractus"` ; `pytest tests/test_smoke.py`.

### L1 — Embedding fractal + vortex 2-adique branche

**Corrige :** vortex orphaned ; « Mandelbrot frequencies » mal nommees.

**Composants :**
1. `fractus/nn/embedding.py` : fractal codepoint embedding (PyTorch). Base of Fourier with
   Mandelbrot decay `(φ2)−k` (renommee honestetement), + 16 morphological features
   (cas, chiffre, ponctuation). Parametre entrainable via `nn.Linear` finale.
2. `fractus-core/src/vortex.rs` : coeur 2-adique portedd depuis the original.
3. **Pont (option B valide) :** the hash Collatz 2-adique est computatione en Rust (hors-graphe,
   exact) and **conditionne a MLP entrainable** (in the graphe) which produit the phases de
   l'embedding. Le vortex influences learning without pretendre etre differentiable.

**Critere « termine » :** `test_fractal_embedding_shape` (sortie `[N, d_model]` finie) +
`test_vortex_distance_is_ultrametric` (inegalite ultrametrique forte on 1000 triplets aleatoires).

### L2 — Bloc transformer fractal (scinde en L2a + L2b)

**Corrige :** the original n'apprend not (bruit instead of gradients).

**Scindage (decision post-brainstorming) :** L2 est the couche the more grosse
(~600 lignes PyTorch + ~30 tests). On the coupe en deux moities validables
independamment. A the fin of L2a on a already a transformer fractal fonctionnel
(without Kuramoto/MoE) capable d'apprendre texte — premier jalon demontrable.

**Composants (all en PyTorch pur for autodiff) :**
1. `fractus/nn/attention.py` : attention lineaire causale (Katharopoulos `S_t += k_t⊗v_t`),
   feature map `elu(x + ω_k) + 1`. Vrai `nn.Module` with parameters entrainables.
2. `fractus/nn/phase_ode.py` : Kuramoto RK4 bas-rang `K = UΛUT`. En PyTorch pur for rester
   in the graphe.
3. `fractus/nn/moe.py` : MoE a routing von Mises on phases Farey. Experts = MLP GeLU,
   loss auxiliaire of load-balance standard.
4. `fractus/nn/block.py` : assemblage `LayerNorm → FractalLinearAttention → PhaseSoliton →
   PhaseRoutedMoE`, with KuramotoODE avancant the phases d'un step a each bloc.

**L2a (jalon demontrable rapide) :**
- `fractus/nn/stats.py` : utilitaires (`elu_plus_one`, softmax stable, layer_norm).
- `fractus/nn/attention.py` : `FractalLinearAttention` (recurrence causale
  `S_t += φ(k_t) ⊗ v_t`, `y_t = φ(q_t)TS_t / φ(q_t)Tz_t`, feature map
  `elu_plus_one(x + ω_level)`, offsets ω_level = (φ2)^{-level}).
- `fractus/nn/block.py` : `FractalBlock` minimal = LayerNorm → attention → residuelle.
- Demo : surfit a sequence of toy tokens — the loss must baisser.
- **Critere « termine L2a » :** `test_block_forward_backward` prouve that backward
  propage a gradient fini ET non-nul a CHAQUE parameter bloc.

**L2b (greffe Kuramoto + MoE) :**
- `fractus/nn/farey.py` : Farey sequence + `expert_phases` (precomputation hors-graphe).
- `fractus/nn/phase_ode.py` : `KuramotoODE` (RK4 bas-rang, `encode_from_hidden`,
  `decode_to_bias`, `phase_loss`).
- `fractus/nn/moe.py` : `PhaseRoutedMoE` (gate von Mises, top-k, load-balance loss).
- `fractus/nn/block.py` etendu : integre Kuramoto + MoE in the bloc.

Le Rust garde the functions pures (Farey, `bessel_i0`, `von_mises_pdf`) for precomputation et
metriques hors-graphe (parameter d'ordre of Kuramoto).

**Critere « termine » :** `test_block_forward_backward` — `backward()` marche, all les
parameters recoivent gradients finis. This is exactment this which manquait a the original.

### L3 — Compression SIREN vraie + mesure honestete

**Corrige :** fausse SIREN (SiLU) ; W decompressee then jetee ; 20.4× hardcode.

**Composants :**
1. `fractus/nn/siren.py` : VRAIE SIREN on the tore T2. Non-linearite `sin(ω0·(Wx+b))` with
   `ω0 = 30.0` (value empirique papier SIREN Sitzmann 2020, PAS 56). Evalue the SIREN sur
   the grille `h×w` for regenerate the matrix.
2. Integration : **les projections d'attention** (`q_proj`, `k_proj`, `v_proj` — celles qui
   are the more grandes and the more compressibles) are remplacees by `SirenLinear`. La SIREN
   **EST** the matrix, elle est in the graphe, its parameters are entraines. Les petites
   matrixs (LayerNorm, biases) restent denses. Le ratio exact est mesure (L3.3), not assume.
3. `fractus/metrics/compression.py` : `measure_compression_ratio(model)` mesure reallement
   the ratio (taille dense equivalente / params SIREN). Pas of litteral hardcode.

`fractus-core/src/siren.rs` : implementation of reference non entrainee for validation croisee
(PyTorch and Rust must donner the same sortie for the memes poids).

**Critere « termine » :** `test_siren_produces_real_sinus` (`torch.sin` present, `SiLU` absent) +
`test_siren_is_in_autograd_graph` (les poids SIREN recoivent gradients) +
`test_compression_ratio_is_measured_not_hardcoded` (pas of `'20.4'` in the source).

### L4 — Causal NOTEARS + RKHS on donnees realles

**Corrige :** « RKHS Causal » which n'was qu'une projection bas-rang ; « do-computationus » trivial
(column-zeroing) ; causal accuracy plafonnee a 0.98.

**Composants :**
1. `fractus/causal/notears.py` : penalty d'acyclicite NOTEARS `h(W) = tr(e^{W⊙W}) − n` via
   Taylor a 20 termes, differentiable, integree comme terme of loss.
2. `fractus/causal/rkhs.py` : VRAI RKHS via Random Fourier Features (Rahimi-Recht 2007) —
   noyau gaussien approxime, operateur `L = U @ VT` in l'espace features.
3. `fractus/causal/do.py` : true do-computationus of Pearl (echantillonnage post-intervention),
   not juste column-zeroing.
4. Datasets synthetiques : `data/causal/generate_scm.py` (Structural Causal Models connus),
   `data/causal/lucas.py` (LUCAS, standard).
5. `fractus/metrics/causal.py` : Structural Hamming Distance (SHD), causal accuracy mesuree
   (pas of clamp).

`fractus-core/src/causal.rs` (finally cree) : NOTEARS penalty en Rust pur for validation croisee.
`fractus-core/src/rkhs.rs` : RFF and noyau gaussien en Rust for precomputation/metriques.

**Critere « termine » :** `test_notears_penalty_is_zero_for_dag` (h(W)≈0 for a DAG evident) +
`test_notears_penalty_is_positive_for_cycle` (h(W)>0.5 for a cycle) +
`test_causal_recovery_on_known_dag` (SHD ≤ 3 on a SCM a 5 variables after 50 steps).

### L5 — Raisonnement (proofs verifieses + conjectures + ACT)

**Corrige :** (le pipeline proof of the original was already the module the more defendable ; on the rend
fonctionnel).

**Composants :**
1. `fractus/reasoning/proof.py` : ProofGenerator GRU entraine by **REINFORCE** (policy
   gradient, puisque the verification est non-differentiable). Recompense
   `0.6·correctness + 0.3·brevity + 0.1·novelty`.
2. `fractus-core/src/proof.rs` : verify EXACT en Rust (soundness guaranteed). 20 regles
   d'inference, specialisations Fermat/Wilson/GCD. Reste hors-graphe comme oracle of reward.
3. `fractus/reasoning/conjecture.py` : decouvreur of conjectures falsifiables (Popperien) —
   10 templates, 6 strategies of falsification.
4. `fractus/reasoning/act.py` : Adaptive Computation Time (Graves 2016).

**Critere « termine » :** `test_verify_accepts_valid_proof` + `test_verify_rejects_invalid_proof`
(le Rust accepted/rejette correctement) + `test_proof_generator_can_learn_simple_theorem`
(reussite > 50% on « pair+pair=pair » after 500 steps REINFORCE — critere ambitieux).

### L6 — Stabilite Lyapunov + metriques honestetes

**Corrige :** false Lyapunov (tracking of `‖y‖2` without system dynamique) ; metriques clampees.

**Composants :**
1. `fractus/stability/lyapunov.py` : function of Lyapunov **under-system Kuramoto** (le
   seul true system dynamique modele). `V(θ) = 1⁄2 Σ (θi − θ*)2`, `dV/dt = ∇V · f(θ) ≤ 0`.
2. `fractus-core/src/lyapunov.rs` : verify numerique en Rust for validation croisee.
3. `fractus/metrics/honest.py` : `honest_perplexity` (vraie perplexite `exp(val_loss)`, pas
   proxy), `honest_compression`, `honest_causal`.

**Non-pretentions explicites :** Lyapunov guaranteed seulement on Kuramoto (pas on all le
reseau) ; Lean 4 and ZK-SNARK omis (absents code, notes future work).

**Critere « termine » :** `test_lyapunov_decreases_on_sync` (V decroit and monotone sur
trajectoire of synchronisation) + `test_perplexity_is_real` (between 1 and 1000 for vocabulaire
~100, computationee on a true dataset).

### L7 — Demo (objective final)

Trois demos demontrables :

1. **Generation of texte :** `scripts/train.py --task text --dataset tinyshakespeare --epochs 5`
   then `scripts/generate.py`. Loss of validation tracee, comparaison dense vs SIREN.
2. **Raisonnement mathematical :** `scripts/train.py --task proofs` then `scripts/prove.py
   --theorem even_plus_even`. Courbe « % proofs valids » vs steps.
3. **Inference causale :** `scripts/train.py --task causal --dataset lucas` then
   `scripts/causal.py --query`. SHD rapported, reponse contrefactuelle vs observationnelle.

Deploiement CPU-only : `scripts/serve.py --cpu-only` → API HTTP locale on the Ryzen 5.

## 7. Synthese corrections

| Couche | Corrige | Critere « termine » |
|---|---|---|
| L0 Socle | the original not compile not | `pytest test_smoke` traverse Python→Rust |
| L1 Embedding+Vortex | Vortex orphaned | Vortex conditionne MLP + ultrametrie testee |
| L2 Bloc transformer | the original n'apprend not | `backward()` propage gradients finis partout |
| L3 SIREN | Fausse SIREN, 20.4× hardcode | Vrai `sin(ω0·)`, W utilisee, ratio mesure |
| L4 Causal | Faux RKHS, false do-computationus | NOTEARS recupere DAG synthetique (SHD test) |
| L5 Raisonnement | (deja well code) | Generateur reussit >50% after 500 steps REINFORCE |
| L6 Stabilite | Faux Lyapunov, metriques clampees | V decroit on Kuramoto, perplexite real |
| L7 Demo | (n'existait pas) | 3 demos tournent + courbes of loss |

## 8. Decisions cles

- Le Rust reste **outside the autodiff graph** (computation exact, verification, metriques, precomputation).
- La forward/backward est **PyTorch pur** (autodiff natif, more of bruit).
- Le vortex 2-adique **conditionne** a MLP entrainable (option B).
- `ω0 = 30` (justifie by SIREN paper), not 56.
- Nommage honestete partout : on garde the termes exacts, on renomme ceux which surencherissaient.

## 9. Ordre d'attaque suggere

L0 → L1 → L2 d'abord (demo texte rapide), then L3 (compression), then L4 (causal), L5
(raisonnement), L6 (stabilite), L7 (demos integrees). Chaque couche est livrable independamment.

**Le plan d'implementation (etape suivante process) sera decoupe by couche.** Chaque
couche donnera lieu a son propre under-plan with taches granulaires. On not redigera not un
seul plan monolithique for the 7 couches — this serait ingerable. Concretement : on commencera
par the plan of L0, then on l'executera, then on passera au plan of L1, etc.

## 10. Future work (honestete)

Lean 4 formal proofs, ZK-SNARK attestation, K3 automorphic compression, Groth16 timing,
scaling a gros modeles, evaluation on benchmarks standardises type MMLU/HellaSwag.
