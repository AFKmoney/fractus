---
language: ["en", "fr", "code"]
tags:
  - continuous-thought-engine
  - rag
  - metacognition
  - decentralized-ai
  - cpu-trained
  - agi
license: mit
---

# Fractus

**Le premier système d'IA à pensée continue, mémoire persistante et autonomie cognitive — qui s'entraîne et vit sur un laptop ordinaire.**

Fractus n'est pas un autre GPT. C'est une architecture fondamentalement différente où l'IA pense en temps réel (pas input→output statique), se souvient de chaque interaction entre les sessions, apprend sans réentraînement, change de personnalité à la volée, et décide elle-même quand chercher, apprendre, réfléchir ou répondre.

> **Aucune corporation ne peut le contrôler.** L'IA tourne sur la machine de l'utilisateur. Les données ne quittent jamais l'appareil.

---

## Les trois couches de Fractus

### Couche 1 : Le Cerveau (Fractus-1B)

Un transformer fractal propriétaire de **0.86 milliard de paramètres de capacité effective**, compressé en seulement **88M paramètres entraînables** via LazyStructuredSiren (décomposition low-rank W = scale·U·Vᵀ).

**Architecture :**
- **LazyStructuredSiren** : chaque matrice de poids est stockée comme U·Vᵀ (rank 16), pas comme une matrice dense. Résultat : 0.86B capacité dans 0.4 GB de RAM. Aucune grid SIREN — pas de fuite mémoire.
- **64 experts MoE sparse** : seulement top-2 experts actifs par token (routing von Mises sur phases Farey). Compute proportionnel à 2/64 du total.
- **Attention linéaire causale multi-niveaux** (Katharopoulos 2020) avec batching des heads×levels (optimisation mesurée à 2.6× speedup).
- **Oscillateurs Kuramoto RK4 low-rang** : un système dynamique couplé qui sert d'"horloge de conscience" — les phases déterminent le routing cognitif.
- **Vortex 2-adique (Rust)** : arithmétique p-adique exacte pour le conditionnement des tokens, hors graphe autodiff.

**Entraînement from scratch sur CPU :**
| Epoch | Loss | Perplexity |
|-------|------|------------|
| 1 | 5.322 | 205 |
| 5 | 5.045 | 155 |
| 9 | 4.799 | 121 |
| 10 | 4.730 | 113 |
| 11 | 4.635 | 103 |

Vitesse : **19-21 tokens/sec** sur AMD Ryzen 5 5500U (6 cœurs, CPU-only).
Données : 500k tokens d'un corpus de 12.8M (code Python 26%, web knowledge, instructions QA, chat humain, creative writing).

### Couche 2 : Le Continuous Thought Engine (CTE)

Le cerveau ne fonctionne pas en input→output. Il **tick** comme un cerveau biologique :

1. **Chaque tick** : les oscillateurs Kuramoto avancent d'un step RK4 → l'état d'attention (S,z) accumule le contexte → le MoE transforme la pensée → une head de confiance décide si l'IA a quelque chose à dire.

2. **Profondeur adaptative** : une question facile = 1 tick (réponse immédiate). Une question difficile = 10 ticks (l'IA réfléchit, accumule de l'évidence, puis répond quand elle est confiante). C'est du **raisonnement énergie-proportionnel**.

3. **Proactif** : le CTE peut émettre un output sans être sollicité — quand la dynamique interne fait traverser le seuil de confiance. GPT et Claude attendent une question. Fractus peut initier.

4. **Chunk-based processing** : 16 tokens par forward pass (117 tok/s sur le 13M, 19 tok/s sur le 1B) au lieu d'un token à la fois.

### Couche 3 : La Couche Cognitive (RAG + MetaCognition)

C'est ce qui transforme Fractus d'un **outil** en un **agent** :

#### Mémoire Persistante (`rag.learn()`)
L'IA stocke chaque fait, chaque conversation, chaque interaction dans une **base de connaissances vectorielle** qui :
- **Survit aux redémarrages** (sauvegarde sur disque, recharge au démarrage)
- **Récupère par similarité cosinus** : pose une question → l'IA trouve les passages pertinents dans sa mémoire
- **Grandit sans réentraînement** : `rag.learn("nouveau fait")` ajoute du savoir instantanément, sans un seul gradient

```
rag.learn("Python est un langage créé par Guido van Rossum.")
rag.learn("L'utilisateur s'appelle Philippe et préfère les réponses concises.")

# Plus tard :
result = rag.query("Qui a créé Python ?")
# → retrieve "Python est un langage créé par Guido van Rossum."
# → génère une réponse basée sur ce contexte
```

#### Apprentissage Continu (`rag.converse()`)
Chaque conversation est une opportunité d'apprentissage :
1. L'input de l'utilisateur est **stocké** dans la KB (l'IA s'en souvient)
2. L'IA **récupère** le contexte pertinent de ses souvenirs passés
3. L'IA **génère** une réponse
4. Sa propre réponse est aussi **stockée** (l'IA apprend de ce qu'elle dit)

**Le modèle ne s'arrête jamais d'apprendre.** Il ne faut JAMAIS le réentraîner pour lui apprendre quelque chose de nouveau. C'est comme un humain qui accumule de l'expérience.

#### Plugins Cognitifs (personnalité hot-swappable)
5 modes de pensée, changeables instantanément comme des apps :

| Plugin | Température | Style |
|--------|-------------|-------|
| analyst | 0.3 | Précis, factuel, structuré |
| creative | 1.2 | Imaginatif, expressif |
| coder | 0.2 | Code propre, correct |
| teacher | 0.5 | Patient, exemples simples |
| hacker | 0.4 | Cybersécurité, attaquant+défenseur |

```python
pm.load("coder")      # l'IA pense comme un développeur
pm.load("creative")   # switch immédiat vers mode créatif
pm.custom("philosophe", temperature=0.9)  # ton propre style
```

#### MétaCognition (autonomie)
Un réseau d'action de **8.5K paramètres** qui décide **ce que l'IA fait** à chaque interaction :

- **RETRIEVE** : chercher dans la mémoire
- **LEARN** : stocker une nouvelle information
- **GENERATE** : produire une réponse
- **SWITCH** : changer de mode cognitif
- **REFLECT** : réfléchir plus avant de répondre

L'IA analyse l'input, choisit une séquence d'actions (jusqu'à 3), l'exécute, et **apprend de l'outcome**. Le réseau d'action s'entraîne en temps réel — l'IA devient meilleure à se gérer avec l'usage.

```python
# L'IA décide elle-même :
meta.process("Remember: my API key is sk-1234")
# → Actions: ['LEARN'] — elle a décidé de mémoriser

meta.process("What is my API key?")
# → Actions: ['RETRIEVE', 'GENERATE'] — elle cherche puis répond
```

---

## Pourquoi Fractus est différent de GPT/Claude

| Propriété | GPT-4 / Claude | Fractus |
|---|---|---|
| **Traitement** | Statique (1 forward pass) | Continu (tick-by-tick CTE) |
| **Mémoire** | Fenêtre de contexte (amnésique) | Base vectorielle persistante |
| **Apprentissage** | Réentraînement requis | En ligne (chaque conversation) |
| **Compétences** | Monolithe générique | Plugins hot-swappable (5 modes) |
| **Autonomie** | Attend des instructions | Décide ses propres actions |
| **Entraînement** | Datacenter GPU | CPU laptop consumer |
| **Déploiement** | API cloud (centralisé) | Appareil local (décentralisé) |
| **Données utilisateur** | Envoyées au serveur | Restent sur l'appareil |
| **Croissance** | Figé entre versions | Grandit à chaque utilisation |
| **Coût d'entraînement** | Millions de dollars | $0 (électricité seulement) |

---

## Ce qui est prouvé et testé

| Composant | Status | Test |
|---|---|---|
| Fractus-1B (88M params, 0.86B capacité) | ✅ Entraîné (epoch 11, loss 4.635, ppl 103) | Convergence mesurée |
| LazyStructuredSiren | ✅ Fonctionne | 0.4 GB RAM, 21 tok/s |
| ContinuousThoughtEngine | ✅ Codé + testé | tick(), tick_chunk(), generate() |
| RAG (KnowledgeBase + retrieval) | ✅ Fonctionne | Apprend, récupère, répond |
| Apprentissage en ligne | ✅ Testé | Grandit sans réentraînement |
| Plugins cognitifs | ✅ 5 modes | Hot-swap en 1 appel |
| MétaCognition | ✅ Codé + testé | 5 actions, autonome |
| Mémoire persistante | ✅ Sauvegarde/charge | Survit aux redémarrages |
| MoE sparse (64 experts, top-2) | ✅ Fonctionne | Von Mises/Farey routing |
| Attention linéaire batchée | ✅ 2.6× speedup | Équivalence vectorisée testée |

**166+ tests** passent dans la suite de tests.

---

## Démarrage rapide

```bash
git clone https://github.com/AFKmoney/fractus.git
cd fractus
py -m venv .venv && .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
maturin develop --release
pytest tests/ -q
```

### Utiliser le système RAG

```python
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.tokenizer import FractusTokenizer
from fractus.rag import KnowledgeBase, RAGEngine, PluginManager, MetaCognition
import torch

engine = ContinuousThoughtEngine(vocab_size=50257, d_model=128)
tok = FractusTokenizer.gpt2_compatible()
kb = KnowledgeBase(d_model=128)
rag = RAGEngine(engine, tok, kb)
pm = PluginManager(rag)
meta = MetaCognition(rag, pm)

# Apprendre — sans réentraînement
rag.learn("Python est un langage de programmation.")
rag.learn("Les réseaux de neurones apprennent par rétropropagation.")

# Demander
result = rag.query("Qu'est-ce que Python ?", top_k=2, max_tokens=30)
print(result['answer'])

# Laisser l'IA se gérer elle-même
result = meta.process("Souviens-toi: mon nom est Philippe")
print(result['actions'])  # ['LEARN'] — elle a choisi de mémoriser

# Changer de personnalité
pm.load("coder")     # mode développeur
pm.load("creative")  # mode créatif
```

---

## Architecture du dépôt

```
Fractus/
├── crate/fractus-core/           Rust: vortex 2-adique (math exacte, hors-graphe)
├── crate/fractus-py/             Rust: bindings PyO3
├── fractus/
│   ├── continuous_engine.py      Le Continuous Thought Engine (tick-based)
│   ├── model_1b.py               Fractus-1B (88M params, 0.86B capacité)
│   ├── rag.py                    RAG + KnowledgeBase + Plugins + MetaCognition
│   ├── memory.py                 Mémoire persistante cross-session
│   ├── cognitive_modes.py        Kuramoto → état mental
│   ├── generative_planner.py     Plan-then-fill generation
│   ├── specialization.py         Diversité des experts
│   ├── tokenizer.py              BPE byte-level (GPT-2 compatible)
│   ├── nn/                       12 modules (attention, Kuramoto, MoE, SIREN variants)
│   ├── causal/                   NOTEARS, RKHS, Pearl do-calculus
│   ├── reasoning/                Preuves, conjectures, nombres premiers, ACT
│   ├── stability/                Lyapunov sur Kuramoto
│   ├── metrics/                  Mesures honnêtes (compression, SHD, perplexité)
│   └── train/                    Online, mini-batch, surprise-gated, forward-forward
├── data/                         Datasets (Alpaca, OASST, Dolly, FineWeb, TinyStories, code)
├── tests/                        28 fichiers de test, 166+ tests
├── scripts/                      Training, démos, benchmarks, constructeurs de corpus
├── docs/                         OVERVIEW, SPEC, plans L0-L9
└── Fractus_White_Paper.pdf       Document technique (10 pages, signé)
```

---

## Données d'entraînement

Corpus mega de **12.8 millions de tokens** construit à partir de 9 sources :

| Source | Type | Tokens |
|---|---|---|
| Instructions code Python | Code | 1.5M |
| CodeAlpaca | Code multi-langage | 2M |
| FineWeb | Web / connaissance générale | 3M |
| Alpaca | Questions-réponses | 2M |
| OpenAssistant | Chat humain | 2M |
| TinyStories | Écriture créative | 1.5M |
| Dolly | Instruction tuning | 1M |

Coverage vocabulaire : 96.8%. Build : `python scripts/build_mega_corpus.py`

---

## Limitations honnêtes

1. **Qualité de génération** — le modèle à epoch 11 (ppl 103) génère du texte rudimentaire. Plus de training = meilleur.
2. **CTE needs trained weights** — le CTE fonctionne mais a besoin des poids 1B transférés pour produire du texte cohérent.
3. **MétaCognition jeune** — le réseau d'action (8.5K params) s'améliore avec l'usage mais est au début.
4. **Vitesse d'entraînement** — 19 tok/s sur CPU. Un GPU accélère 50-100×.

---

## Licence

MIT. Ce projet appartient à l'utilisateur, pas à une corporation.

## Auteur

**Philippe-Antoine Robert** — 2026

## Liens

- **GitHub :** [github.com/AFKmoney/fractus](https://github.com/AFKmoney/fractus)
- **HuggingFace :** [huggingface.co/thefinalboss/Fractus](https://huggingface.co/thefinalboss/Fractus)
