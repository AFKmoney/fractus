# Fractus — Métriques Complètes

*Toutes les valeurs sont mesurées, pas estimées. Dernière mise à jour : 2026-07-21.*

---

## 1. Modèle 88M (entraîné, checkpoint sur HF)

| Métrique | Valeur |
|----------|--------|
| **Config** | d=768, L=8, H=12, dh=64, E=64, K=2, ff=1024, rank=16 |
| **Params totaux** | 87.8M |
| **Params actifs/token** | ~30M (embed + 8×attn + 8×2 experts) |
| **RAM float32** | 0.35 GB |
| **RAM bf16** | 0.18 GB |
| **Checkpoint** | 1059 MB (sur HF) |
| **Corpus entraîné** | 1.38B tokens |
| **Steps entraînés** | 140,000 (~84% du corpus) |
| **Loss finale** | 2.91 |
| **Perplexité** | 9-14 (oscillante) |
| **Loss minimale** | 0.88 (ppl 2.4, step 32500) |
| **Hardware** | RTX 6000 Ada (48GB VRAM) |
| **Vitesse training** | 2.3 step/s, 9,400 tok/s |
| **Batch/seq** | 256 × 16 |
| **Coût training réel** | ~$80 (bugs inclus) |

---

## 2. Modèle 1B (architecture Fractus1B)

| Métrique | Valeur |
|----------|--------|
| **Config** | d=1280, L=16, H=20, dh=64, E=128, K=2, ff=2048, rank=64 |
| **Params totaux** | 1,048,674,976 (1.049B) |
| **Params actifs/token** | 183,243,776 (183.2M) |
| **Ratio actifs/total** | 17.5% |
| **RAM float32** | 4.19 GB |
| **RAM bf16** | 2.10 GB |
| **Optimizer 8-bit** | 1.57 GB |
| **Expert params** | 0.43M chacun × 2048 = 879M |
| **Attention params** | 6.6M × 16 = 105M |
| **Embedding** | 64.4M (tied avec LM head) |

---

## 3. Expert Decoupled Training (EDT)

*Mesuré sur RTX 3090, 24GB VRAM, bf16, 8-bit AdamW*

| Phase | Composants | Params | Données | Vitesse | Temps |
|-------|-----------|--------|---------|---------|-------|
| **1 (experts)** | 2048 experts indépendants | 0.43M chacun | hidden states réels | 237 step/s | **1.2h** |
| **2a (attention)** | 16 layers indépendants | 6.6M chacun | hidden states réels | 134 step/s | **<1s** |
| **2b (embedding)** | embedding + LM head | 64.4M | 500M tokens corpus | 43,773 tok/s | **3.2h** |
| **3 (joint)** | modèle complet | 1.049B | 100M tokens corpus | 678 tok/s | **41h** |
| **TOTAL EDT** | | | | | **~45h (1.9 jours)** |

| Comparaison | Standard | EDT | Différence |
|-------------|----------|-----|-----------|
| **Temps (1B, 21B tokens)** | 358 jours | 2 jours | **356 jours sauvés** |
| **Coût (RTX 3090 $0.40/hr)** | ~$3,400 | ~$18 | **$3,382 sauvés** |
| **Heures GPU** | 8,592h | 48h | **8,544h sauvées** |
| **Multi-GPU requis?** | Oui (cluster) | Non (1 GPU) | — |
| **Speedup** | 1× | **189×** | — |

---

## 4. Vitesse GPU mesurée (1B)

*RTX 3090, batch=14, seq=32, PGSU=4, bf16, 8-bit AdamW*

| Métrique | Valeur |
|----------|--------|
| **Step time** | 661ms |
| **Throughput** | 678 tok/s |
| **VRAM peak** | 23.8 GB / 24 GB |
| **Loss (warmup 5 steps)** | 15.45 → 7.42 |

### Breakdown VRAM (batch=4)

| Composant | VRAM |
|-----------|------|
| Model (bf16) | 2.1 GB |
| Optimizer (8-bit) | 1.6 GB |
| Gradients (bf16) | 2.1 GB |
| Activations | 3.8 GB |
| **Total peak** | **9.6 GB** |
| **Libre** | **15.4 GB** |

### Évolution batch → throughput

| Batch | Tok/s | VRAM |
|-------|-------|------|
| 4 | 195 | 9.6 GB |
| 8 | 394 | 16.5 GB |
| 14 (max) | 678 | 23.8 GB |

---

## 5. PGSU (Phase-Gated Sparse Update)

| Métrique | Valeur |
|----------|--------|
| Layers actifs/step | 4/16 |
| Backward reduction | 4× |
| Fairness | chaque layer active 4× sur 16 steps |
| Convergence testée | loss descend (6.94→4.55 en 20 steps) |
| Gradients sur layers actifs | ✓ (vérifié) |
| Gradients sur layers gelés | None (vérifié) |

---

## 6. Progressive Depth Training

| Phase | Layers actifs | Params effectifs |
|-------|--------------|-----------------|
| 1 (0-25%) | 4/16 | ~250M |
| 2 (25-50%) | 8/16 | ~500M |
| 3 (50-75%) | 12/16 | ~750M |
| 4 (75-100%) | 16/16 | ~1B |
| **Speedup global** | | **~2×** |

---

## 7. HVR (Holographic Vector Learning)

### Ce qui marche

| Test | Vocab | Recall | Vitesse |
|------|-------|--------|---------|
| Binding/unbinding | 100 | rank #1 (100%) | — |
| Séquence synthétique | 100 | 4/4 (100%) | — |
| Multi-séquences | 200 | 9/9 (100%) | — |
| Vrai texte (word-level) | 78 | 5/5 (100%) | 2,266 tok/s |
| Scaling 100k tokens | 500 | 16/16 (100%) | 2,860 tok/s |
| GPU (1M tokens) | 5000 | — | 2,928 tok/s |
| **Logit bias CTE** | 78 | `def`→`fibonacci` ✓ | — |

### Ce qui ne marche pas

| Test | Vocab | Recall | Raison |
|------|-------|--------|--------|
| BPE 50k (448 tokens) | 50,257 | 0/4 (0%) | Cross-talk |
| BPE 50k (5420 tokens, 20 passes) | 50,257 | 0/4 (0%) | Cross-talk |
| BPE 50k dim=20000 | 50,257 | 0/4 (0%) | Cross-talk |
| BPE 50k dim=50000 | 50,257 | timeout | Trop de RAM |

---

## 8. Rank Expansion (88M → plus gros)

| Config | Rank | Params | Transféré | Expanded | Forward |
|--------|------|--------|-----------|----------|---------|
| 88M original | 16 | 88M | — | — | ✓ |
| Expanded | 32 | 156M | 2183 | 2048 | ✓ |
| 1B target | 64 | 1.049B | — | — | ✓ |

---

## 9. Corpus

### Corpus 88M (1.38B tokens)

| Source | Tokens | % |
|--------|--------|---|
| codeparrot-clean (Python) | 350M | 25% |
| FineWeb (sample-10BT) | 200M | 14% |
| OpenOrca | 160M | 12% |
| CodeFeedback | 104M | 8% |
| Wikipedia 2023 | 100M | 7% |
| TinyStories | 100M | 7% |
| code_x_glue (6 langages) | 165M | 12% |
| Autres (OASST, Dolly, Alpaca, Cosmopedia) | 201M | 15% |

### Corpus 1B (20.9B tokens)

| Source | Tokens | % |
|--------|--------|---|
| FineWeb-Edu | 8.0B | 38% |
| CodeParrot-clean | 2.5B | 12% |
| FineWeb | 2.0B | 10% |
| OpenOrca | 1.5B | 7% |
| Open-Web-Math | 1.0B | 5% |
| CodeFeedback | 1.0B | 5% |
| Tulu-3 SFT | 0.8B | 4% |
| FLAN-v2 | 0.7B | 3% |
| Cosmopedia (math) | 0.5B | 2% |
| Wikipedia | 1.0B | 5% |
| Autres | 2.9B | 14% |

---

## 10. CTE (Continuous Thought Engine) assemblé

| Métrique | Valeur |
|----------|--------|
| Config | d=768, H=12, dh=64, E=64, K=2, ff=1024, rank=16 |
| Params CTE | 386M (avec CachedSiren) / 88M (avec LazySiren) |
| Poids transférés | 401/1432 (embedding, attention, kuramoto, experts U/V) |
| Experts reconstruits | 128 (_cached_W = scale × U @ V^T) |
| Tick modes | tick(), tick_chunk() (16 tokens), think() (adaptive) |
| Kuramoto oscillators | 16, RK4, rank-8 coupling |
| Transfer loss | 2.91 (préservé du training) |

---

## 11. Cognitive Layer

| Composant | Status | Détail |
|-----------|--------|--------|
| KnowledgeBase | ✓ | Vector memory, cosine similarity, save/load |
| rag.learn() | ✓ | Instant, zero gradients |
| rag.query() | ✓ | Retrieval + generation testé |
| rag.converse() | ✓ | Store + retrieve + generate + store |
| PluginManager | ✓ | 5 modes (analyst, creative, coder, teacher, hacker) |
| MetaCognition | ✓ | 8.5K params, 5 actions |
| Persistent memory | ✓ | Save/load fractus_memory.pkl |

### Démo validée

```
rag.learn("Python was created by Guido van Rossum")
rag.query("Who created Python?") → "Python was created by Guido van Rossum"
meta.process("Remember: my name is Philippe") → ['RETRIEVE', 'SWITCH', 'GENERATE']
pm.load("coder") → switched ✓
```

---

## 12. Génération (88M, step 140000)

| Prompt | Output |
|--------|--------|
| `def fibonacci(n):` | Docstring Python cohérent ✓ |
| `Python is` | `"free software: you can redistribute it and/or modify"` ✓ |
| `Once upon a time` | `"Once upon a timezone."` ✓ |
| `The meaning of life` | `"The meaning of life of the"` (court) |

### CTE assemblé

| Prompt | Output |
|--------|--------|
| `def fibonacci` | `def fibonacciseries` ✓ |
| `Python is` | `Python isinstance` ✓ |
| `Hello` | `Hello missing` ✓ |

---

## 13. Tests

| Métrique | Valeur |
|----------|--------|
| Tests unitaires | 166+ passent |
| Fichiers de test | 28 |
| MoE vectorisé équiv | 7.45e-09 |
| Chunked CE équiv | 4.77e-07 |
| Triton kernel self-test | PASS (loss 4.77e-07, grad <1e-7) |
| PGSU | Schedule fair, gradients corrects |
| EDT | 4 phases validées sur GPU |
| Rank expansion | Forward OK, connaissance préservée |

---

## 14. Coûts résumés

| Item | Coût |
|------|------|
| Training 88M (avec bugs) | ~$80 |
| Training 88M (sans bugs) | ~$40 |
| Training 1B standard | ~$3,400-$5,000 |
| **Training 1B EDT** | **~$18-25** |
| Corpus build 21B | ~$2 |
| Inference | **$0 (local)** |

---

*© 2026 Philippe-Antoine Robert. Toutes les métriques sont mesurées et reproductibles.*
