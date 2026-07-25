# Adapted Chinchilla Scaling Law for Fractus

## Why Standard Chinchilla Doesn't Apply

Chinchilla (Hoffmann et al. 2022) derived the optimal token-to-parameter ratio by training **dense transformer models** where:
- ALL parameters are active for EVERY token
- Weights are full-rank dense matrices
- Attention is softmax (O(L²) compute)
- The loss curve follows: L = A/N^α + B/P^β

The optimal point balances the decreasing loss from more data (N) against the increasing capacity from more parameters (P), yielding:

```
N_optimal = 20 × P_total
```

**Fractus violates every assumption of this derivation:**

| Assumption | Standard Transformer | Fractus |
|-----------|---------------------|---------|
| Active params per token | = P_total (all active) | = P_active (2/128 experts) |
| Weight structure | Dense (full rank) | LazyStructuredSiren (rank r) |
| Attention | Softmax O(L²) | Linear O(L) |
| Routing | None (dense) | Kuramoto phase → von Mises gate |

---

## Deriving the Fractus Scaling Law

### Step 1: Separate the Model into Components

Fractus has two types of parameters with different training dynamics:

**Shared parameters** (trained on EVERY token):
- Embedding: P_embed = 64.4M
- Attention layers: P_attn = 6.6M × 16 layers = 105.6M
- Kuramoto + norms: P_kur ≈ 0.1M
- Total shared: P_shared = 170.1M

**Expert parameters** (only trained when their expert is selected):
- Per expert pair (w1 + w2): P_expert = 0.43M
- Total experts: N_experts = 128 per layer × 16 layers = 2048
- Total expert params: P_expert_total = 879.0M

### Step 2: Chinchilla for Shared Parameters

Shared parameters see EVERY token. Standard Chinchilla applies directly:

```
N_shared = 20 × P_shared = 20 × 170.1M = 3.4B tokens
```

### Step 3: Chinchilla for Expert Parameters

Each expert only sees tokens routed to it. With top_k=2 and 128 experts per layer, assuming roughly uniform routing:

```
Probability expert i is selected per token = top_k / N_experts = 2/128 = 1/64
```

Each expert has P_expert = 0.43M parameters. For each expert to reach Chinchilla optimality, it needs to see:

```
Tokens per expert = 20 × P_expert = 20 × 0.43M = 8.6M tokens
```

Since each token activates top_k=2 experts, and there are 128 experts per layer, the corpus must be large enough that each expert receives its 8.6M tokens:

```
Tokens routed to expert i = N_corpus × (top_k / N_experts)
```

Setting this equal to the Chinchilla requirement:

```
N_corpus × (2/128) ≥ 20 × 0.43M
N_corpus ≥ 20 × 0.43M × 128 / 2
N_corpus ≥ 20 × 0.43M × 64
N_corpus ≥ 550.4M tokens per layer
```

But we have 16 layers, each with 128 independent experts. The corpus is shared across layers (each token passes through all 16 layers). So:

```
N_expert_total = (N_experts_per_layer × N_layers × 20 × P_expert) / top_k
N_expert_total = (128 × 16 × 20 × 0.43M) / 2
N_expert_total = 8.8B tokens
```

### Step 4: The Binding Constraint

The optimal corpus size is the MAXIMUM of the two requirements:

```
N_optimal = max(N_shared, N_expert_total)
N_optimal = max(3.4B, 8.8B)
N_optimal = 8.8B tokens
```

### Step 5: Correction for LazyStructuredSiren

LazyStructuredSiren stores W = scale · U · V^T where U(out, rank) and V(in, rank).

A rank-r matrix has **r × (m + n) - r²** degrees of freedom (the dimension of the manifold of rank-r matrices). For r << min(m, n), this is ≈ r × (m + n), which equals the stored parameter count.

**This means: for low-rank weights, capacity = parameters. No correction needed.**

The Siren structure does NOT reduce information capacity below the parameter count — it constrains the WEIGHT SPACE to the rank-r manifold, but within that manifold, every parameter is free.

### Step 6: Correction for Linear Attention

Standard Chinchilla assumes softmax attention with O(L²) compute per token pair. Fractus uses linear attention (Katharopoulos 2020) with O(L) compute.

This affects the COMPUTE-optimal point (FLOPs per token), not the DATA-optimal point (tokens needed). The scaling law for DATA depends on model capacity, not compute per token.

**Linear attention does NOT change N_optimal** — it only makes each token cheaper to process.

### Step 7: Correction for Kuramoto Routing

Standard MoE uses LEARNED routing (the gate is a linear layer trained by backprop). Fractus uses Kuramoto phase routing — the gate is determined by oscillator phases, which evolve deterministically.

This has two effects:
1. **Faster routing convergence**: deterministic routing reaches steady state faster than learned routing (which needs gradient descent to find good assignments)
2. **Non-uniform routing**: the von Mises distribution means some experts get more tokens than others. The worst-case expert may see fewer tokens.

For a conservative estimate, we apply a routing diversity factor:

```
routing_diversity = 1.2 (20% overhead for non-uniform routing)
```

### Final Formula

```
N_optimal(Fractus) = routing_diversity × max(
    20 × P_shared,
    (N_experts_per_layer × N_layers × 20 × P_expert) / top_k
)
```

---

## Application to Fractus Configurations

### Fractus 88M (current)

| Parameter | Value |
|-----------|-------|
| P_shared (embed + attn + kur) | 38.5M |
| P_expert (per pair) | 0.22M |
| N_experts per layer | 64 |
| N_layers | 8 |
| top_k | 2 |

```
N_shared = 20 × 38.5M = 0.77B
N_expert = (64 × 8 × 20 × 0.22M) / 2 = 1.13B
N_optimal = 1.2 × max(0.77B, 1.13B) = 1.2 × 1.13B = 1.35B tokens
```

**Actual corpus used: 1.38B tokens → 102% of optimal. The 88M was Chinchilla-optimal!**

### Fractus 1B (target)

| Parameter | Value |
|-----------|-------|
| P_shared (embed + attn + kur) | 170.1M |
| P_expert (per pair) | 0.43M |
| N_experts per layer | 128 |
| N_layers | 16 |
| top_k | 2 |

```
N_shared = 20 × 170.1M = 3.40B
N_expert = (128 × 16 × 20 × 0.43M) / 2 = 8.80B
N_optimal = 1.2 × max(3.40B, 8.80B) = 1.2 × 8.80B = 10.56B tokens
```

**Fractus 1B optimal: ~10.6B tokens, NOT 21B.**

### Comparison Table

| Method | N_optimal for 1B | Reduction vs standard |
|--------|-----------------|----------------------|
| Standard Chinchilla (total params) | 21.0B | 1.0× |
| Active params only | 3.7B | 5.7× (too aggressive) |
| **Fractus-adapted Chinchilla** | **10.6B** | **2.0× reduction** |

The adapted law gives **half the tokens** of standard Chinchilla — because sparse MoE experts don't all need to see the same data.

---

## Impact on EDT Training Time

With the adapted Chinchilla (10.6B instead of 21B):

| EDT Phase | Standard (21B) | Adapted (10.6B) | Savings |
|-----------|----------------|-----------------|---------|
| Phase 1 (experts) | 1.2h | 1.2h (synthetic) | — |
| Phase 2b (embedding) | 3.2h (500M) | 3.2h (500M) | — |
| Phase 3 (joint) | 41h (100M) | 41h (100M) | — |
| **Corpus build** | **7-8h** | **3-4h** | **50%** |

The main savings are in corpus preparation time, not training time (EDT phases don't depend on corpus size for Phases 1-2, and Phase 3 uses only 100M tokens regardless).

---

## The General Formula (for any Fractus-like architecture)

For a model with:
- P_s = shared params (embedding + attention + norms)
- P_e = expert params (per expert pair)
- E = experts per layer
- L = number of layers
- K = top_k (active experts per token)
- D = routing diversity factor (1.0 for uniform, 1.2 for von Mises)

```
N_optimal = D × max(
    20 × P_s,
    (E × L × 20 × P_e) / K
)
```

This is the **Fractal Chinchilla Law** — adapted for sparse MoE architectures with low-rank weights and deterministic routing.

---

## Citation

```
Robert, P.-A. (2026). "Fractal Chinchilla: Adapted Scaling Law for
Sparse MoE Architectures with Low-Rank Weights."
In Fractus: A Continuous Cognitive Agent.
https://github.com/AFKmoney/fractus
```

---

*© 2026 Philippe-Antoine Robert. MIT License.*
