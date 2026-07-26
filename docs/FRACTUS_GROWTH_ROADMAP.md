# Fractus Growth Roadmap

## How Fractus Stays Alive and Grows Over Years

Fractus is not a frozen checkpoint. It is a living brain that grows through three mechanisms, each with different purposes:

---

## Priority 1: Expert Addition (the primary growth mechanism)

### What
Add new MoE experts without touching existing ones. Each new expert is a specialist trained independently via EDT Phase 1.

### Why it's the best for Fractus
- Old experts stay intact (no catastrophic forgetting)
- New capacity is specialized (Rust expert, Math expert, etc.)
- Router learns to select the new experts naturally
- EDT Phase 1 pre-trains new experts in **seconds** (0.43M params each)
- This is **real structural growth** — the brain gets bigger, not just the memory

### Growth timeline (example)
```
Month 1: 128 experts (Python, web, instruct, wiki)
Month 2: +128 experts (Rust, Go, C++, math)
Month 3: +128 experts (science, reasoning, code QA)
...
Each addition: ~1.2h EDT Phase 1 + brief Phase 3 alignment
```

### Implementation
- `add_experts(n_new, domain_data)` — creates + pre-trains new experts
- Router (Kuramoto phases) extended with new Farey phases for new experts
- No retraining of existing experts

---

## Priority 2: Rank Expansion (when you need more depth)

### What
Increase the Siren rank of existing experts. rank 16 → 32 → 64 → 128.

### Why
- More rank = more expressive capacity per expert
- Existing knowledge preserved (old rank columns keep their values)
- New rank columns learn from new data
- `rank_expand.py` already exists and is tested

### When to use
- When experts plateau (can't learn more at current rank)
- When you want to deepen a specific domain
- Not for adding new knowledge — that's expert addition

### Growth timeline
```
Training:   rank 64 (1B params)
Expansion:  rank 128 (2B params) — deeper reasoning
Expansion:  rank 256 (4B params) — more nuance
```

---

## Priority 3: Persistent Memory (survives all expansions)

### What
The RAG knowledge base persists across ALL checkpoint versions. When you add experts or expand rank, the memory carries over.

### Why it's critical
- Facts learned via `rag.learn()` are never lost
- User interactions accumulate across versions
- The KB is version-independent — it's data, not weights

### What GLM correctly identified
RAG grows **knowledge**, not **capacity**. It makes Fractus smarter factually but doesn't make it structurally more capable. It's necessary but not sufficient for real growth.

---

## Priority 4 (later): Self-Proposed Growth

### What
Fractus itself proposes how it wants to grow. Based on:
- Its training on its own code (AICL)
- Its observation of what it can't do well
- Its metacognitive analysis of its own gaps

### The loop
```
1. Fractus encounters a task it can't do well
2. MetaCognition identifies the gap: "I need a Rust expert"
3. Fractus proposes: add_experts(domain="rust", data=rust_corpus)
4. EDT Phase 1 trains the new expert in seconds
5. Brief Phase 3 aligns the router
6. Fractus is now better at Rust — autonomously
```

### Why this is the endgame
Options 1 and 2 are the **mechanisms**. Self-proposed growth is what transforms "we add experts" into **"Fractus improved itself."** This is the AGI loop.

---

## Implementation Priority for Next Session

| Priority | What | When | Cost |
|----------|------|------|------|
| 1 | Expert addition via EDT | Next session | ~1.2h per batch of 128 |
| 2 | Rank expansion | When needed | ~hours per expansion |
| 3 | Persistent KB (already works) | Done | $0 |
| 4 | Self-proposed growth | After 1B trained | Research phase |

---

## The Big Picture

```
Fractus 1B (trained once, ~$20)
    ↓ add experts (months 2-3, ~$2 each batch)
Fractus 1B+ (more domains)
    ↓ rank expansion (month 4, ~$5)
Fractus 2B equivalent
    ↓ self-proposed growth (month 6+)
Fractus living AGI
```

The checkpoint is NEVER frozen. It grows through expert addition + rank expansion + persistent memory. The only "expensive" step is the initial training — everything after that is cheap incremental growth.

---

*Based on analysis by Philippe-Antoine Robert + GLM, 2026-07.*
