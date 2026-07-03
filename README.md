# Fractus

**A continuous-thought AI with persistent memory, cognitive modes, and autonomous self-management — runs on any laptop.**

Fractus is not another GPT clone. It is a fundamentally different architecture built on five pillars:

1. **Continuous Thought Engine** — thinks tick-by-tick like a dynamical system, not a static forward pass
2. **Retrieval-Augmented Generation (RAG)** — a vector knowledge base that provides unlimited external memory
3. **Online Learning** — learns new facts from every conversation without retraining
4. **Cognitive Plugins** — hot-swappable personality modules (coder, creative, hacker, teacher, analyst)
5. **MetaCognition** — the AI decides for itself when to retrieve, learn, switch mode, or reflect

---

## Quick test

```bash
git clone https://github.com/AFKmoney/fractus.git
cd fractus
py -m venv .venv && .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
maturin develop --release

python -c "
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

# Teach it
rag.learn('Python is a programming language.')
rag.learn('Neural networks learn via backpropagation.')

# Ask it
result = rag.query('What is Python?', top_k=2, max_tokens=30)
print(result['answer'])

# Let it manage itself
result = meta.process('Remember: my name is Philippe')
print(result['actions'])  # AI chose: LEARN
result = meta.process('What do you know about me?')
print(result['actions'])  # AI chose: RETRIEVE → GENERATE
"
```

---

## How it works

### The brain (13M params, trained)

The ContinuousThoughtEngine is a fractal transformer with:
- **Causal linear attention** (Katharopoulos 2020) with batched heads × levels
- **Kuramoto oscillators** (low-rank RK4) as a "consciousness clock"
- **Sparse MoE** (4 experts, top-2, von Mises/Farey routing)
- **Tick-based processing**: each tick advances the oscillators, updates the attention state, routes through experts, and estimates confidence
- **Chunk-based training**: processes 16 tokens per forward at 117 tokens/sec on CPU

### The memory (unlimited, grows with use)

The KnowledgeBase stores text chunks + their embeddings (computed by the engine's own embedding layer). Retrieval is cosine similarity. Key operations:
- `learn(text)` — adds knowledge instantly, no retraining
- `converse(input)` — learns from every interaction, responds, learns from its own response
- The KB saves to disk (`data/fractus_memory.pkl`) and reloads on restart

### The personality (hot-swappable, 5 built-in modes)

CognitivePlugins change the engine's behavior at runtime:
- **analyst** — precise, factual, structured (temp=0.3)
- **creative** — imaginative, expressive (temp=1.2)
- **coder** — clean, correct code (temp=0.2)
- **teacher** — patient, simple explanations (temp=0.5)
- **hacker** — cybersecurity mindset (temp=0.4)
- `custom(name, ...)` — create your own

### The autonomy (the AI manages itself)

MetaCognition gives the AI a tiny action network (8.5K params) that decides:
- **RETRIEVE** — search memory for relevant knowledge
- **LEARN** — store new information permanently
- **GENERATE** — produce an answer
- **SWITCH** — change cognitive mode (analyst → coder → creative...)
- **REFLECT** — think more before answering

The action network trains online from feedback — the AI gets better at self-management through use.

---

## Architecture

```
Fractus/
├── crate/fractus-core/           Rust: 2-adic vortex (exact math, off-graph)
├── crate/fractus-py/             Rust: PyO3 bindings
├── fractus/
│   ├── continuous_engine.py      The tick-based reasoning engine (13M params)
│   ├── model_1b.py               The 1B-capacity model architecture (LazyStructuredSiren)
│   ├── rag.py                    RAG + KnowledgeBase + Plugins + MetaCognition
│   ├── memory.py                 Persistent cross-session memory
│   ├── cognitive_modes.py        Kuramoto phase → mental state classifier
│   ├── generative_planner.py     Plan-then-fill generation
│   ├── specialization.py         Expert diversity loss
│   ├── tokenizer.py              GPT-2 byte-level BPE
│   ├── nn/                       embedding, attention, Kuramoto, MoE, SIREN variants
│   ├── causal/                   NOTEARS, RKHS, Pearl do-calculus
│   ├── reasoning/                proofs, conjectures, primes, ACT
│   ├── stability/                Lyapunov on Kuramoto
│   ├── metrics/                  honest measurements (compression, SHD, perplexity)
│   └── train/                    online, mini-batch, surprise-gated, forward-forward
├── data/                         quality datasets (Alpaca, OASST, Dolly, FineWeb, TinyStories)
├── tests/                        28 test files, 166+ tests
├── scripts/                      training, demos, benchmarks, dataset builder, white paper
├── docs/                         OVERVIEW, SPEC, layer plans, white paper PDF
├── Fractus_White_Paper.pdf       10-page technical document (signed)
└── MODEL_CARD.md                 HuggingFace model card
```

---

## Training data

The model was trained on 500k tokens from a 45M-token quality corpus:

| Source | Type | Tokens |
|---|---|---|
| FineWeb (sample-10BT) | Web text (general knowledge) | 20M available |
| Alpaca | Instruction QA pairs | 6M available |
| OpenAssistant | Human chat conversations | 10M available |
| TinyStories | Creative writing | 8M available |
| Dolly | Instruction tuning | 1.5M available |

Build your own subset: `python scripts/build_large_dataset.py`

---

## Measured performance

| Config | Tokens/sec | Hardware |
|---|---|---|
| 13M, single-token tick | 25 | Ryzen 5 5500U |
| 13M, chunk-based (16 tok) | 117 | Ryzen 5 5500U |
| 1B (88M trainable, LazySiren) | 5-8 | Ryzen 5 5500U |

### Training optimizations (all profile-measured)
| Optimization | Before → After | Speedup |
|---|---|---|
| Batch heads × levels in attention | 17.3ms → 6.6ms | 2.6× |
| Chunk-based tick (16 tokens) | 25 → 117 tok/s | 4.7× |
| LazyStructuredSiren (vs grid SIREN) | 43s → 5.9s/step | 7.3× |

---

## Why Fractus is different from GPT/Claude

| Property | GPT / Claude | Fractus |
|---|---|---|
| Processing | Static (1 forward) | Continuous (ticks) |
| Memory | Context window | Persistent KB + retrieval |
| Learning | Retraining needed | Online (every conversation) |
| Skills | Generic monolith | Hot-swappable plugins |
| Autonomy | Waits for instructions | Decides actions itself |
| Training | Datacenter GPU | Consumer CPU |
| Deployment | Cloud API | Local device |

---

## Documentation

| Document | Content |
|---|---|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | Complete A-to-Z walkthrough |
| [Fractus_White_Paper.pdf](Fractus_White_Paper.pdf) | 10-page technical paper |
| [docs/SPEC.md](docs/SPEC.md) | Full specification (L0-L7) |
| [docs/2026-06-26-fractus-L9-continuous-thought-engine.md](docs/2026-06-26-fractus-L9-continuous-thought-engine.md) | The 5 innovations |
| [docs/2026-06-26-fractus-L8-lightweight-training.md](docs/2026-06-26-fractus-L8-lightweight-training.md) | Training optimizations |

---

## Honest limitations

1. **Generation quality** — the model needs more training epochs for coherent text generation
2. **RAG quality** — the retrieval works but embeddings need training for precision
3. **MetaCognition is early** — the action net is 8.5K params, improves with use
4. **1B training is slow** — 5-8 tok/s on CPU (GPU would give 50-100×)

---

## License

MIT. This project belongs to the user, not a corporation.

## Author

**Philippe-Antoine Robert** — 2026

## Links

- **GitHub:** [github.com/AFKmoney/fractus](https://github.com/AFKmoney/fractus)
- **HuggingFace:** [huggingface.co/thefinalboss/Fractus](https://huggingface.co/thefinalboss/Fractus)
