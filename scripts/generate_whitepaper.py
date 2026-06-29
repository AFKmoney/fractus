#!/usr/bin/env python
"""Generate the Fractus White Paper PDF.

A complete technical document covering the entire Fractus project:
architecture, innovations, measured results, and comparison with GPT/Claude.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, Image
)
from reportlab.pdfgen import canvas
from reportlab.lib import colors

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fractus_White_Paper.pdf")

# Colors
PRIMARY = HexColor("#1a1a2e")
ACCENT = HexColor("#16213e")
HIGHLIGHT = HexColor("#0f3460")
LIGHT_BG = HexColor("#f8f9fa")
TEXT_COLOR = HexColor("#2c3e50")
MUTED = HexColor("#7f8c8d")
RULE_COLOR = HexColor("#e0e0e0")

# Styles
styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    'CustomTitle', parent=styles['Title'],
    fontSize=28, textColor=PRIMARY, spaceAfter=6*mm,
    fontName='Helvetica-Bold', alignment=TA_CENTER,
)
style_subtitle = ParagraphStyle(
    'CustomSubtitle', parent=styles['Normal'],
    fontSize=14, textColor=MUTED, spaceAfter=12*mm,
    alignment=TA_CENTER, fontName='Helvetica',
)
style_author = ParagraphStyle(
    'Author', parent=styles['Normal'],
    fontSize=11, textColor=TEXT_COLOR, alignment=TA_CENTER,
    fontName='Helvetica', spaceAfter=2*mm,
)
style_h1 = ParagraphStyle(
    'H1', parent=styles['Heading1'],
    fontSize=18, textColor=PRIMARY, spaceBefore=10*mm, spaceAfter=4*mm,
    fontName='Helvetica-Bold',
)
style_h2 = ParagraphStyle(
    'H2', parent=styles['Heading2'],
    fontSize=14, textColor=ACCENT, spaceBefore=6*mm, spaceAfter=3*mm,
    fontName='Helvetica-Bold',
)
style_h3 = ParagraphStyle(
    'H3', parent=styles['Heading3'],
    fontSize=12, textColor=HIGHLIGHT, spaceBefore=4*mm, spaceAfter=2*mm,
    fontName='Helvetica-Bold',
)
style_body = ParagraphStyle(
    'Body', parent=styles['Normal'],
    fontSize=10, textColor=TEXT_COLOR, spaceAfter=3*mm,
    alignment=TA_JUSTIFY, fontName='Helvetica',
    leading=14,
)
style_code = ParagraphStyle(
    'Code', parent=styles['Code'],
    fontSize=8.5, textColor=HexColor("#c0392b"), spaceAfter=3*mm,
    fontName='Courier', leftIndent=8*mm, leading=11,
    backColor=HexColor("#fdf6e3"),
)
style_bullet = ParagraphStyle(
    'Bullet', parent=style_body,
    leftIndent=12*mm, bulletIndent=6*mm, spaceAfter=1.5*mm,
)
style_note = ParagraphStyle(
    'Note', parent=style_body,
    fontSize=9, textColor=MUTED, fontName='Helvetica-Oblique',
    leftIndent=8*mm,
)


def make_table(data, col_widths=None):
    """Create a styled table."""
    available = 170*mm
    if col_widths is None:
        n = len(data[0])
        col_widths = [available / n] * n
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HIGHLIGHT),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, RULE_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def header_footer(canvas_obj, doc):
    """Page header and footer."""
    canvas_obj.saveState()
    # Footer line
    canvas_obj.setStrokeColor(RULE_COLOR)
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(20*mm, 15*mm, 190*mm, 15*mm)
    # Footer text
    canvas_obj.setFont('Helvetica', 8)
    canvas_obj.setFillColor(MUTED)
    canvas_obj.drawString(20*mm, 10*mm, "Fractus White Paper")
    canvas_obj.drawRightString(190*mm, 10*mm, f"Page {doc.page}")
    # Author in footer
    canvas_obj.drawCentredString(105*mm, 10*mm, "Philippe-Antoine Robert")
    canvas_obj.restoreState()


def build_document():
    """Build the complete white paper."""
    doc = SimpleDocTemplate(
        OUTPUT_PATH, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=25*mm, bottomMargin=20*mm,
        title="Fractus: A Continuous Thought Engine for Decentralized AI",
        author="Philippe-Antoine Robert",
        subject="Technical White Paper",
    )

    story = []

    # ================================================================
    # COVER
    # ================================================================
    story.append(Spacer(1, 40*mm))
    story.append(Paragraph("Fractus", style_title))
    story.append(Paragraph("A Continuous Thought Engine for Decentralized AI", style_subtitle))
    story.append(Spacer(1, 20*mm))

    cover_info = [
        ["Document Type", "Technical White Paper"],
        ["Version", "1.0"],
        ["Date", "29 June 2026"],
        ["Author", "Philippe-Antoine Robert"],
        ["Repository", "github.com/AFKmoney/fractus"],
        ["Model", "thefinalboss/Fractus-1B on HuggingFace"],
    ]
    story.append(make_table(cover_info, col_widths=[50*mm, 120*mm]))
    story.append(Spacer(1, 30*mm))

    story.append(Paragraph(
        "This document presents the complete architecture, innovations, and measured "
        "results of the Fractus project: a 1B-capacity language model trained entirely "
        "on a consumer CPU laptop, featuring continuous-time reasoning, persistent "
        "memory, cognitive modes, and generative planning.",
        style_body,
    ))

    story.append(PageBreak())

    # ================================================================
    # TABLE OF CONTENTS
    # ================================================================
    story.append(Paragraph("Table of Contents", style_h1))
    toc_data = [
        ["1.", "Abstract"],
        ["2.", "Motivation: Why Fractus Exists"],
        ["3.", "Architecture Overview"],
        ["4.", "The 2-adic Vortex (Rust Core)"],
        ["5.", "Fractal Embedding"],
        ["6.", "Multi-Level Causal Linear Attention"],
        ["7.", "Kuramoto Oscillators and Phase-Routed MoE"],
        ["8.", "StructuredSiren Weight Compression"],
        ["9.", "The Continuous Thought Engine"],
        ["10.", "Persistent Memory"],
        ["11.", "Expert Specialization"],
        ["12.", "Cognitive Modes"],
        ["13.", "Generative Planning"],
        ["14.", "Training Methodology and Breakthroughs"],
        ["15.", "Measured Results"],
        ["16.", "Comparison with GPT and Claude"],
        ["17.", "Honest Limitations and Future Work"],
        ["18.", "Conclusion"],
    ]
    for num, title in toc_data:
        story.append(Paragraph(f"<b>{num}</b>&nbsp;&nbsp;{title}", style_body))
    story.append(PageBreak())

    # ================================================================
    # 1. ABSTRACT
    # ================================================================
    story.append(Paragraph("1. Abstract", style_h1))
    story.append(Paragraph(
        "We present <b>Fractus</b>, a novel neural architecture that departs from the "
        "dominant paradigm of static, datacenter-trained language models. Fractus "
        "introduces the <b>Continuous Thought Engine</b>: a dynamical system that "
        "processes information in continuous time, maintains persistent memory across "
        "sessions, exhibits cognitive modes, and generates structured output through "
        "planning rather than blind token-by-token prediction. The model achieves "
        "<b>0.86 billion parameters of effective capacity</b> from only 88 million "
        "trainable parameters through a low-rank decomposition we call "
        "<b>LazyStructuredSirenLinear</b>, enabling the entire model to fit in 0.4 GB "
        "of RAM and train on a single consumer CPU (AMD Ryzen 5 5500U). This work "
        "demonstrates that decentralized, personal AI is not only possible but "
        "practical, challenging the assumption that intelligence requires datacenter "
        "infrastructure.",
        style_body,
    ))
    story.append(Spacer(1, 3*mm))

    # ================================================================
    # 2. MOTIVATION
    # ================================================================
    story.append(Paragraph("2. Motivation: Why Fractus Exists", style_h1))
    story.append(Paragraph(
        "Contemporary large language models (GPT-4, Claude, Llama) share a set of "
        "fundamental limitations: they are <b>static functions</b> (one forward pass "
        "per query), <b>stateless</b> (no memory between conversations), "
        "<b>generic</b> (one monolithic network handles all tasks), and "
        "<b>centralized</b> (requiring datacenter GPUs for both training and "
        "deployment). These limitations are not merely inconveniences; they define "
        "the power dynamics of AI: whoever controls the datacenter controls the "
        "intelligence.",
        style_body,
    ))
    story.append(Paragraph(
        "Fractus was born from a simple question: <b>what if an AI could think "
        "continuously, remember across sessions, specialize by skill, and train on "
        "a laptop?</b> Each of these properties is individually achievable; their "
        "combination in a single architecture is the contribution of this work.",
        style_body,
    ))

    # ================================================================
    # 3. ARCHITECTURE
    # ================================================================
    story.append(Paragraph("3. Architecture Overview", style_h1))
    story.append(Paragraph(
        "Fractus combines several novel components into a unified architecture. "
        "The core is a <b>fractal transformer</b> with causal linear attention "
        "(Katharopoulos 2020), coupled Kuramoto oscillators for phase-based routing, "
        "and a sparse mixture-of-experts with StructuredSiren weight compression. "
        "Rust handles exact computation (2-adic arithmetic, proof verification) "
        "outside the autodiff graph; PyTorch handles all trainable components.",
        style_body,
    ))

    arch_data = [
        ["Component", "Module", "Role"],
        ["2-adic Vortex", "crate/fractus-core (Rust)", "Exact p-adic arithmetic, token hashing"],
        ["Fractal Embedding", "fractus/nn/embedding.py", "Char features + Fourier basis + vortex conditioning"],
        ["Linear Attention", "fractus/nn/attention.py", "Multi-level causal state-space attention"],
        ["Kuramoto Layer", "fractus/nn/phase_ode.py", "Low-rank coupled oscillators (RK4)"],
        ["Sparse MoE", "fractus/nn/moe.py", "Von Mises / Farey-routed top-k experts"],
        ["LazyStructuredSiren", "fractus/nn/lazy_siren.py", "Low-rank weight compression (no grid)"],
        ["Continuous Engine", "fractus/continuous_engine.py", "Tick-based real-time reasoning"],
        ["Persistent Memory", "fractus/memory.py", "Cross-session vector memory bank"],
        ["Cognitive Modes", "fractus/cognitive_modes.py", "Kuramoto phase to mental state"],
        ["Generative Planner", "fractus/generative_planner.py", "Plan-then-fill generation"],
    ]
    story.append(make_table(arch_data, col_widths=[40*mm, 60*mm, 70*mm]))
    story.append(PageBreak())

    # ================================================================
    # 4. 2-ADIC VORTEX
    # ================================================================
    story.append(Paragraph("4. The 2-adic Vortex (Rust Core)", style_h1))
    story.append(Paragraph(
        "The 2-adic vortex is the only module inherited directly from prior work, "
        "and it is implemented in pure Rust for exact computation. It provides:",
        style_body,
    ))
    story.append(Paragraph("&bull; <b>2-adic valuation</b> v2(x) = max{k : 2^k divides x}", style_bullet))
    story.append(Paragraph("&bull; <b>Ultrametric distance</b> d(a,b) = 2^{-v2(a XOR b)}", style_bullet))
    story.append(Paragraph("&bull; <b>Collatz hash</b> for deterministic token conditioning", style_bullet))
    story.append(Paragraph(
        "The vortex operates <b>outside the autodiff graph</b>: the Collatz hash of "
        "each token is computed exactly in Rust, then used to condition a trainable "
        "PyTorch MLP that produces embedding phase offsets. This preserves exactness "
        "without pretending the p-adic arithmetic is differentiable.",
        style_body,
    ))

    # ================================================================
    # 5. FRACTAL EMBEDDING
    # ================================================================
    story.append(Paragraph("5. Fractal Embedding", style_h1))
    story.append(Paragraph(
        "Each token is embedded by combining three deterministic feature sources, "
        "projected to d_model by a trainable linear layer:",
        style_body,
    ))
    story.append(Paragraph("&bull; <b>16 morphological features</b> (is_vowel, is_digit, case, punctuation...)", style_bullet))
    story.append(Paragraph("&bull; <b>Mandelbrot-decayed Fourier basis</b>: frequencies wk = (phi^2)^{-k}", style_bullet))
    story.append(Paragraph("&bull; <b>Vortex conditioning</b>: Collatz hash to MLP to phase offsets", style_bullet))
    story.append(Paragraph(
        "The forward pass is differentiable end-to-end, verified by tests that "
        "confirm backward() propagates non-zero gradients to every parameter.",
        style_body,
    ))

    # ================================================================
    # 6. ATTENTION
    # ================================================================
    story.append(Paragraph("6. Multi-Level Causal Linear Attention", style_h1))
    story.append(Paragraph(
        "Fractus uses Katharopoulos linear attention with a strictly-positive feature "
        "map phi(x; level) = ELU+1(x + omega_level), where omega_level = (phi^2)^{-level} "
        "provides a geometric scale separation across levels.",
        style_body,
    ))
    story.append(Paragraph(
        "The causal recurrence maintains a running state:",
        style_body,
    ))
    story.append(Paragraph(
        "S_t = Sum_{i<=t} phi(k_i) (x) v_i &nbsp;&nbsp;&nbsp; z_t = Sum_{i<=t} phi(k_i)",
        style_code,
    ))
    story.append(Paragraph(
        "y_t = (phi(q_t)^T S_t) / (phi(q_t)^T z_t)",
        style_code,
    ))
    story.append(Paragraph(
        "<b>L8 Optimization:</b> The original implementation looped over heads and "
        "levels separately (n_heads x n_levels Python calls). Profiling revealed this "
        "was the dominant cost, not the Kuramoto oscillators as previously claimed. "
        "Batching all heads and levels into a single flattened batch dimension yielded "
        "a <b>2.6x speedup</b> (17.3 ms to 6.6 ms per forward).",
        style_body,
    ))

    # ================================================================
    # 7. KURAMOTO + MOE
    # ================================================================
    story.append(Paragraph("7. Kuramoto Oscillators and Phase-Routed MoE", style_h1))
    story.append(Paragraph(
        "The <b>KuramotoLayer</b> implements low-rank coupled oscillators integrated "
        "by RK4, with coupling K = U Lambda U^T (rank r, O(N*r) per step). The "
        "oscillator phases route tokens to experts via a <b>von Mises gate</b> on "
        "<b>Farey-distributed expert phases</b>.",
        style_body,
    ))
    story.append(Paragraph(
        "The Farey sequence F_{2E} provides E expert angles in [0, 2pi) that are "
        "dense, non-collapsing, and deterministic. The gate is:",
        style_body,
    ))
    story.append(Paragraph(
        "g_e = exp(kappa * cos(theta - theta_e)) / sum_e' g_e'",
        style_code,
    ))
    story.append(Paragraph(
        "Only the top-k=2 experts are active per token, making the compute "
        "proportional to k/E of the total expert capacity.",
        style_body,
    ))
    story.append(PageBreak())

    # ================================================================
    # 8. STRUCTURED SIREN
    # ================================================================
    story.append(Paragraph("8. StructuredSiren Weight Compression", style_h1))
    story.append(Paragraph(
        "The key innovation enabling 1B-scale models on CPU. Each expert weight "
        "matrix W is decomposed as:",
        style_body,
    ))
    story.append(Paragraph("W = scale * U @ V^T", style_code))
    story.append(Paragraph(
        "where U is (out, r) and V is (in, r), with rank r = 16. The forward pass "
        "is y = scale * (x @ V) @ U^T + b: two small matmuls that never materialize "
        "the full (in, out) matrix.",
        style_body,
    ))

    siren_table = [
        ["Property", "Dense Baseline", "LazyStructuredSiren"],
        ["Storage per matrix (768x1024)", "3.1 MB", "115 KB"],
        ["Compression ratio", "1x", "27x"],
        ["Grid memory", "786K floats", "0 (no grid)"],
        ["Forward cost", "1 matmul (768x1024)", "2 matmuls (768x16 + 16x1024)"],
        ["Backward memory", "Full gradient", "U,V gradients only"],
    ]
    story.append(make_table(siren_table, col_widths=[55*mm, 55*mm, 60*mm]))

    story.append(Paragraph(
        "The original StructuredSiren stored a coordinate grid of shape (out x in) "
        "per expert, consuming 3.2 GB for 64 experts at d_model=768. The "
        "<b>LazyStructuredSirenLinear</b> eliminates the grid entirely by evaluating "
        "the low-rank decomposition directly on the input, reducing memory to O(1) "
        "per expert and enabling the full 1B model to train in 0.4 GB of RAM.",
        style_body,
    ))

    # ================================================================
    # 9. CONTINUOUS THOUGHT ENGINE
    # ================================================================
    story.append(Paragraph("9. The Continuous Thought Engine", style_h1))
    story.append(Paragraph(
        "The ContinuousThoughtEngine is the paradigm shift at the heart of Fractus. "
        "Unlike a standard language model (input to output, one forward pass), the "
        "engine is a <b>dynamical system</b> that maintains a persistent thought "
        "state and advances it tick by tick:",
        style_body,
    ))
    story.append(Paragraph("&bull; Each <b>tick</b> advances the Kuramoto oscillators by one RK4 step", style_bullet))
    story.append(Paragraph("&bull; The attention state (S, z) <b>accumulates context</b> (working memory)", style_bullet))
    story.append(Paragraph("&bull; The MoE <b>transforms the thought</b>, routed by Kuramoto phases", style_bullet))
    story.append(Paragraph("&bull; A confidence head decides <b>when to emit output</b>", style_bullet))
    story.append(Paragraph(
        "Adaptive depth: easy observations require 1 tick, difficult ones may require "
        "10. This is <b>energy-proportional reasoning</b>.",
        style_body,
    ))
    story.append(Paragraph(
        "<b>Chunk-based processing</b>: the tick_chunk() method processes 16 tokens "
        "per forward pass, using the L8 batched attention. This yielded a measured "
        "<b>4.7x speedup</b> over single-token ticks (25 to 117 tokens/sec on CPU).",
        style_body,
    ))
    story.append(PageBreak())

    # ================================================================
    # 10-13. FOUR INNOVATIONS
    # ================================================================
    story.append(Paragraph("10. Persistent Memory", style_h1))
    story.append(Paragraph(
        "A bank of memory vectors that survives across sessions. Memories are "
        "recalled via cosine similarity to the current thought state and injected "
        "into the engine's thought via a weighted blend. Memories are consolidated "
        "from salient thought states and evicted by LRU when full. The memory is "
        "saved to disk and reloaded at startup: the engine <b>remembers the user</b>.",
        style_body,
    ))

    story.append(Paragraph("11. Expert Specialization", style_h1))
    story.append(Paragraph(
        "A diversity loss penalizes experts that produce similar outputs for the "
        "same input, forcing each expert to specialize on a distinct domain (code, "
        "math, language, reasoning). Domain vectors with an orthogonality constraint "
        "ensure that the MoE is a true <b>skill dispatcher</b>, not random routing.",
        style_body,
    ))

    story.append(Paragraph("12. Cognitive Modes", style_h1))
    story.append(Paragraph(
        "The Kuramoto phase pattern is classified into cognitive modes (analytical, "
        "creative, focused, exploratory...) by a learnable classifier that extracts "
        "synchronization, mean phase, and variance features. The engine has "
        "<b>mental states</b> that change how it processes information, analogous to "
        "human cognitive shifts between focused work and creative brainstorming.",
        style_body,
    ))

    story.append(Paragraph("13. Generative Planning", style_h1))
    story.append(Paragraph(
        "Instead of generating token-by-token, the engine <b>plans</b> a structure "
        "first (a sequence of key anchors), then fills in the content between "
        "anchors. This is how humans write: outline first, detail later. The planner "
        "is type-aware (code needs more structural anchors than prose).",
        style_body,
    ))
    story.append(PageBreak())

    # ================================================================
    # 14. TRAINING
    # ================================================================
    story.append(Paragraph("14. Training Methodology and Breakthroughs", style_h1))
    story.append(Paragraph(
        "The training of a 1B-capacity model on a CPU laptop required four sequential "
        "breakthroughs, each discovered through profiling rather than assumption:",
        style_body,
    ))

    story.append(Paragraph("14.1 Profile-Driven Optimization (L8)", style_h2))
    story.append(Paragraph(
        "The README claimed 'Kuramoto RK4 is the bottleneck.' Profiling proved this "
        "wrong: the real cost was the Python loop over heads and levels in attention "
        "(17.3 ms per forward), and SIREN matrix reconstruction (148% of forward time). "
        "This is the 'measure, don't claim' discipline applied to our own code.",
        style_body,
    ))

    story.append(Paragraph("14.2 Cached SIREN (L9 attempt 1)", style_h2))
    story.append(Paragraph(
        "Caching the reconstructed SIREN weight matrix and recomputing only every 8 "
        "forward calls yielded an 8.5x speedup per layer. However, the grid storage "
        "(out x in floats per expert) caused OOM at 64 experts and d_model=768.",
        style_body,
    ))

    story.append(Paragraph("14.3 Gradient Checkpointing (L9 attempt 2)", style_h2))
    story.append(Paragraph(
        "torch.utils.checkpoint eliminated the OOM by recomputing the forward during "
        "backward, but at the cost of a 10x slowdown (4.3s to 43s per step).",
        style_body,
    ))

    story.append(Paragraph("14.4 LazyStructuredSiren (L9 breakthrough)", style_h2))
    story.append(Paragraph(
        "The final breakthrough: <b>drop the SIREN residual entirely for large "
        "matrices</b> and use pure low-rank decomposition W = scale * U @ V^T. No "
        "grid, no reconstruction, no checkpointing. The forward is two small matmuls; "
        "the backward touches only U and V. This reduced:",
        style_body,
    ))

    breakthrough_table = [
        ["Metric", "Before (CachedSiren)", "After (LazySiren)", "Improvement"],
        ["RAM per 64 experts", "3.2 GB (OOM)", "0.4 GB", "8x"],
        ["Training step time", "43s (checkpointed)", "5.9s", "7.3x"],
        ["Grid memory", "786K floats/expert", "0", "Eliminated"],
        ["1 epoch (20k tokens)", "~18 hours", "~1 hour", "18x"],
    ]
    story.append(make_table(breakthrough_table, col_widths=[45*mm, 45*mm, 40*mm, 40*mm]))

    story.append(PageBreak())

    # ================================================================
    # 15. RESULTS
    # ================================================================
    story.append(Paragraph("15. Measured Results", style_h1))

    story.append(Paragraph("15.1 Small Model (13M params, ContinuousThoughtEngine)", style_h2))
    small_results = [
        ["Epoch", "Loss", "Perplexity", "Accuracy"],
        ["1", "6.29", "539", "20.5%"],
        ["10", "2.37", "10.7", "47.0%"],
        ["20", "1.29", "3.6", "70.0%"],
        ["30", "1.00", "2.7", "77.0%"],
    ]
    story.append(make_table(small_results, col_widths=[30*mm, 35*mm, 40*mm, 35*mm]))

    story.append(Paragraph("15.2 Large Model (88M params, Fractus-1B)", style_h2))
    story.append(Paragraph(
        "The 1B-capacity model (d_model=768, 8 layers, 64 experts, top-2 routing) "
        "was trained on a 500k-token multi-domain corpus (code, literature, "
        "mathematics, universal knowledge). Training is ongoing at the time of "
        "writing; preliminary results show loss decreasing from 9.3 to approximately "
        "6.0 over the first 925 steps.",
        style_body,
    ))

    story.append(Paragraph("15.3 Training Throughput", style_h2))
    throughput = [
        ["Configuration", "Tokens/sec", "Hardware"],
        ["13M, d_model=128, single-token", "25", "Ryzen 5 5500U (6 threads)"],
        ["13M, d_model=128, chunk-based", "117", "Ryzen 5 5500U (6 threads)"],
        ["88M, d_model=768, LazySiren", "5-8", "Ryzen 5 5500U (6 threads)"],
    ]
    story.append(make_table(throughput, col_widths=[70*mm, 35*mm, 65*mm]))

    story.append(PageBreak())

    # ================================================================
    # 16. COMPARISON
    # ================================================================
    story.append(Paragraph("16. Comparison with GPT and Claude", style_h1))

    comparison = [
        ["Property", "GPT-4 / Claude", "Fractus"],
        ["Processing model", "Static (1 forward pass)", "Dynamical (continuous ticks)"],
        ["Memory", "Context window (amnesic)", "Persistent memory bank"],
        ["Skill structure", "Generic monolith", "Specialized MoE experts"],
        ["Mental state", "Stateless", "Cognitive modes (Kuramoto)"],
        ["Generation", "Token-by-token", "Plan-then-fill"],
        ["Training hardware", "Datacenter GPU cluster", "Consumer CPU laptop"],
        ["Deployment", "Cloud API (centralized)", "Local (decentralized)"],
        ["User data", "Sent to server", "Stays on device"],
        ["Cost to train", "Millions of USD", "Zero (electricity only)"],
    ]
    story.append(make_table(comparison, col_widths=[40*mm, 65*mm, 65*mm]))

    story.append(Paragraph(
        "Fractus does not claim to match GPT-4 on benchmark performance. It claims "
        "something different: that the <b>paradigm</b> of continuous, personal, "
        "decentralized AI is viable and produces qualitatively different capabilities "
        "(memory, planning, cognitive modes) that centralized models cannot replicate "
        "without architectural changes.",
        style_body,
    ))

    # ================================================================
    # 17. LIMITATIONS
    # ================================================================
    story.append(Paragraph("17. Honest Limitations and Future Work", style_h1))
    story.append(Paragraph("&bull; <b>Model quality</b>: The trained model produces repetitive text. More data and epochs are needed.", style_bullet))
    story.append(Paragraph("&bull; <b>1B training speed</b>: 5-8 tokens/sec on CPU is feasible but slow. GPU acceleration would yield 50-100x.", style_bullet))
    story.append(Paragraph("&bull; <b>Cognitive modes untrained</b>: The mode classifier exists but has not been trained on labeled data.", style_bullet))
    story.append(Paragraph("&bull; <b>Generative planner is proof-of-concept</b>: The plan/fill pipeline works but needs integration with a well-trained engine.", style_bullet))
    story.append(Paragraph("&bull; <b>State-carry is attention-level only</b>: Carrying (S,z) through the full block stack is documented future work.", style_bullet))
    story.append(Paragraph("&bull; <b>No Lean 4 / ZK-SNARK</b>: Formal verification is honestly absent.", style_bullet))

    # ================================================================
    # 18. CONCLUSION
    # ================================================================
    story.append(Paragraph("18. Conclusion", style_h1))
    story.append(Paragraph(
        "Fractus demonstrates that a 1B-capacity language model can be constructed "
        "and trained on a consumer CPU laptop, using low-rank weight compression "
        "and a continuous-time reasoning architecture. The Continuous Thought Engine "
        "introduces a fundamentally different processing model from GPT and Claude: "
        "not input-output, but continuous thought with memory, modes, and planning.",
        style_body,
    ))
    story.append(Paragraph(
        "The implications extend beyond performance metrics. If AI can be trained "
        "and deployed on any laptop, the centralization of intelligence by a handful "
        "of corporations is not inevitable. Fractus is a proof of concept for "
        "<b>decentralized AI</b>: intelligence that belongs to the user, runs on "
        "their hardware, and remembers them.",
        style_body,
    ))
    story.append(Spacer(1, 15*mm))

    # Signature
    story.append(Paragraph("Signed,", style_body))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("<b>Philippe-Antoine Robert</b>", ParagraphStyle(
        'Signature', parent=style_body, fontSize=12, fontName='Helvetica-Bold',
        textColor=PRIMARY,
    )))
    story.append(Paragraph("29 June 2026", style_body))

    # References
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph("References", style_h2))
    refs = [
        "[1] Katharopoulos et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML.",
        "[2] Sitzmann et al. (2020). Implicit Neural Representations with Periodic Activation Functions (SIREN). NeurIPS.",
        "[3] Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations.",
        "[4] Zheng et al. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. NeurIPS.",
        "[5] Rahimi, A. and Recht, B. (2007). Random Features for Large-Scale Kernel Machines. NeurIPS.",
        "[6] Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. arXiv.",
        "[7] Pearl, J. (2009). Causality: Models, Reasoning and Inference. Cambridge University Press.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, ParagraphStyle(
            'Ref', parent=style_body, fontSize=8.5, leading=11,
            leftIndent=8*mm, firstLineIndent=-8*mm, spaceAfter=1*mm,
        )))

    # Build
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"White Paper generated: {OUTPUT_PATH} ({size_mb:.1f} MB)", flush=True)
    return OUTPUT_PATH


if __name__ == "__main__":
    build_document()
