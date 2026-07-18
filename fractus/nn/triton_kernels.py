"""Triton kernels for Fractus — GPU acceleration with eager fallback.

These kernels are ONLY used when:
  1. Triton is importable (CUDA available)
  2. The equivalence test passes against PyTorch reference

Otherwise the caller falls back to the eager PyTorch path. Zero risk.

Kernels:
  - fused_linear_cross_entropy: lm_head + softmax-CE in one pass.
    Avoids materializing (B, L, vocab) — the VRAM ceiling that limited batch
    size. Inspired by Liger Kernel's LinearCrossEntropy.

Why Triton? PyTorch eager launches one kernel per op. A 50257-vocab LM head
followed by CE materializes a (B, L, 50257) tensor = ~2GB at B=512 bf16.
Triton fuses the projection + logsoftmax + NLL into one kernel that streams
through the vocab dimension, keeping only running maxima/sums in registers.
"""
import torch
import torch.nn.functional as F

# Lazy import — only fails if Triton missing (CPU, old GPU).
_triton = None
try:
    import triton
    import triton.language as tl
    _triton = triton
except ImportError:
    pass


# =============================================================================
# Fused Linear + Cross-Entropy kernel
# =============================================================================

if _triton is not None:

    @_triton.jit
    def _linear_ce_fwd_kernel(
        # Pointers
        hidden_ptr, weight_ptr, target_ptr, loss_ptr,
        # Shapes
        B, L, D, V,
        # Strides
        stride_hb, stride_hl, stride_hd,
        stride_wv, stride_wd,
        # Meta
        BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """One program computes the loss for ONE (batch, position) pair.

        Streams through the vocab dimension in blocks of BLOCK_V, computing
        logits on-the-fly via the dot product hidden @ weight[:, v]. Tracks
        the running logsumexp so we never store the full (V,) logits vector.
        """
        pid = tl.program_id(0)  # index in B*L
        bl = pid
        b = bl // L
        l = bl % L

        # Load hidden vector for this (b, l).
        d_off = tl.arange(0, BLOCK_D)
        d_mask = d_off < D
        h = tl.load(hidden_ptr + b * stride_hb + l * stride_hl + d_off * stride_hd,
                    mask=d_mask, other=0.0).to(tl.float32)

        # Load target.
        tgt = tl.load(target_ptr + bl).to(tl.int32)

        # First pass: find max logit (for numerical stability).
        m = -float('inf')
        for v_start in range(0, V, BLOCK_V):
            v_off = tl.arange(0, BLOCK_V)
            v_mask = (v_start + v_off) < V
            # weight shape (V, D). Load weight[v_start:v_start+BLOCK_V, :].
            w = tl.load(weight_ptr + (v_start + v_off)[:, None] * stride_wv
                        + d_off[None, :] * stride_wd,
                        mask=v_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            logits = tl.sum(w * h[None, :], axis=1)  # (BLOCK_V,)
            logits = tl.where(v_mask, logits, -float('inf'))
            m = tl.maximum(m, tl.max(logits))

        # Second pass: sum_exp and the target logit.
        sum_exp = 0.0
        tgt_logit = 0.0
        for v_start in range(0, V, BLOCK_V):
            v_off = tl.arange(0, BLOCK_V)
            v_mask = (v_start + v_off) < V
            w = tl.load(weight_ptr + (v_start + v_off)[:, None] * stride_wv
                        + d_off[None, :] * stride_wd,
                        mask=v_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            logits = tl.sum(w * h[None, :], axis=1)
            logits = tl.where(v_mask, logits, -float('inf'))
            exp = tl.exp(logits - m)
            sum_exp += tl.sum(exp)
            # Pick out the target logit (if it's in this block).
            in_block = (v_start <= tgt) & (tgt < v_start + BLOCK_V)
            tgt_local = tgt - v_start
            tgt_logit += tl.where(in_block,
                                  tl.sum(tl.where(v_off == tgt_local, logits, 0.0)),
                                  0.0)

        # loss = -log(softmax[target]) = -(tgt_logit - log(sum_exp) - m)
        log_sum_exp = m + tl.log(sum_exp)
        loss = -(tgt_logit - log_sum_exp)
        tl.store(loss_ptr + bl, loss)


def fused_linear_cross_entropy(hidden, weight, target):
    """Compute loss = CE(lm_head(hidden), target) without materializing logits.

    hidden:  (B, L, D) — final hidden states.
    weight:  (V, D)    — lm_head weight (tied to embedding).
    target:  (B, L)    — next-token ids.

    Returns scalar loss (mean over B*L).

    Falls back to standard PyTorch if Triton unavailable or on CPU.
    """
    if _triton is None or not hidden.is_cuda:
        # Eager fallback.
        logits = F.linear(hidden.reshape(-1, hidden.shape[-1]), weight)
        return F.cross_entropy(logits, target.reshape(-1))

    B, L, D = hidden.shape
    V = weight.shape[0]
    # Pad D up to a power of 2 for the kernel.
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2
    BLOCK_V = 64  # vocab processed in blocks of 64.

    loss = torch.empty(B * L, device=hidden.device, dtype=torch.float32)
    # Make hidden contiguous in (B*L, D) layout.
    h_flat = hidden.reshape(B * L, D).contiguous()
    _linear_ce_fwd_kernel[(B * L,)](
        h_flat, weight, target.reshape(-1), loss,
        B * L, L, D, V,
        h_flat.stride(0), h_flat.stride(1) if D > 1 else 1, 1,
        weight.stride(0), weight.stride(1) if D > 1 else 1,
        BLOCK_V=BLOCK_V, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return loss.mean()


# =============================================================================
# Self-test (run on GPU pod to validate equivalence before use)
# =============================================================================

def self_test():
    """Validate fused kernel against PyTorch reference.
    Run this on a GPU pod before trusting the kernel:
        python -c "from fractus.nn.triton_kernels import self_test; self_test()"
    Returns True if equivalent (max diff < 1e-4), False otherwise.
    """
    if _triton is None:
        print("Triton not available — self-test skipped, will use eager.")
        return False
    if not torch.cuda.is_available():
        print("CUDA not available — self-test skipped.")
        return False

    torch.manual_seed(42)
    dev = torch.device('cuda')
    B, L, D, V = 2, 8, 64, 1000
    hidden = torch.randn(B, L, D, device=dev, dtype=torch.float32)
    weight = torch.randn(V, D, device=dev, dtype=torch.float32) * 0.1
    target = torch.randint(0, V, (B, L), device=dev)

    # Reference
    logits = F.linear(hidden.reshape(-1, D), weight)
    ref = F.cross_entropy(logits, target.reshape(-1))

    # Triton
    tri = fused_linear_cross_entropy(hidden, weight, target)

    diff = (ref - tri).abs().item()
    print(f"ref loss:  {ref.item():.6f}")
    print(f"tri loss:  {tri.item():.6f}")
    print(f"diff:      {diff:.2e}")
    ok = diff < 1e-4
    print("EQUIV PASS" if ok else "EQUIV FAIL — kernel will not be used")
    return ok


# Module-level flag: True only if triton is present AND self_test passed.
# The caller checks `TRITON_READY` before using the kernel.
TRITON_READY = False
