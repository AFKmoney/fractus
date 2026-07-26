"""Fractus Growth Module — add experts and expand rank without retraining.

Two mechanisms for continuous checkpoint growth:

1. add_experts(model, n_new, data, device)
   Creates new MoE experts, pre-trains them via EDT Phase 1, extends the router.
   Old experts are untouched. The brain grows wider.

2. expand_rank(model, target_rank)
   Increases the Siren rank of ALL experts. Old rank columns preserved.
   New columns initialized to zero (contribute nothing until trained).
   The brain grows deeper.

3. save_grown_model(model, path, growth_log)
   Saves the model with a growth log documenting what was added.

Usage:
    from fractus.growth import FractusGrowth

    growth = FractusGrowth(model, tok, device)

    # Add 128 experts trained on Rust code
    growth.add_experts(n_new=128, data=rust_tokens, domain="rust")

    # Expand rank 64 → 128 for deeper capacity
    growth.expand_rank(target_rank=128)

    # Save the grown model
    growth.save("checkpoints/fractus_grown.pt")
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FractusGrowth:
    """Continuous growth manager for Fractus models.

    Allows adding experts and expanding rank without retraining the
    existing model. This is what makes Fractus a living brain.
    """

    def __init__(self, model, tokenizer=None, device="cpu"):
        self.model = model
        self.tok = tokenizer
        self.device = torch.device(device)
        self.growth_log = []

        # Record initial state.
        self.initial_experts = model.blocks[0].moe.n_experts if hasattr(model, 'blocks') else 0
        self.initial_rank = (
            model.blocks[0].moe.experts_w1[0].rank
            if hasattr(model, 'blocks') and hasattr(model.blocks[0].moe.experts_w1[0], 'rank')
            else 0
        )
        self.n_layers = len(model.blocks) if hasattr(model, 'blocks') else 0

    # ===================================================================
    # MECHANISM 1: ADD EXPERTS
    # ===================================================================

    def add_experts(self, n_new=128, data=None, domain="unknown",
                    steps_per_expert=2000, batch_size=64, lr=1e-3):
        """Add new MoE experts to every layer and pre-train them.

        The new experts are:
        1. Created with random LazyStructuredSiren weights
        2. Pre-trained on the provided data (or synthetic if no data)
        3. Added to the model's expert lists
        4. The router's Farey phases are extended

        OLD EXPERTS ARE NEVER TOUCHED.

        Args:
            n_new: number of new experts to add per layer.
            data: token IDs (1D tensor) for pre-training, or None for synthetic.
            domain: name of the domain (for logging).
            steps_per_expert: EDT Phase 1 steps per expert.
            batch_size: batch size for pre-training.
            lr: learning rate.
        """
        from fractus.nn.lazy_siren import LazyStructuredSirenLinear

        print(f"\n{'='*60}", flush=True)
        print(f"GROWTH: Adding {n_new} experts (domain: {domain})", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()

        # Generate hidden states for pre-training.
        hidden_states = self._generate_hidden_states(data, n_samples=5000)

        for layer_idx in range(self.n_layers):
            moe = self.model.blocks[layer_idx].moe
            old_count = moe.n_experts
            d_model = moe.d_model
            d_ff = moe.d_ff
            rank = moe.experts_w1[0].rank if hasattr(moe.experts_w1[0], 'rank') else 16

            # Create new experts.
            for i in range(n_new):
                new_w1 = LazyStructuredSirenLinear(d_model, d_ff, rank=rank)
                new_w2 = LazyStructuredSirenLinear(d_ff, d_model, rank=rank)
                moe.experts_w1.append(new_w1)
                moe.experts_w2.append(new_w2)

            # Update expert count.
            moe.n_experts = old_count + n_new

            # Pre-train the new experts (EDT Phase 1).
            for i in range(n_new):
                expert_idx = old_count + i
                w1 = moe.experts_w1[expert_idx]
                w2 = moe.experts_w2[expert_idx]
                self._pretrain_expert(w1, w2, hidden_states,
                                      steps=steps_per_expert,
                                      batch_size=batch_size, lr=lr)

            print(f"  Layer {layer_idx+1}/{self.n_layers}: "
                  f"{old_count} → {old_count + n_new} experts", flush=True)

        # Extend the Farey phases for the router.
        self._extend_farey_phases(n_new)

        elapsed = time.time() - t0
        total_new = n_new * self.n_layers
        self.growth_log.append({
            "type": "add_experts",
            "n_new": n_new,
            "domain": domain,
            "total_new_experts": total_new,
            "time_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nGrowth complete: +{total_new} experts in {elapsed:.0f}s", flush=True)
        print(f"  Total params: {n_params/1e6:.0f}M", flush=True)
        return self.model

    def _generate_hidden_states(self, data, n_samples=5000):
        """Generate hidden states from the embedding for expert pre-training."""
        d_model = self.model.d_model
        seq_len = 16

        if data is not None and self.tok is not None and len(data) > 100:
            # Real hidden states from embedding.
            states = []
            self.model.eval()
            with torch.no_grad():
                for _ in range(n_samples // 64):
                    idx = torch.randint(0, len(data) - seq_len - 1, (64,))
                    tokens = torch.stack([data[i:i+seq_len] for i in idx]).to(self.device)
                    h = self.model.embed(tokens)
                    states.append(h.cpu())
            return torch.cat(states, dim=0)
        else:
            # Synthetic fallback.
            print("  (using synthetic hidden states — provide data for real training)",
                  flush=True)
            return torch.randn(n_samples, seq_len, d_model)

    def _pretrain_expert(self, w1, w2, hidden_states, steps=2000,
                         batch_size=64, lr=1e-3):
        """Pre-train one expert on hidden states (EDT Phase 1)."""
        params = list(w1.parameters()) + list(w2.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

        n_samples = len(hidden_states)
        for _ in range(steps):
            idx = torch.randint(0, n_samples - 1, (batch_size,))
            h_in = hidden_states[idx].to(self.device)
            h_target = hidden_states[idx + 1].to(self.device)

            opt.zero_grad()
            h1 = w1(h_in)
            h1_act = F.gelu(h1)
            h_out = w2(h1_act)
            loss = F.mse_loss(h_out, h_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

    def _extend_farey_phases(self, n_new):
        """Extend the Farey phase distribution for new experts."""
        from fractus.nn.farey import expert_phases

        for layer_idx in range(self.n_layers):
            moe = self.model.blocks[layer_idx].moe
            old_count = moe.n_experts - n_new
            new_count = moe.n_experts

            # Regenerate phases for the new total count.
            new_phases = expert_phases(new_count)
            moe.expert_phases = torch.tensor(new_phases, dtype=torch.float32,
                                              device=moe.expert_phases.device)

    # ===================================================================
    # MECHANISM 2: RANK EXPANSION
    # ===================================================================

    def expand_rank(self, target_rank=128):
        """Expand the Siren rank of ALL experts.

        Old rank columns are preserved (knowledge kept).
        New rank columns are initialized to zero (learn during fine-tuning).

        Args:
            target_rank: the new rank (must be > current rank).
        """
        print(f"\n{'='*60}", flush=True)
        print(f"GROWTH: Expanding rank → {target_rank}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        expanded = 0

        for layer_idx in range(self.n_layers):
            moe = self.model.blocks[layer_idx].moe

            for expert_list in [moe.experts_w1, moe.experts_w2]:
                for expert in expert_list:
                    if not hasattr(expert, 'rank'):
                        continue
                    current_rank = expert.rank
                    if current_rank >= target_rank:
                        continue

                    # Expand U: (out, current_rank) → (out, target_rank)
                    old_U = expert.U.data  # (out, current_rank)
                    old_V = expert.V.data  # (in, current_rank)

                    new_U = torch.zeros(old_U.shape[0], target_rank,
                                        dtype=old_U.dtype, device=old_U.device)
                    new_V = torch.zeros(old_V.shape[0], target_rank,
                                        dtype=old_V.dtype, device=old_V.device)

                    # Copy old values (preserve knowledge).
                    new_U[:, :current_rank] = old_U
                    new_V[:, :current_rank] = old_V
                    # New columns stay zero (contribute nothing until trained).

                    # Replace parameters.
                    expert.U = nn.Parameter(new_U)
                    expert.V = nn.Parameter(new_V)
                    expert.rank = target_rank
                    expanded += 1

            print(f"  Layer {layer_idx+1}/{self.n_layers}: rank expanded", flush=True)

        elapsed = time.time() - t0
        n_params = sum(p.numel() for p in self.model.parameters())

        self.growth_log.append({
            "type": "expand_rank",
            "target_rank": target_rank,
            "experts_expanded": expanded,
            "time_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        print(f"\nRank expansion complete: {expanded} experts in {elapsed:.0f}s", flush=True)
        print(f"  Total params: {n_params/1e6:.0f}M", flush=True)
        return self.model

    # ===================================================================
    # MECHANISM 3: SAVE GROWN MODEL
    # ===================================================================

    def save(self, path):
        """Save the grown model with growth log."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        n_params = sum(p.numel() for p in self.model.parameters())
        torch.save({
            "model_state": self.model.state_dict(),
            "growth_log": self.growth_log,
            "n_params": n_params,
            "initial_experts": self.initial_experts,
            "initial_rank": self.initial_rank,
            "current_experts": self.model.blocks[0].moe.n_experts if hasattr(self.model, 'blocks') else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, path)

        size_mb = os.path.getsize(path) / 1e6
        print(f"Saved grown model: {path} ({size_mb:.0f}MB)", flush=True)
        print(f"  Params: {n_params/1e6:.0f}M", flush=True)
        print(f"  Growth events: {len(self.growth_log)}", flush=True)
        for event in self.growth_log:
            if event["type"] == "add_experts":
                print(f"    +{event['total_new_experts']} experts ({event['domain']})", flush=True)
            elif event["type"] == "expand_rank":
                print(f"    rank → {event['target_rank']}", flush=True)

    # ===================================================================
    # UTILITY: Print current state
    # ===================================================================

    def status(self):
        """Print current model status."""
        n_params = sum(p.numel() for p in self.model.parameters())
        n_experts = self.model.blocks[0].moe.n_experts if hasattr(self.model, 'blocks') else 0
        rank = (
            self.model.blocks[0].moe.experts_w1[0].rank
            if hasattr(self.model, 'blocks') and hasattr(self.model.blocks[0].moe.experts_w1[0], 'rank')
            else '?'
        )

        print(f"\nFractus Status:", flush=True)
        print(f"  Params:   {n_params/1e6:.0f}M ({n_params/1e9:.3f}B)", flush=True)
        print(f"  Experts:  {n_experts} per layer × {self.n_layers} layers = {n_experts * self.n_layers}", flush=True)
        print(f"  Rank:     {rank}", flush=True)
        print(f"  Growth:   {len(self.growth_log)} events", flush=True)
