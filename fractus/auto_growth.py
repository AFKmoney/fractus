"""Fractus Self-Growth + Memory Management.

Fractus decides ITSELF when to:
- Add experts (when it encounters a new domain it can't handle)
- Expand rank (when existing experts plateau)
- Forget memories (when they're wrong, outdated, or harmful)
- Modify memories (correct mistakes, update facts)

This is the autonomous growth loop — the AGI layer.
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class SelfGrowthPolicy(nn.Module):
    """A small policy network that decides HOW Fractus should grow.

    Inputs: Fractus's current state (performance metrics, domain coverage,
    expert utilization, memory size).
    Outputs: growth actions (ADD_EXPERT, EXPAND_RANK, NOTHING).

    This is NOT trained by backprop — it uses a simple heuristic policy
    that can be refined over time. The goal is autonomy, not optimization.
    """

    ACTIONS = ["NOTHING", "ADD_EXPERT", "EXPAND_RANK", "FORGET", "MODIFY"]

    def __init__(self, n_domains=10):
        super().__init__()
        # Tiny network: 5 input features → 5 actions
        # Features: [loss_trend, expert_utilization_entropy, memory_size,
        #            domain_coverage, confidence_avg]
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, len(self.ACTIONS)),
        )

    def forward(self, features):
        """features: (5,) tensor → action logits (5,)"""
        return self.net(features)

    def decide(self, features):
        """Decide what to do based on current state.

        Args:
            features: dict with keys:
                - loss_trend: float (negative = improving, positive = degrading)
                - expert_utilization: float (0-1, entropy of expert selection)
                - memory_size: int (number of memories)
                - domain_coverage: float (0-1, how many domains are covered)
                - confidence_avg: float (0-1, average generation confidence)

        Returns:
            action: str (one of ACTIONS)
            reason: str (why this action was chosen)
        """
        f = torch.tensor([
            features.get("loss_trend", 0.0),
            features.get("expert_utilization", 0.5),
            min(features.get("memory_size", 0) / 1000.0, 1.0),
            features.get("domain_coverage", 0.5),
            features.get("confidence_avg", 0.5),
        ], dtype=torch.float32)

        # Heuristic policy (not learned — transparent and debuggable).
        loss_trend = features.get("loss_trend", 0.0)
        util = features.get("expert_utilization", 0.5)
        coverage = features.get("domain_coverage", 0.5)
        mem_size = features.get("memory_size", 0)

        # Decision tree.
        if loss_trend > 0.1 and coverage < 0.7:
            return "ADD_EXPERT", f"Loss degrading ({loss_trend:+.2f}) and domain coverage low ({coverage:.0%}) — need new specialists"

        if loss_trend > 0.1 and coverage >= 0.7:
            return "EXPAND_RANK", f"Loss degrading ({loss_trend:+.2f}) but domains covered — existing experts need more depth"

        if mem_size > 5000:
            return "FORGET", f"Memory large ({mem_size} entries) — consolidate and forget irrelevant entries"

        return "NOTHING", "Model is healthy — no growth needed"


class MemoryManager:
    """Manages Fractus's persistent memory — store, retrieve, forget, modify.

    This wraps the KnowledgeBase with higher-level operations:
    - forget(pattern): remove memories matching a pattern
    - modify(old, new): replace an old memory with a corrected version
    - consolidate(): merge similar memories, remove duplicates
    - importance_score(memory): how important is this memory?
    """

    def __init__(self, kb):
        self.kb = kb

    def forget(self, pattern: str = None, index: int = None,
               source: str = None, older_than_days: int = None):
        """Remove memories matching criteria.

        Args:
            pattern: remove memories containing this text (case-insensitive)
            index: remove the memory at this specific index
            source: remove all memories from this source
            older_than_days: remove memories older than N days

        Returns:
            Number of memories removed.
        """
        if not self.kb.chunks:
            return 0

        to_keep = []
        removed = 0

        for i, (text, src) in enumerate(zip(self.kb.chunks,
                                             self.kb.sources if self.kb.sources else [""] * len(self.kb.chunks))):
            keep = True

            if index is not None and i == index:
                keep = False

            if pattern and pattern.lower() in text.lower():
                keep = False

            if source and src == source:
                keep = False

            if keep:
                to_keep.append(i)

        # Remove in reverse order (don't mess up indices).
        to_remove = sorted([i for i in range(len(self.kb.chunks)) if i not in to_keep],
                          reverse=True)
        for i in to_remove:
            self.kb.chunks.pop(i)
            if i < len(self.kb.embeddings):
                self.kb.embeddings.pop(i)
            if i < len(self.kb.sources):
                self.kb.sources.pop(i)
            removed += 1

        return removed

    def modify(self, old_pattern: str, new_text: str, source: str = "correction"):
        """Replace memories containing old_pattern with new_text.

        Args:
            old_pattern: find memories containing this text
            new_text: the corrected text to replace them with
            source: source label for the correction

        Returns:
            Number of memories modified.
        """
        if not self.kb.chunks:
            return 0

        modified = 0
        for i, text in enumerate(self.kb.chunks):
            if old_pattern.lower() in text.lower():
                self.kb.chunks[i] = new_text
                if i < len(self.kb.sources):
                    self.kb.sources[i] = source
                modified += 1

        return modified

    def consolidate(self, similarity_threshold: float = 0.9):
        """Merge near-duplicate memories.

        Removes memories that are >90% similar to an earlier one.
        Keeps the longer version.

        Returns:
            Number of duplicates removed.
        """
        if len(self.kb.chunks) < 2 or not self.kb.embeddings:
            return 0

        removed = 0
        to_keep = list(range(len(self.kb.chunks)))

        bank = np.array(self.kb.embeddings, dtype=np.float32)
        norms = np.linalg.norm(bank, axis=1, keepdims=True)
        normalized = bank / (norms + 1e-10)

        for i in range(len(self.kb.chunks)):
            if i not in to_keep:
                continue
            for j in range(i + 1, len(self.kb.chunks)):
                if j not in to_keep:
                    continue
                sim = float(normalized[i] @ normalized[j])
                if sim > similarity_threshold:
                    # Keep the longer one.
                    if len(self.kb.chunks[j]) > len(self.kb.chunks[i]):
                        to_keep.remove(i)
                    else:
                        to_keep.remove(j)
                    removed += 1

        # Rebuild without duplicates.
        self.kb.chunks = [self.kb.chunks[i] for i in sorted(to_keep)]
        self.kb.embeddings = [self.kb.embeddings[i] for i in sorted(to_keep)
                              if i < len(self.kb.embeddings)]
        self.kb.sources = [self.kb.sources[i] for i in sorted(to_keep)
                           if self.kb.sources and i < len(self.kb.sources)]

        return removed

    def importance_score(self, text: str) -> float:
        """Estimate how important a memory is.

        Factors:
        - Length (longer = more information)
        - Specificity (contains proper nouns, numbers, code)
        - Recency (if timestamp available)

        Returns: 0.0 to 1.0
        """
        score = 0.0

        # Length factor.
        score += min(len(text) / 500.0, 0.3)

        # Specificity: contains numbers?
        if any(c.isdigit() for c in text):
            score += 0.1

        # Specificity: contains code-like patterns?
        if "def " in text or "class " in text or "import " in text:
            score += 0.2

        # Specificity: contains factual statements?
        factual_markers = [" is ", " are ", " was ", " created ", " built ", " made "]
        if any(marker in text.lower() for marker in factual_markers):
            score += 0.15

        # Specificity: contains questions (less important to keep)?
        if "?" in text:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def prune_low_importance(self, max_memories: int = 1000):
        """Remove least important memories when over the limit.

        Args:
            max_memories: maximum memories to keep.
        Returns:
            Number removed.
        """
        if len(self.kb.chunks) <= max_memories:
            return 0

        # Score all memories.
        scores = [(self.importance_score(text), i)
                  for i, text in enumerate(self.kb.chunks)]
        scores.sort()  # lowest importance first

        # Remove the lowest-scoring ones.
        n_to_remove = len(self.kb.chunks) - max_memories
        to_remove = set(idx for _, idx in scores[:n_to_remove])

        kept_chunks = []
        kept_embeddings = []
        kept_sources = []

        for i in range(len(self.kb.chunks)):
            if i not in to_remove:
                kept_chunks.append(self.kb.chunks[i])
                if i < len(self.kb.embeddings):
                    kept_embeddings.append(self.kb.embeddings[i])
                if self.kb.sources and i < len(self.kb.sources):
                    kept_sources.append(self.kb.sources[i])

        removed = len(self.kb.chunks) - len(kept_chunks)
        self.kb.chunks = kept_chunks
        self.kb.embeddings = kept_embeddings
        self.kb.sources = kept_sources

        return removed


class FractusSelfGrowth:
    """The autonomous growth loop.

    Combines SelfGrowthPolicy + MemoryManager + FractusGrowth to let
    Fractus decide how to improve itself.

    Usage:
        self_growth = FractusSelfGrowth(model, tok, kb, device)
        action = self_growth.evaluate(metrics)
        if action:
            self_growth.execute(action, data)
    """

    def __init__(self, model, tokenizer, kb, device="cpu"):
        self.model = model
        self.tok = tokenizer
        self.kb = kb
        self.device = torch.device(device)
        self.policy = SelfGrowthPolicy()
        self.memory_mgr = MemoryManager(kb)

        # Track metrics over time.
        self.loss_history = []
        self.expert_usage = {}

    def evaluate(self, metrics: dict):
        """Evaluate current state and decide if growth is needed.

        Args:
            metrics: dict with current performance metrics.

        Returns:
            dict with action + reason, or None if nothing to do.
        """
        action, reason = self.policy.decide(metrics)

        if action == "NOTHING":
            return None

        return {
            "action": action,
            "reason": reason,
            "metrics": metrics,
        }

    def execute(self, decision: dict, data=None, domain=None):
        """Execute a growth decision.

        Args:
            decision: dict from evaluate()
            data: token tensor for expert training (if ADD_EXPERT)
            domain: domain name for new experts (if ADD_EXPERT)
        """
        from fractus.growth import FractusGrowth

        action = decision["action"]
        reason = decision["reason"]

        print(f"[SelfGrowth] Action: {action}", flush=True)
        print(f"[SelfGrowth] Reason: {reason}", flush=True)

        if action == "ADD_EXPERT":
            growth = FractusGrowth(self.model, self.tok, str(self.device))
            n_new = 32  # add 32 experts per growth event
            growth.add_experts(
                n_new=n_new,
                data=data,
                domain=domain or "auto_detected",
                steps_per_expert=2000,
            )
            print(f"[SelfGrowth] Added {n_new} experts per layer", flush=True)

        elif action == "EXPAND_RANK":
            growth = FractusGrowth(self.model, self.tok, str(self.device))
            current_rank = self.model.blocks[0].moe.experts_w1[0].rank
            new_rank = current_rank * 2
            growth.expand_rank(target_rank=new_rank)
            print(f"[SelfGrowth] Expanded rank {current_rank} → {new_rank}", flush=True)

        elif action == "FORGET":
            # Consolidate + prune low importance.
            duplicates = self.memory_mgr.consolidate()
            pruned = self.memory_mgr.prune_low_importance(max_memories=500)
            print(f"[SelfGrowth] Forgot {duplicates} duplicates + {pruned} low-importance memories",
                  flush=True)

        elif action == "MODIFY":
            # This would be triggered by MetaCognition when it detects a correction.
            print(f"[SelfGrowth] Memory modification queued (use memory_mgr.modify() directly)",
                  flush=True)

    def user_forget(self, pattern: str = None, source: str = None):
        """User-requested forgetting.

        The user can tell Fractus to forget specific things.
        """
        removed = self.memory_mgr.forget(pattern=pattern, source=source)
        print(f"[SelfGrowth] Forgot {removed} memories matching '{pattern or source}'", flush=True)
        return removed

    def user_correct(self, old_pattern: str, new_text: str):
        """User-requested correction.

        The user corrects a wrong memory.
        """
        modified = self.memory_mgr.modify(old_pattern, new_text, source="user_correction")
        print(f"[SelfGrowth] Corrected {modified} memories: '{old_pattern[:30]}' → '{new_text[:30]}'",
              flush=True)
        return modified
