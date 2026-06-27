"""Online trainer for the ContinuousThoughtEngine.

THE TRAINING BREAKTHROUGH. No batches. No BPTT. One observation at a time,
one gradient at a time. The model learns as it "sees" data, like a human.

    for each token in the data stream:
        1. Feed the token to the engine (tick).
        2. The engine produces a prediction + confidence.
        3. Compute the loss (was the prediction right?).
        4. Backward + step IMMEDIATELY (online SGD, 1 sample at a time).
        5. The thought state is carried forward (detached — no BPTT).

WHY THIS IS FAST:
    - Each step processes ONE token (not B×L).
    - The forward is tiny (1 token, 1 tick).
    - The backward is tiny (1 sample).
    - No batching, no padding, no sequence masking.

This is the training method that makes the Continuous Thought Engine
trainable on ANY CPU, because the per-step cost is minimal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineTrainer:
    """Online SGD trainer for the ContinuousThoughtEngine.

    Args:
        engine:           a ContinuousThoughtEngine.
        lr:               learning rate.
        weight_decay:     AdamW weight decay.
        confidence_threshold: the engine emits output when confidence exceeds this.
    """

    def __init__(
        self,
        engine,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        self.engine = engine
        self.optimizer = torch.optim.AdamW(engine.parameters(), lr=lr,
                                           weight_decay=weight_decay)
        self.step_count = 0
        self.losses = []

    def train_on_stream(self, token_ids: torch.Tensor, max_ticks: int = 3) -> dict:
        """Train on a stream of tokens, one at a time.

        token_ids: (L,) a 1D tensor of token ids (the data stream).
        max_ticks: max thinking ticks per token.

        Returns a dict with average loss, accuracy, and steps.
        """
        self.engine.train()
        self.engine.reset_thought(batch_size=1)

        total_loss = 0.0
        correct = 0
        total = 0

        for t in range(len(token_ids) - 1):
            obs = token_ids[t:t + 1]  # (1,) current token
            target = token_ids[t + 1]  # scalar, next token

            # Think: tick until confidence or max_ticks.
            for tick in range(max_ticks):
                logits, conf = self.engine.tick(obs if tick == 0 else None)
                if conf.item() > 0.5:
                    break

            # Online loss: did we predict the next token?
            loss = F.cross_entropy(logits, target.unsqueeze(0))

            # Immediate backward + step (online SGD).
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.engine.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=-1).item()
            if pred == target.item():
                correct += 1
            total += 1
            self.step_count += 1
            self.losses.append(loss.item())

        return {
            "avg_loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "steps": total,
        }

    def train_step_batch(self, input_ids: torch.Tensor, target_ids: torch.Tensor,
                         max_ticks: int = 3) -> dict:
        """Train on a small batch using the think() method.

        input_ids:  (B, L) token ids.
        target_ids: (B, L) next-token targets.
        """
        self.engine.train()
        self.engine.reset_thought(batch_size=input_ids.shape[0])

        # Use think() to process the whole sequence.
        logits = self.engine.think(input_ids, max_ticks=max_ticks, confidence_threshold=0.5)
        # The logits are (B, L, vocab) — but think() only produces output when
        # confident. For training we compute loss on ALL positions.
        loss = F.cross_entropy(
            logits.reshape(-1, self.engine.vocab_size),
            target_ids.reshape(-1),
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.engine.parameters(), 1.0)
        self.optimizer.step()
        self.step_count += 1

        return {"loss": loss.item()}
