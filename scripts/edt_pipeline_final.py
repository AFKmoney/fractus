"""Shard reader for EDT — reads .npy shards directly, no fusion needed.

This replaces the old `torch.load('corpus.pt')` which crashed on files >50GB.
The training reads shards one at a time via mmap (zero RAM cost).
"""
import os, glob, random
import numpy as np
import torch


class ShardCorpusReader:
    """Read tokenized corpus from .npy shards without loading everything in RAM.

    Usage:
        reader = ShardCorpusReader('data/fractus_1b_shards/')
        tokens = reader.get_random_batch(batch_size=8, seq_len=32, device='cuda')
        # tokens: (batch_size, seq_len) long tensor
    """

    def __init__(self, shard_dir: str):
        self.shards = sorted(glob.glob(os.path.join(shard_dir, "*_*.npy")))
        if not self.shards:
            raise FileNotFoundError(f"No shards found in {shard_dir}")

        # Index: for each shard, get its length (mmap, no RAM).
        self.shard_lengths = []
        for s in self.shards:
            arr = np.load(s, mmap_mode="r")
            self.shard_lengths.append(len(arr))

        self.total_tokens = sum(self.shard_lengths)
        print(f"  ShardCorpusReader: {len(self.shards)} shards, "
              f"{self.total_tokens/1e9:.2f}B tokens", flush=True)

    def get_random_batch(self, batch_size: int, seq_len: int, device="cpu"):
        """Get a random batch of token sequences from the shards.

        Picks random positions across all shards, reads seq_len tokens from each.
        """
        batch_tokens = []
        for _ in range(batch_size):
            # Pick a random shard (weighted by size).
            shard_idx = random.choices(
                range(len(self.shards)),
                weights=self.shard_lengths
            )[0]

            shard = np.load(self.shards[shard_idx], mmap_mode="r")
            shard_len = self.shard_lengths[shard_idx]

            # Pick a random position.
            if shard_len <= seq_len + 1:
                continue
            start = random.randint(0, shard_len - seq_len - 1)

            # Read seq_len tokens.
            tokens = shard[start:start + seq_len + 1]
            batch_tokens.append(tokens)

        if not batch_tokens:
            # Fallback if all shards too small (shouldn't happen).
            shard = np.load(self.shards[0], mmap_mode="r")
            tokens = shard[:seq_len + 1]
            batch_tokens = [tokens] * batch_size

        # Convert to tensor: inp and tgt.
        data = np.stack(batch_tokens)  # (batch_size, seq_len+1)
        data_tensor = torch.from_numpy(data.astype(np.int64)).to(device)

        inp = data_tensor[:, :-1]  # (batch_size, seq_len)
        tgt = data_tensor[:, 1:]   # (batch_size, seq_len)

        return inp, tgt

    def get_random_tokens(self, n: int, device="cpu"):
        """Get n random token IDs from the corpus (for embedding training)."""
        tokens = []
        for _ in range(n):
            shard_idx = random.choices(
                range(len(self.shards)),
                weights=self.shard_lengths
            )[0]
            shard = np.load(self.shards[shard_idx], mmap_mode="r")
            pos = random.randint(0, self.shard_lengths[shard_idx] - 1)
            tokens.append(shard[pos])

        return torch.tensor(tokens, dtype=torch.long, device=device)

    def stream_tokens(self, batch_size=128, seq_len=64, device="cpu"):
        """Generator that yields batches sequentially through the corpus."""
        for shard_path in self.shards:
            shard = np.load(shard_path, mmap_mode="r")
            shard_len = len(shard)

            for start in range(0, shard_len - seq_len - 1, seq_len * batch_size):
                batch_tokens = []
                for b in range(batch_size):
                    pos = start + b * seq_len
                    if pos + seq_len + 1 >= shard_len:
                        break
                    batch_tokens.append(shard[pos:pos + seq_len + 1])

                if len(batch_tokens) < batch_size:
                    break

                data = np.stack(batch_tokens)
                data_tensor = torch.from_numpy(data.astype(np.int64)).to(device)
                yield data_tensor[:, :-1], data_tensor[:, 1:]
