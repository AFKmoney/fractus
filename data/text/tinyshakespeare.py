"""Dataset tinyshakespeare for l'entrainement du transformer fractal.

Charge le texte, encode en ids (niveau caractere), decoupe en sequences
de longueur fixe for l'entrainement par batch.
"""

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "text", "tinyshakespeare.txt",
)


class TinyShakespeareDataset(Dataset):
    """Dataset tinyshakespeare niveau caractere.

    Args:
        seq_len  : longueur des sequences.
        path     : chemin du fichier texte (defaut : data/text/tinyshakespeare.txt).
        vocab    : vocabulaire optionnel (sinon construit depuis le texte).
    """

    def __init__(
        self,
        seq_len: int = 64,
        path: Optional[str] = None,
        vocab: Optional[dict] = None,
    ):
        if path is None:
            path = DEFAULT_PATH
        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read()
        self.seq_len = seq_len

        # Vocabulaire : mapping char → id.
        if vocab is not None:
            self.char_to_id = vocab
        else:
            chars = sorted(set(self.text))
            self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)

        # Encoder tout le texte en ids.
        self.ids = torch.tensor(
            [self.char_to_id[c] for c in self.text if c in self.char_to_id],
            dtype=torch.long,
        )
        # Nombre de sequences possibles.
        self.n_seqs = (len(self.ids) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retourne (input_ids, target_ids) ou target = input decale de 1."""
        start = idx * self.seq_len
        end = start + self.seq_len
        input_ids = self.ids[start:end]
        target_ids = self.ids[start + 1:end + 1]
        return input_ids, target_ids

    def decode(self, ids: torch.Tensor) -> str:
        """Decode des ids en texte."""
        return "".join(self.id_to_char.get(int(i), "?") for i in ids)
