"""16 features morphologiques déterministes par token.

Porté depuis FNN v5.0 (src/embedding.rs, CharClassFeatures). Le token id est
interprété comme un codepoint Unicode ; pour les ids < 128 ce sont des
caractères ASCII, au-delà on dérive les features de la valeur numérique.

Ces features n'ont AUCUN paramètre entraînable — elles sont calculées
déterministiquement puis concaténées à la base de Fourier dans FractalEmbedding.
"""

import torch


class CharClassFeatures:
    """Extraction de 16 features morphologiques à partir d'un token id.

    Features (index : signification) :
        0  : is_vowel          (a, e, i, o, u)
        1  : is_consonant      (lettre non voyelle)
        2  : is_digit          (0-9)
        3  : is_space          (0x20)
        4  : is_uppercase
        5  : is_lowercase
        6  : is_punctuation    (!"#$%...)
        7  : is_alphabetic
        8  : is_numeric        (alias de is_digit ici)
        9  : is_whitespace     (espace, tab, newline)
        10 : is_control        (codepoint < 32 ou == 127)
        11 : digit_value       (0-9, ou 0 si pas un chiffre)
        12 : char_category     (catégorie Unicode simplifiée comme float)
        13 : position_in_alphabet (0-25, ou -1 si pas une lettre ; on encode -1→0)
        14 : is_ascii          (codepoint < 128)
        15 : parity            (token id pair = 1, impair = 0)
    """

    N_FEATURES = 16

    VOWELS = frozenset(b"aeiouAEIOU")

    @staticmethod
    def extract(token_id: int) -> torch.Tensor:
        """Retourne un tenseur float32 de forme (16,)."""
        f = torch.zeros(CharClassFeatures.N_FEATURES, dtype=torch.float32)

        # On interprète l'octet de poids faible comme un caractère potentiel.
        as_byte = (token_id & 0xFF)

        # 0: is_vowel
        is_vowel = float(as_byte in CharClassFeatures.VOWELS)
        f[0] = is_vowel

        # 1: is_consonant (lettre alphabétique non voyelle)
        is_alpha = (
            (0x41 <= as_byte <= 0x5A) or  # A-Z
            (0x61 <= as_byte <= 0x7A)     # a-z
        )
        f[1] = float(is_alpha and is_vowel < 0.5)

        # 2: is_digit
        is_digit = 0x30 <= as_byte <= 0x39
        f[2] = float(is_digit)

        # 3: is_space (0x20)
        f[3] = float(as_byte == 0x20)

        # 4: is_uppercase
        f[4] = float(0x41 <= as_byte <= 0x5A)

        # 5: is_lowercase
        f[5] = float(0x61 <= as_byte <= 0x7A)

        # 6: is_punctuation (ASCII punctuation ranges)
        is_punct = (
            (0x21 <= as_byte <= 0x2F) or
            (0x3A <= as_byte <= 0x40) or
            (0x5B <= as_byte <= 0x60) or
            (0x7B <= as_byte <= 0x7E)
        )
        f[6] = float(is_punct)

        # 7: is_alphabetic
        f[7] = float(is_alpha)

        # 8: is_numeric (alias de is_digit ici)
        f[8] = float(is_digit)

        # 9: is_whitespace (espace, tab 0x09, newline 0x0A, CR 0x0D)
        f[9] = float(as_byte in (0x09, 0x0A, 0x0D, 0x20))

        # 10: is_control (codepoint < 32 ou == 127)
        f[10] = float(as_byte < 0x20 or as_byte == 0x7F)

        # 11: digit_value
        f[11] = float(as_byte - 0x30) if is_digit else 0.0

        # 12: char_category simplifié : 1.0 lettre, 2.0 chiffre, 3.0 ponctuation,
        #     4.0 espace, 5.0 contrôle, 0.0 autre.
        if is_alpha:
            f[12] = 1.0
        elif is_digit:
            f[12] = 2.0
        elif is_punct:
            f[12] = 3.0
        elif f[9] > 0.5:
            f[12] = 4.0
        elif f[10] > 0.5:
            f[12] = 5.0

        # 13: position_in_alphabet (0-25, ou 0 si pas une lettre)
        if 0x41 <= as_byte <= 0x5A:
            f[13] = float(as_byte - 0x41)
        elif 0x61 <= as_byte <= 0x7A:
            f[13] = float(as_byte - 0x61)

        # 14: is_ascii
        f[14] = float(token_id < 128)

        # 15: parity (token id pair)
        f[15] = float((token_id % 2) == 0)

        return f

    @staticmethod
    def extract_batch(token_ids: torch.Tensor) -> torch.Tensor:
        """Version vectorisée : token_ids de forme (N,) → features (N, 16).

        Comme le calcul est déterministe et indépendant par token, on peut
        précalculer une lookup table une fois pour toute la taille du vocab.
        """
        ids_list = token_ids.tolist()
        rows = [CharClassFeatures.extract(int(i)) for i in ids_list]
        return torch.stack(rows, dim=0)
