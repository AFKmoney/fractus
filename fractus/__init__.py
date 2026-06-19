"""fractus — réfonte unifiée de FNN + OMNI-FRACTAL.

L0 : seul le pont natif `_core` est exposé. Les modules nn/, causal/, reasoning/
seront ajoutés dans les couches ultérieures (L1+).
"""

__version__ = "0.1.0"

# Le module natif fractus._core est construit par maturin et placé ici.
# On l'importe explicitement pour qu'il soit accessible via `from fractus import _core`.
try:
    from fractus import _core  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Le module natif fractus._core est introuvable. "
        "As-tu lancé `maturin develop` ?"
    ) from e
