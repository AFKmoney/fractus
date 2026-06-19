"""fractus — réfonte unifiée de FNN + OMNI-FRACTAL.

L0 : seul le pont natif `_core` est exposé (lazily, voir plus bas). Les modules
nn/, causal/, reasoning/ seront ajoutés dans les couches ultérieures (L1+).

Conception : les sous-modules purs-Python (fractus.nn, etc.) doivent rester
importables même si le module natif Rust n'est pas construit (utile pour les
tests unitaires PyTorch). On n'import donc PAS `_core` au niveau module — on
l'expose via __getattr__ paresseux, qui ne lève que si quelqu'un y accède
réellement sans avoir lancé `maturin develop`.
"""

__version__ = "0.1.0"


def __getattr__(name):
    # Import paresseux du module natif fractus._core.
    # Ne se déclenche que si quelqu'un fait `from fractus import _core`
    # ou `fractus._core`. Les imports `import fractus.nn` ne passent pas ici.
    #
    # On utilise importlib.import_module et non `from fractus import _core`,
    # car cette dernière forme re-déclencherait __getattr__ → récursion infinie.
    if name == "_core":
        import importlib
        try:
            return importlib.import_module("fractus._core")
        except ImportError as e:
            raise ImportError(
                "Le module natif fractus._core est introuvable. "
                "As-tu lancé `maturin develop` ?"
            ) from e
    raise AttributeError(f"module 'fractus' has no attribute {name!r}")
