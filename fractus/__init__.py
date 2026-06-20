"""fractus — refonte unifiee of the original + the original design.

L0 : seul the pont natif `_core` est expose (lazily, voir more bas). Les modules
nn/, causal/, reasoning/ will be ajoutes in the couches ulterieures (L1+).

Conception : the under-modules purs-Python (fractus.nn, etc.) must rester
importables same si the module natif Rust n'est not construit (utile for les
tests unitaires PyTorch). On n'import therefore PAS `_core` au niveau module — on
l'expose via __getattr__ paresseux, which not leve that si quelqu'un y accede
reallement without have lance `maturin develop`.
"""

__version__ = "0.1.0"


def __getattr__(name):
    # Import paresseux module natif fractus._core.
    # Ne se declenche that si quelqu'un does `from fractus import _core`
    # or `fractus._core`. Les imports `import fractus.nn` not passent not ici.
    #
    # On utilise importlib.import_module and non `from fractus import _core`,
    # because cette derniere shape re-declencherait __getattr__ → recursion infinie.
    if name == "_core":
        import importlib
        try:
            return importlib.import_module("fractus._core")
        except ImportError as e:
            raise ImportError(
                "Le module natif fractus._core est introuvable. "
                "As-tu lance `maturin develop` ?"
            ) from e
    raise AttributeError(f"module 'fractus' has no attribute {name!r}")
