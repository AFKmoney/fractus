"""fractus — refonte unifiee de FNN + the original design.

L0 : seul le pont natif `_core` est expose (lazily, voir plus bas). Les modules
nn/, causal/, reasoning/ seront ajoutes in les couches ulterieures (L1+).

Conception : les under-modules purs-Python (fractus.nn, etc.) must rester
importables meme si le module natif Rust n'est pas construit (utile for les
tests unitaires PyTorch). On n'import therefore PAS `_core` au niveau module — on
l'expose via __getattr__ paresseux, qui ne leve que si quelqu'un y accede
reellement without avoir lance `maturin develop`.
"""

__version__ = "0.1.0"


def __getattr__(name):
    # Import paresseux du module natif fractus._core.
    # Ne se declenche que si quelqu'un fait `from fractus import _core`
    # ou `fractus._core`. Les imports `import fractus.nn` ne passent pas ici.
    #
    # On utilise importlib.import_module et non `from fractus import _core`,
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
