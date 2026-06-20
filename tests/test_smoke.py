"""Test fume : prouve que la plomberie Python → PyTorch → Rust tient.

Ces tests ne valident no logical mathematical — juste que les briques
communiquent. Si un de ces tests echoue, rien d'autre ne can marcher.
"""


def test_torch_available():
    """PyTorch est installe et fonctionnel."""
    import torch
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.sum().item() == 6.0


def test_numpy_available():
    """NumPy est installe (necessaire for le pont tenseurs)."""
    import numpy as np
    a = np.array([1, 2, 3])
    assert a.sum() == 6


def test_rust_bridge_import():
    """Le module natif fractus._core est bien construit et importable."""
    from fractus import _core
    assert hasattr(_core, "add")


def test_rust_bridge_add():
    """Python can appeler du Rust et recuperer le bon result."""
    from fractus import _core
    assert _core.add(2, 3) == 5
    assert _core.add(-10, 4) == -6


def test_torch_numpy_interop():
    """PyTorch et numpy s'echangent des tenseurs (necessaire for le pont Rust)."""
    import numpy as np
    import torch
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = torch.from_numpy(arr)
    assert t.dtype == torch.float32
    # Retour vers numpy
    back = t.numpy()
    assert np.allclose(back, arr)
