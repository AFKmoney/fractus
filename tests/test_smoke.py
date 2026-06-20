"""Test fume : prouve that the plomberie Python → PyTorch → Rust tient.

Ces tests not validnt no logical mathematical — juste that the briques
communiquent. Si a of these tests echoue, nothing d'other not can marcher.
"""


def test_torch_available():
    """PyTorch est installe and functionnel."""
    import torch
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.sum().item() == 6.0


def test_numpy_available():
    """NumPy est installe (necessaire for the pont tenseurs)."""
    import numpy as np
    a = np.array([1, 2, 3])
    assert a.sum() == 6


def test_rust_bridge_import():
    """Le module natif fractus._core est well construit and importable."""
    from fractus import _core
    assert hasattr(_core, "add")


def test_rust_bridge_add():
    """Python can appeler Rust and recuperer the bon result."""
    from fractus import _core
    assert _core.add(2, 3) == 5
    assert _core.add(-10, 4) == -6


def test_torch_numpy_interop():
    """PyTorch and numpy s'echangent tenseurs (necessaire for the pont Rust)."""
    import numpy as np
    import torch
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = torch.from_numpy(arr)
    assert t.dtype == torch.float32
    # Retour toward numpy
    back = t.numpy()
    assert np.allclose(back, arr)
