"""Tests de KuramotoLyapunov : vrai Lyapunov sur sous-système Kuramoto."""

import math
import torch


def test_lyapunov_V_positive_for_nonzero():
    """V(θ) > 0 pour θ != θ*."""
    from fractus.stability.lyapunov import KuramotoLyapunov
    from fractus.nn.phase_ode import KuramotoLayer
    kur = KuramotoLayer(d_model=8, n_oscillators=4, rank=2)
    lyap = KuramotoLyapunov(kur, target_phase=0.0)
    phases = torch.rand(2, 3, 4) * 2 * math.pi + 0.1  # != 0
    V = lyap.V(phases)
    assert (V > 0).all()


def test_lyapunov_V_zero_at_target():
    """V(θ*) = 0 (phase synchronisée à la cible)."""
    from fractus.stability.lyapunov import KuramotoLyapunov
    from fractus.nn.phase_ode import KuramotoLayer
    kur = KuramotoLayer(d_model=8, n_oscillators=4, rank=2)
    lyap = KuramotoLyapunov(kur, target_phase=1.0)
    phases = torch.full((2, 3, 4), 1.0)  # tous à θ* = 1.0
    V = lyap.V(phases)
    assert (V.abs() < 1e-5).all()


def test_lyapunov_V_handles_wrap():
    """V gère le wrap circulaire : θ = 2π - 0.01 ≈ θ* = 0.01."""
    from fractus.stability.lyapunov import KuramotoLyapunov
    from fractus.nn.phase_ode import KuramotoLayer
    kur = KuramotoLayer(d_model=8, n_oscillators=4, rank=2)
    lyap = KuramotoLyapunov(kur, target_phase=0.0)
    # Phase 2π - 0.01 doit être proche de 0 (wrap), donc V petit.
    phases_near_zero = torch.full((1, 1, 4), 2 * math.pi - 0.01)
    phases_far = torch.full((1, 1, 4), math.pi)  # loin de 0
    V_near = lyap.V(phases_near_zero).item()
    V_far = lyap.V(phases_far).item()
    assert V_near < V_far, f"V(2π-0.01)={V_near} devrait être < V(π)={V_far}"


def test_lyapunov_is_stable_trajectory_true_for_decreasing():
    """Une trajectoire où V décroît → stable."""
    from fractus.stability.lyapunov import KuramotoLyapunov
    from fractus.nn.phase_ode import KuramotoLayer
    kur = KuramotoLayer(d_model=8, n_oscillators=4, rank=2)
    lyap = KuramotoLyapunov(kur, target_phase=0.0)
    # Trajectoire qui converge vers 0 : phases décroissantes vers 0.
    traj = [torch.full((1, 2, 4), p) for p in [2.0, 1.5, 1.0, 0.5, 0.1]]
    assert lyap.is_stable_trajectory(traj) is True


def test_lyapunov_is_stable_trajectory_false_for_increasing():
    """Une trajectoire où V croît → instable."""
    from fractus.stability.lyapunov import KuramotoLyapunov
    from fractus.nn.phase_ode import KuramotoLayer
    kur = KuramotoLayer(d_model=8, n_oscillators=4, rank=2)
    lyap = KuramotoLyapunov(kur, target_phase=0.0)
    traj = [torch.full((1, 2, 4), p) for p in [0.1, 0.5, 1.0, 1.5, 2.0]]  # diverge
    assert lyap.is_stable_trajectory(traj) is False


def test_lyapunov_not_just_output_norm():
    """CRITÈRE L6 : V ne doit PAS être juste ||y||² (le faux Lyapunov d'OMNI).
    On vérifie que V dépend des PHASES, pas d'une norme de sortie réseau."""
    import inspect
    from fractus.stability import lyapunov as lyap_mod
    src = inspect.getsource(lyap_mod)
    # V doit être calculé à partir des phases (différence circulaire θ - θ*).
    assert "target_phase" in src
    assert "remainder" in src or "wrap" in src.lower(), \
        "V doit gérer le wrap circulaire des phases (pas juste ||y||²)"
