import numpy as np

from vehicle_sim.tire import PacejkaTire


def test_zero_slip_has_zero_force() -> None:
    tire = PacejkaTire()
    fx, fy = tire.forces(0.0, 0.0, 3500.0, 1.0)
    assert np.isclose(fx, 0.0)
    assert np.isclose(fy, 0.0)


def test_lateral_force_opposes_positive_slip_angle() -> None:
    tire = PacejkaTire()
    _, fy = tire.forces(0.05, 0.0, 3500.0, 1.0)
    assert fy < 0.0


def test_combined_force_stays_inside_friction_limit() -> None:
    tire = PacejkaTire()
    fx, fy = tire.forces(0.4, 0.4, 3500.0, 0.8)
    assert np.hypot(fx, fy) <= 0.8 * 3500.0 + 1e-8
