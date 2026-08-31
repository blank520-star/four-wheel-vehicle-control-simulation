"""Tire-force models used by the vehicle simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PacejkaParameters:
    """Parameters for one normalized Magic Formula curve."""

    B: float = 10.0
    C: float = 1.9
    D: float = 1.0
    E: float = 0.97
    H: float = 0.0
    V: float = 0.0

    def __post_init__(self) -> None:
        if self.B <= 0.0:
            raise ValueError("Pacejka B must be positive")
        if self.C <= 0.0:
            raise ValueError("Pacejka C must be positive")
        if self.D < 0.0:
            raise ValueError("Pacejka D cannot be negative")


class PacejkaTire:
    """Combined-slip tire model with a friction-ellipse limiter.

    The lateral convention is positive force to the vehicle's left.  The
    supplied slip angle is the velocity angle in the wheel frame, so a
    negative slip angle produces a positive lateral tire force.
    """

    def __init__(
        self,
        longitudinal: PacejkaParameters | None = None,
        lateral: PacejkaParameters | None = None,
    ) -> None:
        self.longitudinal = longitudinal or PacejkaParameters()
        self.lateral = lateral or PacejkaParameters(B=8.5, C=1.3, D=1.0, E=0.8)

    @staticmethod
    def _magic_formula(value: np.ndarray, parameters: PacejkaParameters) -> np.ndarray:
        shifted = value + parameters.H
        bx = parameters.B * shifted
        return parameters.D * np.sin(
            parameters.C
            * np.arctan(bx - parameters.E * (bx - np.arctan(bx)))
        ) + parameters.V

    def forces(
        self,
        slip_angle: float | np.ndarray,
        slip_ratio: float | np.ndarray,
        normal_load: float | np.ndarray,
        friction_coefficient: float | np.ndarray = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return longitudinal and lateral force in the wheel frame.

        ``slip_ratio`` is positive during driven-wheel spin-up.  The output
        is limited to the available friction ellipse, which prevents the
        independent longitudinal and lateral curves from producing an
        impossible combined force.
        """

        alpha = np.asarray(slip_angle, dtype=float)
        kappa = np.asarray(slip_ratio, dtype=float)
        fz = np.maximum(np.asarray(normal_load, dtype=float), 0.0)
        mu = np.maximum(np.asarray(friction_coefficient, dtype=float), 1e-6)

        fx = fz * self._magic_formula(kappa, self.longitudinal)
        fy = -fz * self._magic_formula(alpha, self.lateral)

        limit = np.maximum(mu * fz, 1e-6)
        utilization = np.sqrt((fx / limit) ** 2 + (fy / limit) ** 2)
        scale = np.maximum(utilization, 1.0)
        return fx / scale, fy / scale
