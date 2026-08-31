"""Actuator allocation for front-, rear- and all-wheel drive."""

from __future__ import annotations

import numpy as np

from .model import ControlInput


class TorqueAllocator:
    """Map total drive/brake requests to four wheel actuator commands."""

    def __init__(
        self,
        wheel_radius: float,
        front_track: float,
        rear_track: float,
        max_drive_torque: float = 2400.0,
        max_brake_torque: float = 3600.0,
        drive_mode: str = "awd",
        max_wheel_torque: float | None = None,
    ) -> None:
        if wheel_radius <= 0.0 or front_track <= 0.0 or rear_track <= 0.0:
            raise ValueError("wheel_radius and track widths must be positive")
        if max_drive_torque <= 0.0 or max_brake_torque <= 0.0:
            raise ValueError("torque limits must be positive")
        if drive_mode not in {"fwd", "rwd", "awd"}:
            raise ValueError("drive_mode must be fwd, rwd or awd")
        self.wheel_radius = float(wheel_radius)
        self.front_track = float(front_track)
        self.rear_track = float(rear_track)
        self.max_drive_torque = float(max_drive_torque)
        self.max_brake_torque = float(max_brake_torque)
        self.drive_mode = drive_mode
        self.max_wheel_torque = (
            float(max_wheel_torque)
            if max_wheel_torque is not None
            else self.max_drive_torque
        )

    @property
    def drive_indices(self) -> np.ndarray:
        if self.drive_mode == "fwd":
            return np.array([0, 1], dtype=int)
        if self.drive_mode == "rwd":
            return np.array([2, 3], dtype=int)
        return np.array([0, 1, 2, 3], dtype=int)

    def allocate(self, total_torque: float, yaw_moment: float = 0.0) -> np.ndarray:
        """Allocate a total wheel torque and optional direct yaw moment.

        A positive yaw moment increases the right-side drive force and
        decreases the left-side drive force.  The simple allocation is
        intentionally deterministic and is a baseline for later constrained
        optimization.
        """

        total_torque = float(np.clip(total_torque, -self.max_drive_torque, self.max_drive_torque))
        torque = np.zeros(4, dtype=float)
        indices = self.drive_indices
        torque[indices] = total_torque / len(indices)

        left = np.intersect1d(indices, np.array([0, 2], dtype=int))
        right = np.intersect1d(indices, np.array([1, 3], dtype=int))
        if len(left) and len(right) and yaw_moment != 0.0:
            effective_half_track = 0.25 * (self.front_track + self.rear_track)
            torque_difference = float(yaw_moment) * self.wheel_radius / effective_half_track
            torque[left] -= torque_difference / (2.0 * len(left))
            torque[right] += torque_difference / (2.0 * len(right))

        return np.clip(torque, -self.max_wheel_torque, self.max_wheel_torque)

    def from_throttle(
        self,
        throttle: float,
        steer_front: float = 0.0,
        steer_rear: float = 0.0,
        yaw_moment: float = 0.0,
    ) -> ControlInput:
        """Convert a normalized throttle/brake command into wheel commands."""

        throttle = float(np.clip(throttle, -1.0, 1.0))
        if throttle >= 0.0:
            wheel_torque = self.allocate(throttle * self.max_drive_torque, yaw_moment)
            wheel_brake = np.zeros(4, dtype=float)
        else:
            wheel_torque = np.zeros(4, dtype=float)
            wheel_brake = np.full(4, -throttle * self.max_brake_torque / 4.0)
        return ControlInput(
            steer_front=steer_front,
            steer_rear=steer_rear,
            wheel_torque=wheel_torque,
            wheel_brake=wheel_brake,
        )
