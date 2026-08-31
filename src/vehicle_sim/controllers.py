"""Reusable longitudinal and lateral controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import VehicleState
from .path import Path2D


class PIDController:
    """Discrete PID controller with output limits and anti-windup."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        output_limits: tuple[float, float] = (-1.0, 1.0),
        integral_limits: tuple[float, float] | None = None,
        derivative_filter: float = 0.0,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if output_limits[0] >= output_limits[1]:
            raise ValueError("output_limits must be increasing")
        if integral_limits is not None and integral_limits[0] >= integral_limits[1]:
            raise ValueError("integral_limits must be increasing")
        if not 0.0 <= derivative_filter < 1.0:
            raise ValueError("derivative_filter must be in [0, 1)")
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.output_limits = tuple(float(value) for value in output_limits)
        self.integral_limits = (
            None
            if integral_limits is None
            else tuple(float(value) for value in integral_limits)
        )
        self.derivative_filter = float(derivative_filter)
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0
        self.initialized = False

    def update(self, setpoint: float, measurement: float, dt: float | None = None) -> float:
        step = self.dt if dt is None else float(dt)
        if step <= 0.0:
            raise ValueError("dt must be positive")
        error = float(setpoint - measurement)
        if self.initialized:
            raw_derivative = (error - self.previous_error) / step
        else:
            raw_derivative = 0.0
            self.initialized = True
        self.filtered_derivative = (
            self.derivative_filter * self.filtered_derivative
            + (1.0 - self.derivative_filter) * raw_derivative
        )

        candidate_integral = self.integral + error * step
        if self.integral_limits is not None:
            candidate_integral = float(
                np.clip(candidate_integral, self.integral_limits[0], self.integral_limits[1])
            )
        unsaturated = (
            self.kp * error
            + self.ki * candidate_integral
            + self.kd * self.filtered_derivative
        )
        lower, upper = self.output_limits
        output = float(np.clip(unsaturated, lower, upper))

        saturated_high = unsaturated > upper and error > 0.0
        saturated_low = unsaturated < lower and error < 0.0
        if not (saturated_high or saturated_low):
            self.integral = candidate_integral
        self.previous_error = error
        return output


@dataclass(frozen=True)
class SteeringCommand:
    """Front and rear steering targets in radians."""

    front: float
    rear: float = 0.0


class PurePursuitController:
    """Pure Pursuit path tracker with speed-dependent lookahead."""

    def __init__(
        self,
        wheelbase: float,
        lookahead_distance: float = 3.0,
        max_steering_angle: float = 0.55,
        rear_steer_ratio: float = 0.0,
        lookahead_speed_gain: float = 0.0,
    ) -> None:
        if wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive")
        if lookahead_distance <= 0.0:
            raise ValueError("lookahead_distance must be positive")
        if max_steering_angle <= 0.0:
            raise ValueError("max_steering_angle must be positive")
        if lookahead_speed_gain < 0.0:
            raise ValueError("lookahead_speed_gain cannot be negative")
        self.wheelbase = float(wheelbase)
        self.lookahead_distance = float(lookahead_distance)
        self.max_steering_angle = float(max_steering_angle)
        self.rear_steer_ratio = float(rear_steer_ratio)
        self.lookahead_speed_gain = float(lookahead_speed_gain)

    @staticmethod
    def _to_vehicle_frame(state: VehicleState, point: np.ndarray) -> np.ndarray:
        delta = np.asarray(point, dtype=float) - np.array([state.x, state.y])
        cos_yaw = np.cos(state.yaw)
        sin_yaw = np.sin(state.yaw)
        return np.array(
            [
                cos_yaw * delta[0] + sin_yaw * delta[1],
                -sin_yaw * delta[0] + cos_yaw * delta[1],
            ],
            dtype=float,
        )

    def command(self, state: VehicleState, path: Path2D) -> SteeringCommand:
        projection = path.project(np.array([state.x, state.y]))
        lookahead = self.lookahead_distance + self.lookahead_speed_gain * max(state.v_x, 0.0)
        target = path.lookahead(projection, lookahead)
        target_vehicle = self._to_vehicle_frame(state, target)
        distance_squared = max(float(np.dot(target_vehicle, target_vehicle)), 1e-9)
        curvature = 2.0 * target_vehicle[1] / distance_squared
        front = float(np.arctan(self.wheelbase * curvature))
        front = float(np.clip(front, -self.max_steering_angle, self.max_steering_angle))
        rear = float(np.clip(-self.rear_steer_ratio * front, -self.max_steering_angle, self.max_steering_angle))
        return SteeringCommand(front=front, rear=rear)
