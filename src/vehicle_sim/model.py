"""Four-wheel planar vehicle model with wheel-speed dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tire import PacejkaTire


WHEEL_NAMES = ("fl", "fr", "rl", "rr")


@dataclass
class VehicleParams:
    """Physical and actuator parameters for the vehicle model."""

    mass: float = 1200.0
    yaw_inertia: float = 1800.0
    front_length: float = 1.2
    rear_length: float = 1.6
    front_track: float = 1.5
    rear_track: float = 1.5
    center_of_mass_height: float = 0.55
    wheel_radius: float = 0.31
    wheel_inertia: float = 1.8
    gravity: float = 9.81
    air_density: float = 1.225
    drag_coefficient: float = 0.32
    frontal_area: float = 2.2
    downforce_coefficient: float = 0.0
    rolling_resistance: float = 0.015
    steering_time_constant: float = 0.08
    max_steering_angle: float = 0.55
    slip_speed_epsilon: float = 0.5
    friction_coefficient: float = 1.0
    initial_speed: float = 0.5
    initial_x: float = 0.0
    initial_y: float = 0.0
    initial_yaw: float = 0.0
    tire: PacejkaTire = field(default_factory=PacejkaTire)

    def __post_init__(self) -> None:
        positive = {
            "mass": self.mass,
            "yaw_inertia": self.yaw_inertia,
            "front_length": self.front_length,
            "rear_length": self.rear_length,
            "front_track": self.front_track,
            "rear_track": self.rear_track,
            "wheel_radius": self.wheel_radius,
            "wheel_inertia": self.wheel_inertia,
            "gravity": self.gravity,
            "air_density": self.air_density,
            "frontal_area": self.frontal_area,
            "steering_time_constant": self.steering_time_constant,
            "slip_speed_epsilon": self.slip_speed_epsilon,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.center_of_mass_height < 0.0:
            raise ValueError("center_of_mass_height cannot be negative")
        if self.drag_coefficient < 0.0:
            raise ValueError("drag_coefficient cannot be negative")
        if self.downforce_coefficient < 0.0:
            raise ValueError("downforce_coefficient cannot be negative")
        if self.rolling_resistance < 0.0:
            raise ValueError("rolling_resistance cannot be negative")
        if not 0.0 < self.friction_coefficient:
            raise ValueError("friction_coefficient must be positive")
        if not 0.0 < self.max_steering_angle:
            raise ValueError("max_steering_angle must be positive")

    @property
    def wheelbase(self) -> float:
        return self.front_length + self.rear_length

    @property
    def wheel_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return wheel x/y positions in the vehicle body frame.

        The order is front-left, front-right, rear-left, rear-right.  The
        positive y axis points to the vehicle's left.
        """

        x = np.array(
            [self.front_length, self.front_length, -self.rear_length, -self.rear_length],
            dtype=float,
        )
        y = np.array(
            [
                0.5 * self.front_track,
                -0.5 * self.front_track,
                0.5 * self.rear_track,
                -0.5 * self.rear_track,
            ],
            dtype=float,
        )
        return x, y


@dataclass
class VehicleState:
    """Vehicle state used by the integrator and controllers."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v_x: float = 0.0
    v_y: float = 0.0
    yaw_rate: float = 0.0
    wheel_speed: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    steer_front: float = 0.0
    steer_rear: float = 0.0
    a_x: float = 0.0
    a_y: float = 0.0

    def __post_init__(self) -> None:
        self.wheel_speed = np.asarray(self.wheel_speed, dtype=float).reshape(-1).copy()
        if self.wheel_speed.shape != (4,):
            raise ValueError("wheel_speed must contain four wheel angular speeds")

    def copy(self) -> "VehicleState":
        return VehicleState(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            v_x=self.v_x,
            v_y=self.v_y,
            yaw_rate=self.yaw_rate,
            wheel_speed=self.wheel_speed.copy(),
            steer_front=self.steer_front,
            steer_rear=self.steer_rear,
            a_x=self.a_x,
            a_y=self.a_y,
        )

    def to_vector(self) -> np.ndarray:
        """Return the integrated part of the state as a flat array."""

        return np.array(
            [
                self.x,
                self.y,
                self.yaw,
                self.v_x,
                self.v_y,
                self.yaw_rate,
                *self.wheel_speed,
                self.steer_front,
                self.steer_rear,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, values: np.ndarray) -> "VehicleState":
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape != (12,):
            raise ValueError("integrated vehicle state must contain 12 values")
        return cls(
            x=float(values[0]),
            y=float(values[1]),
            yaw=float(values[2]),
            v_x=float(values[3]),
            v_y=float(values[4]),
            yaw_rate=float(values[5]),
            wheel_speed=values[6:10],
            steer_front=float(values[10]),
            steer_rear=float(values[11]),
        )


@dataclass
class ControlInput:
    """Actuator commands for the four wheels."""

    steer_front: float = 0.0
    steer_rear: float = 0.0
    wheel_torque: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    wheel_brake: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))

    def __post_init__(self) -> None:
        self.wheel_torque = np.asarray(self.wheel_torque, dtype=float).reshape(-1).copy()
        self.wheel_brake = np.asarray(self.wheel_brake, dtype=float).reshape(-1).copy()
        if self.wheel_torque.shape != (4,):
            raise ValueError("wheel_torque must contain four values")
        if self.wheel_brake.shape != (4,):
            raise ValueError("wheel_brake must contain four values")
        if np.any(self.wheel_brake < 0.0):
            raise ValueError("wheel_brake cannot contain negative values")


@dataclass
class WheelMeasurement:
    """Per-wheel quantities calculated from the current vehicle state."""

    slip_angle: np.ndarray
    slip_ratio: np.ndarray
    normal_load: np.ndarray
    longitudinal_force: np.ndarray
    lateral_force: np.ndarray
    body_longitudinal_force: np.ndarray
    body_lateral_force: np.ndarray
    wheel_longitudinal_velocity: np.ndarray
    wheel_lateral_velocity: np.ndarray
    wheel_steering_angle: np.ndarray
    a_x: float
    a_y: float
    yaw_acceleration: float


class FourWheelVehicle:
    """Four-wheel vehicle model integrated with a fixed-step RK4 solver."""

    def __init__(
        self,
        params: VehicleParams | None = None,
        initial_state: VehicleState | None = None,
    ) -> None:
        self.params = params or VehicleParams()
        if initial_state is None:
            wheel_speed = np.full(
                4,
                self.params.initial_speed / self.params.wheel_radius,
                dtype=float,
            )
            initial_state = VehicleState(
                x=self.params.initial_x,
                y=self.params.initial_y,
                yaw=self.params.initial_yaw,
                v_x=self.params.initial_speed,
                wheel_speed=wheel_speed,
            )
        self.state = initial_state.copy()
        self._last_a_x = float(self.state.a_x)
        self._last_a_y = float(self.state.a_y)
        self.last_measurement = self._evaluate(self.state.to_vector(), ControlInput())

    def reset(self, state: VehicleState | None = None) -> VehicleState:
        """Reset the vehicle and return a copy of its new state."""

        if state is None:
            wheel_speed = np.full(
                4,
                self.params.initial_speed / self.params.wheel_radius,
                dtype=float,
            )
            state = VehicleState(
                x=self.params.initial_x,
                y=self.params.initial_y,
                yaw=self.params.initial_yaw,
                v_x=self.params.initial_speed,
                wheel_speed=wheel_speed,
            )
        self.state = state.copy()
        self._last_a_x = float(self.state.a_x)
        self._last_a_y = float(self.state.a_y)
        self.last_measurement = self._evaluate(self.state.to_vector(), ControlInput())
        return self.state.copy()

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _normal_loads(self, v_x: float) -> np.ndarray:
        """Estimate four vertical loads with quasi-static load transfer."""

        p = self.params
        total_load = p.mass * p.gravity + 0.5 * p.air_density * p.downforce_coefficient * p.frontal_area * v_x**2
        front_each = 0.5 * total_load * p.rear_length / p.wheelbase
        rear_each = 0.5 * total_load * p.front_length / p.wheelbase

        longitudinal_transfer_per_wheel = (
            0.5 * p.mass * self._last_a_x * p.center_of_mass_height / p.wheelbase
        )
        front_lateral_transfer = (
            0.5
            * p.mass
            * self._last_a_y
            * p.center_of_mass_height
            * p.rear_length
            / (p.wheelbase * p.front_track)
        )
        rear_lateral_transfer = (
            0.5
            * p.mass
            * self._last_a_y
            * p.center_of_mass_height
            * p.front_length
            / (p.wheelbase * p.rear_track)
        )

        loads = np.array(
            [
                front_each - longitudinal_transfer_per_wheel - front_lateral_transfer,
                front_each - longitudinal_transfer_per_wheel + front_lateral_transfer,
                rear_each + longitudinal_transfer_per_wheel - rear_lateral_transfer,
                rear_each + longitudinal_transfer_per_wheel + rear_lateral_transfer,
            ],
            dtype=float,
        )
        loads = np.maximum(loads, 0.05 * total_load / 4.0)
        return loads * (total_load / np.sum(loads))

    def _wheel_kinematics(
        self,
        state: VehicleState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.params
        position_x, position_y = p.wheel_positions
        steering = np.array(
            [state.steer_front, state.steer_front, state.steer_rear, state.steer_rear],
            dtype=float,
        )
        wheel_vx = state.v_x - state.yaw_rate * position_y
        wheel_vy = state.v_y + state.yaw_rate * position_x
        cos_delta = np.cos(steering)
        sin_delta = np.sin(steering)
        longitudinal_velocity = cos_delta * wheel_vx + sin_delta * wheel_vy
        lateral_velocity = -sin_delta * wheel_vx + cos_delta * wheel_vy
        denominator = np.maximum(np.abs(longitudinal_velocity), p.slip_speed_epsilon)
        slip_angle = np.arctan2(lateral_velocity, denominator)
        slip_ratio = (
            p.wheel_radius * state.wheel_speed - longitudinal_velocity
        ) / denominator
        return (
            longitudinal_velocity,
            lateral_velocity,
            slip_angle,
            slip_ratio,
        )

    def _evaluate(
        self,
        vector: np.ndarray,
        control: ControlInput,
    ) -> WheelMeasurement:
        p = self.params
        state = VehicleState.from_vector(vector)
        position_x, position_y = p.wheel_positions
        steering = np.array(
            [state.steer_front, state.steer_front, state.steer_rear, state.steer_rear],
            dtype=float,
        )
        wheel_vx, wheel_vy, slip_angle, slip_ratio = self._wheel_kinematics(state)
        normal_load = self._normal_loads(state.v_x)
        fx, fy = p.tire.forces(
            slip_angle,
            slip_ratio,
            normal_load,
            p.friction_coefficient,
        )

        cos_delta = np.cos(steering)
        sin_delta = np.sin(steering)
        body_fx = fx * cos_delta - fy * sin_delta
        body_fy = fx * sin_delta + fy * cos_delta

        drag = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * state.v_x * abs(state.v_x)
        rolling = p.rolling_resistance * p.mass * p.gravity * np.tanh(state.v_x / 0.2)
        total_fx = float(np.sum(body_fx) - drag - rolling)
        total_fy = float(np.sum(body_fy))
        total_mz = float(np.sum(position_x * body_fy - position_y * body_fx))
        a_x = total_fx / p.mass + state.v_y * state.yaw_rate
        a_y = total_fy / p.mass - state.v_x * state.yaw_rate
        yaw_acceleration = total_mz / p.yaw_inertia

        return WheelMeasurement(
            slip_angle=slip_angle,
            slip_ratio=slip_ratio,
            normal_load=normal_load,
            longitudinal_force=fx,
            lateral_force=fy,
            body_longitudinal_force=body_fx,
            body_lateral_force=body_fy,
            wheel_longitudinal_velocity=wheel_vx,
            wheel_lateral_velocity=wheel_vy,
            wheel_steering_angle=steering,
            a_x=float(a_x),
            a_y=float(a_y),
            yaw_acceleration=float(yaw_acceleration),
        )

    def _derivatives(self, vector: np.ndarray, control: ControlInput) -> np.ndarray:
        p = self.params
        state = VehicleState.from_vector(vector)
        measurement = self._evaluate(vector, control)
        position_x, position_y = p.wheel_positions
        steering_target = np.array(
            [
                np.clip(control.steer_front, -p.max_steering_angle, p.max_steering_angle),
                np.clip(control.steer_front, -p.max_steering_angle, p.max_steering_angle),
                np.clip(control.steer_rear, -p.max_steering_angle, p.max_steering_angle),
                np.clip(control.steer_rear, -p.max_steering_angle, p.max_steering_angle),
            ],
            dtype=float,
        )
        wheel_speed_sign = np.sign(state.wheel_speed)
        moving_wheel_sign = np.sign(measurement.wheel_longitudinal_velocity)
        wheel_speed_sign = np.where(np.abs(state.wheel_speed) > 1e-4, wheel_speed_sign, moving_wheel_sign)
        brake_torque = control.wheel_brake * wheel_speed_sign
        wheel_angular_acceleration = (
            control.wheel_torque
            - brake_torque
            - p.wheel_radius * measurement.longitudinal_force
        ) / p.wheel_inertia

        derivatives = np.zeros(12, dtype=float)
        cos_yaw = np.cos(state.yaw)
        sin_yaw = np.sin(state.yaw)
        derivatives[0] = state.v_x * cos_yaw - state.v_y * sin_yaw
        derivatives[1] = state.v_x * sin_yaw + state.v_y * cos_yaw
        derivatives[2] = state.yaw_rate
        derivatives[3] = measurement.a_x
        derivatives[4] = measurement.a_y
        derivatives[5] = measurement.yaw_acceleration
        derivatives[6:10] = wheel_angular_acceleration
        derivatives[10] = (
            steering_target[0] - state.steer_front
        ) / p.steering_time_constant
        derivatives[11] = (
            steering_target[2] - state.steer_rear
        ) / p.steering_time_constant
        return derivatives

    def step(self, control: ControlInput | None = None, dt: float = 0.01) -> VehicleState:
        """Advance the vehicle by ``dt`` seconds and return its new state."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        control = control or ControlInput()
        vector = self.state.to_vector()
        k1 = self._derivatives(vector, control)
        k2 = self._derivatives(vector + 0.5 * dt * k1, control)
        k3 = self._derivatives(vector + 0.5 * dt * k2, control)
        k4 = self._derivatives(vector + dt * k3, control)
        next_vector = vector + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.state = VehicleState.from_vector(next_vector)
        self.state.yaw = self._wrap_angle(self.state.yaw)
        final_derivative = self._derivatives(self.state.to_vector(), control)
        self._last_a_x = float(final_derivative[3])
        self._last_a_y = float(final_derivative[4])
        self.state.a_x = self._last_a_x
        self.state.a_y = self._last_a_y
        self.last_measurement = self._evaluate(self.state.to_vector(), control)
        self.last_measurement.a_x = self._last_a_x
        self.last_measurement.a_y = self._last_a_y
        self.last_measurement.yaw_acceleration = float(final_derivative[5])
        return self.state.copy()
