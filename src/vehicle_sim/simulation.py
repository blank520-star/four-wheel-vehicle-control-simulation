"""Closed-loop simulation loop and structured result logging."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .allocation import TorqueAllocator
from .controllers import PIDController, PurePursuitController, SteeringCommand
from .model import ControlInput, FourWheelVehicle, VehicleState, WheelMeasurement
from .path import Path2D


@dataclass
class SimulationLog:
    """In-memory log of one closed-loop simulation run."""

    _records: list[dict[str, object]] = field(default_factory=list)

    def append(
        self,
        time: float,
        state: VehicleState,
        measurement: WheelMeasurement,
        control: ControlInput,
        throttle: float,
        target_speed: float,
        projection_distance: float,
        cross_track_error: float,
        target_point: np.ndarray,
    ) -> None:
        self._records.append(
            {
                "time": float(time),
                "x": float(state.x),
                "y": float(state.y),
                "yaw": float(state.yaw),
                "v_x": float(state.v_x),
                "v_y": float(state.v_y),
                "yaw_rate": float(state.yaw_rate),
                "a_x": float(state.a_x),
                "a_y": float(state.a_y),
                "steer_front": float(state.steer_front),
                "steer_rear": float(state.steer_rear),
                "throttle": float(throttle),
                "target_speed": float(target_speed),
                "projection_distance": float(projection_distance),
                "cross_track_error": float(cross_track_error),
                "target_x": float(target_point[0]),
                "target_y": float(target_point[1]),
                "wheel_speed": state.wheel_speed.copy(),
                "slip_angle": measurement.slip_angle.copy(),
                "slip_ratio": measurement.slip_ratio.copy(),
                "normal_load": measurement.normal_load.copy(),
                "longitudinal_force": measurement.longitudinal_force.copy(),
                "lateral_force": measurement.lateral_force.copy(),
                "wheel_torque": control.wheel_torque.copy(),
                "wheel_brake": control.wheel_brake.copy(),
            }
        )

    def __len__(self) -> int:
        return len(self._records)

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the log as NumPy arrays suitable for plotting or export."""

        if not self._records:
            return {}
        result: dict[str, np.ndarray] = {}
        for key in self._records[0]:
            result[key] = np.asarray([record[key] for record in self._records])
        return result

    def to_csv(self, file_path: str | Path) -> Path:
        """Write scalar and four-wheel log values to a CSV file."""

        data = self.as_dict()
        if not data:
            raise ValueError("cannot export an empty simulation log")

        wheel_names = ("fl", "fr", "rl", "rr")
        scalar_keys: list[str] = []
        wheel_keys: list[str] = []
        for key, values in data.items():
            values = np.asarray(values)
            if values.ndim == 1:
                scalar_keys.append(key)
            elif values.ndim == 2 and values.shape[1] == 4:
                wheel_keys.append(key)
            else:
                raise ValueError(f"unsupported log shape for {key}: {values.shape}")

        fieldnames = scalar_keys + [
            f"{key}_{wheel}" for key in wheel_keys for wheel in wheel_names
        ]
        output = Path(file_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(len(self._records)):
                row = {key: float(data[key][index]) for key in scalar_keys}
                for key in wheel_keys:
                    for wheel_index, wheel in enumerate(wheel_names):
                        row[f"{key}_{wheel}"] = float(data[key][index, wheel_index])
                writer.writerow(row)
        return output


class ClosedLoopSimulator:
    """Connect a vehicle, path tracker, speed controller and actuator map."""

    def __init__(
        self,
        vehicle: FourWheelVehicle,
        path: Path2D,
        path_controller: PurePursuitController,
        speed_controller: PIDController,
        dt: float = 0.01,
        target_speed: float = 5.0,
        torque_allocator: TorqueAllocator | None = None,
        yaw_moment: float = 0.0,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if target_speed < 0.0:
            raise ValueError("target_speed cannot be negative")
        self.vehicle = vehicle
        self.path = path
        self.path_controller = path_controller
        self.speed_controller = speed_controller
        self.dt = float(dt)
        self.target_speed = float(target_speed)
        self.torque_allocator = torque_allocator or TorqueAllocator(
            wheel_radius=vehicle.params.wheel_radius,
            front_track=vehicle.params.front_track,
            rear_track=vehicle.params.rear_track,
        )
        self.yaw_moment = float(yaw_moment)
        self.time = 0.0
        self.log = SimulationLog()

    def reset(self, state: VehicleState | None = None) -> None:
        self.vehicle.reset(state)
        self.speed_controller.reset()
        self.time = 0.0
        self.log = SimulationLog()

    def step(self) -> VehicleState:
        state = self.vehicle.state
        steering: SteeringCommand = self.path_controller.command(state, self.path)
        throttle = self.speed_controller.update(self.target_speed, state.v_x, self.dt)
        control = self.torque_allocator.from_throttle(
            throttle,
            steer_front=steering.front,
            steer_rear=steering.rear,
            yaw_moment=self.yaw_moment,
        )
        next_state = self.vehicle.step(control, self.dt)
        projection = self.path.project(np.array([next_state.x, next_state.y]))
        target_point = self.path.lookahead(
            projection,
            self.path_controller.lookahead_distance
            + self.path_controller.lookahead_speed_gain * max(next_state.v_x, 0.0),
        )
        self.time += self.dt
        self.log.append(
            time=self.time,
            state=next_state,
            measurement=self.vehicle.last_measurement,
            control=control,
            throttle=throttle,
            target_speed=self.target_speed,
            projection_distance=projection.distance,
            cross_track_error=projection.cross_track_error,
            target_point=target_point,
        )
        return next_state

    def run(self, duration: float) -> SimulationLog:
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        steps = int(np.ceil(duration / self.dt))
        for _ in range(steps):
            self.step()
        return self.log
