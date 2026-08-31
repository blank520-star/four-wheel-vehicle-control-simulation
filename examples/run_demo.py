"""Run a minimal closed-loop circle-tracking demonstration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vehicle_sim import (  # noqa: E402
    ClosedLoopSimulator,
    FourWheelVehicle,
    PIDController,
    PurePursuitController,
    VehicleParams,
    VehicleState,
)
from vehicle_sim.scenarios import make_circle_path  # noqa: E402
from vehicle_sim.visualization import plot_simulation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    path = make_circle_path(radius=20.0)
    params = VehicleParams(initial_speed=0.5)
    initial_state = VehicleState(
        x=20.0,
        y=0.0,
        yaw=np.pi / 2.0,
        v_x=0.5,
        wheel_speed=np.full(4, 0.5 / params.wheel_radius),
    )
    vehicle = FourWheelVehicle(params, initial_state)
    path_controller = PurePursuitController(
        wheelbase=params.wheelbase,
        lookahead_distance=4.0,
        max_steering_angle=params.max_steering_angle,
        lookahead_speed_gain=0.15,
    )
    speed_controller = PIDController(
        kp=0.35,
        ki=0.12,
        kd=0.02,
        dt=0.01,
        output_limits=(-1.0, 1.0),
        integral_limits=(-2.0, 2.0),
        derivative_filter=0.2,
    )
    simulator = ClosedLoopSimulator(
        vehicle=vehicle,
        path=path,
        path_controller=path_controller,
        speed_controller=speed_controller,
        dt=0.01,
        target_speed=5.0,
    )
    log = simulator.run(args.duration)
    data = log.as_dict()
    print(f"steps: {len(log)}")
    print(f"final speed: {data['v_x'][-1]:.3f} m/s")
    print(f"RMS cross-track error: {np.sqrt(np.mean(data['cross_track_error'] ** 2)):.3f} m")
    if args.save is not None or args.show:
        plot_simulation(log, path=path, save_path=args.save, show=args.show)


if __name__ == "__main__":
    main()
