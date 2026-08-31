"""Matplotlib-based result visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .path import Path2D
from .simulation import SimulationLog


def plot_simulation(
    log: SimulationLog,
    path: Path2D | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot trajectory and key closed-loop signals."""

    import matplotlib.pyplot as plt

    data = log.as_dict()
    if not data:
        raise ValueError("cannot plot an empty simulation log")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    trajectory = axes[0, 0]
    if path is not None:
        samples = path.sample()
        trajectory.plot(samples[:, 0], samples[:, 1], "k--", linewidth=1.0, label="reference")
    trajectory.plot(data["x"], data["y"], color="tab:blue", label="vehicle")
    trajectory.set_title("Trajectory")
    trajectory.set_xlabel("x [m]")
    trajectory.set_ylabel("y [m]")
    trajectory.axis("equal")
    trajectory.grid(True, alpha=0.3)
    trajectory.legend()

    speed = axes[0, 1]
    speed.plot(data["time"], data["v_x"], label="v_x")
    speed.plot(data["time"], data["target_speed"], "k--", label="target")
    speed.set_title("Longitudinal speed")
    speed.set_xlabel("time [s]")
    speed.set_ylabel("speed [m/s]")
    speed.grid(True, alpha=0.3)
    speed.legend()

    tracking = axes[1, 0]
    tracking.plot(data["time"], data["cross_track_error"], label="cross-track error")
    tracking.plot(data["time"], data["yaw_rate"], label="yaw rate")
    tracking.set_title("Tracking and yaw response")
    tracking.set_xlabel("time [s]")
    tracking.grid(True, alpha=0.3)
    tracking.legend()

    commands = axes[1, 1]
    commands.plot(data["time"], data["throttle"], label="throttle / brake")
    commands.plot(data["time"], data["steer_front"], label="front steer [rad]")
    if np.any(np.abs(data["steer_rear"]) > 1e-9):
        commands.plot(data["time"], data["steer_rear"], label="rear steer [rad]")
    commands.set_title("Control commands")
    commands.set_xlabel("time [s]")
    commands.grid(True, alpha=0.3)
    commands.legend()

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
    if show:
        plt.show()
    return figure, axes
