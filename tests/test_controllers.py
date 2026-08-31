import numpy as np

from vehicle_sim.controllers import PIDController, PurePursuitController
from vehicle_sim.model import VehicleState
from vehicle_sim.path import Path2D


def test_pid_respects_output_limits_and_recovers_from_saturation() -> None:
    controller = PIDController(
        kp=2.0,
        ki=1.0,
        kd=0.0,
        dt=0.1,
        output_limits=(-1.0, 1.0),
    )
    outputs = [controller.update(10.0, 0.0) for _ in range(20)]
    assert max(outputs) <= 1.0
    assert min(outputs) >= -1.0
    recovery = controller.update(0.0, 0.0)
    assert recovery <= 1.0


def test_pure_pursuit_points_forward_on_straight_path() -> None:
    path = Path2D(np.array([[0.0, 0.0], [20.0, 0.0]]), loop=False)
    state = VehicleState(x=2.0, y=0.0, yaw=0.0, v_x=5.0)
    command = PurePursuitController(2.8, lookahead_distance=3.0).command(state, path)
    assert np.isclose(command.front, 0.0)
    assert np.isclose(command.rear, 0.0)
