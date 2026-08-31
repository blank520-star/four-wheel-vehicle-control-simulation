import numpy as np

from vehicle_sim import (
    ClosedLoopSimulator,
    FourWheelVehicle,
    PIDController,
    PurePursuitController,
    VehicleParams,
    VehicleState,
)
from vehicle_sim.scenarios import make_circle_path


def test_closed_loop_run_produces_finite_log() -> None:
    params = VehicleParams(initial_speed=0.5)
    path = make_circle_path(radius=15.0, count=120)
    vehicle = FourWheelVehicle(
        params,
        VehicleState(
            x=15.0,
            y=0.0,
            yaw=np.pi / 2.0,
            v_x=0.5,
            wheel_speed=np.full(4, 0.5 / params.wheel_radius),
        ),
    )
    simulator = ClosedLoopSimulator(
        vehicle=vehicle,
        path=path,
        path_controller=PurePursuitController(params.wheelbase, 3.5),
        speed_controller=PIDController(0.4, 0.1, 0.01, 0.01),
        dt=0.01,
        target_speed=3.0,
    )
    log = simulator.run(2.0)
    data = log.as_dict()
    assert len(log) == 200
    assert np.all(np.isfinite(data["x"]))
    assert np.all(np.isfinite(data["wheel_torque"]))
