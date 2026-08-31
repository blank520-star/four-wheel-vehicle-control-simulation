import numpy as np

from vehicle_sim.model import ControlInput, FourWheelVehicle, VehicleParams, VehicleState


def test_vehicle_moves_forward_under_drive_torque() -> None:
    params = VehicleParams(initial_speed=0.0)
    vehicle = FourWheelVehicle(params)
    control = ControlInput(wheel_torque=np.full(4, 120.0))
    for _ in range(100):
        vehicle.step(control, 0.01)
    assert vehicle.state.x > 0.0
    assert vehicle.state.v_x > 0.0
    assert np.all(np.isfinite(vehicle.state.wheel_speed))


def test_vehicle_accepts_rear_steering_and_reports_four_wheels() -> None:
    params = VehicleParams(initial_speed=4.0)
    state = VehicleState(
        v_x=4.0,
        wheel_speed=np.full(4, 4.0 / params.wheel_radius),
    )
    vehicle = FourWheelVehicle(params, state)
    vehicle.step(ControlInput(steer_front=0.1, steer_rear=-0.03), 0.01)
    measurement = vehicle.last_measurement
    assert measurement.slip_angle.shape == (4,)
    assert measurement.normal_load.shape == (4,)
    assert np.all(np.isfinite(measurement.lateral_force))
