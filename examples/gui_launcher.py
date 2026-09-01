"""Desktop launcher for the four-wheel vehicle simulation."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np


if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(APP_DIR / "src"))

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


def run_simulation(duration: float, target_speed: float, radius: float):
    """Run the default circle scenario and return its log and path."""

    path = make_circle_path(radius=radius)
    params = VehicleParams(initial_speed=0.5)
    initial_state = VehicleState(
        x=radius,
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
        target_speed=target_speed,
    )
    return simulator.run(duration), path


class SimulationLauncher(tk.Tk):
    """Small GUI for running and inspecting the default simulation."""

    def __init__(self) -> None:
        super().__init__()
        self.title("四轮车辆控制仿真")
        self.geometry("430x250")
        self.resizable(False, False)
        self.figure = None

        self.duration_var = tk.StringVar(value="20")
        self.target_speed_var = tk.StringVar(value="5.0")
        self.radius_var = tk.StringVar(value="20.0")
        self.status_var = tk.StringVar(value="填写参数后点击“运行仿真”")

        frame = ttk.Frame(self, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)

        self._add_input(frame, 0, "仿真时长 [s]", self.duration_var)
        self._add_input(frame, 1, "目标速度 [m/s]", self.target_speed_var)
        self._add_input(frame, 2, "圆形路径半径 [m]", self.radius_var)

        self.run_button = ttk.Button(frame, text="运行仿真", command=self._run)
        self.run_button.grid(row=3, column=0, columnspan=2, pady=(18, 10))
        ttk.Label(
            frame,
            textvariable=self.status_var,
            wraplength=380,
            justify=tk.LEFT,
        ).grid(row=4, column=0, columnspan=2, sticky="w")
        self.protocol("WM_DELETE_WINDOW", self._close)

    @staticmethod
    def _add_input(frame: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=variable, width=18).grid(
            row=row,
            column=1,
            sticky="e",
            pady=5,
        )

    @staticmethod
    def _positive_value(variable: tk.StringVar, label: str) -> float:
        try:
            value = float(variable.get())
        except ValueError as error:
            raise ValueError(f"{label}必须是数字") from error
        if value <= 0.0:
            raise ValueError(f"{label}必须大于 0")
        return value

    def _run(self) -> None:
        self.run_button.configure(state=tk.DISABLED)
        try:
            duration = self._positive_value(self.duration_var, "仿真时长")
            target_speed = self._positive_value(self.target_speed_var, "目标速度")
            radius = self._positive_value(self.radius_var, "路径半径")
            self.status_var.set("仿真运行中，请稍候……")
            self.update_idletasks()

            log, path = run_simulation(duration, target_speed, radius)
            import matplotlib.pyplot as plt

            if self.figure is not None:
                plt.close(self.figure)
            output_path = APP_DIR / "artifacts" / "latest_demo.png"
            self.figure, _ = plot_simulation(
                log,
                path=path,
                save_path=output_path,
                show=False,
            )
            self.figure.show()
            data = log.as_dict()
            rms_error = float(np.sqrt(np.mean(data["cross_track_error"] ** 2)))
            self.status_var.set(
                f"完成：最终速度 {data['v_x'][-1]:.3f} m/s，"
                f"横向误差 RMS {rms_error:.3f} m\n"
                f"结果已保存到：{output_path}"
            )
        except Exception as error:
            self.status_var.set("仿真失败，请检查输入参数。")
            messagebox.showerror("仿真失败", f"{type(error).__name__}: {error}", parent=self)
        finally:
            self.run_button.configure(state=tk.NORMAL)

    def _close(self) -> None:
        if self.figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self.figure)
        self.destroy()


def main() -> None:
    SimulationLauncher().mainloop()


if __name__ == "__main__":
    main()
