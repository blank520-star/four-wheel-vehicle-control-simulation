"""VOFA-inspired desktop dashboard for the four-wheel vehicle simulation."""

from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import font_manager

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


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


WHEEL_NAMES = ("fl", "fr", "rl", "rr")
WHEEL_LABELS = {"fl": "左前", "fr": "右前", "rl": "左后", "rr": "右后"}
ACCENT = "#2dd4bf"
BLUE = "#60a5fa"
ORANGE = "#fb923c"
PURPLE = "#c084fc"
RED = "#fb7185"
TEXT = "#e5edf5"
MUTED = "#91a4b7"
PANEL = "#17212b"
PLOT = "#0f171f"
GRID = "#314252"
WINDOW = "#0d141b"


def _configure_matplotlib_font() -> None:
    """Prefer a Windows CJK font for Chinese plot labels."""

    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei", "SimSun"):
        if font_name in available:
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_font()


def run_simulation(
    duration: float,
    target_speed: float,
    radius: float,
    lookahead_distance: float = 4.0,
    friction_coefficient: float = 1.0,
):
    """Run the default circle scenario and return its log and path."""

    path = make_circle_path(radius=radius)
    params = VehicleParams(
        initial_speed=0.5,
        friction_coefficient=friction_coefficient,
    )
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
        lookahead_distance=lookahead_distance,
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
    """Single-window simulation dashboard with plots and wheel telemetry."""

    def __init__(self) -> None:
        super().__init__()
        self.title("四轮车辆控制实验台")
        self.geometry("1380x860")
        self.minsize(1120, 720)
        self.configure(background=WINDOW)

        self.last_log = None
        self.last_path = None
        self.wave_figure = Figure(figsize=(8, 5), dpi=100, facecolor=PLOT)
        self.analysis_figure = Figure(figsize=(8, 5), dpi=100, facecolor=PLOT)
        self.wave_canvas = None
        self.analysis_canvas = None

        self.duration_var = tk.StringVar(value="20")
        self.target_speed_var = tk.StringVar(value="5.0")
        self.radius_var = tk.StringVar(value="20.0")
        self.lookahead_var = tk.StringVar(value="4.0")
        self.friction_var = tk.StringVar(value="1.0")
        self.status_var = tk.StringVar(value="就绪 | 等待运行仿真")
        self.metric_vars = {
            "final_speed": tk.StringVar(value="--"),
            "rms_error": tk.StringVar(value="--"),
            "max_slip": tk.StringVar(value="--"),
            "peak_yaw_rate": tk.StringVar(value="--"),
        }

        self._configure_styles()
        self._build_layout()
        self._draw_empty_plots()
        self._append_log("应用已启动。请在左侧设置工况后点击“运行仿真”。")

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("App.TFrame", background=WINDOW)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Card.TFrame", background="#202d39")
        style.configure("Title.TLabel", background=WINDOW, foreground=TEXT, font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=WINDOW, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Heading.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 11, "bold"))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 8))
        style.configure("MetricName.TLabel", background="#202d39", foreground=MUTED, font=("Segoe UI", 8))
        style.configure("MetricValue.TLabel", background="#202d39", foreground=ACCENT, font=("Segoe UI", 14, "bold"))
        style.configure("TButton", background="#263746", foreground=TEXT, borderwidth=0, padding=(10, 7))
        style.map("TButton", background=[("active", "#355064"), ("disabled", "#1b2731")])
        style.configure("Run.TButton", background="#087f70", foreground="#ffffff", font=("Segoe UI", 9, "bold"), padding=(12, 8))
        style.map("Run.TButton", background=[("active", "#0aa58f"), ("disabled", "#1b4b47")])
        style.configure("TEntry", fieldbackground="#0f171f", foreground=TEXT, insertcolor=TEXT, bordercolor="#3a4c5b", padding=5)
        style.configure("TNotebook", background=WINDOW, borderwidth=0)
        style.configure("TNotebook.Tab", background="#1c2a36", foreground=MUTED, padding=(14, 7))
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
        style.configure("Treeview", background="#101a23", fieldbackground="#101a23", foreground=TEXT, rowheight=27, borderwidth=0)
        style.configure("Treeview.Heading", background="#263746", foreground=TEXT, relief="flat", padding=5)
        style.map("Treeview", background=[("selected", "#245056")], foreground=[("selected", "#ffffff")])
        style.configure("TCheckbutton", background=PANEL, foreground=MUTED)

    def _build_layout(self) -> None:
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        header = ttk.Frame(self, style="App.TFrame", padding=(20, 16, 20, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="四轮车辆控制实验台", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Four-wheel vehicle dynamics  /  path tracking  /  telemetry",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(header, textvariable=self.status_var, style="Subtitle.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        body = ttk.Frame(self, style="App.TFrame", padding=(12, 0, 12, 0))
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0)

        self._build_control_panel(body).grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._build_plot_panel(body).grid(row=0, column=1, sticky="nsew")
        self._build_telemetry_panel(body).grid(row=0, column=2, sticky="nsew", padx=(10, 0))

        footer = ttk.Frame(self, style="Panel.TFrame", padding=(14, 8, 14, 8))
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, text="运行日志", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(
            footer,
            height=4,
            background="#101a23",
            foreground="#b7c7d6",
            insertbackground=TEXT,
            relief=tk.FLAT,
            borderwidth=0,
            font=("Consolas", 9),
            state=tk.DISABLED,
        )
        self.log_text.grid(row=1, column=0, sticky="ew", pady=(5, 0))

    def _build_control_panel(self, parent: ttk.Frame) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=14, width=230)
        panel.grid_propagate(False)
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=0)
        ttk.Label(panel, text="工况与参数", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, text="圆形路径闭环测试", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 12))

        inputs = [
            ("仿真时长 [s]", self.duration_var),
            ("目标速度 [m/s]", self.target_speed_var),
            ("路径半径 [m]", self.radius_var),
            ("前视距离 [m]", self.lookahead_var),
            ("路面附着系数 μ", self.friction_var),
        ]
        for row, (label, variable) in enumerate(inputs, start=2):
            self._add_input(panel, row, label, variable)

        ttk.Separator(panel).grid(row=7, column=0, sticky="ew", pady=14)
        ttk.Label(panel, text="操作", style="Heading.TLabel").grid(row=8, column=0, sticky="w")
        self.run_button = ttk.Button(panel, text="▶  运行仿真", style="Run.TButton", command=self._run)
        self.run_button.grid(row=9, column=0, sticky="ew", pady=(10, 6))
        self.reset_button = ttk.Button(panel, text="↺  清空结果", command=self._reset)
        self.reset_button.grid(row=10, column=0, sticky="ew", pady=3)
        self.save_button = ttk.Button(panel, text="⇩  导出最新数据", command=self._save_results, state=tk.DISABLED)
        self.save_button.grid(row=11, column=0, sticky="ew", pady=3)
        self.open_button = ttk.Button(panel, text="□  打开结果目录", command=self._open_output_dir)
        self.open_button.grid(row=12, column=0, sticky="ew", pady=3)

        ttk.Separator(panel).grid(row=13, column=0, sticky="ew", pady=14)
        ttk.Label(panel, text="调试提示", style="Heading.TLabel").grid(row=14, column=0, sticky="w")
        ttk.Label(
            panel,
            text="滚轮缩放，工具栏可平移/保存。\nCSV 中的四轮列名以 _fl、_fr、_rl、_rr 结尾。",
            style="Muted.TLabel",
            wraplength=195,
            justify=tk.LEFT,
        ).grid(row=15, column=0, sticky="w", pady=(7, 0))
        return panel

    def _build_plot_panel(self, parent: ttk.Frame) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="数据监视", style="Heading.TLabel").grid(row=0, column=0, sticky="w", padx=4, pady=(0, 6))

        notebook = ttk.Notebook(panel)
        notebook.grid(row=1, column=0, sticky="nsew")
        wave_tab = ttk.Frame(notebook, style="Panel.TFrame")
        analysis_tab = ttk.Frame(notebook, style="Panel.TFrame")
        notebook.add(wave_tab, text="实时波形")
        notebook.add(analysis_tab, text="轨迹与轮胎")

        wave_tab.rowconfigure(0, weight=1)
        wave_tab.columnconfigure(0, weight=1)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_figure, master=wave_tab)
        self.wave_canvas.draw()
        self.wave_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.wave_canvas, wave_tab, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        analysis_tab.rowconfigure(0, weight=1)
        analysis_tab.columnconfigure(0, weight=1)
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_figure, master=analysis_tab)
        self.analysis_canvas.draw()
        self.analysis_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        analysis_toolbar = NavigationToolbar2Tk(self.analysis_canvas, analysis_tab, pack_toolbar=False)
        analysis_toolbar.update()
        analysis_toolbar.grid(row=1, column=0, sticky="ew")
        return panel

    def _build_telemetry_panel(self, parent: ttk.Frame) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=14, width=300)
        panel.grid_propagate(False)
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="关键指标", style="Heading.TLabel").grid(row=0, column=0, sticky="w")

        metric_frame = ttk.Frame(panel, style="Panel.TFrame")
        metric_frame.grid(row=1, column=0, sticky="ew", pady=(10, 12))
        metric_frame.columnconfigure(0, weight=1)
        metric_frame.columnconfigure(1, weight=1)
        metrics = [
            ("最终速度", "final_speed", "m/s"),
            ("横向误差 RMS", "rms_error", "m"),
            ("最大滑移率", "max_slip", ""),
            ("峰值横摆角速度", "peak_yaw_rate", "rad/s"),
        ]
        for index, (name, key, unit) in enumerate(metrics):
            card = ttk.Frame(metric_frame, style="Card.TFrame", padding=(9, 7))
            card.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
            ttk.Label(card, text=name, style="MetricName.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=self.metric_vars[key], style="MetricValue.TLabel").pack(anchor="w", pady=(3, 0))
            ttk.Label(card, text=unit, style="MetricName.TLabel").pack(anchor="w")

        ttk.Label(panel, text="四轮遥测（末状态）", style="Heading.TLabel").grid(row=2, column=0, sticky="w", pady=(4, 8))
        columns = ("wheel", "load", "slip", "angle")
        self.wheel_tree = ttk.Treeview(panel, columns=columns, show="headings", height=5)
        headings = {"wheel": "车轮", "load": "Fz [N]", "slip": "κ", "angle": "α [rad]"}
        widths = {"wheel": 55, "load": 72, "slip": 58, "angle": 68}
        for column in columns:
            self.wheel_tree.heading(column, text=headings[column])
            self.wheel_tree.column(column, width=widths[column], anchor="center", stretch=False)
        self.wheel_tree.grid(row=3, column=0, sticky="ew")

        ttk.Label(panel, text="读数说明", style="Heading.TLabel").grid(row=4, column=0, sticky="w", pady=(16, 6))
        ttk.Label(
            panel,
            text="κ > 0：驱动轮滑转\nα < 0：车辆左转侧偏约定\nFz：轮胎法向载荷",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).grid(row=5, column=0, sticky="w")
        return panel

    @staticmethod
    def _add_input(panel: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        panel.rowconfigure(row, pad=2)
        ttk.Label(panel, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w")
        ttk.Entry(panel, textvariable=variable, width=12).grid(row=row, column=1, sticky="e", pady=(2, 6))

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
        self.status_var.set("运行中 | 正在计算四轮动力学……")
        self.update_idletasks()
        try:
            duration = self._positive_value(self.duration_var, "仿真时长")
            target_speed = self._positive_value(self.target_speed_var, "目标速度")
            radius = self._positive_value(self.radius_var, "路径半径")
            lookahead = self._positive_value(self.lookahead_var, "前视距离")
            friction = self._positive_value(self.friction_var, "路面附着系数")

            log, path = run_simulation(duration, target_speed, radius, lookahead, friction)
            self.last_log = log
            self.last_path = path
            self._render_results()
            self._save_results(show_message=False)
            self.save_button.configure(state=tk.NORMAL)
            self._append_log(
                f"完成：{len(log)} 步，目标速度 {target_speed:.2f} m/s，"
                f"μ={friction:.2f}。"
            )
        except Exception as error:
            self.status_var.set("错误 | 仿真未完成")
            self._append_log(f"错误：{type(error).__name__}: {error}")
            messagebox.showerror("仿真失败", f"{type(error).__name__}: {error}", parent=self)
        finally:
            self.run_button.configure(state=tk.NORMAL)

    def _render_results(self) -> None:
        data = self.last_log.as_dict()
        rms_error = float(np.sqrt(np.mean(data["cross_track_error"] ** 2)))
        max_slip = float(np.max(np.abs(data["slip_ratio"])))
        peak_yaw_rate = float(np.max(np.abs(data["yaw_rate"])))
        self.metric_vars["final_speed"].set(f"{data['v_x'][-1]:.2f}")
        self.metric_vars["rms_error"].set(f"{rms_error:.3f}")
        self.metric_vars["max_slip"].set(f"{max_slip:.3f}")
        self.metric_vars["peak_yaw_rate"].set(f"{peak_yaw_rate:.3f}")

        for item in self.wheel_tree.get_children():
            self.wheel_tree.delete(item)
        for index, wheel in enumerate(WHEEL_NAMES):
            self.wheel_tree.insert(
                "",
                tk.END,
                values=(
                    WHEEL_LABELS[wheel],
                    f"{data['normal_load'][-1, index]:.0f}",
                    f"{data['slip_ratio'][-1, index]:.3f}",
                    f"{data['slip_angle'][-1, index]:.3f}",
                ),
            )

        self._draw_wave_plots(data)
        self._draw_analysis_plots(data)
        self.status_var.set("完成 | 数据已更新，可用工具栏缩放和保存图像")

    @staticmethod
    def _style_axis(axis, title: str, ylabel: str = "") -> None:
        axis.set_facecolor(PLOT)
        axis.set_title(title, color=TEXT, fontsize=10, loc="left", pad=8)
        axis.set_ylabel(ylabel, color=MUTED, fontsize=8)
        axis.tick_params(colors=MUTED, labelsize=8)
        for spine in axis.spines.values():
            spine.set_color(GRID)
        axis.grid(True, color=GRID, alpha=0.55, linewidth=0.6)

    def _draw_empty_plots(self) -> None:
        for figure, canvas, title in (
            (self.wave_figure, self.wave_canvas, "等待仿真数据"),
            (self.analysis_figure, self.analysis_canvas, "等待仿真数据"),
        ):
            figure.clear()
            axis = figure.add_subplot(111)
            self._style_axis(axis, title)
            axis.text(0.5, 0.5, "左侧设置参数后点击“运行仿真”", color=MUTED, ha="center", va="center", transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
            if canvas is not None:
                canvas.draw_idle()

    def _draw_wave_plots(self, data: dict[str, np.ndarray]) -> None:
        self.wave_figure.clear()
        axes = self.wave_figure.subplots(2, 2, squeeze=False)
        time = data["time"]
        self._style_axis(axes[0, 0], "速度 / 目标", "m/s")
        axes[0, 0].plot(time, data["v_x"], color=BLUE, label="v_x", linewidth=1.4)
        axes[0, 0].plot(time, data["target_speed"], color=MUTED, linestyle="--", label="target", linewidth=1.0)
        axes[0, 0].legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="lower right")

        self._style_axis(axes[0, 1], "路径横向误差", "m")
        axes[0, 1].plot(time, data["cross_track_error"], color=ORANGE, linewidth=1.3)
        axes[0, 1].axhline(0.0, color=MUTED, linewidth=0.7, alpha=0.6)

        self._style_axis(axes[1, 0], "横摆角速度", "rad/s")
        axes[1, 0].plot(time, data["yaw_rate"], color=PURPLE, linewidth=1.3)
        axes[1, 0].axhline(0.0, color=MUTED, linewidth=0.7, alpha=0.6)

        self._style_axis(axes[1, 1], "四轮滑移率", "κ")
        for wheel, color in zip(WHEEL_NAMES, (ACCENT, BLUE, ORANGE, RED)):
            index = WHEEL_NAMES.index(wheel)
            axes[1, 1].plot(time, data["slip_ratio"][:, index], color=color, label=WHEEL_LABELS[wheel], linewidth=1.0)
        axes[1, 1].legend(frameon=False, labelcolor=TEXT, fontsize=8, ncol=2)
        self.wave_figure.tight_layout(pad=1.5)
        self.wave_canvas.draw_idle()

    def _draw_analysis_plots(self, data: dict[str, np.ndarray]) -> None:
        self.analysis_figure.clear()
        axes = self.analysis_figure.subplots(2, 2, squeeze=False)
        self._style_axis(axes[0, 0], "车辆轨迹", "y [m]")
        reference = self.last_path.sample()
        axes[0, 0].plot(reference[:, 0], reference[:, 1], color=MUTED, linestyle="--", label="reference", linewidth=1.0)
        axes[0, 0].plot(data["x"], data["y"], color=ACCENT, label="vehicle", linewidth=1.4)
        axes[0, 0].axis("equal")
        axes[0, 0].legend(frameon=False, labelcolor=TEXT, fontsize=8)

        time = data["time"]
        self._style_axis(axes[0, 1], "四轮法向载荷", "N")
        for wheel, color in zip(WHEEL_NAMES, (ACCENT, BLUE, ORANGE, RED)):
            index = WHEEL_NAMES.index(wheel)
            axes[0, 1].plot(time, data["normal_load"][:, index], color=color, label=WHEEL_LABELS[wheel], linewidth=1.0)
        axes[0, 1].legend(frameon=False, labelcolor=TEXT, fontsize=8, ncol=2)

        self._style_axis(axes[1, 0], "四轮纵向力", "N")
        for wheel, color in zip(WHEEL_NAMES, (ACCENT, BLUE, ORANGE, RED)):
            index = WHEEL_NAMES.index(wheel)
            axes[1, 0].plot(time, data["longitudinal_force"][:, index], color=color, label=WHEEL_LABELS[wheel], linewidth=1.0)
        axes[1, 0].legend(frameon=False, labelcolor=TEXT, fontsize=8, ncol=2)

        self._style_axis(axes[1, 1], "控制输入", "幅值 / rad")
        axes[1, 1].plot(time, data["throttle"], color=ACCENT, label="throttle", linewidth=1.1)
        axes[1, 1].plot(time, data["steer_front"], color=ORANGE, label="front steer", linewidth=1.1)
        axes[1, 1].legend(frameon=False, labelcolor=TEXT, fontsize=8)
        self.analysis_figure.tight_layout(pad=1.5)
        self.analysis_canvas.draw_idle()

    def _save_results(self, show_message: bool = True) -> None:
        if self.last_log is None:
            if show_message:
                messagebox.showinfo("暂无数据", "请先运行一次仿真。", parent=self)
            return
        output_dir = APP_DIR / "artifacts"
        csv_path = self.last_log.to_csv(output_dir / "latest_demo.csv")
        wave_path = output_dir / "latest_wave.png"
        analysis_path = output_dir / "latest_analysis.png"
        self.wave_figure.savefig(wave_path, dpi=150, facecolor=self.wave_figure.get_facecolor())
        self.analysis_figure.savefig(analysis_path, dpi=150, facecolor=self.analysis_figure.get_facecolor())
        if show_message:
            self._append_log(f"已导出：{csv_path}")
            messagebox.showinfo("导出完成", f"CSV 和两张分析图已保存到：\n{output_dir}", parent=self)

    def _open_output_dir(self) -> None:
        output_dir = APP_DIR / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(output_dir))

    def _reset(self) -> None:
        self.last_log = None
        self.last_path = None
        for key in self.metric_vars:
            self.metric_vars[key].set("--")
        for item in self.wheel_tree.get_children():
            self.wheel_tree.delete(item)
        self.save_button.configure(state=tk.DISABLED)
        self._draw_empty_plots()
        self.status_var.set("就绪 | 结果已清空")
        self._append_log("已清空当前结果。")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def main() -> None:
    SimulationLauncher().mainloop()


if __name__ == "__main__":
    main()
