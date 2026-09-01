# 四轮车辆控制仿真工具

一个面向学习、算法验证和工程原型开发的四轮车辆控制仿真工具。

> 当前状态：MVP 已实现，正在进行仿真参数和控制性能校核。

## 快速开始

```powershell
py -3 -m pip install -e .
py -3 examples/run_demo.py --duration 20 --save artifacts/circle_demo.png
py -3 -m pytest
```

示例会运行一个圆形路径闭环仿真，并输出最终速度和横向误差 RMS。绘图依赖 Matplotlib，核心车辆模型和测试只依赖 NumPy。

## Windows 一键启动

如果只想双击运行，可以执行仓库根目录下的 `build_windows_exe.ps1`。脚本会自动安装打包依赖，并生成：

```text
release/FourWheelVehicleSimulation.exe
```

双击该文件后，在窗口中设置仿真时长、目标速度和圆形路径半径，点击“运行仿真”即可。结果图片会保存到 exe 同目录下的 `artifacts/latest_demo.png`。

启动器采用面向调试的单窗口布局：左侧是工况参数和操作按钮，中间提供“实时波形”和“轨迹与轮胎”两个数据视图，右侧显示速度、跟踪误差、滑移率、横摆角速度和四轮末状态，底部保留运行日志。波形工具栏支持平移、缩放和保存。

每次运行还会自动保存：

```text
artifacts/latest_demo.csv
artifacts/latest_wave.png
artifacts/latest_analysis.png
```

## 当前已实现

- 四轮平面动力学模型：车身纵向、横向和横摆运动，以及四轮独立轮速状态；
- Pacejka Magic Formula 轮胎模型和联合滑移摩擦椭圆约束；
- 基于纵向/横向载荷转移、空气动力和滚动阻力的轮胎法向载荷估计；
- 前轮转向、后轮转向比例、转向执行器一阶动态和幅值限制；
- 带抗积分饱和和微分滤波的速度 PID；
- Pure Pursuit 路径跟踪控制器和无第三方几何依赖的二维路径工具；
- 前驱、后驱、全驱扭矩分配，以及基础横摆力矩分配接口；
- 固定步长 RK4 积分、仿真日志、CSV 路径读取和可选 Matplotlib 可视化；
- 仿真日志 CSV 导出，四轮轮速、滑移率、法向载荷和控制量按车轮展开；
- 车辆、轮胎、路径、控制器和闭环仿真的自动化测试。

## 1. 项目目标

本项目用于建立一个结构清晰、参数可配置、便于扩展的车辆控制仿真环境，重点验证以下问题：

- 四轮车辆的运动学与动力学响应；
- 纵向速度控制和横向路径跟踪；
- 轮胎侧偏、纵向滑移和路面附着变化；
- 四轮独立驱动时的扭矩分配与横摆力矩控制；
- 前后轮转向、四轮转向等车辆构型的控制效果；
- 控制器在典型测试工况下的稳定性和跟踪误差。

本项目定位为仿真和研究工具，不直接用于真实车辆的安全控制或自动驾驶量产系统。

## 2. 首版范围

首版采用 Python 实现，核心依赖 NumPy，可选 Matplotlib 用于结果可视化，优先保证模型可读性、可重复运行和结果可视化。

### 2.1 车辆模型

首版采用平面车辆模型，包含车身纵向、横向和横摆运动，并为四个车轮保留独立的轮胎力和车轮转速计算接口。

当前支持以下输入：

- 前轮平均转向角和后轮转向角（后轮转向比例可配置）；
- 四轮独立驱动扭矩和四轮制动力；
- 车辆质量、轴距、轮距、质心位置和质心高度；
- 全局路面附着系数、空气动力和滚动阻力参数。

当前输出以下状态和测量量：

- 车辆位置、航向角、纵向速度和横向速度；
- 横摆角速度、车辆质心侧偏角和横向加速度；
- 四个车轮的轮速、侧偏角、滑移率和轮胎力；
- 四轮垂向载荷、控制输入和轨迹跟踪误差。

### 2.2 轮胎模型

当前已实现并可继续扩展：

1. Pacejka Magic Formula 纵向和横向轮胎力；
2. 联合滑移摩擦椭圆约束；
3. 轮胎峰值附着系数和 Pacejka 参数配置。

线性轮胎模型、每轮独立附着系数和道路坡度列入后续扩展。

### 2.3 控制器

当前已实现：

- 纵向速度 PID 控制器；
- Pure Pursuit 路径跟踪控制器；
- 四轮独立驱动的基础扭矩分配器；

Stanley、横摆角速度反馈和 MPC 接口列入后续扩展。

## 3. 仿真工况

项目将使用标准化工况验证车辆模型和控制器：

- 直线加速和定速行驶；
- 阶跃转向；
- 定半径圆周行驶；
- 蛇形路径跟踪；
- 双移线测试；
- 不同附着系数路面；
- 前后轮转向比例变化；
- 四轮驱动扭矩不均衡和横摆力矩修正。

每个工况应保存输入、状态、轮胎力、控制输出和评价指标，便于重复测试和对比不同控制器。

## 4. 软件结构

```text
four-wheel-vehicle-control-simulation/
├── README.md
├── pyproject.toml
├── src/
│   └── vehicle_sim/
│       ├── model.py         # 车辆状态、四轮动力学和积分器
│       ├── tire.py          # Pacejka 轮胎和联合滑移
│       ├── controllers.py   # PID 和 Pure Pursuit
│       ├── allocation.py    # 驱动形式和横摆力矩分配
│       ├── path.py          # 二维路径和投影
│       ├── simulation.py    # 闭环循环和数据记录
│       ├── scenarios.py     # 圆形、正弦和椭圆路径
│       └── visualization.py # 可选结果绘图
├── tests/                   # 单元测试和回归测试
└── examples/                # 可直接运行的示例
```

## 5. 开发原则

- 模型、控制器、仿真循环和可视化相互解耦；
- 所有物理量明确单位，默认采用 SI 单位制；
- 车辆参数和控制器参数集中在可校验的数据类中管理；
- 每增加一种模型，都提供最小可运行示例和基本测试；
- 仿真结果应能够保存为 CSV 或 NumPy 数据文件；
- 控制器应限制执行器幅值、变化率和积分累积；
- 对低速、零速、低附着和轮胎饱和等边界情况进行单独测试。

## 6. 评价指标

计划使用以下指标比较不同模型和控制器：

- 横向位置误差的最大值、平均值和均方根值；
- 航向角误差和横摆角速度误差；
- 速度跟踪误差；
- 最大横向加速度和最大横摆角速度；
- 轮胎滑移率、侧偏角和摩擦利用率；
- 控制输入峰值、变化率和能量消耗；
- 仿真实时性和单步计算时间。

## 7. 开发路线

### 阶段一：基础仿真框架

- 建立车辆参数对象和状态对象；
- 实现四轮运动学模型；
- 实现固定步长仿真循环和结果记录；
- 完成轨迹、速度和横摆状态可视化。

### 阶段二：车辆动力学

- 加入车身纵向、横向和横摆动力学；
- 加入轮速和纵向滑移率；
- 加入线性轮胎模型和摩擦圆约束；
- 完成阶跃转向、圆周和蛇形工况。

### 阶段三：控制算法

- 实现速度 PID；
- 实现 Pure Pursuit 和 Stanley；
- 增加横摆角速度反馈；
- 建立控制器参数扫描和结果对比工具。

### 阶段四：四轮独立控制

- 实现四轮独立驱动扭矩分配；
- 增加基于横摆力矩的控制分配；
- 增加分离路面和轮胎饱和测试；
- 预留四轮独立转向和 MPC 接口。

### 阶段五：工程化和发布

- 增加单元测试和持续集成；
- 固化示例参数和测试结果；
- 完善数学模型文档；
- 发布首个可复现实验版本。

## 8. 开源参考项目

以下项目用于学习模型结构、控制器接口和测试方法，不直接复制其源代码：

- [Toy_car_controller](https://github.com/ben-du-pont/Toy_car_controller)：Python 四轮模型、Pacejka 轮胎、PID 和 Pure Pursuit，MIT License。
- [TUMFTM/sim_vehicle_dynamics](https://github.com/TUMFTM/sim_vehicle_dynamics)：MATLAB/Simulink 非线性单轨和双轨车辆动力学模型，LGPL-3.0。
- [vehicle-dynamics-models-matlab-simulink](https://github.com/jaykumarpatil9099/vehicle-dynamics-models-matlab-simulink)：包含 14 自由度整车模型和 Pacejka 轮胎模型，MIT License。
- [electronic-differential](https://github.com/meltinglab/electronic-differential)：四个独立驱动轮、虚拟差速器、ESP 和 ABS 控制参考，MIT License，但依赖 MATLAB/Simulink 商业工具箱。
- [gazebo_ros_four_wheel_steering](https://github.com/Kettenhoax/gazebo_ros_four_wheel_steering)：ROS2/Gazebo 四轮转向和四轮驱动插件，仓库内不同文件需分别遵循其许可证说明。
- [TUMFTM/Open-Car-Dynamics](https://github.com/TUMFTM/Open-Car-Dynamics)：模块化 C++ 车辆动力学库，提供 Python 和 ROS2 接口，Apache-2.0。
- [TUMFTM/YawMomentDiagrams](https://github.com/TUMFTM/YawMomentDiagrams)：用于车辆操纵性、稳定性和横摆力矩分析，LGPL-3.0。

## 9. 许可证和第三方代码

本项目的最终许可证待首版代码结构确定后再决定。引用或使用第三方代码时，应：

- 保留原始版权声明和许可证文件；
- 在文档中明确说明使用的项目和修改内容；
- 单独检查代码、数据、模型、图片和商业软件依赖的许可证；
- 不将 MATLAB、CarMaker 或其他商业工具箱误标为开源依赖；
- 不将本项目用于未经验证的真实车辆安全控制。

## 10. 当前待办

- [x] 确定车辆参数和状态变量定义；
- [x] 确定 Python 最低版本和依赖版本；
- [x] 实现第一版四轮车辆模型；
- [ ] 实现直线和阶跃转向示例；
- [x] 增加 PID 和 Pure Pursuit 控制器；
- [x] 增加自动化测试；
- [x] 发布第一个可运行示例。
