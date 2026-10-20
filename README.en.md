<div align="center">

# Scan · Plan · Spray

**From point cloud to real-robot spraying: a full-stack lab for wall operations on robots**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-ros2_control-00599C?logo=cplusplus&logoColor=white)
![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy-22314E?logo=ros&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-sim--2--sim-000000)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-RL-76B900)
![MATLAB](https://img.shields.io/badge/MATLAB-station--planning-ED7E0B)
![License](https://img.shields.io/badge/License-MIT-green)

[简体中文](README.md) | **English**

</div>

![overview](assets/figures/overview.png)

## What this is

Two complete technical tracks for the same task — **automatic line-tracing / spraying on walls**:

- **Humanoid track (Unitree G1)**: cabin point-cloud registration → spray station & stance planning → **dual-posture (stand + crouch)** constant-width zigzag trajectories with per-point IK → a web workstation → whole-body control execution, with the control policy based on NVIDIA **GEAR-SONIC** and one-switch sim/real modes (full-band spray verified on the physical G1).
- **Robot-arm track (OpenArm v1, 7-DOF)**: zero calibration → simulation → real robot, pushed through three layers (MPC / reinforcement learning / ILC), cutting one 36-waypoint run from ≈300 s down to **143.6 s** with zero e-stops.

The four modules are independent and each runs on its own. `g1-wall-spray` consumes exactly the output of `pointcloud-registration`, and its workspace computation inherits the method from `station-planning`.

This repository is an archival record of that work: every module keeps its runnable code, measured results and design documents.

## Modules

| Module | What it does | Entry point |
|---|---|---|
| [pointcloud-registration](pointcloud-registration/) | Register a grayscale robot scan against a semantic reference map + **label transfer**, hand-written in pure numpy | `python register_and_label.py` |
| [station-planning](station-planning/) | **Spray cell decomposition (BCD) and stance planning** on Monte-Carlo reachability | `station_planning.m` (MATLAB) |
| [g1-wall-spray](g1-wall-spray/) | G1 wall-spraying **web workstation + whole-body control** (policy based on GEAR-SONIC; stand+crouch dual posture, verified on the physical G1) | `./启动工作站.sh` → `127.0.0.1:8090` |
| [openarm-zigzag](openarm-zigzag/) | OpenArm 7-DOF **calibration → simulation → real robot**, with MPC/RL/ILC | see the module [README](openarm-zigzag/README.en.md) |

## Results at a glance

| Module | Metric | Value |
|---|---|---|
| pointcloud-registration | Alignment RMSE (full-size data) | **≈ 0.34 mm** |
| | Dependencies | numpy + matplotlib only — no PCL/Open3D |
| station-planning | 10 m × 2 m demo wall | 4 BCD cells → **12 stances**, gapless coverage |
| | Per-stance coverage (±30° waist yaw) | ≈1.1 m wide; reachability 1.12 m² / spray coverage 2.53 m² |
| g1-wall-spray | Per-point IK, dual-posture trajectory | **457/457 converged**, measured gun-wall angle 60–90° |
| | NN-IK universal model seed convergence (iters=60) | **64%** vs 23.8% random MC (no retrain across waist/crouch configs) |
| | V4.1 shallow-crouch+bend finalized (full sim audit 15/15, 2026-10-13) | full band 0.35–1.45 m with pelvis no deeper than ~0.55 m; crouch reference strictly within [0.55, 0.70] |
| | Full-band spray on the physical G1 (2026-10-12 deep-squat version, ×1.0, 155.6 s) | stand+crouch 0.35–1.45 m; zero mode-flapping while standing (V4.1 awaiting its first 0.5× real run) |
| | Base-disturbance compensation (wrist error, world frame) | 32 mm → **4 mm** |
| | Full planning pipeline (03 → packaging → final audit) | ≈ 33 s |
| openarm-zigzag | Full-run duration (MPC baseline → RL+ILC) | ≈300 s → **143.6 s**, zero e-stops |
| | Tracking / run-to-run repeatability | mean 1.19° / RMS **0.04°** |
| | Low-frequency sway suppression (0.5–2 Hz) | 3.8° → 0.7°; integrator peak down 2–5× |

## Repository layout

```
scan-plan-spray/
├── pointcloud-registration/   # point-cloud registration + semantic transfer (pure numpy)
│   ├── register_and_label.py  #   main algorithm (806 lines, full method notes in the header)
│   ├── make_sample_data.py    #   builds the downsampled sample (sample data is bundled)
│   ├── 舱室点云/                #   sample input ply files + ground-truth transform
│   ├── docs/                  #   algorithm design document (docx, in Chinese)
│   └── figures/
├── station-planning/          # spray cell & stance planning (BCD)
│   ├── station_planning.m     #   MATLAB entry: full 10×2 m demo
│   ├── workspace/             #   Python reachability (Tiangong 3.0 FK Monte-Carlo)
│   ├── docs/                  #   design doc V2.0 (PDF) + original plan (MD)
│   └── figures/
├── g1-wall-spray/             # G1 wall-spraying workstation (web + WBC)
│   ├── scripts/               #   numbered pipeline 00–13 + exec/ (daemon/streaming/estop)
│   ├── station/               #   config, NN-IK models, three.js frontend assets
│   ├── media/                 #   result figures + MuJoCo replay video
│   └── README.md              #   module deep-dive (incl. sim/real switch)
├── openarm-zigzag/            # OpenArm 7-DOF experiment platform
│   ├── 标定/ 仿真/ real/ ilc_sim/   # calibration / simulation / real robot / ILC sim
│   ├── docs/                  #   9 technical documents (architecture/PD/RL/MPC/ILC)
│   ├── assets/figures/
│   └── README.md / README.en.md
└── assets/                    # overview figure for this README + its script
```

## Quick start

**pointcloud-registration** (numpy + matplotlib only; sample data bundled)

```bash
cd pointcloud-registration
python register_and_label.py     # coarse → adaptive ICP → label transfer → ply + 4-view png
```

The full 5-million-point clouds are too large to ship; regenerate the downsampled sample with `make_sample_data.py --src <path-to-original-project>`.

**station-planning**

```matlab
run('station_planning.m')   % BCD decomposition + 12 stances + plots for the 10×2 m wall
```

A pure-Python reachability version is provided at `workspace/ws_calc.py` (Tiangong 3.0 right-arm FK Monte-Carlo + alpha-shape boundary).

**g1-wall-spray** (first clone [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) **next to this repo** and set up the conda envs; see the module README)

```bash
cd g1-wall-spray
cp station/env.json station/env.local.json   # fill local interpreters / GR00T path / robot_mode (gitignored)
./启动工作站.sh          # open http://127.0.0.1:8090
# Web flow: ① start robot (test the soft e-stop) → ② pick cloud/stance/posture, plan → ③ confirm motion → ④ execute (0.5× available)
# Sim/real switch: robot_mode in station/env.local.json ("sim" / "real")
```

**openarm-zigzag** (Ubuntu 24.04 + ROS 2 Jazzy; the calibration tool runs hardware-free on vcan0)

```bash
cd openarm-zigzag
cat README.en.md          # five-step quick start: calibrate → MoveIt2 sim → MPC on real → RL → ILC
```

## Module highlights

### pointcloud-registration — registration without calling a library

Given a grayscale robot scan and a hand-held semantic map, everything from the k-d tree to ICP is hand-written numpy. Coarse registration skips generic feature matching and exploits the fact that a cabin is a rectangular room: **four wall corners + the offset door frame act as structural landmarks**, matched by distance-geometry consistency on triplets, solved with Kabsch, and disambiguated by dense overlap scoring. Fine registration is an **adaptive-threshold point-to-plane ICP** (ε = μ + 0.1σ rejection + Lie-algebra updates + dual convergence criteria). Semantic labels are then transferred through a voxel-hash nearest-neighbor pass.

<table>
<tr><td width="50%"><img src="pointcloud-registration/figures/舱室点云_原始扫描黑白.png"/></td>
<td width="50%"><img src="pointcloud-registration/figures/舱室地图机器人扫描版本.png"/></td></tr>
<tr><td align="center">Input: grayscale robot scan (local frame)</td>
<td align="center">Output: registered back + transferred semantic labels</td></tr>
</table>

### station-planning — first "where can the robot reach", then cut the wall

Monte-Carlo FK on the Tiangong 3.0 URDF yields the wrist reachability (alpha-shape boundary); the union of cone-shaped spray footprints defines per-stance coverage. The wall is decomposed with **BCD** along obstacle projections — fences merge their above-wall strip into the left cell, the console gets its own cell on top — then each cell is split into sub-regions with right-offset stances, plus dedicated corner stances: 12 stances cover the full 10 m wall without gaps. The design doc lives in [docs/](station-planning/docs/) (V2.0, 15 pages, in Chinese).

<img src="station-planning/figures/fig_bcd.png" width="70%"/>

### g1-wall-spray — web workstation + whole-body control based on SONIC

A single-file Python web workstation (stdlib HTTP + embedded three.js 3D preview) chains cloud parsing, trajectory planning, motion export and execution dispatch. The **whole-body control policy is based on NVIDIA GEAR-SONIC** (the official GR00T-WholeBodyControl stack): lower-body balance is handled by the SONIC policy, and the upper body is injected as 17 joints @50 Hz with velocity feedforward through the official `zmq_planner_sender` protocol. MuJoCo simulation and the real DDS link share one planning chain — switching is one field in `env.local.json`.

**Current status (V3 final, full-band spray verified on the physical G1, 2026-10-12)**: stand band 8 rows over 0.75–1.45 m → five-segment slow squat (max descent 0.13 m/s; segment-gated height channel keeps the standing phase free of mode flapping) → crouch band 4 rows over 0.35–0.65 m, ×1.0 full run in 155.6 s; two-level soft e-stop and a 0.5× low-speed lane. Crouch-row placement accuracy (no pelvis-pitch feedback yet) is the main remaining gap. A dated capability timeline lives in [g1-wall-spray/README_版本迭代.md](g1-wall-spray/README_版本迭代.md) (Chinese).

<p align="center">
<img src="g1-wall-spray/media/fig_3d_scene.png" width="72%"/><br/>
<i>Dual-posture spray scene (blue = 8-row stand band, green = 4-row crouch band, grey = transition)</i>
</p>

<p align="center">
<img src="g1-wall-spray/media/fig_replay_contactsheet.png" width="98%"/><br/>
<i>MuJoCo replay overview (left = side view / right = front view, 12 timestamps: stand band → squat transition → crouch band)</i>
</p>

Replay video (the same motion executed on the physical G1, ×1.0 full run in 155.6 s):

<video src="g1-wall-spray/media/replay_motion.mp4" controls muted width="60%"></video>

**Multi-station relay (accepted in simulation, 2026-10-20)**: the wall is cut into N
bands of one action width; each station sprays its own band, then the robot steps
along the wall and sprays the next one. A two-station GR00T MuJoCo relay lands the
step exactly (0.500 / 0.500 m) without falling, and station 2 keeps station 1's
landing-height accuracy (steady-state mean -0.036 m vs -0.014 m). The residual is a
~12 cm shift of the spray pattern along the wall (the torso tilts more after walking).

<p align="center">
<img src="g1-wall-spray/media/fig_multi_station_bands.png" width="92%"/><br/>
<i>Multi-station band division: a 0.98 m wall covered seamlessly by 3 stations (band 0.327 m, step 0.327 m)</i>
</p>

<p align="center">
<img src="g1-wall-spray/media/fig_multi_station_sim.png" width="98%"/><br/>
<i>Two-station relay in simulation: base trajectory / landing-height error / along-wall pattern shift / task-servo correction</i>
</p>

### openarm-zigzag — getting "fast" and "stable" at the same time

A full-chain project on the OpenArm v1 7-DOF arm (Damiao DM motors + CAN-FD): a pure-Python reimplementation of the official calibration flow, four simulation routes (MoveIt2 / hand-written numerical IK / MuJoCo / Isaac Lab), and three real-robot routes (JTC baseline / a ros2_control MPC plugin / an RL policy with an ILC feedforward layer). Nine technical documents record the trade-offs along the way: landing published PD-tuning formulas, RL reward and time design, root-causing the 0.5–2 Hz hunting behind the 73.4 s e-stop, and taking ILC from feasibility study to real-robot acceptance.

<img src="openarm-zigzag/assets/figures/moveit_zigzag_right_route.png" width="90%"/>

## Upstream & credits

- [enactic/openarm](https://github.com/enactic/openarm) — OpenArm v1 hardware and the official ROS 2 workspace that `openarm-zigzag` builds upon
- [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) — source of the whole-body control policy (GEAR-SONIC) used by `g1-wall-spray`; upper-body injection follows its official `zmq_planner_sender` protocol
- Unitree G1 official URDF / SDK2; Damiao DM motor family
- The boustrophedon line-tracing scheme references the open-source spray-coating-robot project
- PD-tuning formulas from arXiv:2508.08241

## License

[MIT](LICENSE). This repository contains code that drives real robots and comes with no safety warranty; run your own safety assessment and e-stop drill before touching hardware.

