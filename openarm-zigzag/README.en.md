# OpenArm Zigzag Trajectory Lab

[简体中文](README.md) | **English**

A full-pipeline experimental project for an **OpenArm v1 7-DOF manipulator**
(Damiao DM-series motors over CAN-FD): **zero calibration → simulation → real robot**.

The main task is to have the gripper trace a **zigzag raster scan** on the vertical
plane **x = 0.30 m** in front of the base — a **29 × 53 cm rectangle, 3 cm line
spacing, 36 waypoints** — and to take that trajectory all the way from MoveIt2
simulation onto the physical arm, while solving two problems at once: making it
**fast** and making it **stable**.

![Right-arm zigzag route and TCP speed profile](assets/figures/moveit_zigzag_right_route.png)

> The figure above is produced by `仿真/moveit_sim/make_dense_zigzag_right.py`
> and archived at `assets/figures/moveit_zigzag_right_route.png`.

## What has been achieved (as of 2026-09-23)

| Milestone | Result | Date |
|---|---|---|
| First full zigzag on the real arm (MoveIt2) | speed-scale 0.4, TCP ≈ 2 cm/s, mean TCP deviation **1.3 mm** | 2026-09-07 |
| Dual-arm refactor (left/right configs fully isolated) | `real/{common,left,right}` | 2026-09-08 |
| RL policy accepted on real arm, 60 s segment | tracking error mean **1.16°** / max 3.46°, oscillation window 3.8° → 0.7° | 2026-09-18 |
| First full-length RL run | stopped by the operator at **73.4 s** due to 0.5–2 Hz low-frequency oscillation | 2026-09-21 |
| ILC feed-forward correction table deployed | driver integrator peak reduced **2–5×**, 0.5–2 Hz peak-to-peak max 1.18° (< 1.5° criterion) | 2026-09-22/23 |
| **Full run accepted (RL + ILC)** | **143.6 s**, zero e-stops, converges back to home; tracking mean 1.19° / max 3.53°; run-to-run repeatability RMS **0.04°** | 2026-09-23 |

Total runtime was cut from roughly 300 s (MPC baseline) to **143.6 s**, with the
0.5–2 Hz low-frequency oscillation brought inside the acceptance criterion. That is
the project's headline result so far.

## ⚠️ Safety rules (read before touching the robot)

This repository contains code that **directly drives a real manipulator**.

1. **Collaborators and AI agents never execute commands that move the robot** —
   they write code, configuration and instructions; the operator runs them and
   reports back what happened.
2. **The calibration node and control/MoveIt must never run at the same time**
   (both drive the motors directly and will fight over the same bus).
3. Run `can_up.sh` to configure the CAN interface after every power-up / adapter
   replug.
4. Three-level emergency stop: stop the stream (Ctrl+C) → `/safety/estop` →
   **physically cut power**.
5. Always bring up a new configuration in **short segments** first; never start
   with a full-length run.

See [`docs/架构文档.md`](docs/架构文档.md) §12 (Chinese) for details.

## Repository layout: what is what

Top-level directories keep their original Chinese names (the project's documents,
scripts and runbooks are all written in a Chinese-language environment). The table
below gives the English equivalent and the responsibility of each directory:

| Directory | English | What it is |
|---|---|---|
| `标定/` | `calibration/` | **Zero-position calibration tool** (ROS2 package, v2 official-zero procedure). The principle matches the official `openarm_can-zero-position-calibration`: each joint slowly hits its mechanical limit, both directions are recorded, and a `zero` file is written. Includes a vcan0 software simulator, so the whole flow can be validated without hardware. |
| `仿真/` | `sim/` | **Four simulation routes**: `moveit_sim/` (the MoveIt2 mainline; `dense_zigzag*.json` is produced here), `openarm_sim/` (pure-kinematics RViz sim + FK/IK middleware), `mujoco_sim/` (MuJoCo cross-check + RL validation), `isaaclab_sim/` (Isaac Lab training pipeline). |
| `real/` | `real/` | **Real-robot mainline.** `common/` is shared by both arms (ZeroOffsetHW driver workspace, config generator, 100 Hz black box); `left/` and `right/` hold each arm's config, MPC execution track and RL execution node. |
| `docs/` | `docs/` | Technical documents: architecture, PD tuning, RL training structure, MPC plugin design, ILC feasibility, route comparison and switching protocol. |
| `ilc_sim/` | `ilc_sim/` | **Offline ILC simulation harness** (pure numpy/scipy, consumes the real trajectory file), companion to the ILC document. |
| `assets/figures/` | `assets/figures/` | **All figures live here** — figures are never left lying next to source files. Provenance is documented below. |
| `archive/` | `archive/` | **Abandoned historical route** (Damiao private USB protocol + motorbridge). ⚠️ **Never use anything in this directory on the real robot.** |

**The rule of thumb: a tool lives in the directory of the arm it drives.** In
`real/left/` and `real/right/`, joint names, controller names and MPC topics are
hard-wired to that side to prevent cross-arm mistakes. Both arms use the factory
default CAN IDs, so **only one arm may be connected to a CAN port at a time**.

## Three execution routes (mutually exclusive)

The same task and the same reference trajectory were attempted over three
execution routes. They are **fully separated and mutually exclusive**; the switching
protocol and selection criteria are in
[`docs/RL与MPC方案对比与切换.md`](docs/RL与MPC方案对比与切换.md) (Chinese).

| Route | Injection mechanism | Status |
|---|---|---|
| **① JTC / MoveIt2 baseline** | `FollowJointTrajectory` action (standard ros2_control slot) | First full traverse; speed-scale 0.4, TCP ≈ 2 cm/s |
| **② MPC execution track** | `OpenArmMpcController` plugin takes over the JTC slot | Full traverse completed; one-line rollback |
| **③ RL execution track** (current mainline) | `rl_exec.py` injects on a dedicated topic (bypasses the controller plugin) | Full real-robot run of **143.6 s** accepted |

On top of the RL route sits an **ILC feed-forward correction layer**. It is not a
new controller: it reconstructs the "driver debt torque" offline from real-robot
run data and generates a feed-forward table that compensates for unmodeled load
(cable drag), which removes the 0.5–2 Hz low-frequency oscillation.

```
                ┌── ② MPC plugin ──────┐
Reference ──────┤                      ├──→ ZeroOffsetHW ──→ Damiao motors (CAN-FD)
dense_zigzag_   ├── ③ RL policy (npz) ─┤      (real driver)
right.json      │      + ILC table ────┘
                └── ① JTC / MoveIt2 ───┘
```

## Data flow: who produces what, who consumes it

The fastest way to understand this repository is to follow the artifact chain:

```
标定/zero, zero_right            calibration products (zero + measured soft limits)
        │
        └─→ real/common/gen_real_config.py
                └─→ real/<side>/config/{real_safety.yaml, *_real.urdf}

仿真/isaaclab_sim/stage3_workspace.py       workspace scan (1 cm grid + false-hole repair)
        └─→ workspace.json / workspace_right.json (36 waypoints + grid joint solutions)
                └─→ 仿真/moveit_sim/make_dense_zigzag{,_right}.py
                        └─→ dense_zigzag{,_right}.json (dense joint trajectory)

dense_zigzag_right.json
        ├─→ 仿真/isaaclab_sim/rl/zigzag_ref.py
        │       └─→ ref_right.npz (50 Hz reference: q_ref/qd_ref/phase/metadata)
        │               ├─→ ② MPC: mpc_exec.py --ref
        │               ├─→ ③ RL : rl_exec.py (policy + reference + feed-forward)
        │               └─→ MuJoCo validation: mujoco_sim/rl_validate.py
        └─→ ilc_sim/ilc_sim.py (offline ILC feasibility study)

/joint_states ──→ real/common/js_blackbox.py (100 Hz CSV; oscillation attribution / ILC error signal)
```

**Critical contract**: `real/<side>/config/d_prev_sidecar.json` is the config
generator's idempotency record. **Deleting it doubles the zero offset `d` — do not
delete it.**

## Figures and data provenance

All figures are archived in `assets/figures/`. Where each one comes from:

| Figure | Content | How it was produced |
|---|---|---|
| [`moveit_zigzag_right_route.png`](assets/figures/moveit_zigzag_right_route.png) | Right-arm zigzag route, 4 panels: workspace rectangle + waypoints, joint trajectories, TCP speed profile | `仿真/moveit_sim/make_dense_zigzag_right.py` (generated 2026-09-11) |
| [`workspace_left.svg`](assets/figures/workspace_left.svg) | Left-arm reachable set + same-branch execution domain + largest inscribed rectangle + zigzag | `仿真/isaaclab_sim/stage3_workspace.py` (default `--side left`) |
| [`workspace_right.svg`](assets/figures/workspace_right.svg) | Same for the right arm (limits from that arm's measured `real_safety.yaml`) | `仿真/isaaclab_sim/stage3_workspace.py --side right` |
| [`workspace_compare_20260908_old_vs_new.png`](assets/figures/workspace_compare_20260908_old_vs_new.png) | Workspace rectangle before/after: 17×51 cm → **29×53 cm** | Hand-made comparison figure by the author (2026-09-08, at the end of the "false hole repair" work). No generating script is committed |
| [`workspace_scan_20260907_before_fix.png`](assets/figures/workspace_scan_20260907_before_fix.png) | Intermediate state during the false-hole repair (19×45 cm) | Same as above (2026-09-07 snapshot; its Chinese labels render as boxes because of a missing CJK font, which is why the 09-08 version supersedes it) |
| [`ref_right_profile.png`](assets/figures/ref_right_profile.png) | RL reference profile: joint position/velocity/acceleration + TCP rate and speed tiers | `仿真/isaaclab_sim/rl/zigzag_ref.py` (generated 2026-09-14) |
| [`real_run_report.png`](assets/figures/real_run_report.png) | Real-robot run report: per-phase J2 tracking error and velocity ripple | `real/right/rl_track/analyze_real_run.py <rl_exec.csv> <blackbox.csv>` (run 2026-09-15) |
| [`ilc_learn_curves.png`](assets/figures/ilc_learn_curves.png) | ILC learning curves and the Δτ correction (iteration 1) | `real/right/rl_track/ilc_learn.py` (generated 2026-09-23) |
| [`ilc_sim_curves.png`](assets/figures/ilc_sim_curves.png) | Offline ILC convergence curves (experiments A/B/C: with/without Q filter, high-gain divergence control) | `ilc_sim/ilc_sim.py` (companion to [`docs/ILC可行性_实施路径文档.md`](docs/ILC可行性_实施路径文档.md) ch. 7) |

**Committed data artifacts** have equally explicit provenance: `ref_right.npz`
(`zigzag_ref.py`), `actor_right.npz` (`export_actor.py`, exported from a training
checkpoint), `grav_ff_right*.npz` (`仿真/mujoco_sim/make_grav_ff.py`; ILC iteration
tables are synthesized by `ilc_learn.py`), and `rl_exec_*.csv` (command-side record
of each real-robot run, written by `rl_exec.py --log-csv`).

## Quick start

### 1) Calibration (runs on vcan0 without hardware)

```bash
cd 标定
./openarm_calibration/build.sh                          # build (conda users need not deactivate)
./openarm_calibration/run_simulation.sh true left      # vcan0 + 8-motor simulator, full flow
# Real hardware: replug the adapter, run ./openarm_calibration/scripts/can_up.sh, then run_calibration.sh
# After recalibration you MUST re-run real/common/gen_real_config.py to regenerate the real configs
```

### 2) Simulation (MoveIt2 mainline)

```bash
cd 仿真/moveit_sim
./start_zigzag_sim.sh          # left arm (use start_zigzag_sim_right.sh for the right arm)
# In another terminal, execute the zigzag:
./run_moveit_zigzag.sh         # or run zigzag_moveit.py manually as described in the README
```

### 3) Real robot (MPC execution track, right arm)

```bash
# 0) can_up.sh → 1) source the official workspace and this workspace
cd real/right/mpc_track
/usr/bin/python3 mpc_exec.py \
    --ref ../../../仿真/moveit_sim/dense_zigzag_right.json --speed-scale 0.4
# The full power-up / bring-up sequence is at the end of real/right/mpc_track/README.md
```

### 4) RL route (right arm, current mainline)

```bash
conda activate isaaclab
cd 仿真/isaaclab_sim/rl
python zigzag_ref.py                     # generate the reference (speed tiers + smoothing, ≈157 s)
python train.py --headless --num_envs 2048 --max_iterations 2000
python play.py --checkpoint logs/rsl_rl/openarm_zigzag/<run>/model_final.pt

# MuJoCo validation (same actor npz, same reference)
conda activate mujoco
cd 仿真/mujoco_sim && python rl_validate.py --actor ../isaaclab_sim/rl/logs/.../actor_final.npz

# Real robot (executed by the operator, in segments; see real/right/rl_track/README.md)
cd real/right/rl_track && /usr/bin/python3 rl_exec.py --max-t 15 --log-csv
```

### 5) Offline ILC simulation

```bash
cd ilc_sim && /usr/bin/python3 ilc_sim.py
# writes ilc_sim_results.json + assets/figures/ilc_sim_curves.png
```

## Environment

- Ubuntu 24.04 + **ROS 2 Jazzy** (`/opt/ros/jazzy`)
- The official workspace `openarm_ros2_ws` (provides the `openarm_can` C++ library,
  `openarm_description`, `openarm_bimanual_moveit_config`; source it before use)
- CAN adapter: a **gs_usb-class** device (e.g. candleLight, USB ID `1d50:606f`),
  CAN-FD at 1M/5M. Damiao's private USB adapters (DM-USB2FDCAN etc.) **do not work** —
  they never create a `can0` interface.
- Simulation: MuJoCo (conda env), Isaac Lab / Isaac Sim (conda env; training needs an
  RTX GPU)
- Offline ILC simulation: numpy / scipy / matplotlib
- Always build with the system Python:
  `--cmake-args -DPython3_EXECUTABLE=/usr/bin/python3` (prevents conda hijacking;
  mujoco/isaaclab are the exceptions and use their own conda environments)

**Hardware**: OpenArm v1, one arm = 7 DOF + gripper. For the left arm, ESC index →
joint: ESC1/2 = DM8009 (shoulder), ESC3/4 = DM4340 (shoulder roll / elbow),
ESC5/6/7 = DM4310 (wrist), ESC8 = DM4310 (gripper; it locks in place after enabling
and does not track the trajectory).

## Documentation index

| Document | Content |
|---|---|
| [real/交接_20260922_ILC就绪与复验指引.md](real/交接_20260922_ILC就绪与复验指引.md) | **Current highest-authority handover**: ILC feed-forward table ready, 0923 full-run acceptance read-out, next steps |
| [real/交接_20260921_RL全程验收进行时.md](real/交接_20260921_RL全程验收进行时.md) | Root cause of the 9-21 full-run abort (0.5–2 Hz hunting), incident-prevention checklist, on-robot runbook |
| [docs/架构文档.md](docs/架构文档.md) | **Layered architecture** (hardware / driver / middleware / application), runtime interfaces, parameter snapshot, safety discipline |
| [docs/RL训练结构详解.md](docs/RL训练结构详解.md) | Complete RL structure snapshot: observation / action / control law / reward formulas + deployment mapping. **Read before changing the training strategy** |
| [docs/PD参数整定流程.md](docs/PD参数整定流程.md) | PD tuning procedure + pre-training offline screening + authoritative twin + version history + training incident post-mortems |
| [docs/RL与MPC方案对比与切换.md](docs/RL与MPC方案对比与切换.md) | Route comparison, strong config-layer isolation and the switching protocol |
| [docs/MPC控制器插件_线路一改造设计.md](docs/MPC控制器插件_线路一改造设计.md) | MPC plugin taking over the JTC slot: model / objective / constraint spec, build acceptance, one-line rollback |
| [docs/ILC可行性_实施路径文档.md](docs/ILC可行性_实施路径文档.md) | ILC feed-forward layer: design, tuning, staged rollout (English version: [`archive/ILC_feasibility_implementation_plan_en.md`](archive/ILC_feasibility_implementation_plan_en.md)) |
| [docs/RL轨迹提速_奖励与时间设计.md](docs/RL轨迹提速_奖励与时间设计.md) | Reward design history, speed tiers and time allocation, pitfalls (numbers defer to the structure document) |
| [docs/仓库整理记录.md](docs/仓库整理记录.md) | Changelog of repository reorganizations (what was removed, moved, and size changes) |
| [仿真/PLAN.md](仿真/PLAN.md) | Overall simulation plan and stage breakdown |
| [仿真/moveit_sim/README.md](仿真/moveit_sim/README.md) | MoveIt2 route usage and implementation details (incl. §4.7 the TOTG acceleration-limit patch) |
| [仿真/isaaclab_sim/rl/README.md](仿真/isaaclab_sim/rl/README.md) | RL training pipeline usage + stability fixes + list of pitfalls |
| [仿真/isaaclab_sim/rl/logs/rsl_rl/openarm_zigzag/VERSIONS.md](仿真/isaaclab_sim/rl/logs/rsl_rl/openarm_zigzag/VERSIONS.md) | **Training version index**: v1 (600-iteration baseline) / v2 (anti-stick-slip) / v3 (curriculum, currently deployed) |
| [real/right/rl_track/README.md](real/right/rl_track/README.md) | RL real-robot execution node: deployment checklist, segmented validation, exception handling, gravity feed-forward channel |
| [real/right/BRINGUP_右臂.md](real/right/BRINGUP_右臂.md) | Right-arm power-up / CAN / calibration / debugging manual (stages A–G) |
| [real/common/README.md](real/common/README.md) | Shared dual-arm layer: driver workspace, config generator, arm-swapping rules |
| [标定/README.md](标定/README.md) | Zero-calibration tool and the v2 official-zero procedure |
| [archive/README.md](archive/README.md) | Notes on the abandoned motorbridge route |

> Most documents are written in Chinese; the handover, architecture and design docs
> are the primary sources of engineering detail.

## Repository conventions

Only **source, configuration, documentation and a small set of artifacts** are
committed (calibration results, deployed `*.npz` policies and references, figures,
and each real-robot run's `rl_exec_*.csv`). colcon build output, raw real-robot
logs, Isaac Sim USD geometry assets and intermediate training checkpoints are not
committed — all of them can be regenerated from what is in the repository.

**The complete exclusion list, with the reason and rebuild command for each entry,
is in [`.gitignore`](.gitignore)** — that single file is the authoritative source.

## Origin and acknowledgements

- **Robot and official stack**: [enactic/openarm](https://github.com/enactic/openarm/)
  — the URDF/description packages, the `openarm_can` C++ library and the bimanual
  MoveIt2 configuration all come from upstream. The zero-calibration procedure is
  aligned 1:1 with the official `openarm_can-zero-position-calibration`.
- **This project's code**: `标定/`, `仿真/`, `real/`, `ilc_sim/` and `docs/` are the
  experimental engineering built on top of that official stack, including a custom
  numerical IK solver, an MPC controller plugin, the RL training and deployment
  pipeline, and the ILC feed-forward correction layer.
- **Nature of the documents**: `docs/` and the handover notes are **dated engineering
  records** — they include failed attempts, post-mortems and parameter history, and
  are not final specifications. Where they disagree, the most recent one wins
  (currently `real/交接_20260922_ILC就绪与复验指引.md`).

## License

MIT — see [LICENSE](LICENSE).

> This repository contains code that directly drives a real manipulator. It is
> provided "AS IS", without warranty of any kind. Users are responsible for their
> own hardware and personal safety.
