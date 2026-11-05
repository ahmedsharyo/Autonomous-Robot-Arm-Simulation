# Autonomous Robot Arm Simulation

A CPU-oriented MuJoCo tabletop arm project with independently implemented serial-chain FK, a geometric Jacobian, damped least-squares IK, bounded SciPy least-squares IK, torque-level tracking, smooth quintic trajectories, collision queries, task orchestration, reproducible benchmarks, plotting, and video recording.

## Status

The repository is implemented, but this checkout's execution host could not install its dependencies because network access is unavailable. Therefore no runtime tests, numerical benchmark results, pick-and-place success, or MP4 are claimed as completed. Run the commands below in a Python 3.11+ environment with package access. The pick-and-place demo verifies real finger/object contact and final object position; it may correctly report failure when the chosen model geometry or waypoints do not produce a physical grasp.

## Architecture and model

`assets/robot.xml` defines a custom six-revolute-joint arm, a two-finger actuated gripper, a tabletop, one free cube, and a fixed obstacle. The arm motors are torque limited; the gripper uses position actuators. `kinematics.py` independently evaluates rigid transformations and the world-frame geometric Jacobian. `simulation.py` wraps MuJoCo state and stepping. `ik.py` implements DLS and bounded least squares. `control.py` distinguishes task-space joint-velocity generation from the joint-space PD torque controller used by the torque motors. `trajectory.py` provides quintic rest-to-rest interpolation. `collision.py` refreshes MuJoCo's state before reading contacts. `manipulation.py` sequences grasp and placement waypoints and checks physical contacts plus final object location.

The independent FK model mirrors the XML transform chain. Joint coordinates are in radians and positions in meters. Position-only IK is prioritized; end-effector orientation is not constrained. Collision queries are discrete-state checks. Trajectory collision validation should sample sufficiently densely, and runtime monitoring remains necessary. The supplied pick-and-place pose is a starting configuration; a failed physical grasp is surfaced as failure rather than simulated by moving the object artificially.

## Install (macOS Apple Silicon)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

MuJoCo's official Python package supports CPU operation. Offscreen rendering may require a supported macOS graphics context; the recording script uses MuJoCo's renderer and reports any renderer setup error.

## Run

```bash
python -m scripts.reach_demo
python -m scripts.pick_place_demo
python -m scripts.benchmark_ik
python -m scripts.evaluate
python -m scripts.record_demo
pytest
```

`reach_demo` computes a reachable pose and tracks it with a bounded PD torque controller. `pick_place_demo` runs the eight named manipulation phases. `benchmark_ik` evaluates both solvers on identical 40-target seeded scenarios and writes per-trial CSV, JSON summaries, and a comparison plot. The current benchmark samples targets from known reachable configurations, then perturbs the initial seed; it does not represent arbitrary workspace targets or obstacle-free planning. Runtime is solver wall time, not simulated task duration. The evaluation script reports measured values only after it runs.

## Algorithms

For each joint, FK composes a fixed translation and an axis-angle rotation. The geometric Jacobian uses each joint's world-frame axis `a_i` and origin `o_i`, with linear column `a_i × (p_ee-o_i)` and angular column `a_i`. DLS solves `Jᵀ(JJᵀ+λ²I)⁻¹e`, caps each update, and clips to joint bounds. The optimization method uses SciPy `least_squares` with box bounds and a small regularizer toward the starting posture. Both methods report the true unregularized endpoint residual and success threshold.

For actuator execution, desired joint trajectories are converted to bounded torques by a PD tracking law. The Cartesian velocity mapping is exposed separately and is not misused as a torque command. Quintic time scaling has zero velocity and acceleration at both ends.

## Evaluation and results

The benchmark uses seed `17` by default and records each target/solver outcome, endpoint residual, solver runtime, and iteration/function-evaluation count. Summaries include mean, median, standard deviation, 95th percentile, and success rate. No result table is populated here because dependencies could not be installed and no measurements were obtained. Generate actual results with `python -m scripts.benchmark_ik`.

## Tests

Tests cover independent Jacobian finite differences, FK/MuJoCo pose agreement, reachable and unreachable IK behavior, trajectory endpoint smoothness, and refreshed collision queries. MuJoCo-dependent tests skip if MuJoCo is absent. The state transition test checks the declared order; full physical end-to-end pick-and-place still needs execution on a dependency-equipped host.

## Limitations and next work

This first implementation does not include a collision-free path planner or formal swept-volume trajectory certification. The arm geometry and grasp pose are custom and require validation in the target MuJoCo build. Contact-based grasping can be sensitive to contact geometry, actuator direction, and initial pose. Orientation-constrained IK, trajectory logging plots, execution metrics, and a successful verified recording should be added only after the installed simulation is run and tuned. Do not interpret solver success as collision-free execution or a successful grasp.
