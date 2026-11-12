# OpenArm Port Plan

Goal: replace YuMi with OpenArm for R2R2R data generation from human demonstration videos.

Demonstration: `demonstrations/pick_up_pringle.mp4` (right-hand pickup, handoff to the left hand, and placement). The current tracked clip is `demonstrations/pick_up_pringle_10fps.mp4`, and the reconstructed object is under `outputs/pringles/`.

The initial milestone was right-arm replay with the OpenArm head camera. The current development task is video-driven bimanual handoff, preserving the demonstration's original timing and object motion.

## Current Behavior

| Mode | Inputs used for robot motion | Robot behavior |
|---|---|---|
| `right-pick` (default) | Recorded object poses and saved `grasps.txt` | The robot's right arm follows the remaining object trajectory, including motion originally performed by the human's left hand. The left arm does not participate. |
| `bimanual-track` | Recorded object poses, reconstructed hands, and prepared per-hand grasp/contact metadata | The robot's arms follow the object during their respective inferred contact intervals, including an overlap for handoff. |

The single-arm mode does not read the human right-hand trajectory or stop at the human's handoff. Its release interval is currently too short to execute a complete release-and-retreat sequence before reset; it should not be described as a validated pick-and-place task.

The bimanual mode uses reconstructed hands to estimate contact timing and rank grasps. It does not reproduce the human wrists frame by frame. During contact, each TCP target is the recorded object pose multiplied by a selected, fixed object-relative grasp. Approach and retreat use generic interpolation.

Both modes replay object poses kinematically. During the main replay interval, robot joint states are also written directly from IK. Passing replay checks does not establish collision-free motion or a handoff supported by contact forces.

## How the Existing R2R2R Simulation Works

```text
scripts/run.py
 ├─ Scene configuration: robot, table, object USD, rendering camera (MultiTiledCameraCfg)
 └─ Simulator, for example TigerPickR2R2R
     ├─ Read outputs/<object>/track/keyframes.txt and state_rigid_*/grasps.txt
     ├─ State machine: setup -> grasp -> follow object trajectory -> release
     ├─ IK: JaxMP using the robot URDF (controllers/jaxmp_diff_ik_controller.py)
     └─ Gripper: set finger joint positions directly
```

The initial port was based on `real2render2real/isaaclab_viser/franka_simulators/franka_coffee_maker.py`: one arm, gripper joints selected with `[-2:]`, and saved grasp orientations. The tiger implementation instead overrides the saved orientation with a top-down grasp.

## YuMi and OpenArm Differences

| Item | YuMi | OpenArm | Implementation |
|---|---|---|---|
| IK target | `left/right_dummy_joint`, 0.13 m ahead of link7 | `openarm_{left,right}_hand_tcp_joint`, 0.08 m ahead of the hand | Simulator controller setup |
| TCP pose link | `left/right_dummy_point` | `openarm_{left,right}_hand_tcp` | TCP lookup |
| Joint mapping | Fixed indices such as `[1,3,5,...,-2,-4]` | Joint-name lookup | Joint maps and `_ik_wrapped()` |
| Gripper | `[-4:-2]`, `[-2:]`; open position 0.024 m | `openarm_*_finger_joint1/2`; range 0-0.044 m per finger | `_set_gripper_positions()` |
| Finger closing axis | x | y | Grasp orientation |
| Approximate opening | 50 mm; insufficient for the can body | 88 mm; can diameter approximately 72 mm | Grasp width filtering |
| Assets | `data/yumi_*` | `Stanley_ws/lerobot_openarm/model/openarm_description.urdf` and meshes | Local robot assets |

## Phase A: OpenArm Right-Arm Pringles Replay

These steps record the original porting plan. The dedicated OpenArm entry point is now `scripts/openarm/run_openarm_pringles.py`.

### Step 0: Environment and Assets

- Use the `r2r2r` conda environment with Isaac Sim 4.5 and the repository's IsaacLab 0.36.6.
- Do not mix it with `Stanley_ws/IsaacLab` 0.54.4, which targets Isaac Sim 5.x. The R2R2R-specific `MultiTiledCameraCfg` is provided by this repository's IsaacLab fork.
- Copy the OpenArm URDF and meshes into `data/openarm_description/`.

### Step 1: Verify Robot Loading First

- Prefer URDF import with `UrdfFileCfg`. The base simulator supplies the same URDF to IK and Viser.
- The fallback considered was `v1_camera_isaac.usd`, created with a newer Isaac Sim version; compatibility with 4.5 would need testing.
- Check joint names, finger limits and mimic behavior, TCP frames, and IK reachability with a small diagnostic script.

### Step 2: Robot Configuration

File: `configs/articulation_configs/openarm_cfg.py`.

- Start from `OPENARM_BI_CFG` for velocity limits, effort limits, and stiffness.
- Set a home configuration with the grippers in front of the robot and above the workspace.

### Step 3: Scene Configuration

Files: `configs/scene_configs/openarm_*_cfg.py`.

- Match the real table height and location. The OpenArm shoulder is 0.698 m above its base.
- Convert the cleaned Pringles mesh with `convert_mesh.py` and place it within the right arm's reachable workspace.
- Render from the head-camera pose, using the camera mount in `v1_camera_isaac.usd` as a reference.

### Step 4: OpenArm Support in `dev_grasp`

- Add a gripper-only URDF containing the hand, two fingers, and TCP; retain the YuMi option.
- Use a TCP offset of 0.08 m and a per-finger width slider limit of 0.044 m.
- Regenerate grasps previously saved for YuMi.
- Use the same TCP frame in the grasp editor and simulation IK, avoiding an extra orientation conversion.

### Step 5: Single-Arm Simulator

File: `openarm_simulators/openarm_pringles_pick.py`, initially adapted from the Franka implementation.

- Use the OpenArm URDF, `num_ees=1`, and target `openarm_right_hand_tcp_joint`.
- Map arm and finger joints by name and use saved grasp orientations.
- Use `outputs/pringles/dig/2026-10-30_181519/config.yml` as the current DiG configuration.
- Check that object-placement randomization is reachable; the original template used a range of approximately +/-0.23 m.
- Adapt the TCP height threshold and replay acceptance checks to the OpenArm workspace.
- The legacy mode uses `part_deltas[4:]`, omitting the first four source frames, during which the can was approximately stationary.

### Step 6: Entry Point

The original plan was to add an OpenArm scene, simulator, and URDF selection to `scripts/run.py`. The implemented entry point is `scripts/openarm/run_openarm_pringles.py`. Start with two environments on the 16 GB GPU.

### Step 7: Staged Validation

1. Load the robot and verify IK against test targets.
2. Replay the measured object trajectory and check reachability over the full motion, including wrist rotation.
3. Inspect complete rollouts and distinguish kinematic replay checks from dynamics-based manipulation validation.
4. Inspect rendered images before accepting data for training.

## Phase B: Bimanual Handoff

- Use `num_ees=2`, with separate object-relative grasps for the right and left grippers.
- Follow the demonstration sequence: right grasp -> object motion -> left grasp -> right release and retreat -> left completes its interval.
- Infer timing from reconstructed hands and finger-to-object distance. Do not use the earlier manually proposed frame windows as labels.
- Preserve the recorded object translations, rotations, and source timing.

## Progress and Evidence

### Steps 0-1: Completed on 2026-11-05

- Copied `openarm_bimanual.urdf` and `meshes/` from `Stanley_ws/lerobot_openarm/model` into `data/openarm_description/`, changing only mesh references to `meshes/<name>.stl`. These local assets are not included in the tracked source files and must be restored on another machine.
- `scripts/openarm/check_openarm_load.py` loads and drives the robot in Isaac Sim 4.5 and compares TCP poses against URDF forward kinematics:
  - All 18 joints are present: seven arm joints and two finger joints per side. Both TCP links are present. Set `merge_fixed_joints=False` to retain them.
  - Keeping mimic joints did not drive the second finger correctly. Set `convert_mimic_joints_to_normal_joints=True` and drive both fingers. The importer changes `finger_joint2` limits to `[-0.009, 0.053]`; restore `[0, 0.044]` with `write_joint_position_limit_to_sim`.
  - Isaac TCP positions matched URDF FK to the reported precision of 0.0 mm, and joint targets were reached.
  - Use actuator regex `openarm_(left|right)_joint[1-7]`. The broader `openarm_.*_joint[1-7]` also matches `finger_joint1`.
- With the same URDF, sequential JaxMP IK for a forward-facing right TCP target converged to the reported precision of 0 mm / 0 degrees. Low, downward-facing targets could get stuck near joint limits. Side grasps were more suitable in these tests.
- A fixed-orientation reachability plot was fragmented, suggesting local IK convergence issues rather than a definitive workspace boundary. Full-trajectory checks remain necessary; initialization, iteration count, or orientation weights may need adjustment.

### Real-Robot References for Step 3

The following values came from the local `Stanley_ws/IsaacLab` configuration:

- Head camera `body_cam`, attached to `openarm_body_link0`: `pos=(0.05621, -0.00305, 0.83062)`, `rot=(0.18720, -0.68244, 0.69543, -0.12490)` in ROS convention; 640 x 480; focal length 24; horizontal aperture 25.2956. Reference: `stack_ik_abs_visuomotor_env_cfg.py:169`.
- Workspace pad: height 0.28 m; dimensions 0.48 x 0.57 m; near edge 0.03 m from the base. Reference: `stack_joint_pos_env_cfg.py:133-150`.
- The real-task can reference was approximately `(0.55, 0.05)`. Reference: `openarm_task_modes.py:2278`.

### Steps 2-7: Pipeline Connected and Initially Tested on 2026-11-05

This milestone established loading, rendering, and data recording. It did not establish manipulation feasibility.

- Robot and scene files: `configs/articulation_configs/openarm_cfg.py`, `configs/scene_configs/openarm_base_cfg.py`, and `openarm_scene_cfg.py`.
- Head-camera poses are stored in `data/camera_poses/openarm_head_cam.h5`. `scripts/openarm/check_openarm_scene.py` tests the scene and saves a head-camera image.
- Pringles USD preparation:
  - The upstream README's `--collision-approximation none` produced no collision shape for the dynamic object, allowing it to pass through the table. The local asset uses `convexHull`.
  - The unscanned bottom formed a pointed collision hull and made the can tip over. Approximately 1.8 cm of bottom artifacts were removed below `z=-0.116`, above which the cross-section was complete.
  - Earlier assets were retained under `outputs/pringles/_unused/`.
- Grasp editor: `--gripper openarm` selects `data/openarm_description/openarm_hand.urdf`, a 0.08 m TCP offset, and finger positions from 0 to 0.044 m. Grasp candidates allow a total width of 0.088 m; YuMi's 0.045 m limit rejected can-body candidates. Changes are recorded in `patches/rsrd.patch`.
- Single-arm replay differences from the YuMi tiger example:
  - Retain the saved grasp orientation.
  - Rotate the initial object placement to favor a right-side approach of approximately 40 degrees, with +/-0.15 rad yaw variation. Nominal object XY is `(0.33, -0.10)`.
  - Retain the legacy interpolation factor of three, originally described as 10-to-30 FPS interpolation. This mode does not derive playback timing from the source video FPS as the bimanual mode does.
  - Clamp IK solutions to joint limits before writing them.
- Parallel-gripper symmetry: a 180-degree roll about the approach axis represents an equivalent grasp. An offline search over 90 combinations favored a TCP x-axis pointing upward for the right arm. The legacy loader flips grasps whose TCP x-axis points downward in the object frame.
- Base simulator fixes:
  - Retain the URDF path when importing the robot so the IK controller can use it.
  - Call `sim.render()` in `render_wrapped_impl`; otherwise camera images remain gray.
- Historical run: `run_openarm_pringles.py --auto-start --num-envs 2 --max-successes 4` accepted five of six rollouts under its replay checks. Each contained 301 head-camera images at 640 x 480 and `robot_data.h5`: 16 logged joints, both TCP poses, and `gripper_binary_cmd` ordered `[left, right]`. These counts are historical replay results, not physical pick-and-place success measurements.
- Remaining issues at this stage: additional saved grasps, noisy rendering, and validation before large-scale generation. The short single-arm release interval is also still unresolved.

### Viser Object Display Fix on 2026-11-06

- The head-camera images contained the can, but the browser's 3D scene only contained its coordinate frame.
- Added the textured OBJ used to build the Isaac USD as a child of the Viser object frame. It follows simulated poses and the selected environment.

### Cancellation and GPU Cleanup on 2026-11-06

- Isaac physics callbacks could swallow a `KeyboardInterrupt` raised by the signal handler. Signals now set a stop flag, checked by both the simulation loop and browser-client wait loop.
- The previous implementation forked eight writers after CUDA initialization. Orphaned writers could retain GPU resources; an ordinary Boolean shutdown flag was not shared across processes, and unbounded `join()` could hang.
- OpenArm now uses `spawn` for CPU writers and a main guard to avoid starting Isaac again when a worker imports the entry script. Single-arm runs with two environments use two writers; bimanual runs use one writer to preserve HDF5 ordering.
- Cleanup closes data writers and Viser before SimulationApp. Writers receive sentinels and have a bounded shutdown, escalating to terminate/kill after the grace period.
- A CPU supervisor manages a separate process group for each run. One Ctrl+C requests cleanup; after 10 seconds it sends terminate, followed by kill after another two seconds. A second Ctrl+C forces immediate termination. Normal exit and crashes also trigger cleanup of remaining group members.
- An interrupted episode may be incomplete. Previously completed `successes/` are retained. Use `--no-save` for inspection without writer processes.
- Historical validation: nine process/writer regression checks passed, covering unresponsive process trees, orphan cleanup, hard exit, IPC cleanup, and queued image/HDF5 writes. An actual two-environment Isaac run exited 2.17 seconds after SIGINT during saving, with no residual processes and GPU memory returning to approximately 1033 MiB. Local evidence: `/tmp/openarm-shutdown-yi48w5hb/`.

### Video-Driven Bimanual Correction on 2026-11-12

The earlier `bimanual-lift` prototype used manually chosen grasps and a synthetic vertical lift. It was withdrawn because it did not satisfy the video-driven project objective. Its archive is at `/tmp/openarm-scripted-prototype-withdrawn/`; its metrics must not be used as evidence for video-driven handoff.

The current implementation preserves the user's chosen demonstration sequence: right-hand pickup -> left-hand reception -> placement.

- Default mode remains `--task right-pick`. Select the new mode with `--task bimanual-track --bimanual-data <metadata.json>`.
- The new mode shares the existing object tracking, scale conversion, rendering augmentations, and data logger. It uses two JaxMP end effectors and retains the recorded object's relative XYZ and rotation trajectory.
- `prepare_bimanual_track.py` reads RSRD `hands` and `part_deltas`, converts registered-frame hand keypoints into object-local metres, and estimates contact using thumb/index distance to the reconstructed surface. This distance-based contact inference is a local extension, not an official handoff-labeling component.
- Default contact thresholds are 0.04 m to enter and 0.055 m to leave contact, with isolated runs shorter than three frames removed. These are configurable engineering parameters; inferred contact is not ground truth.
- The official antipodal sampler proposes grasps, ranked using observed hand positions and a wrist-derived approach direction. Each arm uses a selected fixed grasp per rollout. Approach and retreat are generic smooth transitions, currently approximately 1.35 seconds at 60 Hz.
- A separate JSON records source SHA256, scale, FPS, visibility, distances, inferred contact intervals, and grasp candidates. It does not overwrite `keyframes.txt`, meshes, or `grasps.txt`. Missing hands or a mismatched source causes an error.
- Bimanual replay retains all source frames, without the single-arm mode's four-frame crop. Resampling uses source FPS and simulation dt. The current 81-frame, 10 FPS clip produces 481 poses at 60 Hz, preserving measured poses and elapsed time between them.
- Before replay, native dual-EE JaxMP IK screens candidates at sampled source contact frames. Only grasp choice and rigid initial placement change; relative object motion remains recorded. `--placement-attempts` controls the placement budget, default eight.
- Acceptance is per environment: feasible members replay; infeasible members hold their arms at home and their data is rejected. A wholly infeasible batch raises an error before motion. The outer launcher preserves failure status even if Isaac fast shutdown exits with code zero while unwinding an exception.
- Current inferred, zero-based contact intervals are right frames 3-38 and left frames 29-58, including a shared interval at 29-38. These values were calculated from this clip, not hardcoded. After its interval, a released arm retreats rather than following the other arm's object motion.
- The scene currently supports two hands on one tracked rigid object. Multi-object and articulated-object generalization, complete collision checks, and reliable support/release detection are unfinished.

Validation so far:

- One no-save run accepted both environments after screening. Another run found only one feasible environment after eight placement attempts, showing that feasibility selection is not yet reliable.
- A later saved run under `/tmp/openarm-video-handoff-review-20261112-v3/` accepted two environments under kinematic replay checks. Each has 633 images, `joint_angles (633,16)`, both TCP poses in `ee_poses (633,14)`, and `gripper_binary_cmd (633,2)`.
- The left gripper was closed for 180 steps and the right for 216, matching the inferred source intervals at 60 Hz. All saved numerical values were finite, and image counts matched state counts.
- Maximum measured contact-period position error was approximately 5.2 mm; rotation error was approximately 0.378 rad. Position and orientation acceptance thresholds are 0.02 m and 0.45 rad, respectively, with TCP height and joint-limit checks as well.
- Inspection found that parts of the handoff leave the head camera's view. Camera visibility filtering is unfinished. These outputs are review artifacts, not accepted training data.
- Saved episodes include `replay_validation.json`, recording source hash, intervals, selected grasps, and maximum contact errors. It explicitly sets `collision_validated=false` and `force_supported_handoff_validated=false`. The inherited `successes/` directory name means only that replay checks passed.
- Ten video regression checks passed: contact inference, missing detections, coordinate/scale conversion, source consistency, independent hand timing, regrasp timing, both TCP targets following translation/rotation, and duration-preserving resampling. Rejected environments are also checked to stay at home.
- Six launcher/process regression checks passed, including cancellation, cleanup isolation, safe spawn imports, and preserving failures across Isaac shutdown.

## Running the Current Bimanual Replay

Run from the repository root in the `r2r2r` conda environment. Setup references: [installation](install_rtx5080_conda.md) and [full pipeline](run_rtx5080_conda.md). The robot, camera, reconstruction, and tracking assets must already be present locally.

For a new metadata file, run:

```bash
conda activate r2r2r
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 python scripts/openarm/prepare_bimanual_track.py \
  --track outputs/pringles/track/keyframes.txt \
  --dig-config outputs/pringles/dig/2026-10-30_181519/config.yml \
  --mesh outputs/pringles/state_rigid_20261104_193955/graspable_part_mesh.obj \
  --video demonstrations/pick_up_pringle_10fps.mp4 \
  --output outputs/pringles/bimanual/video_handoff.json
```

The current workspace already has `outputs/pringles/bimanual/video_handoff.json`. Skip preparation if it matches the current track. The preparation script refuses to overwrite an existing output; use a new output filename when regenerating, and pass that filename to replay. If video metadata is unavailable, specify the actual tracking clip's FPS with `--source-fps`.

Inspect one episode without saving:

```bash
PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES python scripts/openarm/run_openarm_pringles.py \
  --task bimanual-track \
  --bimanual-data outputs/pringles/bimanual/video_handoff.json \
  --num-envs 2 --max-episodes 1 --no-save
```

Open `http://localhost:8080` to connect and start. Initial grasp/placement screening can take time. Add `--auto-start` to skip waiting for a browser, or remove `--no-save` to write data to `output_data/openarm_pringles_bimanual_track/`. Use `--output-dir` to select a different output folder. Use `--max-episodes` to bound attempts; `--max-successes` limits accepted saved environments and does not bound a no-save run.

For the legacy right-arm replay, use `--task right-pick`; it does not use the bimanual metadata. A new demonstration requires its own tracking and metadata preparation.

## Remaining Risks and Next Work

1. Add collision constraints or rejection for arm-arm, arm-object, and arm-table intersections, including approach and retreat.
2. Validate contact inference and placement/support timing; ensure neither hand releases prematurely. Complete the legacy single-arm release/retreat interval.
3. Improve grasp/placement feasibility selection and validate complete trajectories rather than relying on sampled IK screening alone.
4. Add camera visibility checks and improve noisy rendering before generating training datasets.
5. Validate transfer to additional videos and tasks. Current evidence covers one rigid-object handoff, not general bimanual manipulation.
6. The local can mesh has gaps from unscanned and reflective regions. Improved capture or a new scan is still desirable. The missing collider and pointed-bottom issues were addressed locally, but full collision behavior remains unvalidated.
7. Isaac Sim 4.5 loading has been verified for the current OpenArm assets. Compatibility with other asset exports or a migration to newer IsaacLab versions would require separate validation.
