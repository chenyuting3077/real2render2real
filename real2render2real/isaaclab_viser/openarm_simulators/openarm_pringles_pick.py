"""OpenArm right-arm pick of the scanned Pringles can (docs/openarm_port_plan.md, phase A).

Task logic follows yumi_simulators/yumi_tiger_pick_r2r2r.py (same state machine, kinematic object
replay of the tracked trajectory, IK while grasped). Robot-specific parts follow the Franka port,
adapted to OpenArm:

* IK (JaxMP) on data/openarm_description/openarm_bimanual.urdf, single target
  ``openarm_right_hand_tcp_joint``; the left arm holds its pose.
* Joint mappings between JaxMP and IsaacLab are built from joint names (no hardcoded index lists).
* Grasps come from ``dev_grasp.py --gripper openarm``: the saved pose *is* the OpenArm TCP frame,
  so it is used as the IK target directly (no YuMi->hand conversion like the Franka port needs).
* The can is yawed so the saved grasp approaches from the robot's side (the right arm cannot reach
  around the can), instead of the tiger task's uniform 360 deg yaw.
* The 10 fps track is upsampled back to 30 fps so per-step motion matches the original R2R2R data.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as onp
import torch
from scipy.spatial.transform import Rotation, Slerp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from jaxmp.extras.urdf_loader import load_urdf as load_jaxmp_urdf

import real2render2real.utils.transforms as tf
from real2render2real.isaaclab_viser.base import IsaacLabViser
from real2render2real.isaaclab_viser.configs.articulation_configs.openarm_cfg import (
    OPENARM_GRIPPER_OPEN,
    fix_openarm_finger_limits,
)
from real2render2real.isaaclab_viser.configs.scene_configs.openarm_base_cfg import HEAD_CAM_POSES_FILE, PAD_HEIGHT
from real2render2real.isaaclab_viser.controllers.jaxmp_diff_ik_controller import JaxMPBatchedController
from trajgen.traj_interp import generate_directional_starts, traj_interp_batch
from trajgen.traj_resampling import generate_uniform_control_points_batch

dir_path = os.path.dirname(os.path.realpath(__file__))
outputs_dir = os.path.join(dir_path, "../../../outputs")

ARM_JOINTS = {side: [f"openarm_{side}_joint{i}" for i in range(1, 8)] for side in ("left", "right")}
FINGER_JOINTS = {side: [f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"] for side in ("left", "right")}
TCP_BODIES = {side: f"openarm_{side}_hand_tcp" for side in ("left", "right")}
ACTIVE_SIDE = "right"


@dataclass
class ManipulationConfig:
    """Configuration for manipulation phases (steps); same phases as the YuMi tiger pick."""

    setup_phase_steps: int = 15  # End of initial setup, begin trajectory
    grasp_phase_steps: int = 96  # Close gripper at step
    release_offset_steps: int = 1  # Steps in trajectory after release
    resampled_part_deltas_length: int = None
    ee_retracts: Dict[str, float] = None
    resampled: bool = False

    def __post_init__(self):
        # Offsets along the TCP approach (z) axis: start backed off, close at the grasp, back off after release.
        self.ee_retracts = {"start": -0.15, "grasp": 0.0, "release": -0.1}

    @property
    def release_phase_steps(self):
        def calculate(resampled: bool = False) -> int:
            return self.grasp_phase_steps + (self.resampled_part_deltas_length if resampled else self.part_deltas_length)

        return calculate

    @property
    def total_steps(self):
        def calculate(resampled: bool = False) -> int:
            return self.release_phase_steps(resampled) + self.release_offset_steps

        return calculate

    def set_part_deltas_length(self, length: int):
        self.part_deltas_length = length

    def set_resampled_part_deltas_length(self, length: int):
        self.resampled_part_deltas_length = length
        self.resampled = True


class ManipulationStateMachine:
    """Approach -> close -> follow trajectory -> release, as in the YuMi tiger pick."""

    def __init__(self, config: ManipulationConfig, lateral_rand: float = 0.05):
        self.config = config
        self.lateral_rand = lateral_rand
        self.gripper_closed = False
        self.ee_goal_offset = [0.0, 0.0, 0.0, 0, -1, 0, 0]
        self._randomize_ee_pose_offset()

    def update(self, count: int):
        self.gripper_closed = False
        if count <= self.config.setup_phase_steps:
            return

        self.ee_goal_offset[2] = self._interpolate_height(
            count,
            self.config.setup_phase_steps + 8,
            self.config.grasp_phase_steps - 2,
            self.config.ee_retracts["start"],
            self.config.ee_retracts["grasp"],
        )
        # Lateral approach offset decays to zero by the grasp.
        self.ee_rand_goal_offset[0] *= 0.95 ** (1.0 / (count - self.config.setup_phase_steps))
        self.ee_rand_goal_offset[1] *= 0.95 ** (1.0 / (count - self.config.setup_phase_steps))
        self.ee_goal_offset[0] = self.ee_goal_offset[2] * self.ee_rand_goal_offset[0] / self.config.ee_retracts["start"]
        self.ee_goal_offset[1] = self.ee_goal_offset[2] * self.ee_rand_goal_offset[1] / self.config.ee_retracts["start"]

        if count > self.config.grasp_phase_steps - 9:
            self.gripper_closed = True

        release = self.config.release_phase_steps(self.config.resampled)
        if count > release + 3:
            self.ee_goal_offset[2] = self._interpolate_height(
                count, release + 5, self.config.total_steps(self.config.resampled),
                self.config.ee_retracts["grasp"], self.config.ee_retracts["release"],
            )
        if count > release:
            self.gripper_closed = False

    @staticmethod
    def _interpolate_height(count, start_count, end_count, start_height, end_height) -> float:
        if count <= start_count:
            return start_height
        if count >= end_count:
            return end_height
        t = (count - start_count) / (end_count - start_count)
        t = t * t * (3 - 2 * t)  # smoothstep
        return start_height + t * (end_height - start_height)

    def _randomize_ee_pose_offset(self):
        self.ee_rand_goal_offset = ((torch.rand((2,)) * 2 - 1) * self.lateral_rand).tolist()


def upsample_part_deltas(part_deltas: torch.Tensor, factor: int) -> torch.Tensor:
    """(T, P, 7) wxyz_xyz -> ((T-1)*factor+1, P, 7): slerp rotations, lerp translations."""
    if factor <= 1:
        return part_deltas
    pd = part_deltas.cpu().numpy()
    T, P, _ = pd.shape
    t_old = onp.arange(T)
    t_new = onp.linspace(0, T - 1, (T - 1) * factor + 1)
    out = onp.zeros((len(t_new), P, 7))
    for p in range(P):
        rots = Rotation.from_quat(pd[:, p, [1, 2, 3, 0]])  # scipy wants xyzw
        out[:, p, :4] = Slerp(t_old, rots)(t_new).as_quat()[:, [3, 0, 1, 2]]
        for k in range(4, 7):
            out[:, p, k] = onp.interp(t_new, t_old, pd[:, p, k])
    return torch.tensor(out, dtype=part_deltas.dtype, device=part_deltas.device)


class OpenArmPringlesPick(IsaacLabViser):
    def __init__(
        self,
        *args,
        dig_config_path: str = f"{outputs_dir}/pringles/dig/2026-10-30_181519/config.yml",
        auto_start: bool = False,
        max_successful_envs: int = 1000,
        traj_upsample: int = 3,
        obj_xy_rand: float = 0.05,
        approach_yaw: float = -0.70,
        approach_yaw_rand: float = 0.15,
        **kwargs,
    ):
        """
        Args:
            auto_start: start immediately instead of waiting for a viser browser client.
            max_successful_envs: stop after this many successful episodes are saved.
            traj_upsample: interpolate the tracked trajectory by this factor (10 fps track -> 30 fps).
            obj_xy_rand: uniform +/- range (m) of the can's start position.
            approach_yaw: grasp approach direction (rad, world yaw; -0.70 = 40 deg toward the right, best in an offline IK search).
            approach_yaw_rand: uniform +/- range (rad) around approach_yaw.
        """
        self.debug_marker_vis = False
        self.dig_config_path = Path(dig_config_path)
        self.ns_output_dir = self.dig_config_path.parent.parent.parent
        self.camera_poses_file = HEAD_CAM_POSES_FILE  # read by IsaacLabViser -> CameraManager
        self.auto_start = auto_start
        self.max_successful_envs = max_successful_envs
        self.traj_upsample = traj_upsample
        self.obj_xy_rand = obj_xy_rand
        self.approach_yaw = approach_yaw
        self.approach_yaw_rand = approach_yaw_rand
        super().__init__(*args, **kwargs)

        self.robot = self.scene.articulations["robot"]
        fix_openarm_finger_limits(self.robot)

        self.load_track_data()
        self.state_machine = ManipulationStateMachine(ManipulationConfig())
        self.state_machine.config.set_part_deltas_length(self.part_deltas.shape[0])
        self.grasp_perturb = None
        self.grasped_obj_loc_augment = False
        self.grasp_idx = 0
        self.run_simulator()

    # ------------------------------------------------------------------ setup

    def _setup_joint_maps(self):
        names = list(self.robot.joint_names)
        jaxmp_names = list(load_jaxmp_urdf(None, Path(self.urdf_path["robot"])).actuated_joint_names)
        self.isaac_ids = {n: i for i, n in enumerate(names)}
        self.jaxmp_ids = {n: i for i, n in enumerate(jaxmp_names)}
        self.arm_isaac = {s: [self.isaac_ids[n] for n in ARM_JOINTS[s]] for s in ARM_JOINTS}
        self.arm_jaxmp = {s: [self.jaxmp_ids[n] for n in ARM_JOINTS[s]] for s in ARM_JOINTS}
        self.finger_isaac = {s: [self.isaac_ids[n] for n in FINGER_JOINTS[s]] for s in FINGER_JOINTS}
        self.all_finger_isaac = self.finger_isaac["left"] + self.finger_isaac["right"]
        default = self.robot.data.default_joint_pos[0].cpu().numpy()
        # Home pose in JaxMP's actuated order (finger_joint2 is a mimic joint there).
        self.home_pose_jaxmp = onp.array([default[self.isaac_ids[n]] for n in jaxmp_names])
        self.tcp_body_ids = {s: self.robot.body_names.index(TCP_BODIES[s]) for s in TCP_BODIES}
        # Logged joints: both arms + one finger per hand, in IsaacLab order.
        self.log_joint_ids = sorted(self.arm_isaac["left"] + self.arm_isaac["right"]
                                    + [self.finger_isaac["left"][0], self.finger_isaac["right"][0]])

    def run_simulator(self):
        """Main simulation loop."""
        self._setup_joint_maps()
        self.controller = JaxMPBatchedController(
            urdf_path=self.urdf_path["robot"],
            num_envs=self.scene.num_envs,
            num_ees=1,
            target_names=[f"openarm_{ACTIVE_SIDE}_hand_tcp_joint"],
            home_pose=self.home_pose_jaxmp,
        )
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"])
        self.robot_entity_cfg.resolve(self.scene)

        count = 0
        sim_dt = self.sim.get_physics_dt()
        self.success_envs = None

        def keep_running():  # `successful_envs` only exists when data saving is on
            return self.data_logger is None or self.successful_envs.value < self.max_successful_envs

        while self.simulation_app.is_running() and keep_running():
            sim_start_time = time.time()
            self._handle_client_connection()
            self._update_rendering(sim_start_time)

            if count % self.state_machine.config.total_steps(self.state_machine.config.resampled) == 0:
                self._handle_reset()
                count = 0

            self.state_machine.update(count)
            self._update_object_states(count)
            if count > self.state_machine.config.setup_phase_steps:
                self._handle_manipulation(count)
                if count > self.state_machine.config.setup_phase_steps + 9:
                    self._log_data(count - 9)
            else:
                self._handle_setup_phase(count)

            self._update_sim_stats(sim_start_time, sim_dt)
            count += 1

    def load_track_data(self):
        """Load the tracked object trajectory and grasps."""
        track_data_path = self.ns_output_dir / "track/keyframes.txt"
        dpt = json.loads((self.dig_config_path.parent / "dataparser_transforms.json").read_text())
        self._scale = dpt["scale"]

        data = json.loads(track_data_path.read_text())
        part_deltas = torch.tensor(data["part_deltas"]).cuda()[4:]
        self.part_deltas = upsample_part_deltas(part_deltas, self.traj_upsample)
        self.T_objreg_objinit = torch.tensor(data["T_objreg_objinit"]).cuda()
        self.T_world_objinit = torch.tensor(data["T_world_objinit"]).cuda()
        print(f"[INFO]: Track {track_data_path}: {part_deltas.shape[0]} frames -> {self.part_deltas.shape[0]} steps")
        self._load_grasp_data()

    def _load_grasp_data(self):
        rigid_state_dirs = sorted(
            [p for p in self.ns_output_dir.glob("state_rigid_*") if p.is_dir()], key=lambda x: x.stem.split("_")[3]
        )
        self.grasped_idxs, self.grasp_data = [], []
        for idx, rigid_state_dir in enumerate(rigid_state_dirs):
            grasp_file = rigid_state_dir / "grasps.txt"
            if grasp_file.exists():
                self.grasped_idxs.append(idx)
                self.grasp_data.append(json.loads(grasp_file.read_text()))
        if not self.grasped_idxs:
            raise ValueError("No grasp data found in rigid state directories")
        if len(self.grasped_idxs) > 1:
            raise ValueError("Single-arm task: expected grasps on exactly one part")
        for g in self.grasp_data[0]:
            if not 0.0 <= g["width"] <= OPENARM_GRIPPER_OPEN:
                raise ValueError(f"Grasp width {g['width']} outside OpenArm finger range; save grasps with "
                                 f"`dev_grasp.py --gripper openarm`")
            # The parallel gripper is symmetric about its approach axis, so a grasp rolled by 180 deg is the
            # same grasp. The right arm reaches side grasps far better with the TCP x axis pointing up
            # (offline IK search), so flip grasps whose x axis points down (object frame, z up).
            rot = Rotation.from_quat(onp.array(g["orientation"])[[1, 2, 3, 0]])
            if rot.apply([1.0, 0.0, 0.0])[2] < 0:
                flipped = (rot * Rotation.from_euler("z", onp.pi)).as_quat()
                g["orientation"] = [float(flipped[3]), float(flipped[0]), float(flipped[1]), float(flipped[2])]
                print(f"[INFO]: Grasp at {onp.round(g['position'], 3).tolist()} rolled 180 deg about its approach axis (TCP x now up)")

    def _setup_viser_gui(self):
        super()._setup_viser_gui()
        self.rigid_objects_viser_frame = []
        for name, rigid_object in self.scene.rigid_objects.items():
            self.rigid_objects_viser_frame.append(
                self.viser_server.scene.add_frame(
                    name,
                    position=rigid_object.data.default_root_state[self.env][:3].cpu().numpy(),
                    wxyz=rigid_object.data.default_root_state[self.env][3:7].cpu().numpy(),
                    axes_length=0.05,
                    axes_radius=0.003,
                )
            )

    def _setup_viser_scene(self):
        super()._setup_viser_scene()
        self.tf_size_handle = 0.2
        self.transform_handles = {
            "ee": self.viser_server.scene.add_frame(
                "tf_ee_env",
                axes_length=0.5 * self.tf_size_handle,
                axes_radius=0.01 * self.tf_size_handle,
                origin_radius=0.1 * self.tf_size_handle,
            )
        }

    def _handle_client_connection(self):
        """Wait for a viser client unless auto_start (headless batch runs)."""
        if self.auto_start:
            return
        while self.client is None:
            clients = self.viser_server.get_clients()
            self.client = clients[0] if len(clients) > 0 else None
            time.sleep(0.1)

    def _update_rendering(self, sim_start_time: float):
        render_start_time = time.time()
        self.render_wrapped_impl()
        # IsaacLabViser.render_wrapped_impl only reads the camera buffers; physics steps run with render=False,
        # so the scene has to be rendered here (the YuMi simulators call sim.render() in their override).
        self.sim.render()
        self.render_time_ms.value = (time.time() - render_start_time) * 1e3

    # ------------------------------------------------------------------ reset

    def randomize_grasp_rot(self):
        """Small perturbation of the grasp rotation, as in the YuMi tasks."""
        device = self.scene.env_origins.device
        perturb_x = tf.SE3.from_rotation(tf.SO3.from_x_radians((torch.randn((1,)) * 0.05).to(device)))
        perturb_y = tf.SE3.from_rotation(tf.SO3.from_y_radians((torch.randn((1,)) * 0.05).to(device)))
        self.grasp_perturb = perturb_x @ perturb_y

    def _handle_reset(self):
        if self.success_envs is not None:
            print(f"[INFO]: Success Envs: {self.success_envs}")
            if self.data_logger is not None:
                self.data_logger.redir_data(self.success_envs)

        # Pick the grasp first: the can's yaw depends on it.
        self.grasp_idx = torch.randint(len(self.grasp_data[0]), (1,)).item()
        self._reset_robot_state()
        self._reset_object_state()
        self._update_object_states(0)
        self.scene.reset()
        self.controller.reset()
        self.state_machine._randomize_ee_pose_offset()
        self.randomize_lighting()
        self.randomize_viewaug()
        self.randomize_grasp_rot()
        self.randomize_skybox_rotation()
        print("[INFO]: Resetting state...")
        self.success_envs = torch.ones((self.scene.num_envs,), device=self.scene.env_origins.device, dtype=bool)

    def _reset_robot_state(self):
        for robot in self.scene.articulations.values():
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += self.scene.env_origins
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())

    def _approach_yaw(self, part_rot_wxyz: torch.Tensor) -> torch.Tensor:
        """Yaw (num_envs,) that makes the saved grasp approach along +x (away from the robot) +/- noise."""
        g = self.grasp_data[0][self.grasp_idx]
        approach_obj = Rotation.from_quat(onp.array(g["orientation"])[[1, 2, 3, 0]]).apply([0.0, 0.0, 1.0])
        part_rot = Rotation.from_quat(part_rot_wxyz.cpu().numpy()[[1, 2, 3, 0]])
        a = part_rot.apply(approach_obj)
        noise = (torch.rand((self.scene.num_envs,)) * 2 - 1) * self.approach_yaw_rand
        return self.approach_yaw + noise - float(onp.arctan2(a[1], a[0]))

    def _reset_object_state(self):
        """Place the can on the pad: random xy, yaw so the grasp faces the right arm."""
        device = self.scene.env_origins.device
        random_xyz = (torch.rand_like(self.scene.env_origins) * 2 - 1) * self.obj_xy_rand
        random_xyz[:, 2] = 0.0

        self.parts_init_state = {}
        for group_idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
            base_transform = tf.SE3(self.T_world_objinit[0:1]) @ tf.SE3(self.T_objreg_objinit[0:1])
            if self.grasped_obj_loc_augment and group_idx in self.grasped_idxs:
                new_starts = generate_directional_starts(
                    self.part_deltas[:, group_idx], self.scene.num_envs, magnitude=0.09, direction_weight=0.7, perp_variation=0.15
                )
                self.new_trajs = traj_interp_batch(
                    traj=self.part_deltas[:, group_idx].cpu().numpy(), new_starts=new_starts.cpu().numpy(), proportion=0.6
                )
                self.new_trajs = generate_uniform_control_points_batch(self.new_trajs, proportion=1.1, tension=0.1)
                self.state_machine.config.set_resampled_part_deltas_length(self.new_trajs.shape[1])

            current_transform = tf.SE3(wxyz_xyz=self.T_world_objinit[group_idx]) @ tf.SE3(wxyz_xyz=self.T_objreg_objinit[group_idx])
            p2o_transform = base_transform.inverse() @ current_transform
            p2o_7vec = tf.SE3(p2o_transform.wxyz_xyz[0].unsqueeze(0).repeat(self.scene.num_envs, 1))
            p2o_7vec.wxyz_xyz[:, 4:] /= self._scale

            yaw = self._approach_yaw(p2o_7vec.wxyz_xyz[0, :4])
            part_inits = tf.SE3.from_rotation(tf.SO3.from_z_radians(yaw.to(device))) @ p2o_7vec
            objects_root_state = rigid_object.data.default_root_state.clone()
            objects_root_state[:, :3] += self.scene.env_origins + part_inits.wxyz_xyz[:, 4:] + random_xyz
            objects_root_state[:, 3:7] = part_inits.wxyz_xyz[:, :4]
            rigid_object.write_root_state_to_sim(objects_root_state)
            self.parts_init_state[list(self.scene.rigid_objects.keys())[group_idx]] = rigid_object.data.root_state_w.clone()[:, :7]

    def _update_object_states(self, count: int):
        """Kinematically replay the tracked trajectory (object-frame deltas) from each env's start pose."""
        cfg = self.state_machine.config
        release = cfg.release_phase_steps(cfg.resampled)
        for group_idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
            objects_root_state = torch.zeros_like(rigid_object.data.root_state_w)
            xyz_wxyz_init = self.parts_init_state[list(self.scene.rigid_objects.keys())[group_idx]]
            obj_init = tf.SE3(torch.cat([xyz_wxyz_init[:, 3:7], xyz_wxyz_init[:, :3]], dim=-1))

            augment = self.grasped_obj_loc_augment and group_idx in self.grasped_idxs
            if cfg.grasp_phase_steps <= count < release:
                if augment:
                    traj = torch.tensor(self.new_trajs[:, count - cfg.grasp_phase_steps], device=self.part_deltas.device)
                else:
                    step = min(count - cfg.grasp_phase_steps, self.part_deltas.shape[0] - 1)
                    traj = self.part_deltas[step, group_idx].unsqueeze(0).repeat(self.scene.num_envs, 1)
            elif count >= release:
                traj = (torch.tensor(self.new_trajs[:, -1], device=self.part_deltas.device) if augment
                        else self.part_deltas[-1, group_idx].unsqueeze(0).repeat(self.scene.num_envs, 1))
            else:
                traj = (torch.tensor(self.new_trajs[:, 0], device=self.part_deltas.device) if augment
                        else self.part_deltas[0, group_idx].unsqueeze(0).repeat(self.scene.num_envs, 1))
            traj = traj.clone()
            traj[:, 4:] /= self._scale
            obj_state = obj_init @ tf.SE3(traj)
            objects_root_state[:, :3] = obj_state.wxyz_xyz[:, 4:]
            objects_root_state[:, 3:7] = obj_state.wxyz_xyz[:, :4]
            rigid_object.write_root_state_to_sim(objects_root_state)
            self._update_object_visualization(rigid_object, group_idx)

    def _update_object_visualization(self, rigid_object, idx: int):
        if not self.init_viser:
            return
        self.rigid_objects_viser_frame[idx].position = (
            rigid_object.data.root_state_w[self.env][:3].cpu().numpy() - self.scene.env_origins.cpu().numpy()[self.env]
        )
        self.rigid_objects_viser_frame[idx].wxyz = rigid_object.data.root_state_w[self.env][3:7].cpu().numpy()

    # ------------------------------------------------------------- manipulation

    def _tcp_pose_w(self, side: str = ACTIVE_SIDE) -> torch.Tensor:
        return self.robot.data.body_state_w[:, self.tcp_body_ids[side], 0:7]

    def _get_ee_poses(self, side: str = ACTIVE_SIDE) -> Tuple[torch.Tensor, torch.Tensor]:
        """TCP pose in the robot base frame."""
        ee_pose = self._tcp_pose_w(side)
        root_pose = self.robot.data.root_state_w[:, 0:7]
        return subtract_frame_transforms(root_pose[:, 0:3], root_pose[:, 3:7], ee_pose[:, 0:3], ee_pose[:, 3:7])

    def _handle_manipulation(self, count: int):
        cfg = self.state_machine.config
        joint_pos_des = self._update_robot_manipulation(count)
        self.sim.step(render=False)

        # Fail if the active TCP goes into the pad.
        too_low = self._tcp_pose_w()[:, 2] < PAD_HEIGHT + 0.005
        self.success_envs = self.success_envs & ~too_low

        if cfg.grasp_phase_steps <= count < cfg.release_phase_steps(cfg.resampled):
            self._handle_grasp_phase(joint_pos_des)
            self._target_state_error()
            self._joint_pos_limit_check()
        elif count == cfg.grasp_phase_steps - 1:
            self._target_state_error()

    def _update_robot_manipulation(self, count: int):
        rigid_object = self.scene.rigid_objects[list(self.scene.rigid_objects.keys())[self.grasped_idxs[0]]]
        rigid_object.update(self.sim.get_physics_dt())
        target_poses = self._calculate_target_poses(count, rigid_object)
        self._update_transform_handles(target_poses)
        joint_pos_des = self._ik_wrapped(target_poses)
        if self._should_apply_actions(count):
            self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
            self.robot.write_data_to_sim()
        return joint_pos_des

    def _calculate_target_poses(self, count: int, rigid_object) -> onp.ndarray:
        """TCP target = object pose @ saved grasp (OpenArm TCP frame) @ approach offset."""
        device = self.scene.env_origins.device
        obj_tf = tf.SE3(torch.cat([rigid_object.data.root_state_w[:, 3:7],
                                   rigid_object.data.root_state_w[:, :3] - self.scene.env_origins], dim=-1))
        g = self.grasp_data[0][self.grasp_idx]
        grasp_tf = tf.SE3(torch.tensor(g["orientation"] + g["position"], dtype=torch.float32, device=device).unsqueeze(0))
        goal_offset_tf = tf.SE3.from_rotation_and_translation(
            tf.SO3(torch.tensor([1.0, 0, 0, 0], device=device).unsqueeze(0)),
            torch.tensor(self.state_machine.ee_goal_offset[:3], dtype=torch.float32, device=device).unsqueeze(0),
        )
        cfg = self.state_machine.config
        if self.grasp_perturb is not None:
            perturbed = grasp_tf @ self.grasp_perturb
            if cfg.grasp_phase_steps <= count < cfg.release_phase_steps(cfg.resampled):
                jitter_x = tf.SE3.from_rotation(tf.SO3.from_x_radians((torch.randn((1,)) * 0.003).to(device)))
                jitter_y = tf.SE3.from_rotation(tf.SO3.from_y_radians((torch.randn((1,)) * 0.003).to(device)))
                perturbed = perturbed @ jitter_x @ jitter_y
            ee_target = obj_tf @ perturbed @ goal_offset_tf
        else:
            ee_target = obj_tf @ grasp_tf @ goal_offset_tf

        target_poses = onp.zeros((self.scene.num_envs, 1, 7))
        target_poses[:, 0, :] = torch.cat([ee_target.wxyz_xyz[:, 4:], ee_target.wxyz_xyz[:, :4]], dim=1).cpu().numpy()
        self.target_poses = target_poses
        return target_poses

    def _update_transform_handles(self, target_poses: onp.ndarray):
        if not self.init_viser:
            return
        self.transform_handles["ee"].position = target_poses[self.env, 0, :3]
        self.transform_handles["ee"].wxyz = target_poses[self.env, 0, 3:7]

    def _ik_wrapped(self, target_poses: onp.ndarray) -> torch.Tensor:
        """IK for the active arm; the other arm holds its current pose."""
        joints = torch.tensor(onp.array(self.controller.compute_ik(target_poses)), dtype=torch.float32, device=self.robot.device)
        joint_pos_des = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids].clone()
        joint_pos_des[:, self.arm_isaac[ACTIVE_SIDE]] = joints[:, self.arm_jaxmp[ACTIVE_SIDE]]
        # JaxMP limits are soft (cost terms), so solutions can overshoot by ~1e-3 rad; clamp to the URDF range.
        limits = self.robot.data.joint_pos_limits
        joint_pos_des = torch.max(torch.min(joint_pos_des, limits[..., 1]), limits[..., 0])
        self._set_gripper_positions(joint_pos_des)
        return joint_pos_des

    def _set_gripper_positions(self, joint_pos_des: torch.Tensor):
        joint_pos_des[:, self.all_finger_isaac] = OPENARM_GRIPPER_OPEN
        if self.state_machine.gripper_closed:
            joint_pos_des[:, self.finger_isaac[ACTIVE_SIDE]] = self.grasp_data[0][self.grasp_idx]["width"]

    def _should_apply_actions(self, count: int) -> bool:
        cfg = self.state_machine.config
        setup_complete = count > cfg.setup_phase_steps + 7
        pre_grasp = count < cfg.grasp_phase_steps
        post_release = count >= cfg.release_phase_steps(cfg.resampled)
        return (setup_complete and pre_grasp) or post_release

    def _handle_grasp_phase(self, joint_pos_des: torch.Tensor):
        """While grasped, write the IK solution directly (as the YuMi tasks do)."""
        joint_vel = torch.zeros_like(self.robot.data.joint_vel[:, self.robot_entity_cfg.joint_ids])
        self.robot.write_joint_state_to_sim(joint_pos_des, joint_vel)

    def _target_state_error(self):
        target = torch.tensor(self.target_poses[:, 0, :], dtype=torch.float32, device=self.robot.device)
        pos_b = self._tcp_pose_w()[:, :3] - self.scene.env_origins
        quat_b = self._tcp_pose_w()[:, 3:7]
        pos_error = (target[:, :3] - pos_b).norm(dim=-1)
        quat_error = 2 * torch.arccos(torch.abs(torch.sum(target[:, 3:7] * quat_b, dim=-1)).clamp(-1, 1))
        self.target_state_errors = torch.stack([pos_error, quat_error], dim=-1)
        error_mask = (pos_error > 0.02) | (quat_error > 0.45)
        self.success_envs = self.success_envs & ~error_mask
        if error_mask.any().item():
            print(f"Target state error exceeded in environments: {torch.where(error_mask)[0].cpu().numpy().tolist()} "
                  f"(pos {pos_error[error_mask].cpu().numpy().round(3).tolist()} m, "
                  f"rot {quat_error[error_mask].cpu().numpy().round(2).tolist()} rad). Marking environments as failed.")

    def _joint_pos_limit_check(self):
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        limits = self.robot.data.joint_pos_limits[:, self.robot_entity_cfg.joint_ids]
        exceeded = torch.any((joint_pos < limits[..., 0] - 1e-4) | (joint_pos > limits[..., 1] + 1e-4), dim=1)
        self.success_envs = self.success_envs & ~exceeded
        if exceeded.any().item():
            print(f"URDF joint limits exceeded in environments: {torch.where(exceeded)[0].cpu().numpy().tolist()}. Marking environments as failed.")

    def _handle_setup_phase(self, count: int):
        cfg = self.state_machine.config
        if count < cfg.setup_phase_steps - 10:
            self._random_perturb_config()
        if count == cfg.setup_phase_steps - 10:
            self._reset_robot_state()
        for idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
            rigid_object.update(self.sim.get_physics_dt())
            self._update_object_visualization(rigid_object, idx)
        self.sim.step(render=False)

    def _random_perturb_config(self):
        target = self.robot.data.default_joint_pos + torch.randn_like(self.robot.data.joint_pos) * 0.015
        target = target.clamp_(self.robot.data.soft_joint_pos_limits[..., 0], self.robot.data.soft_joint_pos_limits[..., 1])
        target[:, self.all_finger_isaac] = OPENARM_GRIPPER_OPEN
        self.robot.set_joint_position_target(target)
        self.robot.write_data_to_sim()
        for group_idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
            state = torch.zeros_like(rigid_object.data.root_state_w)
            state[:, :7] = self.parts_init_state[list(self.scene.rigid_objects.keys())[group_idx]]
            rigid_object.write_root_state_to_sim(state)

    def _update_sim_stats(self, sim_start_time: float, sim_dt: float):
        if self.init_viser:
            self.robot.update(sim_dt)
            joint_dict = {n: self.robot.data.joint_pos[self.env][i].item() for i, n in enumerate(self.robot.data.joint_names)}
            self.urdf_vis["robot"].update_cfg(joint_dict)
            self.isaac_viewport_camera.update(0, force_recompute=True)
        self.sim_step_time_ms.value = (time.time() - sim_start_time) * 1e3

    def _log_data(self, count: int):
        """Log both arms (left idle) so the format also fits the later bimanual hand-off task."""
        if self.data_logger is None:
            return
        pos_l, quat_l = self._get_ee_poses("left")
        pos_r, quat_r = self._get_ee_poses("right")
        closed = torch.tensor(self.state_machine.gripper_closed, device=self.robot.device).repeat(self.scene.num_envs)
        robot_data = {
            "joint_names": [self.robot.data.joint_names[i] for i in self.log_joint_ids],
            "joint_angles": self.robot.data.joint_pos[:, self.log_joint_ids].clone().cpu().numpy(),
            "ee_pos": torch.cat([pos_l, quat_l, pos_r, quat_r], dim=1).cpu().numpy(),
            "gripper_binary_cmd": torch.stack([torch.zeros_like(closed), closed], dim=-1).cpu().numpy(),  # [left, right]
        }
        self.data_logger.save_data(
            self.camera_manager.buffers, robot_data, count - self.state_machine.config.setup_phase_steps - 1, self.output_dir
        )
        stats = self.data_logger.get_stats()
        self.save_time_ms.value = int(stats["save_time"] * 1e3)
        self.images_per_second.value = stats["images_per_second"]
        self.successful_envs.value = stats["total_successful_envs"]
