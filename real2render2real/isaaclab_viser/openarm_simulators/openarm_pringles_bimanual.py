"""R2R2R bimanual replay: recorded object motion, per-hand grasps and video contact timing.

No synthetic lift height, fixed XY trajectory, hand-written grasp coordinates,
or manually selected frame intervals are used in this replay mode.
"""

from pathlib import Path
import json

import numpy as np
import torch
import jax
from scipy.spatial.transform import Rotation, Slerp

from real2render2real.isaaclab_viser.configs.articulation_configs.openarm_cfg import OPENARM_GRIPPER_OPEN
from real2render2real.isaaclab_viser.configs.scene_configs.openarm_base_cfg import PAD_HEIGHT
from real2render2real.isaaclab_viser.controllers.jaxmp_diff_ik_controller import JaxMPBatchedController
from real2render2real.isaaclab_viser.openarm_simulators.openarm_pringles_pick import (
    ManipulationConfig, OpenArmPringlesPick,
)
from real2render2real.isaaclab_viser.openarm_simulators.video_bimanual import (
    SIDES, VideoContactTimeline, load_bimanual_data, pose_matrix,
)


class OpenArmPringlesBimanual(OpenArmPringlesPick):
    def __init__(self, *args, bimanual_data, placement_attempts=8, **kwargs):
        self.bimanual_data_path = Path(bimanual_data)
        self.home_ee_poses = None
        self.selected_grasp_indices = None
        self.max_contact_errors = None
        self.placement_attempts = placement_attempts
        self.replay_feasible = None
        # One writer preserves the HDF5 order for both arms and all frames.
        kwargs["data_logger_num_processes"] = 1
        kwargs.setdefault("track_start_frame", 0)
        super().__init__(*args, **kwargs)

    def load_track_data(self):
        source = json.loads(self.bimanual_data_path.read_text())["source"]
        fps = source.get("fps")
        if fps is None or fps <= 0:
            raise ValueError("Source video FPS is missing; regenerate bimanual metadata with --video")
        self.traj_upsample = 1/(self.sim.get_physics_dt()*fps)
        super().load_track_data()

    def _load_grasp_data(self):
        track_path = self.ns_output_dir / "track/keyframes.txt"
        self.video_data = load_bimanual_data(self.bimanual_data_path, track_path, self._scale)
        if self.video_data["source"]["object_index"] != 0 or self.part_deltas.shape[1] != 1:
            raise ValueError("This scene currently supports both hands on one tracked rigid object")
        self.video_grasps = {
            side: self.video_data["hands"][side]["grasp_candidates"][self.video_data["hands"][side]["selected_grasp"]]
            for side in SIDES
        }
        # Keep the parent's object placement/augmentation machinery. The second
        # grasp is consumed explicitly by the left-hand target, never duplicated.
        self.grasped_idxs = [0]
        self.grasp_data = [[self.video_grasps["right"]]]
        for side in SIDES:
            print(f"[INFO]: Video {side}-hand contact source frames: "
                  f"{self.video_data['hands'][side]['intervals']}; selected grasp is a video-ranked estimate.", flush=True)

    def _create_state_machine(self):
        config = ManipulationConfig()
        config.release_offset_steps = config.grasp_phase_steps-config.setup_phase_steps
        retained = self.video_data["source"]["frame_count"]-self.track_start_frame
        factor = (len(self.part_deltas)-1)/(retained-1)
        return VideoContactTimeline(self.video_data, config, raw_start=self.track_start_frame, upsample=factor)

    def _create_controller(self):
        return JaxMPBatchedController(
            urdf_path=self.urdf_path["robot"], num_envs=self.scene.num_envs, num_ees=2,
            target_names=[f"openarm_{side}_hand_tcp_joint" for side in SIDES], home_pose=self.home_pose_jaxmp,
        )

    def _approach_yaw(self, part_rot_wxyz):
        # Sample a rigid initial placement as in R2R2R, rather than imposing a
        # task-specific approach angle or editing the recorded motion itself.
        return torch.rand(self.scene.num_envs, device=self.scene.env_origins.device) * (2*np.pi)

    def _grasp_for(self, side, env):
        if self.selected_grasp_indices is None:
            return self.video_grasps[side]
        index = int(self.selected_grasp_indices[env, SIDES.index(side)])
        return self.video_data["hands"][side]["grasp_candidates"][index]

    def _screen_video_grasps(self):
        """Test video-ranked candidates on recorded contact poses using the native IK.

        This changes grasp choice and rigid initial placement only. The input
        object's relative translations, rotations and contact timing are untouched.
        """
        kin = self.controller.kin
        indices = [kin.joint_names.index(f"openarm_{side}_hand_tcp_joint") for side in SIDES]
        fk = jax.jit(jax.vmap(lambda q: kin.forward_kinematics(q)[np.array(indices)]))
        home_fk = np.asarray(kin.forward_kinematics(self.controller.rest_pose))[indices]
        home_targets = np.concatenate([home_fk[:, 4:], home_fk[:, :4]], axis=-1)
        pool = [self.video_data["hands"][side]["grasp_candidates"] for side in SIDES]
        scores = np.full((self.scene.num_envs, 2, max(map(len, pool))), np.inf)
        # Evenly spaced source contact frames determine screening poses.
        # No task-specific frame numbers or motion waypoints are supplied.
        chosen = set()
        contact_masks = []
        for i, side in enumerate(SIDES):
            raw_mask = np.array(self.video_data["hands"][side]["contact"], dtype=bool)[self.track_start_frame:]
            ids = np.flatnonzero(raw_mask)
            for raw in np.unique(np.linspace(ids[0], ids[-1], min(7, len(ids))).round().astype(int)):
                chosen.add(int(round(raw*self.state_machine.upsample)))
            contact_masks.append(raw_mask)
        frames = sorted(min(f, len(self.part_deltas)-1) for f in chosen)
        initial = self.parts_init_state["pringles"].cpu().numpy().copy()
        initial[:, :3] -= self.scene.env_origins.cpu().numpy()
        initial_tf = [pose_matrix(np.r_[p[3:], p[:3]]) for p in initial]
        for candidate in range(max(map(len, pool))):
            if self.stop_requested():
                raise InterruptedError("Cancelled during video grasp screening")
            self.controller.reset()
            worst = np.zeros((self.scene.num_envs, 2))
            for frame in frames:
                delta = self.part_deltas[frame, 0].cpu().numpy().copy()
                delta[4:] /= self._scale
                targets = np.tile(home_targets, (self.scene.num_envs, 1, 1))
                raw = min(int(frame/self.state_machine.upsample), len(contact_masks[0])-1)
                for i in range(2):
                    if not contact_masks[i][raw]:
                        continue
                    grasp = pool[i][min(candidate, len(pool[i])-1)]
                    grasp_tf = pose_matrix(grasp["orientation"]+grasp["position"])
                    for env in range(self.scene.num_envs):
                        pose = initial_tf[env] @ pose_matrix(delta) @ grasp_tf
                        targets[env, i] = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]]
                q = np.asarray(self.controller.compute_ik(targets))
                q = np.clip(q, np.asarray(kin.limits_lower), np.asarray(kin.limits_upper))
                actual = np.asarray(fk(q))
                for i in range(2):
                    if not contact_masks[i][raw]:
                        continue
                    ep = np.linalg.norm(actual[:, i, 4:]-targets[:, i, :3], axis=-1)
                    qa, qt = actual[:, i, :4], targets[:, i, 3:]
                    er = 2*np.arccos(np.clip(np.abs(np.sum(qa*qt, axis=-1)), 0, 1))
                    worst[:, i] = np.maximum(worst[:, i], np.maximum(ep/.02, er/.45))
            for i in range(2):
                if candidate < len(pool[i]):
                    scores[:, i, candidate] = worst[:, i]
        best = scores.argmin(axis=-1)
        quality = np.take_along_axis(scores, best[..., None], axis=-1)[..., 0].max(axis=-1)
        return best, quality

    def _handle_reset(self):
        super()._handle_reset()
        self.home_ee_poses = None
        best_quality = np.full(self.scene.num_envs, np.inf)
        best_indices = np.zeros((self.scene.num_envs, 2), dtype=int)
        best_initial = self.parts_init_state["pringles"].clone()
        for attempt in range(self.placement_attempts):
            if attempt:
                self._reset_object_state()
            indices, quality = self._screen_video_grasps()
            improved = quality < best_quality
            best_quality[improved] = quality[improved]
            best_indices[improved] = indices[improved]
            mask = torch.tensor(improved, dtype=torch.bool, device=self.robot.device)
            best_initial[mask] = self.parts_init_state["pringles"][mask]
            print(f"[INFO]: Video grasp/placement screening {attempt+1}: "
                  f"{int((best_quality <= 1).sum())}/{self.scene.num_envs} environments meet sampled IK limits.", flush=True)
            if np.all(best_quality <= 1):
                break
        self.replay_feasible = best_quality <= 1
        if not np.any(self.replay_feasible):
            raise RuntimeError("No acceptable pair of video-ranked grasps was found in this batch. "
                               "Recorded object motion is retained; grasp/placement feasibility requires further work.")
        if not np.all(self.replay_feasible):
            rejected = np.flatnonzero(~self.replay_feasible).tolist()
            print(f"[WARNING]: Holding infeasible environments {rejected} at home; "
                  "their data will be rejected, while feasible environments replay the video.", flush=True)
        self.success_envs &= torch.tensor(self.replay_feasible, device=self.robot.device)
        self.selected_grasp_indices = best_indices
        self.parts_init_state["pringles"] = best_initial
        self._update_object_states(0)
        self.controller.reset()
        self.max_contact_errors = np.zeros((self.scene.num_envs, 2, 2))

    def _finish_episode(self):
        if self.success_envs is not None and self.max_contact_errors is not None:
            print(f"[INFO]: Video contact max errors [position metres, rotation radians], "
                  f"per environment [left, right]: {self.max_contact_errors.round(6).tolist()}", flush=True)
            if self.data_logger is not None:
                for env in range(self.scene.num_envs):
                    report = dict(
                        validation="kinematic_replay_only", collision_validated=False,
                        force_supported_handoff_validated=False,
                        track_sha256=self.video_data["source"]["track_sha256"],
                        source_fps=self.video_data["source"]["fps"],
                        contact_intervals={s: self.video_data["hands"][s]["intervals"] for s in SIDES},
                        selected_grasp_indices=self.selected_grasp_indices[env].tolist(),
                        max_contact_errors={s: dict(position_m=float(self.max_contact_errors[env, i, 0]),
                                                    rotation_rad=float(self.max_contact_errors[env, i, 1]))
                                            for i, s in enumerate(SIDES)},
                        passes_replay_checks=bool(self.success_envs[env].item()),
                    )
                    folder = Path(self.output_dir) / f"env_{env}"
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "replay_validation.json").write_text(json.dumps(report, indent=2)+"\n")
        super()._finish_episode()

    def _setup_viser_scene(self):
        super()._setup_viser_scene()
        self.transform_handles["ee"].remove()
        self.transform_handles = {
            side: self.viser_server.scene.add_frame(f"/targets/{side}_tcp", axes_length=.06, axes_radius=.002)
            for side in SIDES
        }

    def _calculate_target_poses(self, count, rigid_object):
        if self.home_ee_poses is None:
            self.home_ee_poses = {}
            for side in SIDES:
                xyz, quat = self._get_ee_poses(side)
                self.home_ee_poses[side] = np.concatenate([xyz.cpu().numpy(), quat.cpu().numpy()], axis=1)
        actual = rigid_object.data.root_state_w[:, :7].cpu().numpy().copy()
        actual[:, :3] -= self.scene.env_origins.cpu().numpy()
        target_poses = np.zeros((self.scene.num_envs, 2, 7))
        for i, side in enumerate(SIDES):
            stage, fraction, anchor = self.state_machine.phase(side, count)
            fraction = float(fraction*fraction*(3-2*fraction))
            for env in range(self.scene.num_envs):
                grasp = self._grasp_for(side, env)
                grasp_tf = pose_matrix(grasp["orientation"] + grasp["position"])
                home = self.home_ee_poses[side][env]
                home_tf = pose_matrix(np.r_[home[3:], home[:3]])
                if stage == "idle" or not self.replay_feasible[env]:
                    result = home_tf
                else:
                    if stage == "retreat":
                        initial = self.parts_init_state["pringles"][env].cpu().numpy().copy()
                        initial[:3] -= self.scene.env_origins[env].cpu().numpy()
                        obj_tf = pose_matrix(np.r_[initial[3:], initial[:3]])
                        delta = self.part_deltas[min(int(round(anchor)), len(self.part_deltas)-1), 0].cpu().numpy().copy()
                        delta[4:] /= self._scale
                        obj_tf = obj_tf @ pose_matrix(delta)
                    else:
                        obj_tf = pose_matrix(np.r_[actual[env, 3:], actual[env, :3]])
                    desired = obj_tf @ grasp_tf
                    if stage == "grasp":
                        result = desired
                    else:
                        # Generic pre/post grasp transitions around the observed
                        # contact intervals. Recorded object poses stay unchanged.
                        alpha = fraction if stage == "approach" else 1-fraction
                        result = np.eye(4)
                        result[:3, 3] = home_tf[:3, 3]*(1-alpha) + desired[:3, 3]*alpha
                        result[:3, :3] = Slerp([0, 1], Rotation.from_matrix(
                            np.stack([home_tf[:3, :3], desired[:3, :3]])
                        ))([alpha]).as_matrix()[0]
                target_poses[env, i] = np.r_[result[:3, 3], Rotation.from_matrix(result[:3, :3]).as_quat()[[3, 0, 1, 2]]]
        self.target_poses = target_poses
        return target_poses

    def _update_transform_handles(self, target_poses):
        for i, side in enumerate(SIDES):
            self.transform_handles[side].position = target_poses[self.env, i, :3]
            self.transform_handles[side].wxyz = target_poses[self.env, i, 3:]

    def _ik_wrapped(self, target_poses):
        joints = torch.tensor(np.asarray(self.controller.compute_ik(target_poses)), dtype=torch.float32, device=self.robot.device)
        result = self.robot.data.joint_pos.clone()
        for side in SIDES:
            result[:, self.arm_isaac[side]] = joints[:, self.arm_jaxmp[side]]
        limits = self.robot.data.joint_pos_limits
        result = torch.max(torch.min(result, limits[..., 1]), limits[..., 0])
        invalid = torch.tensor(~self.replay_feasible, device=self.robot.device)
        result[invalid] = self.robot.data.default_joint_pos[invalid]
        self._set_gripper_positions(result)
        return result

    def _set_gripper_positions(self, positions):
        positions[:, self.all_finger_isaac] = OPENARM_GRIPPER_OPEN
        for side, closed in (("left", self.state_machine.l_gripper_closed), ("right", self.state_machine.r_gripper_closed)):
            if closed:
                for env in range(self.scene.num_envs):
                    if self.replay_feasible[env]:
                        positions[env, self.finger_isaac[side]] = self._grasp_for(side, env)["width"]

    def _target_state_error(self):
        errors = []
        failed = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.robot.device)
        for i, side in enumerate(SIDES):
            target = torch.tensor(self.target_poses[:, i], dtype=torch.float32, device=self.robot.device)
            actual = self._tcp_pose_w(side)
            xyz = actual[:, :3]-self.scene.env_origins
            ep = torch.linalg.norm(xyz-target[:, :3], dim=-1)
            er = 2*torch.arccos(torch.abs(torch.sum(actual[:, 3:]*target[:, 3:], dim=-1)).clamp(0, 1))
            errors.extend([ep, er])
            holding = self.state_machine.l_gripper_closed if side == "left" else self.state_machine.r_gripper_closed
            if holding:
                current = torch.stack([ep, er], dim=-1).cpu().numpy()
                self.max_contact_errors[:, i] = np.maximum(self.max_contact_errors[:, i], current)
            failed |= (((ep > .02) | (er > .45)) & holding) | (actual[:, 2] < PAD_HEIGHT+.005)
        self.target_state_errors = torch.stack(errors, dim=-1)
        newly_failed = failed & self.success_envs
        if newly_failed.any().item():
            print(f"[WARNING]: Video bimanual replay rejected environments "
                  f"{torch.where(newly_failed)[0].cpu().tolist()} due to TCP/height error; "
                  "object trajectory will not be replaced or flattened.", flush=True)
        self.success_envs &= ~failed

    def _log_data(self, count):
        if self.data_logger is None:
            return
        lp, lq = self._get_ee_poses("left")
        rp, rq = self._get_ee_poses("right")
        closed = np.tile([self.state_machine.l_gripper_closed, self.state_machine.r_gripper_closed], (self.scene.num_envs, 1))
        data = dict(joint_names=[self.robot.joint_names[i] for i in self.log_joint_ids],
                    joint_angles=self.robot.data.joint_pos[:, self.log_joint_ids].cpu().numpy(),
                    ee_pos=torch.cat([lp, lq, rp, rq], dim=1).cpu().numpy(), gripper_binary_cmd=closed)
        self.data_logger.save_data(self.camera_manager.buffers, data,
                                  count-self.state_machine.config.setup_phase_steps-1, self.output_dir)
        stats = self.data_logger.get_stats()
        self.images_per_second.value = stats["images_per_second"]
        self.successful_envs.value = stats["total_successful_envs"]
