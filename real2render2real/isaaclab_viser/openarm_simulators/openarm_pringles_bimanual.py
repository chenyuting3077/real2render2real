"""R2R2R bimanual replay: recorded object motion, per-hand grasps and video contact timing.

No synthetic lift height, fixed XY trajectory, hand-written grasp coordinates,
or manually selected frame intervals are used in this replay mode.
"""

from pathlib import Path
import json

import numpy as np
import torch
import jax
import trimesh

from real2render2real.isaaclab_viser.configs.articulation_configs.openarm_cfg import OPENARM_GRIPPER_OPEN
from real2render2real.isaaclab_viser.configs.scene_configs.openarm_base_cfg import PAD_HEIGHT
from real2render2real.isaaclab_viser.configs.scene_configs.openarm_scene_cfg import PRINGLES_MESH
from real2render2real.isaaclab_viser.controllers.jaxmp_diff_ik_controller import JaxMPBatchedController
from real2render2real.isaaclab_viser.openarm_simulators.openarm_pringles_pick import (
    OpenArmPringlesPick,
)
from real2render2real.isaaclab_viser.openarm_simulators.cardboard_handoff import (
    CardboardHandoffConfig, cardboard_target, compatible_grasp_indices, hand_approach_offset, reference_grasp_frames,
)
from real2render2real.isaaclab_viser.openarm_simulators.video_bimanual import (
    SIDES, ObservedApproach, VideoContactTimeline, load_bimanual_data, pose_matrix,
)


class OpenArmPringlesBimanual(OpenArmPringlesPick):
    def __init__(self, *args, bimanual_data, placement_attempts=8, augment_placement=False, vertical_only=False, **kwargs):
        self.vertical_only = vertical_only
        self.bimanual_data_path = Path(bimanual_data)
        self.home_ee_poses = None
        self.selected_grasp_indices = None
        self.selected_grasps = None
        self.max_contact_errors = None
        self.placement_attempts = placement_attempts
        self.augment_placement = augment_placement
        self.replay_feasible = None
        # One writer preserves the HDF5 order for both arms and all frames.
        kwargs["data_logger_num_processes"] = 1
        kwargs.setdefault("track_start_frame", 0)
        if not augment_placement:
            kwargs.setdefault("obj_xy_rand", 0.0)
            kwargs.setdefault("approach_yaw_rand", 0.0)
        super().__init__(*args, **kwargs)

    def load_track_data(self):
        source = json.loads(self.bimanual_data_path.read_text())["source"]
        fps = source.get("fps")
        if fps is None or fps <= 0:
            raise ValueError("Source video FPS is missing; regenerate bimanual metadata with --video")
        self.traj_upsample = 1/(self.sim.get_physics_dt()*fps)
        super().load_track_data()
        if self.vertical_only:
            # This single-part scene places part 0 with yaw only. Its Z axis
            # is the table normal in every environment. All replay consumers
            # (IK screening, object writes, reach and release anchors) use
            # this resampled tensor; the source track remains available to
            # reconstruct the observed human reaches.
            self.part_deltas[..., 4:6] = self.part_deltas[:1, :, 4:6].clone()
            print("[INFO]: Vertical-only replay: fixed first-frame XY, tracked Z and rotation.", flush=True)

    def _load_grasp_data(self):
        track_path = self.ns_output_dir / "track/keyframes.txt"
        self.video_data = load_bimanual_data(self.bimanual_data_path, track_path, self._scale)
        if self.video_data["source"]["object_index"] != 0 or self.part_deltas.shape[1] != 1:
            raise ValueError("This scene currently supports both hands on one tracked rigid object")
        collision_source = trimesh.load(PRINGLES_MESH, force="mesh")
        self.observed_approach = ObservedApproach(json.loads(track_path.read_text()), self.video_data,
                                                  collision_source.vertices, self.track_start_frame, self._scale)
        self.valid_grasp_indices = {
            side: compatible_grasp_indices(collision_source.vertices, self.video_data["hands"][side]["grasp_candidates"])
            for side in SIDES
        }
        for side in SIDES:
            if not self.valid_grasp_indices[side]:
                raise ValueError(f"No {side}-hand grasp is compatible with the collision source; regenerate video metadata")
            print(f"[INFO]: Video grasp geometry: {side} valid source indices "
                  f"{self.valid_grasp_indices[side]}", flush=True)
        self.runtime_grasps = {side: self.video_data["hands"][side]["grasp_candidates"] for side in SIDES}
        self.video_grasps = {
            side: self.runtime_grasps[side][self.valid_grasp_indices[side][0]]
            for side in SIDES
        }
        # Keep the parent's object placement/augmentation machinery. The second
        # grasp is consumed explicitly by the left-hand target, never duplicated.
        self.grasped_idxs = [0]
        # Preserve the object's existing OpenArm registration, rather than
        # transplanting CardboardLift's absolute zero-yaw placement. This file
        # is the same asset-specific reference used by the legacy right arm.
        reference_path = Path(PRINGLES_MESH).parent / "grasps.txt"
        self.grasp_data = [json.loads(reference_path.read_text())]
        if not self.grasp_data[0]:
            raise ValueError("The OpenArm object-placement reference contains no saved grasp")
        for side in SIDES:
            print(f"[INFO]: Video {side}-hand contact source frames: "
                  f"{self.video_data['hands'][side]['intervals']}; selected grasp is a video-ranked estimate.", flush=True)

    def _create_state_machine(self):
        config = CardboardHandoffConfig()
        retained = self.video_data["source"]["frame_count"]-self.track_start_frame
        factor = (len(self.part_deltas)-1)/(retained-1)
        return VideoContactTimeline(self.video_data, config, raw_start=self.track_start_frame, upsample=factor)

    def _create_controller(self):
        return JaxMPBatchedController(
            urdf_path=self.urdf_path["robot"], num_envs=self.scene.num_envs, num_ees=2,
            target_names=[f"openarm_{side}_hand_tcp_joint" for side in SIDES], home_pose=self.home_pose_jaxmp,
        )

    def _approach_yaw(self, part_rot_wxyz):
        return super()._approach_yaw(part_rot_wxyz)

    def _grasp_for(self, side, env):
        if self.selected_grasps is None:
            return self.video_grasps[side]
        return self.selected_grasps[env][side]

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
        # Build a bounded per-hand shortlist, then verify the selected fixed
        # pair with the same dual-EE solver as CardboardLift. Left and right
        # may need different grasp axes; matching their indices would
        # impose an artificial coupling absent from the upstream task.
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
        placed = [{side: reference_grasp_frames(self.runtime_grasps[side], self.valid_grasp_indices[side],
                                                initial_tf[env][:3, :3], pose_matrix(home_fk[i])[:3, :3])
                   for i, side in enumerate(SIDES)} for env in range(self.scene.num_envs)]
        pool = [placed[0][side] for side in SIDES]
        pair_count = max(map(len, pool))
        scores = np.full((self.scene.num_envs, 2, pair_count), np.inf)

        def evaluate(pair_indices):
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
                    for env in range(self.scene.num_envs):
                        grasp = placed[env][SIDES[i]][pair_indices[env, i]]
                        targets[env, i] = cardboard_target(initial_tf[env] @ pose_matrix(delta), grasp,
                                                         self.state_machine.config.ee_retracts["grasp"])
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
            return worst

        for candidate in range(pair_count):
            pair = np.tile([min(candidate, len(p)-1) for p in pool], (self.scene.num_envs, 1))
            errors = evaluate(pair)
            for i in range(2):
                if candidate < len(pool[i]):
                    scores[:, i, candidate] = errors[:, i]
            if np.all(np.any(scores <= 1, axis=-1)):
                break
        feasible = scores <= 1
        # Keep the first feasible video-ranked candidate; use lowest error
        # only to diagnose an infeasible hand. Always solve the final pair
        # again: per-hand feasibility alone is not an acceptance criterion.
        best = np.where(feasible.any(axis=-1), feasible.argmax(axis=-1), scores.argmin(axis=-1))
        selected = [{side: placed[env][side][best[env, i]] for i, side in enumerate(SIDES)}
                    for env in range(self.scene.num_envs)]
        return best, evaluate(best).max(axis=-1), selected

    def _handle_reset(self):
        super()._handle_reset()
        self.home_ee_poses = None
        best_quality = np.full(self.scene.num_envs, np.inf)
        best_indices = np.zeros((self.scene.num_envs, 2), dtype=int)
        best_initial = self.parts_init_state["pringles"].clone()
        best_grasps = [None]*self.scene.num_envs
        attempts = self.placement_attempts if self.augment_placement else 1
        for attempt in range(attempts):
            if attempt:
                self._reset_object_state()
            indices, quality, grasps = self._screen_video_grasps()
            improved = quality < best_quality
            best_quality[improved] = quality[improved]
            best_indices[improved] = indices[improved]
            for env in np.flatnonzero(improved):
                best_grasps[env] = grasps[env]
            mask = torch.tensor(improved, dtype=torch.bool, device=self.robot.device)
            best_initial[mask] = self.parts_init_state["pringles"][mask]
            print(f"[INFO]: Video grasp/placement screening {attempt+1}: "
                  f"{int((best_quality <= 1).sum())}/{self.scene.num_envs} environments meet sampled IK limits; "
                  f"normalized errors {best_quality.round(3).tolist()} (pass <= 1).", flush=True)
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
        self.selected_grasps = best_grasps
        self.parts_init_state["pringles"] = best_initial
        self._update_object_states(0)
        self.controller.reset()
        self.max_contact_errors = np.zeros((self.scene.num_envs, 2, 2))
        self.audit = dict(count=[], joints=[], targets=[], objects=[], ee=[], closed=[])

    def _handle_manipulation(self, count):
        super()._handle_manipulation(count)
        self.audit["count"].append(count)
        self.audit["joints"].append(self.robot.data.joint_pos[:, self.log_joint_ids].cpu().numpy().copy())
        self.audit["targets"].append(self.target_poses.copy())
        obj = self.scene.rigid_objects["pringles"].data.root_state_w[:, :7].clone()
        obj[:, :3] -= self.scene.env_origins
        self.audit["objects"].append(obj.cpu().numpy())
        ee = torch.stack([self._tcp_pose_w(side) for side in SIDES], dim=1).clone()
        ee[:, :, :3] -= self.scene.env_origins[:, None, :]
        self.audit["ee"].append(ee.cpu().numpy())
        self.audit["closed"].append([self.state_machine.l_gripper_closed, self.state_machine.r_gripper_closed])

    def _finish_episode(self):
        if self.success_envs is not None and self.max_contact_errors is not None:
            print(f"[INFO]: Video contact max errors [position metres, rotation radians], "
                  f"per environment [left, right]: {self.max_contact_errors.round(6).tolist()}", flush=True)
            if self.data_logger is not None:
                for env in range(self.scene.num_envs):
                    report = dict(
                        baseline="cardboard_target_composition_with_openarm_reference", target_order=list(SIDES),
                        target_formula="object @ per_hand_grasp @ local_approach_offset",
                        initial_object_pose_xyz_wxyz=(self.parts_init_state["pringles"][env] - torch.cat([
                            self.scene.env_origins[env], torch.zeros(4, device=self.robot.device)])).cpu().tolist(),
                        object_translation="vertical_only" if self.vertical_only else "tracked",
                        validation="kinematic_replay_only", collision_validated=False,
                        force_supported_handoff_validated=False,
                        track_sha256=self.video_data["source"]["track_sha256"],
                        source_fps=self.video_data["source"]["fps"],
                        contact_intervals={s: self.video_data["hands"][s]["intervals"] for s in SIDES},
                        selected_grasp_indices=self.selected_grasp_indices[env].tolist(),
                        grasp_index_space="runtime_home_reference_frames",
                        selected_grasps={side: self._grasp_for(side, env) for side in SIDES},
                        max_contact_errors={s: dict(position_m=float(self.max_contact_errors[env, i, 0]),
                                                    rotation_rad=float(self.max_contact_errors[env, i, 1]))
                                            for i, s in enumerate(SIDES)},
                        passes_replay_checks=bool(self.success_envs[env].item()),
                        task_success=None, requires_trajectory_review=True,
                    )
                    joint_names = [self.robot.joint_names[i] for i in self.log_joint_ids]
                    q = np.asarray(self.audit["joints"])[:, env]
                    steps = np.max(np.abs(np.diff(q, axis=0)), axis=0) if len(q)>1 else np.zeros(q.shape[-1])
                    report["max_joint_step_rad"] = {name: float(steps[i]) for i, name in enumerate(joint_names) if "finger" not in name}
                    folder = Path(self.output_dir) / f"env_{env}"
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "replay_validation.json").write_text(json.dumps(report, indent=2)+"\n")
                    arrays = {k: np.asarray(v) if k in ("count", "closed") else np.asarray(v)[:, env]
                              for k, v in self.audit.items()}
                    np.savez_compressed(folder / "trajectory_audit.npz", **arrays)
            print(f"[INFO]: TCP replay gate only (task success unvalidated): {self.success_envs}", flush=True)
            if self.data_logger is not None:
                self.data_logger.flush()
                self.data_logger.redir_data(self.success_envs, accepted_subdir="replay_checks_passed")

    def _setup_viser_scene(self):
        super()._setup_viser_scene()
        self.transform_handles["ee"].remove()
        self.transform_handles = {
            side: self.viser_server.scene.add_frame(f"/targets/{side}_tcp", axes_length=.06, axes_radius=.002)
            for side in SIDES
        }

    def _setup_viser_gui(self):
        super()._setup_viser_gui()
        if hasattr(self, "successful_envs"):
            self.successful_envs.label = "TCP replay checks passed (not task success): "

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
            reach = self.observed_approach.paths.get(side)
            offset_z, anchor = hand_approach_offset(self.state_machine, side, count)
            if reach is not None:
                frame = (count-self.state_machine.config.grasp_phase_steps)/self.state_machine.upsample
            observed_reach = reach is not None and frame < reach[1]
            if observed_reach:
                anchor = reach[1]*self.state_machine.upsample
                offset_z = self.state_machine.config.ee_retracts["grasp"]
            for env in range(self.scene.num_envs):
                home = self.home_ee_poses[side][env]
                if not self.replay_feasible[env] or (observed_reach and frame <= reach[0]):
                    target_poses[env, i] = home
                    continue
                if anchor is None:
                    obj_tf = pose_matrix(np.r_[actual[env, 3:], actual[env, :3]])
                else:
                    initial = self.parts_init_state["pringles"][env].cpu().numpy().copy()
                    initial[:3] -= self.scene.env_origins[env].cpu().numpy()
                    initial_tf = pose_matrix(np.r_[initial[3:], initial[:3]])
                    delta = self.part_deltas[min(int(round(anchor)), len(self.part_deltas)-1), 0].cpu().numpy().copy()
                    delta[4:] /= self._scale
                    obj_tf = initial_tf @ pose_matrix(delta)
                goal = cardboard_target(obj_tf, self._grasp_for(side, env), offset_z)
                if observed_reach:
                    goal = self.observed_approach.target(side, frame, home, initial_tf, goal)
                target_poses[env, i] = goal
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
