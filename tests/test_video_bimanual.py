"""Video provenance, coordinate frames, independent hand timing and shared-object targets."""

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from real2render2real.isaaclab_viser.openarm_simulators.video_bimanual import (
    SIDES, VideoContactTimeline, contact_intervals, hand_observations, infer_contact,
    load_bimanual_data, pose_matrix, source_digest,
)


def metadata(left, right):
    return {"hands": {side: {"contact": contacts} for side, contacts in zip(SIDES, (left, right))}}


class VideoBimanualTests(unittest.TestCase):
    def test_visible_distant_hand_is_not_grasp_contact(self):
        distances = [np.inf, .1, .02, .03, .04, .05, .08, .1]
        np.testing.assert_array_equal(infer_contact(distances), [False, False, True, True, True, True, False, False])

    def test_missing_or_isolated_detection_is_not_invented_contact(self):
        self.assertFalse(infer_contact([np.inf, .01, np.inf]).any())

    def test_hand_coordinates_remove_object_motion_and_dataset_scale(self):
        deltas = np.array([[[1, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 2, 0, 0]]], dtype=float)
        local = np.zeros((21, 3)); local[4] = [.05, 0, 0]; local[8] = [-.05, 0, 0]
        hands = {str(t): [{"keypoints_3d": (local*2+deltas[t, 0, 4:])[None].tolist()}, None] for t in range(2)}
        data = dict(part_deltas=deltas, T_world_objinit=[[1, 0, 0, 0, 0, 0, 0]],
                    T_objreg_objinit=[[1, 0, 0, 0, 0, 0, 0]], hands=hands)
        points, distances = hand_observations(data, local[[4, 8]], scale=2)
        np.testing.assert_allclose(points["left"], np.tile(local, (2, 1, 1)))
        np.testing.assert_allclose(distances["left"], 0, atol=1e-14)
        self.assertTrue(np.isinf(distances["right"]).all())

    def test_missing_hands_is_an_error_instead_of_a_synthetic_task(self):
        with self.assertRaisesRegex(ValueError, "save-hand"):
            hand_observations({"hands": {}}, np.zeros((3, 3)), scale=1)

    def test_contact_timing_follows_source_frames_and_resampling(self):
        data = metadata([False]*3+[True]*4+[False]*3, [True]*5+[False]*5)
        config = SimpleNamespace(grasp_phase_steps=20, setup_phase_steps=5)
        timeline = VideoContactTimeline(data, config, raw_start=0, upsample=6)
        timeline.update(20)
        self.assertEqual((timeline.l_gripper_closed, timeline.r_gripper_closed), (False, True))
        timeline.update(20+3*6)
        self.assertEqual((timeline.l_gripper_closed, timeline.r_gripper_closed), (True, True))
        timeline.update(20+5*6)
        self.assertEqual((timeline.l_gripper_closed, timeline.r_gripper_closed), (True, False))
        timeline.update(20+7*6)
        self.assertFalse(timeline.gripper_closed)

    def test_contact_ends_after_clip_even_if_last_frame_was_held(self):
        timeline = VideoContactTimeline(metadata([True]*4, [True]*4),
                                       SimpleNamespace(grasp_phase_steps=10, setup_phase_steps=1), raw_start=0, upsample=2)
        timeline.update(18)
        self.assertFalse(timeline.gripper_closed)

    def test_regrasp_takes_precedence_over_previous_retreat(self):
        data = metadata([True]*3+[False]+[True]*3+[False], [True]*8)
        timeline = VideoContactTimeline(data, SimpleNamespace(grasp_phase_steps=10, setup_phase_steps=1), raw_start=0, upsample=1)
        self.assertEqual(timeline.phase("left", 14)[0], "grasp")

    def test_metadata_rejects_changed_source_video_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            track, manifest = Path(tmp)/"track.json", Path(tmp)/"hands.json"
            track.write_text('{"part_deltas": []}')
            grasp = dict(position=[0, 0, 0], orientation=[1, 0, 0, 0], width=.03)
            data = dict(schema_version=1, source=dict(track_sha256=source_digest(track), frame_count=3, scale=2),
                        hands={side:dict(contact=[True]*3, grasp_candidates=[grasp], selected_grasp=0) for side in SIDES})
            manifest.write_text(json.dumps(data))
            load_bimanual_data(manifest, track, 2)
            track.write_text('{"part_deltas": [1]}')
            with self.assertRaisesRegex(ValueError, "match"):
                load_bimanual_data(manifest, track, 2)

    def test_both_runtime_tcp_targets_follow_recorded_translation_and_rotation(self):
        # Execute the real target-construction method with a tensor scene stand-in,
        # avoiding Isaac startup. Changing the recorded object pose must change BOTH targets.
        repo = Path(__file__).resolve().parents[1]
        source = repo/"real2render2real/isaaclab_viser/openarm_simulators/openarm_pringles_bimanual.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_calculate_target_poses")
        namespace = dict(np=np, pose_matrix=pose_matrix, SIDES=SIDES, Rotation=Rotation, Slerp=Slerp)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        state = torch.tensor([[.3, 0., .4, 1., 0., 0., 0.]])
        grasps = {side: dict(position=[0, .03 if side == "left" else -.03, 0], orientation=[1, 0, 0, 0]) for side in SIDES}
        sim = SimpleNamespace(scene=SimpleNamespace(num_envs=1, env_origins=torch.zeros((1, 3))),
                              home_ee_poses={s:np.array([[.1, 0., .5, 1., 0., 0., 0.]]) for s in SIDES},
                              video_grasps=grasps, replay_feasible=np.array([True]),
                              state_machine=SimpleNamespace(phase=lambda side,count:("grasp", 1., None)))
        sim._grasp_for = lambda side, env: grasps[side]
        obj = SimpleNamespace(data=SimpleNamespace(root_state_w=state))
        first = namespace["_calculate_target_poses"](sim, 10, obj).copy()
        state[:, :3] += torch.tensor([.2, -.1, .05])
        translated = namespace["_calculate_target_poses"](sim, 10, obj)
        np.testing.assert_allclose(translated[0, :, :3]-first[0, :, :3], np.tile([.2, -.1, .05], (2, 1)), atol=1e-7)
        quat = Rotation.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
        state[0, 3:] = torch.tensor(quat, dtype=torch.float32)
        rotated = namespace["_calculate_target_poses"](sim, 10, obj)
        np.testing.assert_allclose(rotated[0, 0, :3]-state[0, :3].numpy(), [-.03, 0, 0], atol=1e-7)
        np.testing.assert_allclose(rotated[0, 1, :3]-state[0, :3].numpy(), [.03, 0, 0], atol=1e-7)
        # A rejected batch member must stay at home despite moving object poses.
        sim.replay_feasible[:] = False
        rejected = namespace["_calculate_target_poses"](sim, 10, obj)
        for i, side in enumerate(SIDES):
            np.testing.assert_allclose(rejected[:, i], sim.home_ee_poses[side])

    def test_video_resampling_preserves_measured_poses_and_duration(self):
        repo = Path(__file__).resolve().parents[1]
        source = repo/"real2render2real/isaaclab_viser/openarm_simulators/openarm_pringles_pick.py"
        tree = ast.parse(source.read_text())
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "upsample_part_deltas")
        namespace = dict(torch=torch, onp=np, Rotation=Rotation, Slerp=Slerp)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        measured = np.zeros((81, 1, 7), dtype=np.float32)
        angles = np.linspace(0, .8, 81)
        measured[:, 0, :4] = Rotation.from_euler("z", angles).as_quat()[:, [3, 0, 1, 2]]
        measured[:, 0, 4:] = np.stack([np.sin(angles)*.2, angles*.1, np.sin(angles*2)*.1], axis=-1)
        result = namespace["upsample_part_deltas"](torch.tensor(measured), 60/10).numpy()
        self.assertEqual(len(result), 481)
        np.testing.assert_allclose(result[::6], measured, atol=1e-6)
        self.assertAlmostEqual((len(result)-1)/60, (len(measured)-1)/10)
        self.assertGreater(np.ptp(result[:, 0, 4]), .1)  # Recorded horizontal motion was retained.


if __name__ == "__main__":
    unittest.main()
