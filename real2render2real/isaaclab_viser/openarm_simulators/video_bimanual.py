"""Video-derived hand assignments and grasp candidates for R2R2R rigid objects.

RSRD stores MANO keypoints in the registered first-object frame, in dataset
units. Object deltas move each part in that frame. Never treat hand presence
as grasp contact, nor replace the recorded object motion with a motion primitive.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


SIDES = ("left", "right")


def pose_matrix(wxyz_xyz):
    pose = np.asarray(wxyz_xyz, dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = pose[4:]
    return matrix


def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact_intervals(mask):
    mask = np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    edges = np.flatnonzero(np.diff(mask))
    return [[int(a), int(b-1)] for a, b in zip(edges[::2], edges[1::2])]


def infer_contact(distances, enter=0.04, leave=0.055, min_frames=3):
    """Hysteresis on finger-to-surface distance; absent detections stay unknown/off."""
    if not 0 < enter < leave:
        raise ValueError("Expected 0 < contact enter distance < leave distance")
    active = False
    mask = np.zeros(len(distances), dtype=bool)
    for frame, distance in enumerate(distances):
        if not np.isfinite(distance):
            active = False
        elif active:
            active = distance <= leave
        else:
            active = distance <= enter
        mask[frame] = active
    for start, end in contact_intervals(mask):
        if end-start+1 < min_frames:
            mask[start:end+1] = False
    return mask


def hand_observations(track, mesh_vertices, scale, object_index=0):
    """Return per-hand keypoints in the moving object's local, metric frame."""
    if scale <= 0:
        raise ValueError("Dataset scale must be positive")
    if not track.get("hands"):
        raise ValueError("Track contains no hand reconstruction; rerun tracking with --save-hand")
    deltas = np.asarray(track["part_deltas"], dtype=float)
    if deltas.ndim != 3 or deltas.shape[-1] != 7:
        raise ValueError("Expected part_deltas with shape (frames, parts, 7)")
    initial = [pose_matrix(a) @ pose_matrix(b) for a, b in zip(track["T_world_objinit"], track["T_objreg_objinit"])]
    part_in_base = np.linalg.inv(initial[0]) @ initial[object_index]
    tree = cKDTree(mesh_vertices)
    points = {side: np.full((len(deltas), 21, 3), np.nan) for side in SIDES}
    distances = {side: np.full(len(deltas), np.inf) for side in SIDES}
    for frame, delta in enumerate(deltas[:, object_index]):
        hands = track["hands"].get(str(frame), track["hands"].get(frame))
        if hands is None:
            continue
        inverse = np.linalg.inv(part_in_base @ pose_matrix(delta))
        for side, hand in zip(SIDES, hands):
            if hand is None:
                continue
            keypoints = np.asarray(hand["keypoints_3d"], dtype=float).reshape(-1, 21, 3)
            local = (keypoints @ inverse[:3, :3].T + inverse[:3, 3]) / scale
            scores = tree.query(local[:, [4, 8]].reshape(-1, 3))[0].reshape(-1, 2).mean(axis=1)
            best = int(np.argmin(scores))
            points[side][frame] = local[best]
            distances[side][frame] = scores[best]
    return points, distances


def grasp_candidates(mesh, points, contact, *, max_width=0.088, candidates=12, samples=1000):
    """Use the official antipodal sampler, ranked by observed thumb/index proximity.

    Human wrist direction supplies the preferred approach, and the parallel
    gripper's 180-degree roll symmetry is retained as an alternative candidate.
    These are estimates: robot reachability/collision checks are still required.
    """
    from jaxmp.extras.grasp_antipodal import AntipodalGrasps

    observed = points[contact & np.isfinite(points).all(axis=(1, 2))]
    if len(observed) == 0:
        raise ValueError("No observed contact frames for grasp ranking")
    fingers = observed[:, [4, 8]].mean(axis=1)
    wrist = np.median(observed[:, 0], axis=0)
    grasps = AntipodalGrasps.from_sample_mesh(mesh, max_samples=samples, max_width=max_width)
    centers, axes = np.asarray(grasps.centers), np.asarray(grasps.axes)
    if len(centers) == 0:
        raise ValueError("No antipodal grasps within the robot gripper's width")
    scores = np.median(np.linalg.norm(centers[:, None] - fingers[None], axis=-1), axis=1)
    result = []
    for index in np.argsort(scores):
        center, y = centers[index].astype(float), axes[index].astype(float)
        if not np.isfinite(y).all() or np.linalg.norm(y) < 1e-6:
            continue
        y /= np.linalg.norm(y)
        z = center - wrist
        z -= np.dot(z, y) * y
        if np.linalg.norm(z) < 1e-6:
            continue
        z /= np.linalg.norm(z)
        x = np.cross(y, z)
        rotation = np.stack([x, y, z], axis=1)
        positive = mesh.ray.intersects_location(center[None], y[None], multiple_hits=False)[0]
        negative = mesh.ray.intersects_location(center[None], -y[None], multiple_hits=False)[0]
        if not len(positive) or not len(negative):
            continue
        span = float(np.linalg.norm(positive[0] - negative[0]))
        if not 1e-5 < span <= max_width:
            continue
        for symmetry in (np.eye(3), np.diag([-1., -1., 1.])):
            quat = Rotation.from_matrix(rotation @ symmetry).as_quat()[[3, 0, 1, 2]]
            result.append(dict(position=center.tolist(), orientation=quat.tolist(), width=span/2,
                               video_distance_score_m=float(scores[index])))
        if len(result) >= candidates:
            break
    if not result:
        raise ValueError("Could not construct a finite, width-valid grasp from observed hands")
    return result


def prepare_bimanual_data(track_path, scale, mesh_path, *, object_index=0,
                          enter=0.04, leave=0.055, max_width=0.088, seed=0, source_fps=None):
    import trimesh

    track_path, mesh_path = Path(track_path).resolve(), Path(mesh_path).resolve()
    track = json.loads(track_path.read_text())
    mesh = trimesh.load(mesh_path, force="mesh")
    points, distances = hand_observations(track, mesh.vertices, scale, object_index)
    hands = {}
    for side in SIDES:
        contact = infer_contact(distances[side], enter, leave)
        if not contact.any():
            raise ValueError(f"No supported {side}-hand contact interval; cannot produce a bimanual replay")
        # Save reproducible geometric proposals; no object-specific coordinates.
        np.random.seed(seed)
        proposals = grasp_candidates(mesh, points[side], contact, max_width=max_width)
        hands[side] = dict(contact=contact.tolist(), intervals=contact_intervals(contact),
                           observed=np.isfinite(distances[side]).tolist(),
                           distance_m=[float(x) if np.isfinite(x) else None for x in distances[side]],
                           grasp_candidates=proposals, selected_grasp=0)
    return dict(schema_version=1, source=dict(track_path=str(track_path), track_sha256=source_digest(track_path),
                mesh_path=str(mesh_path), frame_count=len(track["part_deltas"]), scale=float(scale),
                object_index=object_index, hand_frame="registered_first_object", units="metres", fps=source_fps),
                estimation=dict(enter_distance_m=enter, leave_distance_m=leave, seed=seed,
                                robot_feasibility="not-yet-validated"), hands=hands)


def load_bimanual_data(path, track_path, scale, max_finger=0.044):
    data = json.loads(Path(path).read_text())
    source = data["source"]
    if data.get("schema_version") != 1 or source_digest(track_path) != source["track_sha256"]:
        raise ValueError("Bimanual metadata does not match this video track; regenerate it")
    if not np.isclose(source["scale"], scale):
        raise ValueError("Bimanual metadata uses a different dataset scale")
    for side in SIDES:
        hand = data["hands"][side]
        if len(hand["contact"]) != source["frame_count"] or not any(hand["contact"]):
            raise ValueError(f"Invalid {side}-hand contact timeline")
        grasp = hand["grasp_candidates"][hand["selected_grasp"]]
        if not 0 <= grasp["width"] <= max_finger:
            raise ValueError(f"{side}-hand grasp exceeds finger range")
        pose_matrix(grasp["orientation"] + grasp["position"])
    return data


class VideoContactTimeline:
    """Map source-video contact frames to the existing replay clock, without frame windows."""

    def __init__(self, data, config, raw_start=4, upsample=3):
        self.config = config
        self.contacts = np.stack([data["hands"][s]["contact"] for s in SIDES], axis=1)
        self.raw_start, self.upsample = raw_start, upsample
        self.intervals = {s: contact_intervals(self.contacts[raw_start:, i]) for i, s in enumerate(SIDES)}
        self.l_gripper_closed = self.r_gripper_closed = self.gripper_closed = False

    def _randomize_ee_pose_offset(self):
        pass  # Initial object/grasp/view augmentations remain in the original simulator.

    def update(self, count):
        step = count-self.config.grasp_phase_steps
        raw_frame = self.raw_start + int(np.floor(max(step, 0)/self.upsample))
        active = self.contacts[raw_frame] if step >= 0 and raw_frame < len(self.contacts) else [False, False]
        self.l_gripper_closed, self.r_gripper_closed = map(bool, active)
        self.gripper_closed = self.l_gripper_closed or self.r_gripper_closed

    def phase(self, side, count):
        """Return stage, interpolation fraction and source clip anchor for the hand."""
        duration = self.config.grasp_phase_steps-self.config.setup_phase_steps
        windows = []
        for start, end in self.intervals[side]:
            grasp = self.config.grasp_phase_steps + start*self.upsample
            release = self.config.grasp_phase_steps + (end+1)*self.upsample
            approach = max(self.config.setup_phase_steps+1, grasp-duration)
            windows.append((start, end, approach, grasp, release))
            if grasp <= count < release:
                return "grasp", 1., None
        for start, end, approach, grasp, release in windows:
            if approach <= count < grasp:
                return "approach", (count-approach)/max(grasp-approach, 1), start*self.upsample
        past = [window for window in windows if count >= window[-1]]
        if past:
            start, end, approach, grasp, release = past[-1]
            if count < release+duration:
                return "retreat", (count-release)/duration, end*self.upsample
        return "idle", 0., None
