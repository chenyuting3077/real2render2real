"""CardboardLift target conventions, adapted only for per-hand video contact timing.

The upstream reference is yumi_simulators/yumi_cardboard_lift.py:
T_target = T_object @ T_grasp @ T_local_approach_offset, in [left, right] order.
No human wrist path, synthetic object motion, or home-to-grasp interpolation
is introduced here. The video timeline supplies which hand is holding.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


@dataclass
class CardboardHandoffConfig:
    setup_phase_steps: int = 13
    grasp_phase_steps: int = 80
    # Upstream ends immediately after lifting. A handoff needs time to retreat.
    release_offset_steps: int = 67
    ee_retracts: dict = field(default_factory=lambda: dict(start=-0.105, grasp=-0.005, release=-0.1))
    resampled: bool = False
    part_deltas_length: int = 0
    resampled_part_deltas_length: int = 0

    def set_part_deltas_length(self, length):
        self.part_deltas_length = length

    def set_resampled_part_deltas_length(self, length):
        self.resampled_part_deltas_length = length
        self.resampled = True

    def release_phase_steps(self, resampled=False):
        return self.grasp_phase_steps + (self.resampled_part_deltas_length if resampled else self.part_deltas_length)

    def total_steps(self, resampled=False):
        return self.release_phase_steps(resampled) + self.release_offset_steps


def hand_approach_offset(timeline, side, count):
    """Return CardboardLift's local-z offset and optional recorded release anchor."""
    stage, fraction, anchor = timeline.phase(side, count)
    cfg = timeline.config
    retract = cfg.ee_retracts
    smooth = float(fraction*fraction*(3-2*fraction))
    if stage == "grasp":
        return retract["grasp"], None
    if stage == "approach":
        return retract["start"]*(1-smooth) + retract["grasp"]*smooth, None
    if stage == "retreat":
        return retract["grasp"]*(1-smooth) + retract["release"]*smooth, anchor
    # A released arm parks at its recorded release pose, rather than following
    # the other arm or jumping back home when the retreat interval ends.
    past = [end for _, end in timeline.intervals[side]
            if count >= cfg.grasp_phase_steps + (end+1)*timeline.upsample]
    if past:
        return retract["release"], past[-1]*timeline.upsample
    return retract["start"], None


def cardboard_target(object_tf, grasp, offset_z):
    """Exact upstream object @ grasp @ local offset composition, as xyz_wxyz."""
    grasp_tf = np.eye(4)
    grasp_tf[:3, :3] = Rotation.from_quat(np.asarray(grasp["orientation"])[[1, 2, 3, 0]]).as_matrix()
    grasp_tf[:3, 3] = grasp["position"]
    offset_tf = np.eye(4)
    offset_tf[2, 3] = offset_z
    result = object_tf @ grasp_tf @ offset_tf
    return np.r_[result[:3, 3], Rotation.from_matrix(result[:3, :3]).as_quat()[[3, 0, 1, 2]]]


def compatible_grasp_indices(vertices, candidates, max_opening=0.088):
    """Reject candidates unable to span the configured convex collision source.

    This is a geometric validity gate, not arm collision avoidance. It rejects
    the previously observed near-zero-width surface artifacts without imposing
    an object-specific minimum width or hand-specific angle.
    """
    planes = ConvexHull(np.asarray(vertices)).equations
    valid = []
    for index, grasp in enumerate(candidates):
        center = np.asarray(grasp["position"], dtype=float)
        axis = Rotation.from_quat(np.asarray(grasp["orientation"])[[1, 2, 3, 0]]).as_matrix()[:, 1]
        distance = planes[:, :3] @ center + planes[:, 3]
        if np.any(distance > 1e-6):
            continue
        rate = planes[:, :3] @ axis
        lo = np.max(-distance[rate < -1e-10]/rate[rate < -1e-10], initial=-np.inf)
        hi = np.min(-distance[rate > 1e-10]/rate[rate > 1e-10], initial=np.inf)
        span = hi-lo
        opening = 2*float(grasp["width"])
        # Permit small mesh/surface differences, but reject grossly inconsistent
        # openings. The actual full-span width must fit the OpenArm gripper.
        if np.isfinite(span) and 0 < span <= max_opening and 0.5*span <= opening <= max_opening:
            valid.append(index)
    return valid


def reference_grasp_frames(candidates, valid_indices, object_rotation, home_rotation):
    """Choose free gripper roll near this arm's URDF home orientation.

    Antipodal geometry specifies a center and an unoriented closing axis,
    not a preferred robot wrist pose. Preserve those video-derived quantities,
    then project the arm's home approach onto the plane normal to the axis.
    This is an OpenArm posture prior, not an upstream frame initialization.
    """
    result = []
    seen = []
    for index in valid_indices:
        grasp = candidates[index]
        center = np.asarray(grasp["position"])
        local_axis = Rotation.from_quat(np.asarray(grasp["orientation"])[[1, 2, 3, 0]]).as_matrix()[:, 1]
        if any(np.allclose(center, c, atol=1e-6) and abs(local_axis @ a) > 1-1e-6 for c, a in seen):
            continue
        seen.append((center, local_axis))
        axis = object_rotation @ local_axis
        if axis @ home_rotation[:, 1] < 0:
            axis = -axis
        approach = home_rotation[:, 2] - (home_rotation[:, 2] @ axis)*axis
        if np.linalg.norm(approach) < 1e-8:
            x = home_rotation[:, 0] - (home_rotation[:, 0] @ axis)*axis
            x /= np.linalg.norm(x)
            approach = np.cross(x, axis)
        approach /= np.linalg.norm(approach)
        x = np.cross(axis, approach)
        x /= np.linalg.norm(x)
        approach = np.cross(x, axis)
        local_rotation = object_rotation.T @ np.stack([x, axis, approach], axis=-1)
        quat = Rotation.from_matrix(local_rotation).as_quat()[[3, 0, 1, 2]]
        result.append(dict(grasp, orientation=quat.tolist(), source_candidate=index,
                           frame_variant="home_reference", grasp_frame="urdf_home_reference"))
    return result
