"""Base OpenArm scene for R2R2R: robot, real-world workspace pad, head (body) camera, lights.

Workspace pad and head camera match the real setup used in Stanley_ws/IsaacLab
(stack_joint_pos_env_cfg.py, stack_ik_abs_visuomotor_env_cfg.py). See docs/openarm_port_plan.md.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiTiledCameraCfg
from isaaclab.utils import configclass

from real2render2real.isaaclab_viser.configs.articulation_configs.openarm_cfg import OPENARM_CFG

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")

# Workspace pad (the real table top): a full-height box in front of the robot base.
PAD_HEIGHT = 0.28
PAD_SIZE_X = 0.48  # away from the robot
PAD_SIZE_Y = 0.57  # along the robot
PAD_NEAR_EDGE_X = 0.03
PAD_CENTER_X = PAD_NEAR_EDGE_X + PAD_SIZE_X / 2

# Head camera ("body_cam" on openarm_body_link0, which sits at the robot base origin).
# Pose is in ROS convention (x right, y down, z forward); the robot base is the env origin.
HEAD_CAM_POS = (0.05621, -0.00305, 0.83062)
HEAD_CAM_WXYZ_ROS = (0.18720, -0.68244, 0.69543, -0.12490)
HEAD_CAM_WIDTH, HEAD_CAM_HEIGHT = 640, 480
HEAD_CAM_FOCAL_LENGTH = 24.0
HEAD_CAM_HORIZONTAL_APERTURE = 25.2956
# Render camera poses (xyz + wxyz, ROS convention) for CameraManager. Not a scene field:
# InteractiveScene rejects non-asset entries in the scene cfg.
HEAD_CAM_POSES_FILE = os.path.abspath(os.path.join(data_dir, "camera_poses/openarm_head_cam.h5"))


@configclass
class OpenArmBaseCfg(InteractiveSceneCfg):
    """OpenArm scene; subclass it and add the task's rigid objects."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    workspace_pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WorkspacePad",
        spawn=sim_utils.CuboidCfg(
            size=(PAD_SIZE_X, PAD_SIZE_Y, PAD_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.035, 0.035, 0.035), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(PAD_CENTER_X, 0.0, PAD_HEIGHT / 2)),
    )

    robot: ArticulationCfg = OPENARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    viewport_camera = MultiTiledCameraCfg(  # INCLUDE THIS IN ALL CUSTOM CONFIGS TO LINK WITH A VISER VIEWPORT
        prim_path="{ENV_REGEX_NS}/Viewport",
        height=HEAD_CAM_HEIGHT,
        width=HEAD_CAM_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=HEAD_CAM_FOCAL_LENGTH,
            horizontal_aperture=HEAD_CAM_HORIZONTAL_APERTURE,
            clipping_range=(0.01, 20.0),
        ),
        cams_per_env=1,
    )

    light_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LightLeft",
        spawn=sim_utils.CylinderLightCfg(intensity=1003.0, radius=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.46, -0.64, 1.3)),
    )
    light_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LightRight",
        spawn=sim_utils.CylinderLightCfg(intensity=1003.0, radius=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.46, 0.4, 1.3)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, texture_file=f"{data_dir}/assets/skyboxes/12_9_2024_BWW.jpg"),
    )
