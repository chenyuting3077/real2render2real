"""Configuration for the bimanual OpenArm, spawned from URDF (see docs/openarm_port_plan.md).

* :obj:`OPENARM_CFG`: both arms, 7 DoF each, parallel grippers (0-0.044 m per finger).

The same URDF is used for the JaxMP IK controller and the viser visualisation, so the
simulated TCP frames match IK exactly (checked by scripts/openarm/check_openarm_load.py).
"""

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")

OPENARM_URDF_PATH = os.path.abspath(os.path.join(data_dir, "openarm_description/openarm_bimanual.urdf"))

OPENARM_ARM_JOINTS = "openarm_(left|right)_joint[1-7]"
OPENARM_FINGER_JOINTS = "openarm_(left|right)_finger_joint[12]"
OPENARM_GRIPPER_OPEN = 0.044  # per-finger joint position, metres

OPENARM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=OPENARM_URDF_PATH,
        fix_base=True,
        # Keep the fixed openarm_{left,right}_hand_tcp links: they are the IK targets / EE bodies.
        merge_fixed_joints=False,
        # PhysX mimic joints do not follow in Isaac Sim 4.5, so both fingers are driven explicitly
        # (like YuMi). Call fix_openarm_finger_limits() after the scene is created.
        convert_mimic_joints_to_normal_joints=True,
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Same home pose as Stanley_ws/IsaacLab OPENARM_BI_CFG: elbows at 90 deg, forearms forward.
        joint_pos={  # patterns must not overlap (IsaacLab raises on multiple matches)
            "openarm_(left|right)_joint[1-35-7]": 0.0,
            "openarm_(left|right)_joint4": 1.570796,
            OPENARM_FINGER_JOINTS: OPENARM_GRIPPER_OPEN,
        },
    ),
    # Velocity/effort limits from the DM motor spec sheets (Stanley_ws/IsaacLab OPENARM_BI_CFG).
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=[OPENARM_ARM_JOINTS],
            velocity_limit_sim={"openarm_(left|right)_joint[1-4]": 2.175, "openarm_(left|right)_joint[5-7]": 2.61},
            effort_limit_sim={
                "openarm_(left|right)_joint[1-2]": 40.0,
                "openarm_(left|right)_joint[3-4]": 27.0,
                "openarm_(left|right)_joint[5-7]": 7.0,
            },
            stiffness=1000.0,
            damping=10.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=[OPENARM_FINGER_JOINTS],
            velocity_limit_sim=2.0,
            effort_limit_sim=333.33,
            stiffness=3000.0,
            damping=30.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the bimanual OpenArm (from URDF)."""


def fix_openarm_finger_limits(robot) -> None:
    """Restore finger_joint2 limits to the URDF's [0, 0.044].

    Converting the mimic joint to a normal joint makes the URDF importer widen it to [-0.009, 0.053].
    """
    ids = [i for i, n in enumerate(robot.joint_names) if n.endswith("finger_joint2")]
    limits = torch.tensor([[0.0, OPENARM_GRIPPER_OPEN]] * len(ids), device=robot.device)
    robot.write_joint_position_limit_to_sim(limits.unsqueeze(0).repeat(robot.num_instances, 1, 1), joint_ids=ids)
