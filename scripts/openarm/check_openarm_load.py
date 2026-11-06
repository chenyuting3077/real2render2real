"""Step 1 of docs/openarm_port_plan.md: check that OpenArm loads in Isaac Sim 4.5 / IsaacLab 0.36.

Spawns the bimanual OpenArm from URDF, prints joints/bodies, drives the right arm and both
grippers to a few targets, and compares Isaac's TCP poses against URDF forward kinematics
(the same URDF the JaxMP IK controller uses).

    PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES python scripts/openarm/check_openarm_load.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--keep-mimic", action="store_true", help="keep finger_joint2 as a PhysX mimic joint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import numpy as np
import torch
import yourdfpy

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
URDF = os.path.join(REPO, "data/openarm_description/openarm_bimanual.urdf")

OPENARM_URDF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF,
        fix_base=True,
        merge_fixed_joints=False,  # keep openarm_*_hand_tcp links for EE poses
        convert_mimic_joints_to_normal_joints=not args_cli.keep_mimic,  # default: drive both fingers explicitly, like YuMi
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"openarm_(left|right)_joint[1-7]": 0.0, "openarm_.*_finger_joint.*": 0.044},
    ),
    # Gains and limits from Stanley_ws/IsaacLab OPENARM_BI_CFG.
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=["openarm_(left|right)_joint[1-7]"],
            velocity_limit_sim={"openarm_(left|right)_joint[1-4]": 2.175, "openarm_(left|right)_joint[5-7]": 2.61},
            effort_limit_sim={"openarm_(left|right)_joint[1-2]": 40.0, "openarm_(left|right)_joint[3-4]": 27.0, "openarm_(left|right)_joint[5-7]": 7.0},
            stiffness=1000.0,
            damping=10.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=["openarm_.*_finger_joint.*"],
            velocity_limit_sim=2.0,
            effort_limit_sim=333.33,
            stiffness=3000.0,
            damping=30.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


@configclass
class CheckSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = OPENARM_URDF_CFG


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device))
    scene = InteractiveScene(CheckSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]

    names = robot.joint_names
    if not args_cli.keep_mimic:
        # The URDF importer widens converted mimic joints to [-0.009, 0.053]; restore the URDF range.
        f2 = [i for i, n in enumerate(names) if n.endswith("finger_joint2")]
        robot.write_joint_position_limit_to_sim(torch.tensor([[0.0, 0.044]] * len(f2), device=robot.device)[None], joint_ids=f2)
    print("\n==== joints (IsaacLab order) ====")
    lim = robot.data.joint_pos_limits[0].cpu().numpy()
    for i, n in enumerate(names):
        print(f"{i:2d} {n:32s} limits [{lim[i, 0]: .3f}, {lim[i, 1]: .3f}]")
    print("==== bodies ====\n", robot.body_names)
    for need in ["openarm_right_hand_tcp", "openarm_left_hand_tcp"]:
        print(f"body {need}: {'OK' if need in robot.body_names else 'MISSING'}")

    urdf = yourdfpy.URDF.load(URDF)

    def drive(targets: dict, steps: int = 240):
        q = robot.data.default_joint_pos.clone()
        for n, v in targets.items():
            q[:, names.index(n)] = v
        for _ in range(steps):
            robot.set_joint_position_target(q)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        return q

    # Arm-only poses: right arm moves, left stays; fingers open then closed.
    tests = [
        ("zero, fingers open", {}),
        ("right arm bent, fingers open",
         {"openarm_right_joint1": -0.5, "openarm_right_joint2": 0.3, "openarm_right_joint4": 1.2, "openarm_right_joint6": 0.4}),
        ("right arm bent, fingers closed",
         {"openarm_right_joint1": -0.5, "openarm_right_joint2": 0.3, "openarm_right_joint4": 1.2, "openarm_right_joint6": 0.4,
          "openarm_right_finger_joint1": 0.0, **({} if args_cli.keep_mimic else {"openarm_right_finger_joint2": 0.0})}),
    ]
    root = robot.data.root_state_w[0, :3].cpu().numpy()
    tcp_idx = robot.body_names.index("openarm_right_hand_tcp")
    for label, tgt in tests:
        q_cmd = drive(tgt)
        q = robot.data.joint_pos[0].cpu().numpy()
        err = np.abs(q - q_cmd[0].cpu().numpy())
        isaac_tcp = robot.data.body_state_w[0, tcp_idx, :3].cpu().numpy() - root
        urdf.update_cfg({n: float(q[i]) for i, n in enumerate(names) if n in urdf.actuated_joint_names})
        fk_tcp = urdf.get_transform("openarm_right_hand_tcp")[:3, 3]
        fingers = {n: round(float(q[i]), 4) for i, n in enumerate(names) if "right_finger" in n}
        print(f"\n[{label}] max joint tracking err {err.max():.4f} rad/m | right fingers {fingers}")
        print(f"   right TCP isaac {isaac_tcp.round(4)} | urdf FK {fk_tcp.round(4)} | diff {np.linalg.norm(isaac_tcp - fk_tcp) * 1000:.1f} mm")

    print("\nCHECK_DONE")


if __name__ == "__main__":
    main()
    simulation_app.close()
