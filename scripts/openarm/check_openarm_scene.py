"""Steps 2-3 of docs/openarm_port_plan.md: build the OpenArm Pringles scene and render the head camera.

Lets the can settle on the workspace pad, reports its pose, and saves the head-camera image.

    PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES python scripts/openarm/check_openarm_scene.py --headless --out /tmp/head.png
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True, help="where to save the head-camera PNG")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from real2render2real.isaaclab_viser.configs.articulation_configs.openarm_cfg import fix_openarm_finger_limits
from real2render2real.isaaclab_viser.configs.scene_configs.openarm_base_cfg import HEAD_CAM_POSES_FILE, PAD_HEIGHT
from real2render2real.isaaclab_viser.configs.scene_configs.openarm_scene_cfg import OpenArmPringlesPickCfg


def main():
    # Same render settings as IsaacLabViser (base.py).
    render_cfg = sim_utils.RenderCfg(antialiasing_mode="DLAA", enable_dl_denoiser=True, dlss_mode=1, enable_shadows=False)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0", render=render_cfg))
    scene = InteractiveScene(OpenArmPringlesPickCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot, can, cam = scene["robot"], scene["pringles"], scene["viewport_camera"]
    fix_openarm_finger_limits(robot)

    pose = h5py.File(HEAD_CAM_POSES_FILE, "r")["poses"][0]
    xyz = torch.tensor(pose[None, :3], dtype=torch.float32) + scene.env_origins.cpu()
    wxyz = torch.tensor(pose[None, 3:], dtype=torch.float32)

    # Start in the home pose: driving there from the URDF zero pose would sweep the forearm into the pad.
    q_home = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q_home, robot.data.default_joint_vel.clone())
    for i in range(240):  # 2 s: let the can settle
        robot.set_joint_position_target(q_home)
        scene.write_data_to_sim()
        sim.step(render=(i % 10 == 0))
        scene.update(sim.get_physics_dt())
        cam.set_world_poses(xyz, wxyz, convention="ros")

    p = (can.data.root_state_w[0, :3] - scene.env_origins[0]).cpu().numpy()
    q = can.data.root_state_w[0, 3:7].cpu().numpy()
    tilt = np.degrees(2 * np.arccos(np.clip(np.sqrt(q[0] ** 2 + q[3] ** 2), -1, 1)))  # angle of can axis from vertical
    print(f"CAN settled pos {p.round(4)} (pad top {PAD_HEIGHT}) | tilt {tilt:.1f} deg | lin vel {can.data.root_lin_vel_w[0].norm().item():.4f}")
    tcp = robot.body_names.index("openarm_right_hand_tcp")
    print(f"RIGHT TCP at home: {(robot.data.body_state_w[0, tcp, :3] - scene.env_origins[0]).cpu().numpy().round(3)}")

    for _ in range(5):  # a few extra renders so the tiled camera output is fresh
        sim.step(render=True)
        scene.update(sim.get_physics_dt())
    rgb = cam.data.output["rgb"][0].cpu().numpy()
    Image.fromarray(rgb[..., :3].astype(np.uint8)).save(args_cli.out)
    print(f"SAVED {args_cli.out} shape {rgb.shape} mean {rgb[..., :3].mean():.1f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
