"""Generate R2R2R data with OpenArm: right-arm Pringles pick (docs/openarm_port_plan.md, phase A).

    PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES python scripts/openarm/run_openarm_pringles.py

Open http://localhost:8080 to watch and start (or pass --auto-start for unattended runs).
Successful episodes go to output_data/openarm_pringles_pick/successes/.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num-envs", type=int, default=2, help="parallel envs (16 GB GPU: start with 2-4)")
parser.add_argument("--max-successes", type=int, default=1000, help="stop after this many successful episodes")
parser.add_argument("--auto-start", action="store_true", help="do not wait for a viser browser client")
parser.add_argument("--output-dir", default=None, help="default: output_data/openarm_pringles_pick")
parser.add_argument("--no-save", action="store_true", help="run without saving data")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from real2render2real.isaaclab_viser.configs.scene_configs.openarm_scene_cfg import OpenArmPringlesPickCfg
from real2render2real.isaaclab_viser.openarm_simulators.openarm_pringles_pick import OpenArmPringlesPick


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_dir = args_cli.output_dir or os.path.join(repo, "output_data/openarm_pringles_pick")
    scene_config = OpenArmPringlesPickCfg(num_envs=args_cli.num_envs, env_spacing=1.5)
    # The robot is spawned from URDF, so IsaacLabViser takes the URDF path from the scene config.
    OpenArmPringlesPick(
        simulation_app,
        scene_config,
        save_data=not args_cli.no_save,
        output_dir=output_dir,
        auto_start=args_cli.auto_start,
        max_successful_envs=args_cli.max_successes,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
