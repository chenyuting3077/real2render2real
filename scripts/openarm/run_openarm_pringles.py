"""OpenArm R2R2R: replay trajectories extracted from a human demonstration video.

    PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES python scripts/openarm/run_openarm_pringles.py

The Isaac Sim window opens and replay starts without a browser connection by default.
Pass --headless to use http://localhost:8080 instead, or add --auto-start for batch runs.
The default is the original right-arm replay. Bimanual replay requires video-derived
hand assignments and per-hand grasp candidates; it never substitutes a scripted lift.
Ctrl+C requests cleanup; a second Ctrl+C forces this simulation to stop.
"""

import os
import signal
import sys


def _run_simulation():
    # Spawn workers re-import this script and must never create a SimulationApp.
    import argparse
    import time

    from isaaclab.app import AppLauncher

    stopping = False

    def request_stop(_signum, _frame):
        # Raising here can be swallowed by Isaac's native physics callbacks.
        # Let the outer simulation loop stop and run its finally block instead.
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("right-pick", "bimanual-track"), default="right-pick",
                        help="replay video-extracted object motion using one or two arms")
    parser.add_argument("--num-envs", type=int, default=2, help="parallel envs (16 GB GPU: start with 2-4)")
    parser.add_argument("--max-successes", type=int, default=1000, help="stop after this many successful episodes")
    parser.add_argument("--auto-start", action="store_true", help="do not wait for a viser browser client")
    parser.add_argument("--gui", action="store_true", help="explicitly select the default Isaac Sim window")
    parser.add_argument("--output-dir", default=None, help="override the task-specific output_data directory")
    parser.add_argument("--no-save", action="store_true", help="run without saving data")
    parser.add_argument("--max-episodes", type=int, default=0, help="stop after N attempted episodes; 0 runs until cancelled")
    parser.add_argument("--bimanual-data", default=None, help="prepared video-derived hand/grasp assignments JSON")
    parser.add_argument("--placement-attempts", type=int, default=8,
                        help="maximum bimanual placement samples when --augment-placement is enabled")
    parser.add_argument("--seed", type=int, default=None, help="optional seed for reproducible placement/render augmentation")
    parser.add_argument("--vertical-only", action="store_true",
                        help="bimanual replay: hold can XY at the first frame and retain tracked height and orientation")
    parser.add_argument("--augment-placement", action="store_true", help="enable bimanual placement augmentation; default uses the legacy reference placement")
    parser.add_argument("--_failure-status-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dig-config", default=None, help="override the task's DiG config.yml")
    AppLauncher.add_app_launcher_args(parser)
    # This checkout of IsaacLab defaults --headless to True, so merely removing
    # the previous assignment is insufficient to enable the desktop window.
    parser.set_defaults(headless=False)
    args_cli = parser.parse_args()
    if args_cli.num_envs < 1 or args_cli.max_episodes < 0 or args_cli.placement_attempts < 1:
        parser.error("--num-envs/--placement-attempts must be positive; --max-episodes cannot be negative")
    if args_cli.task == "bimanual-track" and not args_cli.bimanual_data:
        parser.error("--task bimanual-track requires --bimanual-data from the video preparation step")
    if args_cli.vertical_only and args_cli.task != "bimanual-track":
        parser.error("--vertical-only requires --task bimanual-track")
    if args_cli.gui and args_cli.headless:
        parser.error("--gui and --headless cannot be used together")
    # AppLauncher also consults HEADLESS; make the selected CLI mode decisive.
    os.environ["HEADLESS"] = "1" if args_cli.headless else "0"
    args_cli.enable_cameras = True
    simulation_app = None
    try:
        launch_started = time.perf_counter()
        mode = "headless" if args_cli.headless else "desktop GUI"
        print(f"[INFO]: Starting Isaac Sim ({mode}). RTX shader warm-up may temporarily "
              "make the window unresponsive; detailed progress is in the Kit log printed below.", flush=True)
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        print(f"[INFO]: Isaac Sim initialized in {time.perf_counter()-launch_started:.1f}s. "
              "Loading OpenArm scene and reconstruction assets...", flush=True)
        # Restore our handlers if Kit replaced them during startup.
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        if stopping:
            return

        if args_cli.seed is not None:
            import numpy as np
            import torch

            np.random.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)

        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        common = dict(save_data=not args_cli.no_save, auto_start=args_cli.auto_start or not args_cli.headless,
                      stop_requested=lambda: stopping, max_episodes=args_cli.max_episodes)
        if args_cli.dig_config:
            common["dig_config_path"] = args_cli.dig_config
        if args_cli.task == "bimanual-track":
            from real2render2real.isaaclab_viser.configs.scene_configs.openarm_scene_cfg import OpenArmPringlesPickCfg
            from real2render2real.isaaclab_viser.openarm_simulators.openarm_pringles_bimanual import OpenArmPringlesBimanual

            output_dir = args_cli.output_dir or os.path.join(repo, "output_data/openarm_pringles_bimanual_track")
            scene_config = OpenArmPringlesPickCfg(num_envs=args_cli.num_envs, env_spacing=1.5)
            OpenArmPringlesBimanual(simulation_app, scene_config, output_dir=output_dir,
                                   bimanual_data=args_cli.bimanual_data,
                                   placement_attempts=args_cli.placement_attempts,
                                   augment_placement=args_cli.augment_placement, vertical_only=args_cli.vertical_only,
                                   max_successful_envs=args_cli.max_successes, **common)
        else:
            from real2render2real.isaaclab_viser.configs.scene_configs.openarm_scene_cfg import OpenArmPringlesPickCfg
            from real2render2real.isaaclab_viser.openarm_simulators.openarm_pringles_pick import OpenArmPringlesPick

            output_dir = args_cli.output_dir or os.path.join(repo, "output_data/openarm_pringles_pick")
            scene_config = OpenArmPringlesPickCfg(num_envs=args_cli.num_envs, env_spacing=1.5)
            OpenArmPringlesPick(simulation_app, scene_config, output_dir=output_dir,
                               data_logger_num_processes=max(1, min(4, args_cli.num_envs)),
                               max_successful_envs=args_cli.max_successes, **common)
        if stopping:
            print("[INFO]: OpenArm simulation cancelled.", flush=True)
        else:
            print("[INFO]: OpenArm replay finished; closing Isaac Sim.", flush=True)
    except KeyboardInterrupt:
        print("[INFO]: OpenArm simulation cancelled.", flush=True)
    except Exception:
        # Kit may hard-exit from close(), bypassing Python's traceback printer.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        if args_cli._failure_status_file:
            from pathlib import Path

            Path(args_cli._failure_status_file).write_text("simulation failed\n")
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()


def main():
    from process_lifecycle import run_supervised
    import tempfile
    from pathlib import Path

    # Kit's fast shutdown can exit(0) while unwinding an exception. Preserve
    # failure status outside Kit, after the supervisor has cleaned its group.
    with tempfile.TemporaryDirectory(prefix="openarm-status-") as status_dir:
        failure = Path(status_dir) / "failed"
        result = run_supervised([sys.executable, os.path.abspath(__file__), "--_simulation-worker",
                                 *sys.argv[1:], "--_failure-status-file", str(failure)])
        return 1 if result == 0 and failure.exists() else result


if __name__ == "__main__":
    if "--_simulation-worker" in sys.argv:
        sys.argv.remove("--_simulation-worker")
        _run_simulation()
    else:
        sys.exit(main())
