"""Exercise cancellation against real processes, including unresponsive children."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]


def process_running(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


class ProcessLifecycleTests(unittest.TestCase):
    def launch(self, worker_code, shutdown_timeout=0.3):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(REPO / 'scripts/openarm')!r}); "
            "from process_lifecycle import run_supervised; "
            f"sys.exit(run_supervised([sys.executable, '-c', {worker_code!r}], "
            f"shutdown_timeout={shutdown_timeout}, terminate_timeout=0.2))"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        self.addCleanup(self.cleanup_process, proc)
        return proc

    @staticmethod
    def cleanup_process(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)
        if proc.stdout:
            proc.stdout.close()

    def wait_file(self, path):
        deadline = time.monotonic() + 10
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(path.exists(), f"Worker never became ready: {path}")

    def assert_stopped(self, pids):
        deadline = time.monotonic() + 3
        while any(process_running(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.02)
        for pid in pids:
            self.assertFalse(process_running(pid), f"Process {pid} survived cleanup")

    def test_normal_exit_preserves_status(self):
        proc = self.launch("raise SystemExit(7)")
        output, _ = proc.communicate(timeout=5)
        self.assertEqual(proc.returncode, 7, output)

    def test_ctrl_c_kills_unresponsive_tree_without_touching_other_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "ready.json"
            child_code = "import signal,time; signal.signal(2,signal.SIG_IGN); signal.signal(15,signal.SIG_IGN); time.sleep(60)"
            worker = (
                "import os,sys,signal,subprocess,time,json; from pathlib import Path; "
                "signal.signal(2,signal.SIG_IGN); signal.signal(15,signal.SIG_IGN); "
                f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                f"Path({str(ready)!r}).write_text(json.dumps([os.getpid(),child.pid])); time.sleep(60)"
            )
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            self.addCleanup(self.cleanup_process, unrelated)
            proc = self.launch(worker)
            self.wait_file(ready)
            pids = json.loads(ready.read_text())
            started = time.monotonic()
            os.kill(proc.pid, signal.SIGINT)
            output, _ = proc.communicate(timeout=5)
            self.assertEqual(proc.returncode, 130, output)
            self.assertLess(time.monotonic() - started, 4)
            self.assert_stopped(pids)
            self.assertIsNone(unrelated.poll())

    def test_normal_parent_exit_also_cleans_orphan(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "child.pid"
            child_code = (
                "import os,signal,time; from pathlib import Path; "
                "signal.signal(15,signal.SIG_IGN); "
                f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
            )
            worker = (
                "import sys,subprocess,time; from pathlib import Path; "
                f"subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                f"ready=Path({str(ready)!r}); "
                "\nwhile not ready.exists(): time.sleep(.01)\n"
            )
            proc = self.launch(worker)
            output, _ = proc.communicate(timeout=5)
            self.assertEqual(proc.returncode, 0, output)
            self.assert_stopped([int(ready.read_text())])

    def test_spawn_import_does_not_start_isaac(self):
        # This test runs even in a Python environment with no Isaac installation.
        import runpy
        namespace = runpy.run_path(str(REPO / "scripts/openarm/run_openarm_pringles.py"), run_name="__mp_main__")
        self.assertIn("_run_simulation", namespace)
        self.assertNotIn("simulation_app", namespace)

    def test_launcher_preserves_failure_if_kit_exits_zero(self):
        import runpy
        namespace = runpy.run_path(str(REPO / "scripts/openarm/run_openarm_pringles.py"), run_name="__mp_main__")
        fake_lifecycle = ModuleType("process_lifecycle")

        def native_exit_zero(command):
            status = Path(command[command.index("--_failure-status-file")+1])
            status.write_text("simulation failed\n")
            return 0

        fake_lifecycle.run_supervised = native_exit_zero
        with mock.patch.dict(sys.modules, {"process_lifecycle": fake_lifecycle}):
            self.assertEqual(namespace["main"](), 1)

    def test_worker_signal_sets_stop_flag_without_raising_in_physics_callback(self):
        import runpy
        namespace = runpy.run_path(str(REPO / "scripts/openarm/run_openarm_pringles.py"), run_name="__mp_main__")
        app = SimpleNamespace(close=mock.Mock())

        class AppLauncher:
            add_app_launcher_args = staticmethod(lambda parser: None)

            def __init__(self, args):
                self.app = app

        def simulate(*args, stop_requested, **kwargs):
            self.assertFalse(stop_requested())
            # Mimic delivery inside a native callback: no exception may escape.
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            self.assertTrue(stop_requested())

        fake_app = ModuleType("isaaclab.app")
        fake_app.AppLauncher = AppLauncher
        fake_cfg = ModuleType("real2render2real.isaaclab_viser.configs.scene_configs.openarm_scene_cfg")
        fake_cfg.OpenArmPringlesPickCfg = lambda **kwargs: kwargs
        fake_sim = ModuleType("real2render2real.isaaclab_viser.openarm_simulators.openarm_pringles_pick")
        fake_sim.OpenArmPringlesPick = simulate
        original = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            with mock.patch.dict(sys.modules, {
                "isaaclab.app": fake_app, fake_cfg.__name__: fake_cfg, fake_sim.__name__: fake_sim,
            }), mock.patch.object(sys, "argv", ["run_openarm_pringles.py", "--no-save"]):
                namespace["_run_simulation"]()
            app.close.assert_called_once_with()
        finally:
            for signum, handler in original.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    unittest.main()
