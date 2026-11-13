"""Check spawned CPU writers, queued data preservation and bounded shutdown."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import h5py
import numpy as np
import torch
from collections import deque

from real2render2real.isaaclab_viser.data.batch_data_logger import BatchDataLogger, DataSaveTask


class BatchDataLoggerTests(unittest.TestCase):
    def test_flush_preserves_frame_order_before_episode_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = BatchDataLogger(num_processes=1, batch_size=2, start_method="spawn")
            try:
                cameras = {"camera_0": deque([{"rgb": torch.zeros((1, 8, 8, 3), dtype=torch.uint8)}])}
                for frame in range(2):
                    data = {"joint_names": ["joint"], "joint_angles": np.array([[float(frame)]]),
                            "ee_pos": np.zeros((1, 7)), "gripper_binary_cmd": np.array([[True, True]])}
                    logger.save_data(cameras, data, frame, tmp)
                logger.flush(timeout=15)
                logger.redir_data(torch.tensor([True]))
                episodes = list((Path(tmp) / "successes").iterdir())
                self.assertEqual(len(episodes), 1)
                self.assertEqual(len(list(episodes[0].glob("camera_0/rgb/*.jpg"))), 2)
                with h5py.File(episodes[0] / "robot_data/robot_data.h5", "r") as data:
                    np.testing.assert_array_equal(data["joint_angles"][:], [[0.], [1.]])
            finally:
                logger.shutdown()

    def test_spawn_drains_images_and_h5_before_stopping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logger = BatchDataLogger(num_processes=1, batch_size=3, start_method="spawn")
            try:
                for i in range(2):
                    logger.task_queue.put(DataSaveTask(
                        file_name=str(root / f"{i}.jpg"), img=np.zeros((8, 8, 3), dtype=np.uint8),
                    ))
                    logger.task_queue.put(DataSaveTask(
                        task_type="robot_data", robot_data_dir=str(root / "robot_data"), is_first_save=i == 0,
                        robot_data={"joint_names": ["joint"], "joint_angles": np.array([float(i)]),
                                    "ee_pos": np.zeros(7), "gripper_binary_cmd": np.array([False, True])},
                    ))
                logger.shutdown(timeout=15)
                self.assertTrue(all(not p.is_alive() and p.exitcode == 0 for p in logger.processes))
                self.assertEqual(len(list(root.glob("*.jpg"))), 2)
                with h5py.File(root / "robot_data/robot_data.h5", "r") as data:
                    np.testing.assert_array_equal(data["joint_angles"][:], [[0.], [1.]])
                self.assertEqual((root / "robot_data/joint_names.txt").read_text(), "joint")
                logger.shutdown()  # Repeated cleanup must be harmless.
            finally:
                logger.shutdown(timeout=0)

    def test_shutdown_before_first_frame_is_bounded(self):
        logger = BatchDataLogger(num_processes=1, start_method="spawn")
        started = time.monotonic()
        logger.shutdown(timeout=0)
        self.assertLess(time.monotonic() - started, 4)
        self.assertTrue(all(not p.is_alive() for p in logger.processes))

    def test_writer_exits_after_parent_hard_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = (
                "import os,time,numpy as np; from pathlib import Path; "
                "from real2render2real.isaaclab_viser.data.batch_data_logger import BatchDataLogger,DataSaveTask; "
                "logger=BatchDataLogger(num_processes=1,start_method='spawn'); "
                f"Path({str(root / 'worker.pid')!r}).write_text(str(logger.processes[0].pid)); "
                f"output=Path({str(root / 'ready.jpg')!r}); "
                "logger.task_queue.put(DataSaveTask(file_name=str(output),img=np.zeros((8,8,3),dtype=np.uint8))); "
                "deadline=time.monotonic()+15; "
                "\nwhile not output.exists() and time.monotonic()<deadline: time.sleep(.05)\n"
                "os._exit(0 if output.exists() else 1)"
            )
            parent = subprocess.run([sys.executable, "-c", code], timeout=25)
            self.assertEqual(parent.returncode, 0)
            pid = int((root / "worker.pid").read_text())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
                except FileNotFoundError:
                    return
                if state == "Z":
                    return
                time.sleep(.05)
            os.kill(pid, signal.SIGKILL)
            self.fail("Spawned writer survived its parent's hard exit")

    def test_clean_shutdown_releases_semaphores_before_native_exit(self):
        code = (
            "import os; from real2render2real.isaaclab_viser.data.batch_data_logger import BatchDataLogger; "
            "logger=BatchDataLogger(num_processes=1,start_method='spawn'); "
            "logger.shutdown(timeout=15); os._exit(0)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=25)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("leaked semaphore", result.stderr)


if __name__ == "__main__":
    unittest.main()
