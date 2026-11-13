import multiprocessing as mp
import numpy as np
import torch
import cv2
import os
import signal
from typing import Dict, Optional, Union, List, Tuple
from dataclasses import dataclass
from queue import Empty, Full
import threading
import time
from datetime import datetime
from enum import Enum
import h5py
from collections import deque

class SaveStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"

@dataclass
class DataSaveTask:
    batch_id: int = 0
    task_type: str = "image"  # "image" or "robot_data"
    # Image data
    img: Optional[np.ndarray] = None
    file_name: Optional[str] = None
    is_depth: bool = False
    # Robot data
    robot_data: Optional[Dict[str, Union[np.ndarray, List[str]]]] = None
    robot_data_dir: Optional[str] = None
    # Flag for first save (to write joint names)
    is_first_save: bool = False
    
@dataclass
class SaveResult:
    file_name: str
    status: SaveStatus
    error_msg: Optional[str] = None
    batch_id: int = 0

class BatchDataLogger:
    def __init__(
        self, 
        num_processes: int = 4, 
        max_queue_size: int = 1000,
        batch_size: int = 10,
        start_method: Optional[str] = None,
    ):
        context = mp.get_context(start_method)
        self.task_queue = context.Queue(maxsize=max_queue_size)
        self.result_queue = context.Queue()
        self._stop_event = context.Event()
        self._closed = False
        self.processes = []
        self.running = True
        self.batch_size = batch_size
        self.current_batch = 0
        
        # Statistics
        self._total_saved = 0
        self._total_failed = 0
        self._total_submitted = 0
        self._total_successful_envs = 0
        self._start_time = time.time()
        self.start_save_time = self._start_time
        
        # Start worker processes
        self._parent_pid = os.getpid()
        for _ in range(num_processes):
            # A static target keeps the simulation, CUDA tensors and result thread out
            # of the spawned worker's arguments. OpenArm selects spawn after CUDA init.
            p = context.Process(
                target=self._worker_process,
                args=(self.task_queue, self.result_queue, self.batch_size, self._parent_pid, self._stop_event),
            )
            p.daemon = True
            p.start()
            self.processes.append(p)
            
        # Start result processing thread
        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()

    @staticmethod
    def _save_data(task: DataSaveTask) -> Tuple[bool, Optional[str]]:
        """Save image and robot data."""
        try:
            success = True
            error_msg = None

            if task.task_type == "image":
                if task.img is not None and task.file_name is not None:
                    if task.is_depth:
                        normalized = cv2.normalize(task.img, None, 0, 255, cv2.NORM_MINMAX)
                        img_to_save = normalized.astype(np.uint8)
                        img_to_save = cv2.applyColorMap(img_to_save, cv2.COLORMAP_VIRIDIS)
                    else:
                        img_to_save = task.img
                        img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                    
                    os.makedirs(os.path.dirname(task.file_name), exist_ok=True)
                    if not cv2.imwrite(task.file_name, img_to_save):
                        return False, "Failed to write image to disk"

            elif task.task_type == "robot_data":
                if task.robot_data is not None and task.robot_data_dir is not None:
                    os.makedirs(task.robot_data_dir, exist_ok=True)
                    
                    try:
                        # Save joint names only on first save
                        if task.is_first_save and "joint_names" in task.robot_data:
                            joint_names_file = os.path.join(task.robot_data_dir, "joint_names.txt")
                            with open(joint_names_file, 'w') as f:
                                f.write('\n'.join(task.robot_data["joint_names"]))
                        
                        # Use h5py for streaming numerical data
                        h5_file = os.path.join(task.robot_data_dir, "robot_data.h5")
                        
                        if task.is_first_save and os.path.exists(h5_file):
                            os.remove(h5_file)
                
                        with h5py.File(h5_file, 'a', libver='latest') as f:
                            if task.is_first_save:
                                f.create_dataset('joint_angles', 
                                            data=task.robot_data["joint_angles"][np.newaxis, :],
                                            maxshape=(None, task.robot_data["joint_angles"].shape[-1]),
                                            chunks=True)
                                f.create_dataset('ee_poses',
                                            data=task.robot_data["ee_pos"][np.newaxis, :],
                                            maxshape=(None, task.robot_data["ee_pos"].shape[-1]),
                                            chunks=True)
                                f.create_dataset('gripper_binary_cmd',
                                            data=task.robot_data["gripper_binary_cmd"][np.newaxis, :],
                                            maxshape=(None, task.robot_data["gripper_binary_cmd"].shape[-1]),
                                            chunks=True)
                            else:
                                for dataset_name, data in [('joint_angles', task.robot_data["joint_angles"]),
                                                        ('ee_poses', task.robot_data["ee_pos"]),
                                                        ('gripper_binary_cmd', task.robot_data["gripper_binary_cmd"])]:
                                    dataset = f[dataset_name]
                                    current_size = dataset.shape[0]
                                    dataset.resize(current_size + 1, axis=0)
                                    dataset[current_size] = data
                            f.flush()
                            
                    except Exception as e:
                        return False, f"Failed to save robot data: {str(e)}"

            return success, error_msg
            
        except Exception as e:
            return False, str(e)
        
    def redir_data(self, success_envs):
        """
        Redirect data by deleting failed environments and moving successful ones
        to a success subdirectory with datetime stamp.
        
        Args:
            success_envs: Boolean tensor indicating success (True) or failure (False) 
                        for each environment
        """
        try:
            # Convert tensor to numpy for easier handling
            success_envs = success_envs.cpu().numpy()
            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            
            # Count total successful environments
            self._total_successful_envs += np.sum(success_envs)
            
            # Create success directory if it doesn't exist
            success_dir = os.path.join(self.output_dir, "successes")
            os.makedirs(success_dir, exist_ok=True)
            
            for env_idx, is_success in enumerate(success_envs):
                env_dir = os.path.join(self.output_dir, f"env_{env_idx}")
                
                if not os.path.exists(env_dir):
                    continue
                    
                if not is_success:
                    # Delete failed environment directory
                    try:
                        import shutil
                        shutil.rmtree(env_dir)
                        print(f"Deleted failed environment directory: {env_dir}")
                    except Exception as e:
                        print(f"Error deleting directory {env_dir}: {e}")
                else:
                    # Move successful environment to success directory with timestamp
                    new_env_dir = os.path.join(
                        success_dir,
                        f"env_{env_idx}_{timestamp}"
                    )
                    try:
                        os.rename(env_dir, new_env_dir)
                        print(f"Moved successful environment: {env_dir} -> {new_env_dir}")
                    except Exception as e:
                        print(f"Error moving directory {env_dir}: {e}")
                        
        except Exception as e:
            print(f"Error in redir_data: {e}")
            
    @staticmethod
    def _worker_process(task_queue, result_queue, batch_size, parent_pid, stop_event) -> None:
        # The parent handles Ctrl+C and drains queued writes before sending sentinels.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Never wait for a result-queue feeder after the parent has disappeared.
        result_queue.cancel_join_thread()
        while not stop_event.is_set():
            if os.getppid() != parent_pid:
                break
            try:
                # Try to get a batch of tasks
                tasks = []
                task = task_queue.get(timeout=1)
                if task is None:
                    break
                tasks.append(task)
                stopping = False
                
                # Try to get more tasks up to batch_size
                while len(tasks) < batch_size:
                    try:
                        task = task_queue.get_nowait()
                        if task is None:
                            stopping = True
                            break
                        tasks.append(task)
                    except Empty:
                        break
                
                # Process batch of tasks
                for task in tasks:
                    success, error_msg = BatchDataLogger._save_data(task)
                    status = SaveStatus.SUCCESS if success else SaveStatus.FAILURE
                    result_queue.put(SaveResult(
                        task.file_name,
                        status,
                        error_msg,
                        task.batch_id
                    ))
                if stopping:
                    break
                    
            except Empty:
                continue
            except Exception as e:
                print(f"Error in worker process: {e}")

    def _process_results(self) -> None:
        """Process results from worker processes."""
        while self.running:
            try:
                result = self.result_queue.get(timeout=1)
                if result.status == SaveStatus.SUCCESS:
                    self._total_saved += 1
                else:
                    self._total_failed += 1
                    print(f"Failed to save {result.file_name}: {result.error_msg}")
            except Empty:
                continue
            except Exception as e:
                print(f"Error processing results: {e}")

    def save_data(
        self, 
        cam_data: Dict[str, deque[Dict[str, torch.Tensor]]], 
        robot_data: Dict[str, Union[List[str], np.ndarray]],
        count: int, 
        output_dir: str
    ) -> None:
        """Queue images and robot data for saving to disk."""
        self.start_save_time = time.time()
        try:
            self.output_dir = output_dir
            
            # Get reference camera for number of environments
            reference_camera = next(iter(cam_data.values()))
            num_envs = reference_camera[0]["rgb"].shape[0]
            
            # Process each environment
            for env in range(num_envs):
                env_dir = os.path.join(output_dir, f"env_{env}")
                
                # Queue robot data first (one save per environment)
                robot_data_dir = os.path.join(env_dir, "robot_data")
                env_robot_data = {
                    "joint_angles": robot_data["joint_angles"][env],
                    "ee_pos": robot_data["ee_pos"][env],
                    "gripper_binary_cmd": robot_data["gripper_binary_cmd"][env] if "gripper_binary_cmd" in robot_data else np.repeat(np.array([False, False])[None,:], num_envs, axis=0)
                }
                
                if count == 0:
                    env_robot_data["joint_names"] = robot_data["joint_names"]
                
                try:
                    self.task_queue.put_nowait(DataSaveTask(
                        batch_id=self.current_batch,
                        task_type="robot_data",
                        robot_data=env_robot_data,
                        robot_data_dir=robot_data_dir,
                        is_first_save=(count == 0)
                    ))
                    self._total_submitted += 1
                except Full:
                    print("Warning: Task queue is full, skipping robot data")
                    continue
                
                # Process each camera
                for camera_name, camera_data in cam_data.items():
                    camera_dir = os.path.join(env_dir, camera_name)
                    
                    # Queue RGB image
                    rgb_dir = os.path.join(camera_dir, "rgb")
                    rgb_file = os.path.join(rgb_dir, f"{count:04d}.jpg")
                    rgb_img = camera_data[0]["rgb"][env].clone().cpu().detach().numpy()
                    
                    try:
                        self.task_queue.put_nowait(DataSaveTask(
                            batch_id=self.current_batch,
                            task_type="image",
                            img=rgb_img,
                            file_name=rgb_file,
                        ))
                        self._total_submitted += 1
                    except Full:
                        print(f"Warning: Task queue is full, skipping {camera_name} RGB")
                        continue

                    # Queue depth image if available
                    if "depth" in camera_data:
                        depth_dir = os.path.join(camera_dir, "depth")
                        depth_file = os.path.join(depth_dir, f"{count:06d}.png")
                        depth_img = camera_data["depth"][env].clone().cpu().detach().numpy()
                        
                        try:
                            self.task_queue.put_nowait(DataSaveTask(
                                batch_id=self.current_batch,
                                task_type="image",
                                img=depth_img,
                                file_name=depth_file,
                                is_depth=True
                            ))
                            self._total_submitted += 1
                        except Full:
                            print(f"Warning: Task queue is full, skipping {camera_name} depth")
            
            self.current_batch += 1
            
        except Exception as e:
            print(f"Error queueing data: {e}")

            
    def get_stats(self) -> Dict:
        """Get current processing statistics."""
        elapsed_time = time.time() - self._start_time
        save_time = time.time() - self.start_save_time
        return {
            "total_saved": self._total_saved,
            "failed_to_save": self._total_failed,
            "total_successful_envs": self._total_successful_envs,
            "elapsed_time": elapsed_time,
            "save_time": save_time,
            "images_per_second": self._total_saved / elapsed_time if elapsed_time > 0 else 0,
            "current_queue_size": 0 if self._closed else self.task_queue.qsize(),
            # "current_batch": self.current_batch
        }

    def flush(self, timeout: float = 10.0) -> None:
        """Wait for submitted writes before an episode directory is moved."""
        deadline = time.monotonic() + timeout
        while self._total_saved + self._total_failed < self._total_submitted:
            if self._closed or time.monotonic() >= deadline:
                raise TimeoutError("Data writes did not finish before the flush deadline")
            time.sleep(0.01)
        if self._total_failed:
            raise RuntimeError(f"{self._total_failed} data writes failed; episode was not promoted to successes")

    def shutdown(self, timeout: float = 5.0) -> None:
        """Drain queued writes, then stop workers with bounded waits. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + max(0.0, timeout)
        # FIFO sentinels stop workers after previously queued tasks. Queue.empty()
        # cannot tell whether workers still have writes in progress.
        for _ in self.processes:
            try:
                self.task_queue.put(None, timeout=max(0.0, deadline - time.monotonic()))
            except Full:
                break
        for p in self.processes:
            p.join(timeout=max(0.0, deadline - time.monotonic()))

        self._stop_event.set()
        remaining = [p for p in self.processes if p.is_alive()]
        for p in remaining:
            p.terminate()
        deadline = time.monotonic() + 1.0
        for p in remaining:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        for p in remaining:
            if p.is_alive():
                p.kill()
        deadline = time.monotonic() + 1.0
        for p in remaining:
            p.join(timeout=max(0.0, deadline - time.monotonic()))

        self.running = False
        self.result_thread.join(timeout=1.1)
        for queue in (self.task_queue, self.result_queue):
            queue.cancel_join_thread()
            queue.close()
        # Kit can hard-exit without running multiprocessing's finalizers. Drop
        # our IPC objects now so their semaphore finalizers run before app.close().
        self.task_queue = None
        self.result_queue = None
        self._stop_event = None
        if remaining:
            print(f"[WARNING]: Forced {len(remaining)} data workers to stop; the current episode may be incomplete.")
