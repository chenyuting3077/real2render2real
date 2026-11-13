"""Supervise one simulation process group without importing Isaac Sim or CUDA."""

import os
import signal
import subprocess
import time


def _signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
        return True
    except ProcessLookupError:
        return False


def _group_exists(pgid):
    return _signal_group(pgid, 0)


def run_supervised(command, shutdown_timeout=10.0, terminate_timeout=2.0):
    """Forward cancellation, bound shutdown, and reap leftover group members.

    The supervisor stays outside Isaac Sim, so native CUDA/rendering deadlocks
    cannot prevent it from escalating to SIGKILL. Only this run's group is used.
    """
    received = []

    def request_stop(signum, _frame):
        received.append(signum)

    previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    process = None
    try:
        process = subprocess.Popen(command, start_new_session=True)
        stop_started = None
        terminate_started = None
        while process.poll() is None:
            now = time.monotonic()
            if received and stop_started is None:
                stop_started = now
                print("[INFO]: Stopping simulation; waiting for queued writes and cleanup...", flush=True)
                _signal_group(process.pid, received[0])
            if len(received) > 1:
                print("[WARNING]: Repeated cancellation: forcing this simulation to stop.", flush=True)
                _signal_group(process.pid, signal.SIGKILL)
                break
            if stop_started is not None and now - stop_started >= shutdown_timeout:
                if terminate_started is None:
                    terminate_started = now
                    print("[WARNING]: Simulation shutdown timed out; terminating its process group.", flush=True)
                    _signal_group(process.pid, signal.SIGTERM)
                elif now - terminate_started >= terminate_timeout:
                    print("[WARNING]: Simulation is still stuck; forcing its process group to exit.", flush=True)
                    _signal_group(process.pid, signal.SIGKILL)
                    break
            time.sleep(0.1)

        returncode = process.wait(timeout=max(1.0, terminate_timeout))
        return 128 + received[0] if received else (128 - returncode if returncode < 0 else returncode)
    finally:
        try:
            if process is not None:
                # Also run after normal completion or a crash: Isaac's hard exit
                # may bypass Python's daemon-process and atexit cleanup.
                if _signal_group(process.pid, signal.SIGTERM):
                    deadline = time.monotonic() + terminate_timeout
                    while _group_exists(process.pid) and time.monotonic() < deadline:
                        time.sleep(0.05)
                    _signal_group(process.pid, signal.SIGKILL)
                if process.poll() is None:
                    process.wait(timeout=max(1.0, terminate_timeout))
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
