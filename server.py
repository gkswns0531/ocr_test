"""vLLM server lifecycle management."""

import os
import signal
import subprocess
import sys
import time

import requests

from config import ModelConfig


def _kill_gpu_orphans() -> None:
    """Find and kill any orphaned processes holding GPU memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return
        for line in result.stdout.strip().splitlines():
            pid_str = line.strip()
            if pid_str.isdigit():
                pid = int(pid_str)
                # Don't kill ourselves
                if pid == os.getpid():
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"[Server] Killed orphaned GPU process {pid}")
                except (ProcessLookupError, PermissionError):
                    pass
    except Exception:
        pass


def _wait_gpu_free(timeout: int = 30) -> None:
    """Wait until GPU memory is mostly freed (< 1 GiB used)."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                used_mib = int(result.stdout.strip())
                if used_mib < 1024:
                    return
        except Exception:
            pass
        time.sleep(1)
    print(f"[Server] Warning: GPU memory not fully freed after {timeout}s")


class VLLMServer:
    """Manages a vLLM OpenAI-compatible API server process."""

    def __init__(self, model_config: ModelConfig, port: int | None = None) -> None:
        self.model_config = model_config
        self.port = port or model_config.port
        self.proc: subprocess.Popen | None = None

    def build_command(self) -> list[str]:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_config.model_id,
            "--port", str(self.port),
        ]
        cmd.extend(self.model_config.vllm_args)
        return cmd

    def start(self, timeout: int = 300) -> None:
        """Start vLLM server and wait until healthy."""
        # Ensure no orphaned GPU processes from previous runs
        _kill_gpu_orphans()
        _wait_gpu_free(timeout=15)

        cmd = self.build_command()
        print(f"[Server] Starting vLLM: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group for clean cleanup
        )
        self._wait_for_ready(timeout)

    def _wait_for_ready(self, timeout: int) -> None:
        """Poll /v1/models until the server is responsive."""
        url = f"http://localhost:{self.port}/v1/models"
        start = time.time()
        while time.time() - start < timeout:
            if self.proc and self.proc.poll() is not None:
                # Read remaining stdout for error diagnosis
                remaining = ""
                if self.proc.stdout:
                    remaining = self.proc.stdout.read().decode(errors="replace")
                if remaining:
                    print(f"[Server] vLLM stdout:\n{remaining[-3000:]}")
                raise RuntimeError(
                    f"vLLM server exited with code {self.proc.returncode}"
                )
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    print(f"[Server] vLLM ready after {time.time() - start:.1f}s")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(2)
        self.stop()
        raise TimeoutError(f"vLLM server not ready after {timeout}s")

    def stop(self) -> None:
        """Terminate the server and ALL child processes (entire process group)."""
        if self.proc:
            print("[Server] Stopping vLLM server...")
            try:
                # Kill the entire process group (catches EngineCore children)
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(3)
                # Force kill if still alive
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            except (ProcessLookupError, PermissionError):
                # Process already dead
                pass
            except Exception:
                # Fallback: kill individual process
                try:
                    self.proc.kill()
                except Exception:
                    pass
            try:
                self.proc.wait(timeout=10)
            except Exception:
                pass
            print("[Server] vLLM server stopped.")
        self.proc = None

        # Clean up any orphaned GPU processes and wait for GPU memory release
        _kill_gpu_orphans()
        _wait_gpu_free(timeout=15)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
