"""Bounded private-process support scorer prototype; no assistant integration.

Only the worker imports model libraries. Waiting callers consume a bounded
slot; a caller that expires in that queue never terminates another request.
An active deadline terminates the actual process and retires its pipes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

from support_contract import canonical_sha256, validate_request, validate_response

ROOT = Path(__file__).resolve().parent
MAX_FRAME_BYTES = 262144
PACKAGES = {"torch": "2.11.0+cu128", "transformers": "5.15.0", "tokenizers": "0.22.2",
            "huggingface-hub": "1.27.0", "safetensors": "0.8.0"}
REVISION = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"
UPSTREAM_SHA = "325292ec98f0e0902ada035d0e0d0395017082bd1b12a7a8e63e2d15df012d5a"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_identity(manifest_path):
    """Cheap parent declaration; actual model bytes are verified by the worker."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["repository"] != "lytang/MiniCheck-Flan-T5-Large" or manifest["revision"] != REVISION:
        raise ValueError("Unexpected model revision")
    files = [{k: item[k] for k in ("name", "sha256", "bytes")} for item in manifest["files"]]
    if len({row["name"] for row in files}) != len(files):
        raise ValueError("Duplicate model files")
    for row in files:
        if Path(row["name"]).name != row["name"] or len(row["sha256"]) != 64:
            raise ValueError("Invalid artifact name or hash")
    if not any(row["name"] == "pytorch_model.bin" and row["sha256"] ==
               "41291881e13c6235ed47149cec903bee9493e45d9d7325587a9fa2e266c526c0" for row in files):
        raise ValueError("Pinned official weights required")
    model = {"repository": manifest["repository"], "revision": manifest["revision"],
             "files": sorted(files, key=lambda x: x["name"]), "packages": PACKAGES}
    sources = [ROOT / "support_client.py", ROOT / "support_worker.py", ROOT / "support_contract.py",
               ROOT.parent / "minicheck_flan_cpu_run.py", ROOT.parent / "minicheck_cpu_run.py"]
    descriptor = {"schema_version": 1, "model": model, "source_hashes": {p.name: file_sha(p) for p in sources},
                  "upstream_sha256": UPSTREAM_SHA, "device": "cpu", "dtype": "float32", "threads": 2,
                  "interop_threads": 1, "max_input_tokens": 2048, "batch_size": 1,
                  "label_token_ids": [3, 209], "decoder_start_token_id": 0, "threshold": 0.5,
                  "prompt": "predict: <document></s><claim>", "truncation": "forbidden"}
    return {"descriptor": descriptor, "contract_sha256": canonical_sha256(descriptor),
            "model_identity_sha256": canonical_sha256(model)}


class WorkerFailure(RuntimeError):
    pass


class _WindowsJob:
    """Own launcher and every descendant before any child can execute."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes, self.wintypes = ctypes, wintypes
        self._close_lock = threading.Lock()
        class Limits(ctypes.Structure):
            _fields_ = [("ProcessTime", ctypes.c_longlong), ("JobTime", ctypes.c_longlong),
                        ("Flags", wintypes.DWORD), ("MinimumWorkingSet", ctypes.c_size_t),
                        ("MaximumWorkingSet", ctypes.c_size_t), ("ActiveProcesses", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("Priority", wintypes.DWORD), ("Scheduling", wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Limits), ("IO", IO), ("ProcessMemory", ctypes.c_size_t),
                        ("JobMemory", ctypes.c_size_t), ("PeakProcessMemory", ctypes.c_size_t), ("PeakJobMemory", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("IsProcessInJob", [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)], wintypes.BOOL),
            ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            ("WaitForSingleObject", [wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ):
            function = getattr(kernel, name)
            function.argtypes, function.restype = args, result
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        self.worker_handle = None
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign_and_resume(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        resume = self.ctypes.WinDLL("ntdll").NtResumeProcess
        resume.argtypes, resume.restype = [self.wintypes.HANDLE], self.wintypes.LONG
        if resume(int(process._handle)) != 0:
            raise WorkerFailure("Cannot resume owned suspended worker")

    def bind_worker(self, pid):
        # Retain a synchronization handle to the actual interpreter, even if
        # the configured venv executable is a small launcher process.
        handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)
        if not handle:
            raise WorkerFailure("Cannot verify actual worker process")
        member = self.wintypes.BOOL()
        if not self.kernel.IsProcessInJob(handle, self.handle, self.ctypes.byref(member)) or not member.value:
            self.kernel.CloseHandle(handle)
            raise WorkerFailure("Actual worker is outside owned job")
        self.worker_handle = handle

    def close(self):
        with self._close_lock:
            if self.handle:
                self.kernel.TerminateJobObject(self.handle, 1)
                self.kernel.CloseHandle(self.handle)
                self.handle = None
            if self.worker_handle:
                result = self.kernel.WaitForSingleObject(self.worker_handle, 1000)
                self.kernel.CloseHandle(self.worker_handle)
                self.worker_handle = None
                if result != 0:
                    raise WorkerFailure("Actual worker did not exit after job termination")


class SupportClient:
    def __init__(self, python_path, manifest_path, *, max_pending=2, startup_timeout_s=120,
                 worker_script=None, identity=None):
        if type(max_pending) is not int or not 0 <= max_pending <= 8:
            raise ValueError("max_pending must be between 0 and 8")
        if not math.isfinite(startup_timeout_s) or not 0 < startup_timeout_s <= 180:
            raise ValueError("Invalid startup timeout")
        self.python_path = str(Path(python_path).resolve())
        self.manifest_path = Path(manifest_path).resolve()
        self.worker_script = Path(worker_script or ROOT / "support_worker.py").resolve()
        self.identity = identity or build_identity(self.manifest_path)
        self.startup_timeout_s = float(startup_timeout_s)
        self._slots = threading.BoundedSemaphore(max_pending + 1)
        self._operation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = None
        self._closed = False
        self.events = []
        self.readiness = None

    def _event(self, **values):
        with self._state_lock:
            self.events.append(values)
            del self.events[:-128]

    @property
    def pid(self):
        with self._state_lock:
            return self._state.get("worker_pid", self._state["process"].pid) if self._state else None

    def _launch(self):
        generation = uuid.uuid4().hex
        args = [self.python_path, "-I", "-B", str(self.worker_script), "--manifest", str(self.manifest_path),
                "--contract-sha256", self.identity["contract_sha256"],
                "--model-identity-sha256", self.identity["model_identity_sha256"], "--generation", generation]
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        with self._state_lock:
            if self._closed:
                raise WorkerFailure("Client closed")
            launch_started = time.monotonic()
            job = _WindowsJob() if os.name == "nt" else None
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | (0x4 if job else 0)  # CREATE_SUSPENDED
            process = None
            try:
                process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           env=env, shell=False, creationflags=flags, start_new_session=os.name != "nt")
                if job:
                    job.assign_and_resume(process)
            except Exception:
                if process:
                    process.kill()
                    process.wait(timeout=1)
                if job:
                    job.close()
                raise
            state = {"process": process, "generation": generation, "responses": queue.Queue(maxsize=2),
                     "stderr_tail": bytearray(), "started": launch_started, "job": job}
            self._state = state

        def reader():
            try:
                while True:
                    line = process.stdout.readline(MAX_FRAME_BYTES + 1)
                    if not line:
                        raise WorkerFailure("Worker pipe closed")
                    if len(line) > MAX_FRAME_BYTES or not line.endswith(b"\n"):
                        raise WorkerFailure("Worker response exceeds frame limit")
                    item = json.loads(line)
                    state["responses"].put_nowait(item)
            except Exception as error:
                try:
                    state["responses"].put_nowait(WorkerFailure(type(error).__name__))
                except queue.Full:
                    pass

        def stderr_reader():
            while chunk := process.stderr.read(1024):
                state["stderr_tail"].extend(chunk)
                del state["stderr_tail"][:-8192]

        threading.Thread(target=reader, daemon=True, name="support-response-" + generation).start()
        threading.Thread(target=stderr_reader, daemon=True, name="support-stderr-" + generation).start()
        return state

    def _stop(self, state):
        if state is None:
            return
        with self._state_lock:
            if self._state is state:
                self._state = None
        process = state["process"]
        if state["job"]:
            state["job"].close()
        elif os.name != "nt" and process.poll() is None:
            import signal
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def _receive(self, state, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Support deadline")
        try:
            message = state["responses"].get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError("Support deadline") from None
        if isinstance(message, Exception):
            raise message
        return message

    def _ensure_started(self, deadline):
        with self._state_lock:
            state = self._state
            if self._closed:
                raise WorkerFailure("Client closed")
        if state is not None and state["process"].poll() is None:
            return state
        self._stop(state)
        state = self._launch()
        try:
            ready = self._receive(state, min(deadline, time.monotonic() + self.startup_timeout_s))
            if (ready.get("kind") != "ready" or ready.get("worker_generation") != state["generation"] or
                    ready.get("contract_sha256") != self.identity["contract_sha256"] or
                    ready.get("model_identity_sha256") != self.identity["model_identity_sha256"] or
                    ready.get("network_attempts") != 0 or ready.get("cuda_attempts") != 0 or
                    ready.get("device") != "cpu" or type(ready.get("worker_pid")) is not int or ready["worker_pid"] <= 0):
                raise WorkerFailure("Invalid worker readiness")
            if state["job"]:
                state["job"].bind_worker(ready["worker_pid"])
            elif ready["worker_pid"] != state["process"].pid:
                raise WorkerFailure("Unexpected worker interpreter process")
            state["worker_pid"] = ready["worker_pid"]
            ready["parent_startup_wall_s"] = time.monotonic() - state["started"]
            self.readiness = ready
            self._event(**{**ready, "kind": "startup", "launcher_pid": state["process"].pid})
            return state
        except Exception:
            self._stop(state)
            raise

    def start(self, timeout_s=None):
        limit = self.startup_timeout_s if timeout_s is None else timeout_s
        if isinstance(limit, bool) or not isinstance(limit, (float, int)) or not math.isfinite(limit) or not 0 < limit <= 180:
            raise ValueError("Invalid startup timeout")
        deadline = time.monotonic() + limit
        if not self._operation_lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise TimeoutError("Worker startup queued past deadline")
        try:
            self._ensure_started(deadline)
            return dict(self.readiness)
        finally:
            self._operation_lock.release()

    def _unchecked(self, request, reason, started, *, state=None, dispatched=False, queue_s=0):
        elapsed = max(0.0, time.monotonic() - started)
        response = {"schema_version": 1, "request_id": request["request_id"],
                    "contract_sha256": request["contract_sha256"], "input_fingerprint": request["input_fingerprint"],
                    "worker_generation": state["generation"] if state else "not_started",
                    "model_identity_sha256": self.identity["model_identity_sha256"],
                    "status": "unchecked", "support_score": None, "input_tokens": None,
                    "elapsed_s": elapsed, "unchecked_reason": reason, "truncated": False,
                    "usage": {"queue_s": queue_s, "tokenize_s": None if dispatched else 0.0,
                              "forward_s": None if dispatched else 0.0, "total_s": elapsed,
                              "forward_calls": None if dispatched else 0, "input_tokens": None,
                              "decoder_steps": None if dispatched else 0, "generated_output_tokens": 0}}
        self._event(kind="request", request_id=request["request_id"], input_fingerprint=request["input_fingerprint"],
                    status="unchecked", reason=reason, dispatched=dispatched, completion_unknown=dispatched,
                    elapsed_s=elapsed, pid=state["process"].pid if state else None)
        return response

    def score(self, request):
        started = time.monotonic()
        request = validate_request(request, expected_contract_sha256=self.identity["contract_sha256"])
        frame = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(frame) > MAX_FRAME_BYTES:
            raise ValueError("Request exceeds frame limit")
        deadline = request["deadline_monotonic"]
        if not self._slots.acquire(blocking=False):
            return self._unchecked(request, "queue_full", started)
        acquired = False
        state = None
        dispatched = False
        try:
            acquired = self._operation_lock.acquire(timeout=max(0, deadline - time.monotonic()))
            waited = time.monotonic() - started
            if not acquired or time.monotonic() >= deadline:
                return self._unchecked(request, "deadline_exceeded", started, queue_s=waited)
            state = self._ensure_started(deadline)
            if time.monotonic() >= deadline:
                raise TimeoutError("Support deadline")
            completion = queue.Queue(maxsize=1)

            def write_request():
                try:
                    state["process"].stdin.write(frame)
                    state["process"].stdin.flush()
                    completion.put(None)
                except Exception as error:
                    completion.put(error)

            dispatched = True
            threading.Thread(target=write_request, daemon=True, name="support-write").start()
            try:
                error = completion.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty:
                raise TimeoutError("Support write deadline") from None
            if error is not None:
                raise WorkerFailure("Support pipe write failed")
            response = self._receive(state, deadline)
            response = validate_response(response, request, expected_worker_generation=state["generation"],
                                         expected_model_identity_sha256=self.identity["model_identity_sha256"],
                                         now=time.monotonic())
            # Queue time belongs to the parent, so retain wire usage and report
            # measured parent work separately in events rather than alter hashes.
            self._event(kind="request", request_id=request["request_id"], input_fingerprint=request["input_fingerprint"],
                        status=response["status"], dispatched=True, completion_unknown=False,
                        queue_s=waited, elapsed_s=time.monotonic() - started, pid=state["process"].pid,
                        usage=response["usage"])
            return response
        except TimeoutError:
            self._stop(state)
            return self._unchecked(request, "deadline_exceeded", started, state=state, dispatched=dispatched)
        except (WorkerFailure, OSError, ValueError, KeyError, TypeError):
            self._stop(state)
            return self._unchecked(request, "worker_failed", started, state=state, dispatched=dispatched)
        finally:
            if acquired:
                self._operation_lock.release()
            self._slots.release()

    def score_batch(self, requests):
        values = list(requests)
        if not 1 <= len(values) <= 4:
            raise ValueError("A conservative batch has between 1 and 4 sequential pairs")
        return [self.score(request) for request in values]

    def close(self):
        with self._state_lock:
            self._closed = True
            state = self._state
        self._stop(state)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
