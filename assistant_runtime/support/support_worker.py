"""Private CPU-only MiniCheck worker. This prototype is not imported by SISU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

BOOT_STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parent))
from support_client import MAX_FRAME_BYTES, PACKAGES, UPSTREAM_SHA, build_identity, file_sha
from support_contract import validate_request


def emit(value):
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("Response exceeds frame cap")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def install_network_guard(attempts):
    def deny(*unused, **kwargs):
        attempts.append("network operation")
        raise RuntimeError("Offline support worker forbids network")
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.socket.sendto = deny
    socket.getaddrinfo = deny
    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            deny()
    sys.addaudithook(audit)


class PinnedScorer:
    def __init__(self, manifest_path):
        self.network, self.cuda = [], []
        install_network_guard(self.network)
        os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                          HF_HOME=str(manifest_path.parent / "offline_hf_home"))
        start = time.monotonic()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared_path = Path(manifest["local_path"])
        model_path = (declared_path if declared_path.is_absolute() else
                      manifest_path.parent / declared_path).resolve()
        if not model_path.is_relative_to(manifest_path.parent):
            raise ValueError("Model outside pinned local cache")
        self.artifact_stats = []
        for item in manifest["files"]:
            path = (model_path / item["name"]).resolve()
            if path.parent != model_path or file_sha(path) != item["sha256"]:
                raise ValueError("Pinned artifact changed")
            stat = path.stat()
            self.artifact_stats.append((path, stat.st_size, stat.st_mtime_ns))
        upstream = manifest_path.parent / ("upstream--" + manifest["upstream_commit"]) / "minicheck/inference.py"
        if file_sha(upstream) != UPSTREAM_SHA:
            raise ValueError("Pinned inference implementation changed")
        self.verification_s = time.monotonic() - start
        start = time.monotonic()
        versions = {name: importlib.metadata.version(name) for name in PACKAGES}
        if versions != PACKAGES:
            raise ValueError("Installed package identity changed")
        import torch
        import psutil
        def deny_cuda(*unused, **kwargs):
            self.cuda.append("CUDA operation")
            raise RuntimeError("Support worker forbids CUDA")
        torch.cuda._lazy_init = deny_cuda
        torch.cuda.is_available = lambda: False
        torch.Tensor.cuda = deny_cuda
        torch.nn.Module.cuda = deny_cuda
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.set_default_device("cpu")
        import transformers.utils.import_utils as imports
        import transformers.utils as utils
        imports.is_torchao_available = lambda *args, **kwargs: False
        utils.is_torchao_available = imports.is_torchao_available
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from minicheck_flan_cpu_run import official_core
        self.imports_s = time.monotonic() - start
        self.process = psutil.Process()
        if os.name == "nt":
            self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        start = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), local_files_only=True,
                    trust_remote_code=False, dtype=torch.float32, weights_only=True)
        model.to("cpu").eval()
        if {p.device.type for p in model.parameters()} != {"cpu"}:
            raise ValueError("Model is not entirely on CPU")
        self.core = official_core(upstream, {"torch": torch, "F": torch.nn.functional})()
        self.core.model_name = "flan-t5-large"
        self.core.model, self.core.tokenizer = model, self.tokenizer
        self.core.max_model_len, self.core.batch_size = 2048, 1
        self.torch = torch
        self.load_s = time.monotonic() - start
        self.package_versions = versions
        self.import_paths = {"torch": torch.__file__, "transformers": sys.modules["transformers"].__file__}
        self.assert_isolated()

    def assert_isolated(self):
        if self.network or self.cuda or self.torch.cuda.is_initialized():
            raise RuntimeError("CPU/offline isolation violated")
        for path, size, modified in self.artifact_stats:
            stat = path.stat()
            if stat.st_size != size or stat.st_mtime_ns != modified:
                raise RuntimeError("Local model artifact changed after initialization")

    def score(self, request, generation, identity):
        self.assert_isolated()
        started = time.monotonic()
        document = "\n\n".join(source["text"] for source in request["sources"])
        encoded = "predict: " + self.tokenizer.eos_token.join([document, request["claim"]["text"]])
        tokens = len(self.tokenizer(encoded, truncation=False, padding=False)["input_ids"])
        tokenize_s = time.monotonic() - started
        response = {"schema_version": 1, "request_id": request["request_id"],
                    "contract_sha256": request["contract_sha256"], "input_fingerprint": request["input_fingerprint"],
                    "worker_generation": generation, "model_identity_sha256": identity["model_identity_sha256"],
                    "status": "unchecked", "support_score": None, "input_tokens": tokens,
                    "elapsed_s": 0.0, "unchecked_reason": None, "truncated": False}
        forward_s, calls = 0.0, 0
        if tokens > 2048:
            response["unchecked_reason"] = "input_too_long"
        elif time.monotonic() >= request["deadline_monotonic"]:
            response["unchecked_reason"] = "deadline_exceeded"
        else:
            forward_started = time.monotonic()
            with self.torch.inference_mode():
                prediction = self.core.inference([document], [request["claim"]["text"]])
            forward_s = time.monotonic() - forward_started
            calls = 1
            score = float(prediction["max_support_prob"])
            if len(prediction["support_prob_per_chunk"]) != 1 or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid support prediction")
            if time.monotonic() >= request["deadline_monotonic"]:
                response["unchecked_reason"] = "deadline_exceeded"
            else:
                response.update(status="scored", support_score=score)
        self.assert_isolated()
        elapsed = time.monotonic() - started
        response["elapsed_s"] = elapsed
        response["usage"] = {"queue_s": 0.0, "tokenize_s": tokenize_s, "forward_s": forward_s,
                             "total_s": elapsed, "forward_calls": calls, "input_tokens": tokens,
                             "decoder_steps": calls, "generated_output_tokens": 0}
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--model-identity-sha256", required=True)
    parser.add_argument("--generation", required=True)
    args = parser.parse_args()
    identity = build_identity(args.manifest)
    if identity["contract_sha256"] != args.contract_sha256 or identity["model_identity_sha256"] != args.model_identity_sha256:
        raise ValueError("Worker identity differs from parent declaration")
    scorer = PinnedScorer(args.manifest.resolve())
    memory = scorer.process.memory_info()
    emit({"kind": "ready", "schema_version": 1, "worker_generation": args.generation,
          "worker_pid": os.getpid(),
          "contract_sha256": args.contract_sha256, "model_identity_sha256": args.model_identity_sha256,
          "device": "cpu", "dtype": "float32", "threads": 2, "network_attempts": len(scorer.network),
          "cuda_attempts": len(scorer.cuda), "verification_s": scorer.verification_s,
          "imports_s": scorer.imports_s, "load_s": scorer.load_s,
          "worker_startup_s": time.monotonic() - BOOT_STARTED, "rss_bytes": memory.rss,
          "peak_rss_bytes": getattr(memory, "peak_wset", None), "package_versions": scorer.package_versions,
          "import_paths": scorer.import_paths})
    while line := sys.stdin.buffer.readline(MAX_FRAME_BYTES + 1):
        if len(line) > MAX_FRAME_BYTES or not line.endswith(b"\n"):
            raise ValueError("Request frame exceeds cap")
        request = validate_request(json.loads(line), expected_contract_sha256=args.contract_sha256)
        emit(scorer.score(request, args.generation, identity))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # No source text, claim or raw exception content is emitted on failure.
        sys.stderr.write("Support worker terminated: " + type(error).__name__ + "\n")
        raise SystemExit(2)
