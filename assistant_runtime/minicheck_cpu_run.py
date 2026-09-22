"""Offline, CPU-only sentence-support diagnostic with pinned official weights.

Uses the official encoder input/softmax contract on short whole source excerpts.
It does not import or run the upstream device_map='auto' wrapper.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024): hasher.update(block)
    return hasher.hexdigest()


def input_text(tokenizer, document, claim):
    if not document.strip() or not claim.strip(): raise ValueError("Nonempty document and claim required")
    return document + tokenizer.eos_token + claim


def summary(rows):
    output = {}
    for stratum in sorted({row["stratum"] for row in rows}):
        selected = [r for r in rows if r["stratum"] == stratum]
        scored = [r for r in selected if r["status"] == "scored"]
        output[stratum] = {"cases": len(selected), "scored": len(scored), "unscored": len(selected) - len(scored),
            "correct_binary_support": sum(r["predicted_support"] == r["expected_binary_support"] for r in scored),
            "by_reference_label": {label: {"cases": sum(r["reference_label"] == label for r in scored),
                "predicted_supported": sum(r["reference_label"] == label and r["predicted_support"] == 1 for r in scored)}
                for label in ("supported", "contradicted", "insufficient_evidence")}}
    return output


def run(declaration, download_manifest, output):
    declaration, download_manifest, output = Path(declaration).resolve(), Path(download_manifest).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError("Keep each diagnostic result immutable")
    data = json.loads(declaration.read_text(encoding="utf-8"))
    if sha(declaration) != declaration.with_suffix(".sha256").read_text(encoding="ascii").strip(): raise ValueError("Diagnostic declaration changed")
    if data["split"] != "development_only" or data["declaration"]["heldout_material"] or data["declaration"]["device"] != "cpu":
        raise ValueError("Only the declared CPU development diagnostic is accepted")
    manifest = json.loads(download_manifest.read_text(encoding="utf-8"))
    if manifest["repository"] != data["declaration"]["model"] or manifest["revision"] != data["declaration"]["model_revision"]:
        raise ValueError("Model revision differs from predeclared diagnostic")
    model_path = Path(manifest["local_path"]).resolve()
    if not model_path.is_relative_to(download_manifest.parent): raise ValueError("Model must stay in the isolated cache")
    for file in manifest["files"]:
        path = model_path / file["name"]
        if sha(path) != file["sha256"]: raise ValueError("Pinned official model artifact changed")
    for source in data["sources"].values():
        if sha(source["source_path"]) != source["source_sha256"]: raise ValueError("Declared source file changed")
    # Process-local isolation; no shared environment or dependency writes.
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(download_manifest.parent / "offline_hf_home"), OMP_NUM_THREADS=str(data["declaration"]["threads"]), MKL_NUM_THREADS=str(data["declaration"]["threads"]))
    attempts = []
    def deny_socket(self, address):
        attempts.append(repr(address)); raise RuntimeError("All network connections are forbidden in the offline CPU diagnostic")
    socket.socket.connect = deny_socket
    socket.socket.connect_ex = deny_socket
    import torch
    import psutil
    cuda_attempts = []
    def deny_cuda(*args, **kwargs):
        cuda_attempts.append("cuda initialization or transfer"); raise RuntimeError("CUDA forbidden for the CPU diagnostic")
    torch.cuda._lazy_init = deny_cuda
    torch.cuda.is_available = lambda: False  # Do not probe accelerator drivers.
    torch.Tensor.cuda = deny_cuda
    torch.nn.Module.cuda = deny_cuda
    torch.set_num_threads(data["declaration"]["threads"])
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    # This isolated full-precision encoder does not use the shared environment's
    # optional torchao quantization integration. Its import otherwise probes a
    # CUDA device even with CUDA_VISIBLE_DEVICES empty on this Windows build.
    import transformers.utils.import_utils as transformer_imports
    import transformers.utils as transformer_utils
    transformer_imports.is_torchao_available = lambda *args, **kwargs: False
    transformer_utils.is_torchao_available = transformer_imports.is_torchao_available
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    process = psutil.Process()
    try:
        if os.name == "nt": process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (psutil.Error, OSError): pass
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, weights_only=True)
    model.to("cpu").eval()
    if {p.device.type for p in model.parameters()} != {"cpu"}: raise RuntimeError("A model parameter is not on CPU")
    if model.config.id2label != {0: "0", 1: "1"}: raise ValueError("Official support label mapping changed")
    loaded = time.perf_counter() - started
    output.mkdir(parents=True)
    rows = []
    for case in data["cases"]:
        document = "\n\n".join(data["sources"][key]["text"] for key in case["source_ids"])
        text = input_text(tokenizer, document, case["claim"])
        tokenized = tokenizer(text, truncation=False, padding=False, return_tensors="pt")
        tokens = int(tokenized["input_ids"].shape[1])
        row = {"id": case["id"], "stratum": case["stratum"], "claim": case["claim"], "source_ids": case["source_ids"],
            "reference_label": case["reference_label"], "expected_binary_support": case["expected_binary_support"],
            "input_tokens": tokens, "input_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "truncated": False}
        if tokens > data["declaration"]["maximum_input_tokens"]:
            row.update(status="unscored_input_exceeds_context", predicted_support=None, support_score=None, elapsed_s=None)
        else:
            begin = time.perf_counter()
            with torch.inference_mode():
                logits = model(**tokenized).logits
                probability = float(torch.softmax(logits, dim=1)[0, 1])
            row.update(status="scored", predicted_support=int(probability > data["declaration"]["binary_threshold"]),
                support_score=probability, elapsed_s=time.perf_counter() - begin)
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("id", "status", "input_tokens", "support_score", "elapsed_s")}), flush=True)
    memory = process.memory_info()
    result = {"schema_version": 1, "kind": "isolated_cpu_minicheck_diagnostic", "status": "complete",
        "declaration_sha256": sha(declaration), "download_manifest_sha256": sha(download_manifest), "runner_sha256": sha(__file__),
        "model_revision": manifest["revision"], "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(), "network_attempts": attempts,
        "cuda_attempts": cuda_attempts, "cuda_initialized": torch.cuda.is_initialized(), "load_s": loaded,
        "inference_total_s": sum(row["elapsed_s"] or 0 for row in rows), "process_rss_bytes": memory.rss,
        "process_peak_rss_bytes": getattr(memory, "peak_wset", None),
        "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")},
        "disabled_optional_integrations": ["torchao (process-local availability override; full-precision CPU encoder only)"],
        "rows": rows, "summary": summary(rows), "limits": data["limits"],
        "deployment_or_learning_change": False, "score_is_calibrated_truth_probability": False}
    if attempts or cuda_attempts or result["cuda_initialized"]: raise RuntimeError("Offline/CPU isolation was violated; no completed result")
    path = output / "results.json"; path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".sha256").write_text(sha(path) + "\n", encoding="ascii")
    print(json.dumps({key: result[key] for key in ("summary", "load_s", "inference_total_s", "process_peak_rss_bytes", "network_attempts", "cuda_attempts")}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); run(args.declaration, args.download_manifest, args.output)
