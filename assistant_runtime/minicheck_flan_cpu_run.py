"""One pinned Flan-T5 comparison, using reviewed official inference methods.

The frozen V2 claims/sources/labels remain byte-identical. Only model architecture
and its official input/scoring contract differ. No automatic device placement.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time

from minicheck_cpu_run import sha, summary

UPSTREAM_SHA256 = "325292ec98f0e0902ada035d0e0d0395017082bd1b12a7a8e63e2d15df012d5a"
METHODS = {"inference", "batch_tokenize", "chunks"}


def official_core(path, namespace):
    """Execute only three inspected, hash-pinned upstream methods; no imports/init."""
    if sha(path) != UPSTREAM_SHA256: raise ValueError("Reviewed upstream implementation changed")
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Inferencer")
    selected = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    if {node.name for node in selected} != METHODS: raise ValueError("Missing reviewed upstream methods")
    subset = ast.Module(body=[ast.ClassDef(name="PinnedCore", bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[])
    values = dict(namespace)
    exec(compile(ast.fix_missing_locations(subset), str(path), "exec"), values)
    return values["PinnedCore"]


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream: json.dump(value, stream, ensure_ascii=False, indent=2); stream.write("\n")
    Path(path).with_suffix(".sha256").write_text(sha(path) + "\n", encoding="ascii")


def declare(source, manifest_path, protocol_path):
    source, manifest_path, protocol_path = Path(source).resolve(), Path(manifest_path).resolve(), Path(protocol_path).resolve()
    data = json.loads(source.read_text(encoding="utf-8")); manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha(source) != source.with_suffix(".sha256").read_text().strip() or data["split"] != "development_only": raise ValueError("Frozen V2 source declaration changed")
    if manifest["repository"] != "lytang/MiniCheck-Flan-T5-Large" or manifest["revision"] != "96eafd01cee2d16cf81aaa2fb226b14f422a37b3":
        raise ValueError("Require inspected official Flan-T5 revision")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = {"schema_version": 1, "kind": "same_claims_official_flan_t5_cpu_comparison", "split": "development_only",
        "source_declaration": str(source), "source_declaration_sha256": sha(source), "download_manifest": str(manifest_path),
        "download_manifest_sha256": sha(manifest_path), "upstream_sha256": UPSTREAM_SHA256,
        "upstream_methods": sorted(METHODS), "model": manifest["repository"], "revision": manifest["revision"],
        "threshold": .5, "maximum_input_tokens": 2048, "truncation": "forbidden; overlong pairs unscored",
        "prompt_contract": "predict: <document></s><claim>", "decoder_start_token_id": 0, "label_token_ids": [3, 209],
        "support_score": "Official softmax over first decoder step logits [3,209], take index 1",
        "device": "cpu", "dtype": "float32", "threads": 2, "original_cases_labels_sources_unchanged": True,
        "fit_or_threshold_tuning": False, "heldout_used": False, "runtime_integration": False,
        "runner_sha256": sha(__file__), "helper_sha256": sha(Path(__file__).with_name("minicheck_cpu_run.py"))}
    write(protocol_path, protocol); print(json.dumps({"protocol_sha256": sha(protocol_path), "cases": len(data["cases"])}))


def run(protocol_path, output):
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError("Use a new immutable result directory")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha(protocol_path) != protocol_path.with_suffix(".sha256").read_text().strip(): raise ValueError("Comparison protocol changed")
    if protocol["runner_sha256"] != sha(__file__) or protocol["helper_sha256"] != sha(Path(__file__).with_name("minicheck_cpu_run.py")): raise ValueError("Frozen CPU experiment code changed")
    source, manifest_path = Path(protocol["source_declaration"]), Path(protocol["download_manifest"])
    if sha(source) != protocol["source_declaration_sha256"] or sha(manifest_path) != protocol["download_manifest_sha256"]: raise ValueError("Frozen inputs changed")
    data, manifest = json.loads(source.read_text(encoding="utf-8")), json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = Path(manifest["local_path"])
    for item in manifest["files"]:
        if sha(model_path / item["name"]) != item["sha256"]: raise ValueError("Official model artifact changed")
    for item in data["sources"].values():
        if sha(item["source_path"]) != item["source_sha256"]: raise ValueError("Original development document changed")
    upstream = manifest_path.parent / ("upstream--" + manifest["upstream_commit"]) / "minicheck/inference.py"
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(manifest_path.parent / "offline_hf_home"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    network, cuda = [], []
    def deny_network(self, address): network.append(repr(address)); raise RuntimeError("Offline critic forbids network")
    socket.socket.connect = deny_network; socket.socket.connect_ex = deny_network
    import torch
    import psutil
    def deny_cuda(*args, **kwargs): cuda.append("CUDA request"); raise RuntimeError("CPU critic forbids CUDA")
    torch.cuda._lazy_init = deny_cuda; torch.cuda.is_available = lambda: False
    torch.Tensor.cuda = deny_cuda; torch.nn.Module.cuda = deny_cuda
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.set_default_device("cpu")
    import transformers.utils.import_utils as imports
    import transformers.utils as utils
    imports.is_torchao_available = lambda *args, **kwargs: False; utils.is_torchao_available = imports.is_torchao_available
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    process = psutil.Process()
    if os.name == "nt": process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False, dtype=torch.float32, weights_only=True)
    model.to("cpu").eval()
    if {p.device.type for p in model.parameters()} != {"cpu"}: raise ValueError("Model escaped CPU")
    core = official_core(upstream, {"torch": torch, "F": torch.nn.functional})()
    core.model_name = "flan-t5-large"; core.model = model; core.tokenizer = tokenizer
    core.max_model_len = 2048; core.batch_size = 1
    load_s = time.perf_counter() - started
    output.mkdir(parents=True)
    rows = []
    for case in data["cases"]:
        document = "\n\n".join(data["sources"][key]["text"] for key in case["source_ids"])
        encoded = "predict: " + tokenizer.eos_token.join([document, case["claim"]])
        tokens = len(tokenizer(encoded, truncation=False, padding=False)["input_ids"])
        row = {"id": case["id"], "stratum": case["stratum"], "claim": case["claim"], "source_ids": case["source_ids"],
            "reference_label": case["reference_label"], "expected_binary_support": case["expected_binary_support"],
            "input_tokens": tokens, "input_text_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "truncated": False}
        if tokens > 2048:
            row.update(status="unscored_input_exceeds_context", predicted_support=None, support_score=None, elapsed_s=None)
        else:
            start = time.perf_counter()
            with torch.inference_mode(): score = core.inference([document], [case["claim"]])
            if len(score["support_prob_per_chunk"]) != 1: raise ValueError("Unexpected source chunk aggregation")
            probability = float(score["max_support_prob"])
            row.update(status="scored", predicted_support=int(probability > .5), support_score=probability, elapsed_s=time.perf_counter() - start)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("id", "status", "support_score", "elapsed_s")}), flush=True)
    memory = process.memory_info()
    result = {"schema_version": 1, "kind": "isolated_cpu_minicheck_flan_comparison", "status": "complete", "protocol_sha256": sha(protocol_path),
        "source_declaration_sha256": sha(source), "model_revision": manifest["revision"], "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": "cpu", "dtype": "float32", "threads": 2, "network_attempts": network, "cuda_attempts": cuda, "cuda_initialized": torch.cuda.is_initialized(),
        "load_s": load_s, "inference_total_s": sum(row["elapsed_s"] or 0 for row in rows), "process_rss_bytes": memory.rss,
        "process_peak_rss_bytes": getattr(memory, "peak_wset", None), "upstream_methods_executed": sorted(METHODS), "upstream_source_sha256": sha(upstream),
        "label_token_ids": [3, 209], "decoded_label_tokens": tokenizer.convert_ids_to_tokens([3, 209]),
        "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")},
        "rows": rows, "summary": summary(rows), "limits": data["limits"], "deployment_or_learning_change": False,
        "score_is_calibrated_truth_probability": False, "model_device_map_auto_used": False}
    if network or cuda or result["cuda_initialized"]: raise RuntimeError("CPU/offline isolation was violated")
    write(output / "results.json", result)
    print(json.dumps({k: result[k] for k in ("summary", "load_s", "inference_total_s", "process_peak_rss_bytes")}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("declare"); prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True); prepare.add_argument("--protocol", type=Path, required=True)
    execute = commands.add_parser("run"); execute.add_argument("--protocol", type=Path, required=True); execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "declare": declare(args.source, args.manifest, args.protocol)
    else: run(args.protocol, args.output)
