#!/usr/bin/env python3
"""Freeze inputs/checkpoints and split complete model/epoch bundles by server.

This planner uses only the Python standard library. It never loads a checkpoint
or starts inference. Epoch ranges use Python's half-open start:stop[:step]
notation: ``0,5,10:21:2`` requests 0, 5, 10, 12, ..., 20.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True


SCHEMA = "root-epoch-scan-1"
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
CHECKPOINT_NAME = re.compile(r"ckpt_(\d+)_1\.pth\.tar\Z")
CONFIG_KEYS = {
    "repo", "input_root", "energies", "flavors", "methods", "tbeta", "td",
    "nstart", "nend", "models",
}
MODEL_KEYS = {
    "name", "run_id", "checkpoint_dir", "input_dim", "output_dim",
    "writer_args", "input_root",
}
# Physics thresholds/methods and execution options belong to the plan/runner.
RESERVED_WRITER_ARGS = {
    "--multi-h5", "--device", "--pid-threads", "--root-chunk-events",
    "--pipeline-depth", "--replace", "--no-resume", "--skip-existing",
    "--truth-clustering", "--tbeta", "--td", "--beta-d-scan", "--scan",
    "--scan-tbeta", "--scan-td", "--scan-workers", "--pandora",
}


def absolute_path(value, base=None):
    """Keep shared /data paths intact rather than resolving mount symlinks."""
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n\0"):
        raise ValueError(f"Expected a nonempty single-line path, got {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base or os.getcwd()) / path
    return Path(os.path.abspath(path))


def require_int(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def require_name(value, label):
    if not isinstance(value, str) or not SAFE_NAME.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, digits, '.', '_' or '-'")
    return value


def unique_sequence(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    if len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
        raise ValueError(f"{label} contains duplicates")
    return value


def read_config(path, repo_override=None):
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    required = CONFIG_KEYS - {"flavors"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    config.setdefault("flavors", ["uu", "dd", "ss"])
    config["repo"] = str(absolute_path(repo_override or config["repo"], path.parent))
    config["input_root"] = str(absolute_path(config["input_root"], config["repo"]))
    for energy in unique_sequence(config["energies"], "energies"):
        require_int(energy, "energy")
    for flavor in unique_sequence(config["flavors"], "flavors"):
        require_name(flavor, "flavor")
    methods = unique_sequence(config["methods"], "methods")
    if set(methods) - {"reco", "truth_clustering"}:
        raise ValueError("methods must contain only reco and/or truth_clustering")
    for key in ("tbeta", "td"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if not 0 < config["tbeta"] < 1 or config["td"] <= 0:
        raise ValueError("Require 0 < tbeta < 1 and td > 0")
    require_int(config["nstart"], "nstart", 0)
    require_int(config["nend"], "nend", -1)
    if config["nend"] != -1:
        require_int(config["nend"], "nend")
    seen = set()
    for model in unique_sequence(config["models"], "models"):
        if not isinstance(model, dict):
            raise ValueError("Each model must be a JSON object")
        unknown = set(model) - MODEL_KEYS
        missing = MODEL_KEYS - {"input_root"} - set(model)
        if unknown or missing:
            raise ValueError(f"Model keys: unknown={sorted(unknown)}, missing={sorted(missing)}")
        name = require_name(model["name"], "model name")
        require_name(model["run_id"], f"{name}.run_id")
        if name in seen:
            raise ValueError(f"Duplicate model name: {name}; give each run a distinct name")
        seen.add(name)
        for key in ("input_dim", "output_dim"):
            require_int(model[key], f"{name}.{key}")
        if model["output_dim"] < 2:
            raise ValueError(f"{name}.output_dim must include beta and at least one coordinate")
        model["checkpoint_dir"] = str(absolute_path(model["checkpoint_dir"], config["repo"]))
        if "input_root" in model:
            model["input_root"] = str(absolute_path(model["input_root"], config["repo"]))
        args = model["writer_args"]
        if not isinstance(args, list) or not all(isinstance(v, str) and v and
                not any(c in v for c in "\r\n\0") for v in args):
            raise ValueError(f"{name}.writer_args must be a list of nonempty strings")
        reserved = {v.split("=", 1)[0] for v in args} & RESERVED_WRITER_ARGS
        if reserved:
            raise ValueError(f"{name}.writer_args overrides planner/runner settings: {sorted(reserved)}")
    return config


def parse_servers(value):
    servers = []
    for token in value.split(","):
        parts = token.strip().split(":")
        if len(parts) > 2 or not parts[0]:
            raise ValueError("--servers uses name[:weight],name[:weight]")
        name = require_name(parts[0], "server name")
        try:
            weight = float(parts[1]) if len(parts) == 2 else 1.0
        except ValueError as error:
            raise ValueError(f"Invalid server weight: {token}") from error
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Server weight must be positive and finite: {token}")
        if any(server["name"] == name for server in servers):
            raise ValueError(f"Duplicate server name: {name}")
        servers.append({"name": name, "weight": weight})
    return servers


def parse_epochs(value):
    if value == "all":
        return None
    epochs = set()
    for token in value.split(","):
        parts = token.strip().split(":")
        if not 1 <= len(parts) <= 3 or any(not re.fullmatch(r"\d+", p) for p in parts):
            raise ValueError("--epochs uses all, integers, or start:stop[:step] (stop excluded)")
        numbers = [int(p) for p in parts]
        if len(numbers) == 1:
            epochs.add(numbers[0])
        else:
            start, stop = numbers[:2]
            step = numbers[2] if len(numbers) == 3 else 1
            if stop <= start or step <= 0:
                raise ValueError(f"Empty or invalid epoch range: {token}")
            epochs.update(range(start, stop, step))
    if not epochs:
        raise ValueError("No epochs requested")
    return epochs


def snapshot_checkpoint(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    identity = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)
    if identity(before) != identity(after):
        raise ValueError(f"Checkpoint changed while being read; retry once training has saved it: {path}")
    if after.st_size == 0:
        raise ValueError(f"Empty checkpoint: {path}")
    return digest.hexdigest(), {"size": after.st_size, "mtime_ns": after.st_mtime_ns}


def discover_checkpoints(model, requested):
    directory = Path(model["checkpoint_dir"])
    if not directory.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {directory}")
    checkpoints = {}
    for path in directory.glob("ckpt_*_1.pth.tar"):
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected checkpoint filename: {path}")
        epoch = int(match.group(1))
        if epoch in checkpoints:
            raise ValueError(f"Duplicate checkpoint epoch {epoch}: {directory}")
        if not path.is_file():
            raise ValueError(f"Checkpoint is not a regular file: {path}")
        checkpoints[epoch] = path
    if not checkpoints:
        raise ValueError(f"No ckpt_<epoch>_1.pth.tar checkpoints in {directory}")
    if requested is not None:
        missing = sorted(requested - set(checkpoints))
        if missing:
            raise ValueError(f"Missing requested epochs for {model['name']}: {missing}")
    for epoch in sorted(checkpoints if requested is None else requested):
        path = checkpoints[epoch]
        digest, stat = snapshot_checkpoint(path)
        yield {"epoch": epoch, "checkpoint": str(path),
               "checkpoint_sha256": digest, "checkpoint_stat": stat}


def discover_inputs(input_root, energy, flavors, limit):
    directory = Path(input_root) / f"{energy}GeV"
    result = []
    seen_paths, seen_stems = set(), set()
    for flavor in flavors:
        paths = sorted(directory.glob(f"{flavor}*.h5"))
        if not paths:
            raise ValueError(f"No {flavor}*.h5 inputs in {directory}")
        if limit:
            paths = paths[:limit]
        for path in paths:
            if not path.is_file():
                raise ValueError(f"Input is not a regular file: {path}")
            if any(c in str(path) for c in "\r\n\0"):
                raise ValueError(f"Input path cannot be written to a line-based list: {path!s}")
            if str(path) in seen_paths or path.stem in seen_stems:
                raise ValueError(f"Duplicate input or ROOT output stem: {path}")
            stat = path.stat()
            if stat.st_size == 0:
                raise ValueError(f"Empty H5 input: {path}")
            result.append({"path": str(path), "size": stat.st_size,
                           "mtime_ns": stat.st_mtime_ns})
            seen_paths.add(str(path))
            seen_stems.add(path.stem)
    return sorted(result, key=lambda item: item["path"])


def make_plan(config, out, servers, epochs, model_names, limit, files_per_job):
    # Imported here to keep help and configuration parsing dependency-free.
    from scan_common import implementation_hashes

    selected = config["models"]
    if model_names is not None:
        names = set(model_names.split(","))
        if "" in names:
            raise ValueError("--models cannot contain an empty name")
        unknown = names - {model["name"] for model in selected}
        if unknown:
            raise ValueError(f"Unknown requested model names: {sorted(unknown)}")
        selected = [model for model in selected if model["name"] in names]
    if not selected:
        raise ValueError("No models selected")
    source_hashes = implementation_hashes(Path(config["repo"]))
    samples, filelists, bundles = {}, {}, []
    for model in selected:
        input_root = model.get("input_root", config["input_root"])
        sample_id = hashlib.sha256(input_root.encode()).hexdigest()[:16]
        for energy in config["energies"]:
            key = (input_root, energy)
            if key not in samples:
                inputs = discover_inputs(input_root, energy, config["flavors"], limit)
                samples[key] = [inputs[i:i + files_per_job]
                                for i in range(0, len(inputs), files_per_job)]
                for shard, group in enumerate(samples[key]):
                    name = out / "filelists" / sample_id / f"{energy}GeV" / f"shard_{shard:03d}.txt"
                    filelists[str(name)] = "".join(item["path"] + "\n" for item in group)
        checkpoints = list(discover_checkpoints(model, epochs))
        for checkpoint in checkpoints:
            jobs = []
            for energy in config["energies"]:
                for method in config["methods"]:
                    for shard, group in enumerate(samples[(input_root, energy)]):
                        identity = [model["name"], model["run_id"], checkpoint["epoch"], energy, method, shard]
                        job_id = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()[:20]
                        output = (out / "roots" / model["name"] / model["run_id"] /
                                  f"epoch_{checkpoint['epoch']:03d}" / f"{energy}GeV" /
                                  method / f"shard_{shard:03d}")
                        writer_args = model["writer_args"] + ["--tbeta", str(config["tbeta"]),
                                                             "--td", str(config["td"])]
                        if method == "truth_clustering":
                            writer_args = writer_args + ["--truth-clustering"]
                        jobs.append({
                            "id": job_id, "model": model["name"], "run_id": model["run_id"],
                            **checkpoint, "energy": energy, "method": method, "shard": shard,
                            "inputs": group,
                            "input_list": str(out / "filelists" / sample_id / f"{energy}GeV" /
                                              f"shard_{shard:03d}.txt"),
                            "output_dir": str(output), "nstart": config["nstart"], "nend": config["nend"],
                            "input_dim": model["input_dim"], "output_dim": model["output_dim"],
                            "writer_args": writer_args,
                        })
            cost = sum(item["size"] for job in jobs for item in job["inputs"])
            bundles.append({"jobs": jobs, "cost": cost, "key": (model["name"], checkpoint["epoch"])})

    # Long bundles first reduces late stragglers. A bundle never crosses servers.
    loads = {server["name"]: 0 for server in servers}
    for bundle in sorted(bundles, key=lambda value: (-value["cost"], value["key"])):
        target = min(servers, key=lambda server: (
            (loads[server["name"]] + bundle["cost"]) / server["weight"],
            loads[server["name"]] / server["weight"], server["name"]))
        for job in bundle["jobs"]:
            job["server"] = target["name"]
        loads[target["name"]] += bundle["cost"]
    jobs = [job for bundle in bundles for job in bundle["jobs"]]
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Job ID collision")
    if len({job["output_dir"] for job in jobs}) != len(jobs):
        raise ValueError("ROOT output directory collision")
    return {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo": config["repo"], "scan_dir": str(out), "servers": servers,
        "implementation": source_hashes, "config": config,
        "selection": {"models": [model["name"] for model in selected],
                      "epochs": "all" if epochs is None else sorted(epochs),
                      "files_per_flavor": limit, "files_per_job": files_per_job},
        "bundle_count": len(bundles), "jobs": jobs,
    }, filelists


def write_plan(out, plan, filelists):
    # Exclusive creation prevents rerunning a command from changing an active scan.
    out.mkdir(parents=True, exist_ok=False)
    try:
        for name, contents in filelists.items():
            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
            path.chmod(0o444)
        target = out / "plan.json"
        with target.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        target.chmod(0o444)
    except BaseException:
        # Only this newly created scan directory is owned by this invocation.
        shutil.rmtree(out)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("scan_config.json"))
    parser.add_argument("--out", required=True, help="New shared /data scan directory; must not exist")
    parser.add_argument("--servers", required=True, help="Logical names and capacity weights: serverA:1,serverB:2")
    parser.add_argument("--repo", help="Override config repo, preserving the supplied shared path")
    parser.add_argument("--models", help="Comma-separated model names; omitted selects all configured models")
    parser.add_argument("--epochs", default="all", help="all or 0,5,10:21:2; range stop is excluded; missing epochs fail")
    parser.add_argument("--files-per-flavor", type=int, default=0,
                        help="First sorted N H5 files per flavor per energy; 0 selects all (default)")
    parser.add_argument("--files-per-job", type=int, default=20,
                        help="H5 files sharing one model load (default: 20)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        require_int(args.files_per_flavor, "--files-per-flavor", 0)
        require_int(args.files_per_job, "--files-per-job")
        config_path = absolute_path(str(args.config))
        config = read_config(config_path, args.repo)
        out = absolute_path(args.out)
        if out.exists() or out.is_symlink():
            raise ValueError(f"Scan directory already exists; use another --out: {out}")
        plan, filelists = make_plan(config, out, parse_servers(args.servers),
                                  parse_epochs(args.epochs), args.models,
                                  args.files_per_flavor, args.files_per_job)
        write_plan(out, plan, filelists)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"plan_scan: {error}\n")
    print(f"Plan: {out / 'plan.json'}")
    print(f"Models: {len(plan['selection']['models'])}; model/epoch bundles: {plan['bundle_count']}; jobs: {len(plan['jobs'])}")
    for server in plan["servers"]:
        jobs = [job for job in plan["jobs"] if job["server"] == server["name"]]
        bundles = {(job["model"], job["run_id"], job["epoch"]) for job in jobs}
        inputs = sum(len(job["inputs"]) for job in jobs)
        print(f"  {server['name']} (weight {server['weight']:g}): {len(bundles)} bundles, {len(jobs)} jobs, {inputs} ROOT files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
