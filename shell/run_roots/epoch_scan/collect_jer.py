#!/usr/bin/env python3
"""Pool frozen ROOT shards and calculate the repository's fixed-energy JER.

Run on any server sharing /data. Event distributions are pooled before calling
pid_jer::Summary; shard resolutions are never averaged. Source ROOT/H5 files
are never modified. Explicit partial collection is excluded from ranking.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
SCHEMA = "root-epoch-scan-1"
TREE_NAMES = ("t", "reco", "prediction", "event", "jet")
_DEPS = {}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependencies(repo):
    key = str(Path(repo).resolve())
    if key not in _DEPS:
        import numpy as np
        import ROOT
        import uproot

        ROOT.gROOT.SetBatch(True)
        ROOT.TH1.AddDirectory(False)
        if key not in sys.path:
            sys.path.insert(0, key)
        spec = importlib.util.spec_from_file_location("epoch_scan_pid_readouts",
            Path(key) / "macro/save_pid_energy_readouts.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        header = Path(key) / "macro/src/pid_jet_resolution_summary.h"
        if not ROOT.gInterpreter.Declare("#include " + json.dumps(str(header))):
            raise RuntimeError(f"Cannot load JER implementation: {header}")
        from save_root import build_parser
        from root_export.runtime import PHYSICS_OPTIONS
        _DEPS[key] = (np, ROOT, uproot, helper, build_parser, PHYSICS_OPTIONS)
    return _DEPS[key]


def _safe_component(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in (".", ".."):
        raise ValueError(f"Unsafe output path component: {value!r}")
    return value


def _identity(item):
    return dict(path=str(Path(item["path"]).expanduser().resolve()),
                size=int(item["size"]),
                modified_ns=int(item.get("modified_ns", item.get("mtime_ns"))))


def _check_identity(item):
    expected = _identity(item)
    stat = Path(expected["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (expected["size"], expected["modified_ns"]):
        raise ValueError(f"Input changed since planning: {expected['path']}")
    return expected


def _settings(job, recorded, deps):
    _, _, _, _, build_parser, physics = deps
    args = build_parser().parse_args([
        job["inputs"][0]["path"], job["checkpoint"], job["output_dir"],
        str(job["nstart"]), str(job["nend"]), "False",
        str(job["input_dim"]), str(job["output_dim"]), *job["writer_args"],
    ])
    args.energy_regression = args.energy_regression_cluster = True
    args.output_schema = "cedric"
    if args.energy_heads == "auto":
        if recorded.get("energy_heads") not in (2, 5):
            raise ValueError("Invalid resolved energy_heads in writer identity")
        args.energy_heads = recorded["energy_heads"]
    else:
        args.energy_heads = int(args.energy_heads)
    if args.pid_readout == "auto":
        if recorded.get("pid_readout") not in ("representative", "legacy"):
            raise ValueError("Invalid resolved pid_readout in writer identity")
        args.pid_readout = recorded["pid_readout"]
    elif args.pid_readout == "none":
        args.pid_readout = "legacy"
    expected = {name: getattr(args, name, None) for name in physics}
    if expected.get("clustering_bn_eps") is None:
        expected.pop("clustering_bn_eps", None)
    if recorded != expected:
        differences = {name: {"planned": expected.get(name), "saved": recorded.get(name)}
                       for name in sorted(expected.keys() | recorded.keys())
                       if expected.get(name) != recorded.get(name)}
        raise ValueError(f"ROOT physics settings differ from plan: {differences}")
    return expected


def _root_info(path, input_identity, job, plan, deps):
    np, _, uproot, helper, _, _ = deps
    with uproot.open(path) as source:
        metadata = helper._json_object(source, "save_root_metadata")
        signature = metadata.get("signature", {})
        if signature.get("version") != "cedric-unified-1":
            raise ValueError("Missing supported writer completion identity")
        if signature.get("checkpoint_sha256") != job["checkpoint_sha256"]:
            raise ValueError("ROOT checkpoint hash differs from plan")
        if [_identity(item) for item in signature.get("inputs", [])] != [input_identity]:
            raise ValueError("ROOT input identity differs from planned H5")
        settings = _settings(job, signature.get("settings", {}), deps)
        implementation = signature.get("implementation", {})
        if not implementation or any(plan.get("implementation", {}).get(name) != digest
                                     for name, digest in implementation.items()):
            raise ValueError("ROOT writer source hashes differ from plan")
        entries = {name: int(source[name].num_entries) for name in TREE_NAMES}
        if metadata.get("tree_entries") != entries:
            raise ValueError("ROOT trees differ from writer completion identity")
        builder = helper._arrays(source["event"], ("event", "MC_dijet_energy"))
        all_ids = builder["event"].astype(np.int64)
        if len(np.unique(all_ids)) != len(all_ids):
            raise ValueError("Duplicate event IDs in source event tree")
        jets = helper._arrays(source["jet"], ("event", "n_jets", "jet_p4"))
        if len(np.unique(jets["event"])) != len(jets["event"]):
            raise ValueError("Duplicate event IDs in source jet tree")
        allowed_ids = set(map(int, all_ids))
        if not set(map(int, jets["event"])).issubset(allowed_ids):
            raise ValueError("Jet rows reference absent source events")
        angles = {int(event): helper._thrust(np.asarray(p4).reshape(-1))
                  if int(count) == 2 else float("nan")
                  for event, count, p4 in zip(jets["event"], jets["n_jets"], jets["jet_p4"])}
        valid = np.isfinite(builder["MC_dijet_energy"]) & (builder["MC_dijet_energy"] > 0)
        event_ids, truth = all_ids[valid], builder["MC_dijet_energy"][valid]
        pid = "energy_pid" in source["reco"].keys()
        if pid != (settings["pid_readout"] == "representative"):
            raise ValueError("Saved PID branches disagree with writer settings")
        fields = ("event", "cluster", "cond_beta", "energy_pid") if pid else (
            "event", "cluster", "cond_beta", "pred_edep", "pred_edep_cluster", "ntrack_hits")
        reco = helper._arrays(source["reco"], fields)
        identities = np.rec.fromarrays((reco["event"], reco["cluster"]))
        if len(np.unique(identities)) != len(identities):
            raise ValueError("Duplicate event/cluster IDs in source reco tree")
        if not set(map(int, reco["event"])).issubset(allowed_ids):
            raise ValueError("Reco rows reference absent source events")
        energies = (reco["energy_pid"].astype(np.float64) if pid else
                    np.where(reco["ntrack_hits"] > 0, reco["pred_edep"],
                             reco["pred_edep_cluster"]).astype(np.float64))
        particle = helper._json_object(source, "particle_eval_metadata", required=pid)
    return dict(settings=settings, events=event_ids, truth=truth, angles=angles,
                reco=reco, energies=energies, pid=pid, particle=particle)


def _publish(source, destination):
    """Atomic per-file publication also works across different /tmp mounts."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"Refusing to replace symlink: {destination}")
    pending = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pending")
    try:
        shutil.copyfile(source, pending)
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def _verify_plan_sources(plan):
    from scan_common import verify_implementation
    verify_implementation(plan)


def collect_group(plan, jobs, beta_threshold=0.0, allow_incomplete=False,
                  output_directory=None):
    """Collect one model/run/epoch/energy/method group and return metric rows."""
    if not jobs:
        raise ValueError("No jobs supplied")
    if not math.isfinite(beta_threshold) or not 0 <= beta_threshold <= 1:
        raise ValueError("beta threshold must be finite and in [0, 1]")
    _verify_plan_sources(plan)
    deps = _dependencies(plan["repo"])
    np, ROOT, uproot, helper, _, _ = deps
    group = {name: jobs[0][name] for name in ("model", "run_id", "epoch", "energy", "method")}
    if any(any(job[name] != value for name, value in group.items()) for job in jobs):
        raise ValueError("collect_group requires exactly one model/epoch/energy/method")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job ID in group")
    if group["method"] not in ("reco", "truth_clustering"):
        raise ValueError("Unsupported evaluation method")
    if len({job["checkpoint_sha256"] for job in jobs}) != 1:
        raise ValueError("One evaluation group contains different checkpoints")
    source_inputs = [str(Path(item["path"]).resolve()) for job in jobs for item in job["inputs"]]
    if len(set(source_inputs)) != len(source_inputs):
        raise ValueError("Input H5 appears in more than one shard of the same group")
    for checkpoint, digest in {(job["checkpoint"], job["checkpoint_sha256"]) for job in jobs}:
        if _sha256(checkpoint) != digest:
            raise ValueError(f"Checkpoint changed since planning: {checkpoint}")
    if output_directory is None:
        output_directory = Path(plan["scan_dir"]) / "summaries"
        for component in (group["model"], group["run_id"], f"epoch_{int(group['epoch']):03d}",
                          f"{int(group['energy'])}GeV", group["method"]):
            output_directory /= _safe_component(component)
    output_directory = Path(output_directory)
    sources = sorted(((str(Path(item["path"]).resolve()), job, item)
                      for job in jobs for item in job["inputs"]), key=lambda row: row[0])
    summaries, failures, loaded = {}, [], []
    event_digest = hashlib.sha256()
    reference_settings = reference_policy = None
    total_events = 0
    for file_index, (input_path, job, item) in enumerate(sources):
        root_path = Path(job["output_dir"]) / (Path(input_path).stem + ".root")
        try:
            identity = _check_identity(item)
            info = _root_info(root_path, identity, job, plan, deps)
            settings = info["settings"]
            if bool(settings["truth_clustering"]) != (group["method"] == "truth_clustering"):
                raise ValueError("ROOT clustering method differs from group")
            if reference_settings is not None and settings != reference_settings:
                raise ValueError("Physics settings differ between shards")
            policy_names = ("species", "energy_policy", "single_track_energy", "pid_seed",
                            "particle_cluster_readout", "particle_cluster_config")
            policy = {key: info["particle"].get(key) for key in policy_names}
            if reference_policy is not None and policy != reference_policy:
                raise ValueError("Energy readout policy differs between shards")
            selected = ~(info["reco"]["cond_beta"] < beta_threshold)
            saved = helper._totals(info["reco"]["event"], info["energies"], selected, info["events"])
            rows = [dict(event=int(event), truth_energy=float(truth),
                         thrust=info["angles"].get(int(event), float("nan")),
                         reco_energy=saved[int(event)])
                    for event, truth in zip(info["events"], info["truth"])]
            modes = {"saved": saved}
            if info["pid"]:
                alternate, _, _ = helper._file_readouts(root_path, info["particle"], rows, beta_threshold)
                modes.update(alternate)
            if summaries and set(modes) != set(summaries):
                raise ValueError("PID/readout availability differs between shards")
            reference_settings, reference_policy = settings, policy
            particle_text = json.dumps(info["particle"], sort_keys=True) if info["pid"] else ""
            # No pooled-summary mutation occurs until this shard is validated.
            for mode, energies in modes.items():
                if mode not in summaries:
                    summaries[mode] = ROOT.pid_jer.Summary(int(group["energy"]), group["method"],
                                                         beta_threshold, mode)
                summary = summaries[mode]
                summary.addInput(file_index, str(root_path.resolve()), info["pid"], particle_text)
                for row in rows:
                    summary.observe(file_index, row["event"], row["thrust"],
                                    energies[row["event"]], row["truth_energy"])
            event_digest.update(input_path.encode("utf-8") + b"\0")
            for row in sorted(rows, key=lambda value: value["event"]):
                angle = row["thrust"]
                event_digest.update(struct.pack("<qdd", row["event"], row["truth_energy"],
                                                angle if math.isfinite(angle) else float("nan")))
            total_events += len(rows)
            loaded.append(dict(job_id=job["id"], input=input_path, root=str(root_path),
                               file_index=file_index, events=len(rows)))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            if not allow_incomplete:
                raise ValueError(f"Cannot collect {root_path}: {error}") from error
            failures.append(dict(job_id=job["id"], input=input_path, root=str(root_path), reason=str(error)))
    complete = not failures
    coverage = dict(schema="root-epoch-scan-coverage-1", group=group,
                    complete=complete, eligible_for_ranking=complete,
                    expected_files=len(sources), collected_files=len(loaded), events=total_events,
                    event_selection_sha256=event_digest.hexdigest(), beta_threshold=beta_threshold,
                    loaded=loaded, failures=failures)
    metrics = []
    with tempfile.TemporaryDirectory(prefix="epoch-scan-jer-") as temporary:
        staging = Path(temporary)
        for mode, summary in summaries.items():
            mode_directory = staging / mode
            summary.write(str(mode_directory), False)
            with uproot.open(mode_directory / "jet_energy_resolution_summary.root") as source:
                data = source["resolution"].arrays(library="np")
                for index in range(len(data["angle_bin"])):
                    angle_bin = int(data["angle_bin"][index])
                    region = "central" if angle_bin == -1 else "forward" if angle_bin == -2 else "angle"
                    metrics.append(dict(group, readout=mode, region=region,
                        **{name: values[index].item() for name, values in data.items()},
                        complete=complete, eligible_for_ranking=complete,
                        expected_files=len(sources), collected_files=len(loaded), events=total_events,
                        event_selection_sha256=event_digest.hexdigest(), beta_threshold=beta_threshold,
                        summary_root=str(output_directory / mode / "jet_energy_resolution_summary.root")))
            for name in ("jet_energy_resolution_summary.root", "jet_energy_resolution_summary.txt"):
                _publish(mode_directory / name, output_directory / mode / name)
        if not metrics:
            metrics.append(dict(group, readout="saved", region="unavailable", valid=False,
                complete=False, eligible_for_ranking=False, expected_files=len(sources),
                collected_files=0, events=0, event_selection_sha256=event_digest.hexdigest(),
                beta_threshold=beta_threshold))
        coverage_path = staging / "coverage.json"
        coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
        _publish(coverage_path, output_directory / "coverage.json")
    return metrics


def _epoch_selection(text):
    from plan_scan import parse_epochs
    return parse_epochs(text or "all")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--beta-threshold", default=0.0, type=float,
                        help="Analysis reco-cluster cut, independent of writer tbeta (default 0)")
    parser.add_argument("--models", nargs="+", help="Only these model IDs (comma or space separated)")
    parser.add_argument("--epochs", default="all",
                        help="all, comma-separated epochs, or start:stop[:step] (stop excluded)")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Publish explicitly partial groups; never eligible for ranking")
    parser.add_argument("--output-csv", type=Path,
                        help="Default SCAN_DIR/metrics.csv; required when filtering models/epochs")
    args = parser.parse_args(argv)
    from scan_common import load_plan
    plan = load_plan(args.plan)
    epochs = _epoch_selection(args.epochs)
    if args.models:
        args.models = [name for value in args.models for name in value.split(",")]
    if (args.models or epochs is not None) and args.output_csv is None:
        parser.error("Filtered collection requires --output-csv to preserve the full metrics.csv")
    if args.models and set(args.models) - {job["model"] for job in plan["jobs"]}:
        parser.error("Unknown model in --models")
    groups = {}
    for job in plan["jobs"]:
        if args.models and job["model"] not in args.models:
            continue
        if epochs is not None and int(job["epoch"]) not in epochs:
            continue
        key = tuple(job[name] for name in ("model", "run_id", "epoch", "energy", "method"))
        groups.setdefault(key, []).append(job)
    if not groups:
        parser.error("No selected jobs")
    rows, selections = [], {}
    for key, jobs in sorted(groups.items()):
        print(f"[collect] model={key[0]} run={key[1]} epoch={key[2]} energy={key[3]} method={key[4]}", flush=True)
        current = collect_group(plan, jobs, args.beta_threshold, args.allow_incomplete)
        representative = current[0]
        if representative["complete"]:
            comparison = (key[3], key[4])
            digest = representative["event_selection_sha256"]
            if comparison in selections and selections[comparison] != digest:
                raise ValueError(f"Evaluation event selection differs across complete groups: {key}")
            selections[comparison] = digest
        rows.extend(current)
    destination = args.output_csv or Path(plan["scan_dir"]) / "metrics.csv"
    fieldnames = list(dict.fromkeys(name for row in rows for name in row))
    with tempfile.TemporaryDirectory(prefix="epoch-scan-metrics-") as temporary:
        staged = Path(temporary) / "metrics.csv"
        with staged.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        _publish(staged, destination)
    print(f"Saved {len(rows)} metric rows from {len(groups)} groups: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"collect_jer: {error}", file=sys.stderr)
        raise SystemExit(1)
