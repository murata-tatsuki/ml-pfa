#!/usr/bin/env python3
"""Collect JER with the output layout recorded in an epoch-scan plan."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_jer as core
from scan_common import load_plan
from scan_layout import summary_directory
from epoch_collect import EPOCH_LAYOUT, collect_epoch_group


def main(argv=None):
    parser = argparse.ArgumentParser(description=core.__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--beta-threshold", default=0.0, type=float,
                        help="Analysis reco-cluster cut, independent of writer tbeta (default 0)")
    parser.add_argument("--models", nargs="+", help="Only these model IDs (comma or space separated)")
    parser.add_argument("--epochs", default="all",
                        help="all, comma-separated epochs, or start:stop[:step] (stop excluded)")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Publish explicitly partial groups; never eligible for ranking")
    parser.add_argument("--output-csv", type=Path,
                        help="Default result directory/metrics.csv; required when filtering models/epochs")
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    epochs = core._epoch_selection(args.epochs)
    if args.models:
        args.models = [name for value in args.models for name in value.split(",")]
    if (args.models or epochs is not None) and args.output_csv is None:
        parser.error("Filtered collection requires --output-csv to preserve the full metrics.csv")
    if args.models and set(args.models) - {job["model"] for job in plan["jobs"]}:
        parser.error("Unknown model in --models")
    groups = {}
    names = ("model", "run_id", "epoch", "energy", "method")
    for job in plan["jobs"]:
        if args.models and job["model"] not in args.models:
            continue
        if epochs is not None and int(job["epoch"]) not in epochs:
            continue
        key = tuple(job[name] for name in names)
        groups.setdefault(key, []).append(job)
    if not groups:
        parser.error("No selected jobs")
    rows, selections = [], {}
    for key, jobs in sorted(groups.items()):
        print(f"[collect] model={key[0]} run={key[1]} epoch={key[2]} energy={key[3]} method={key[4]}",
              flush=True)
        group = dict(zip(names, key))
        if plan.get("layout") == EPOCH_LAYOUT:
            current = collect_epoch_group(plan, jobs, args.beta_threshold, args.allow_incomplete)
        else:
            current = core.collect_group(plan, jobs, args.beta_threshold, args.allow_incomplete,
                                         output_directory=summary_directory(plan, group))
        representative = current[0]
        if representative["complete"]:
            comparison = (key[3], key[4])
            digest = representative["event_selection_sha256"]
            if comparison in selections and selections[comparison] != digest:
                raise ValueError(f"Evaluation event selection differs across complete groups: {key}")
            selections[comparison] = digest
        rows.extend(current)
    result_dir = plan["result_dir"] if plan.get("layout") == EPOCH_LAYOUT else plan["scan_dir"]
    destination = args.output_csv or Path(result_dir) / "metrics.csv"
    fieldnames = list(dict.fromkeys(name for row in rows for name in row))
    with tempfile.TemporaryDirectory(prefix="epoch-scan-metrics-") as temporary:
        staged = Path(temporary) / "metrics.csv"
        with staged.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        core._publish(staged, destination)
    print(f"Saved {len(rows)} metric rows from {len(groups)} groups: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"collect_scan: {error}", file=sys.stderr)
        raise SystemExit(1)
