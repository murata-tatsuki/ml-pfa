"""One-inference threshold scans with resumable, atomic ROOT output.

Inference results are CPU arrays shared by forked beta workers.  Each worker
uses the common Cedric writer and reuses its clustering cache across td values;
there is no separate scan-specific physics filler or ROOT schema.
"""
from __future__ import annotations

import copy
import gc
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from cluster_energy import inference_loader_kwargs, pooling_model
from dataset import ILCDataset
from model import get_model, get_model_branch
from test_yielder_edit import TestYielder

from . import cedric, runtime
from .clustering import _BetaClusterEngine


TBETA_VALUES = tuple(i / 10.0 for i in range(9, 0, -1))
TD_VALUES = tuple(i / 10.0 for i in range(9, 0, -1))


@dataclass(frozen=True)
class _ScanContext:
    args: object
    cache: tuple
    expected_counts: Optional[Tuple[int, int, int]]


_CONTEXT = None


class _CachedYielder:
    """Feed the common writer without another model or data-loader pass."""

    pid_mode = False

    def __init__(self, cache, engine):
        self.cache = cache
        self.engine = engine

    def iter_pred(self, nmax=None, energyRegression=False,
                  energyRegressionCluster=False, energyRegressionWeight=False):
        del energyRegression, energyRegressionCluster, energyRegressionWeight
        for index, pair in enumerate(self.cache):
            if nmax is not None and index >= nmax:
                break
            yield pair

    def cluster_cached(self, event_index, td):
        return self.engine.cluster(event_index, td)


class _NoopModel:
    """Cached predictions need no model in a CPU output worker."""

    def to(self, unused_device):
        return self


def point_name(tbeta, td):
    return f"tbeta{round(tbeta * 100):03d}td{round(td * 100):03d}"


def point_output(args, tbeta, td):
    return (Path(args.outfile).expanduser() / point_name(tbeta, td)
            / (Path(args.datapath).stem + ".root"))


def _point_args(args, tbeta, td):
    result = copy.copy(args)
    result.beta_d_scan = False
    result.tbeta = float(tbeta)
    result.td = float(td)
    return result


def _threshold_values(args):
    beta_values = tuple(getattr(args, "scan_tbeta", None) or TBETA_VALUES)
    td_values = tuple(getattr(args, "scan_td", None) or TD_VALUES)
    for name, values in (("tbeta", beta_values), ("td", td_values)):
        if not values:
            raise ValueError(f"--scan-{name} needs at least one threshold")
        if any(not np.isfinite(value) or value <= 0
               or (name == "tbeta" and value >= 1) for value in values):
            raise ValueError(f"--scan-{name} needs finite positive values"
                             + (" below 1" if name == "tbeta" else ""))
        if len({round(value * 100) for value in values}) != len(values):
            raise ValueError(f"--scan-{name} values must have distinct "
                             "two-decimal output names")
    return beta_values, td_values


def _build_prediction_cache(args):
    """Retain the original scan loader, prediction layout and event selection."""
    event_energy = (args.event_total_energy
                    or cedric.has_event_builder(args.datapath))
    if event_energy and not args.event_total_energy:
        print("Found event builder: storing truth q/qbar directions in the ROOT jet tree")

    input_dim, output_dim = args.input_dim, args.output_dim
    thetaphi = input_dim == 7
    if args.momentum:
        input_dim += 3
        if args.momentum_amp:
            input_dim += 1
    if args.energy_regression:
        output_dim += 1
        if args.energy_regression_cluster:
            output_dim += 1
    if args.energy_regression_weight:
        output_dim += 5

    device = args.device
    if "cuda" in device:
        torch.cuda.set_device(device)
    print(f"Loading model from checkpoint {args.ckpt}")
    model = dataset = yielder = None
    try:
        if args.energy_branch:
            model = get_model_branch(args.ckpt, jit=False, input_dim=input_dim,
                                     output_dim=output_dim).to(device)
        else:
            model = get_model(
                args.ckpt, jit=False, input_dim=input_dim, output_dim=output_dim,
                energy_regression=args.energy_regression,
                energy_regression_cluster=args.energy_regression_cluster,
                energy_regression_weight=args.energy_regression_weight,
                model_variant=args.model_variant,
                **inference_loader_kwargs(args)).to(device)
        if pooling_model(model) is not None:
            raise ValueError("A pooled energy head depends on clustering thresholds "
                             "and cannot use a one-inference threshold scan.")

        print(f"Loading data from {args.datapath} with nstart={args.nstart}, "
              f"nend={args.nend}, timingCut={args.timingCut}")
        dataset = ILCDataset(
            args.datapath, timingCut=args.timingCut, thetaphi=thetaphi,
            test_mode=True, nstart=args.nstart, nend=args.nend,
            pandora=args.pandora, momentum=args.momentum,
            momentumAmp=args.momentum_amp, mctpe=args.mctpe,
            event_energy=event_energy)
        yielder = TestYielder(model=model, dataset=dataset, device=device,
                             pandora=args.pandora, event_energy=event_energy)
        nmax = None if args.nend == -1 else args.nend
        print("[scan] Running one GNN inference pass for this input...")
        cache = tuple(yielder.iter_pred(
            nmax=nmax, energyRegression=args.energy_regression,
            energyRegressionCluster=args.energy_regression_cluster,
            energyRegressionWeight=args.energy_regression_weight))
        print(f"[scan] Cached {len(cache)} events in CPU memory")
        return cache, event_energy
    finally:
        # Release CUDA and DataLoader references before forking.  Workers only
        # receive the CPU Event/Prediction objects retained in cache.
        if yielder is not None:
            yielder.model = None
            yielder.loader = None
        model = dataset = yielder = None
        gc.collect()
        if "cuda" in device:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def _complete(output, args, expected_counts):
    return (not getattr(args, "replace", False)
            and runtime.validate_existing(output, args, args.datapath,
                                          expected_counts=expected_counts,
                                          pid=False))


def _run_beta_group(job):
    context = _CONTEXT
    if context is None:
        raise RuntimeError("No inference cache installed for threshold worker")
    tbeta, td_values = job
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    engine = _BetaClusterEngine(context.cache, tbeta)
    yielder = _CachedYielder(context.cache, engine)
    written = []
    for td in td_values:
        args = _point_args(context.args, tbeta, td)
        # Predictions are already on CPU; a forked worker must not touch CUDA.
        args.device = "cpu"
        output = point_output(args, tbeta, td)
        if _complete(output, args, context.expected_counts):
            print(f"[scan skip] {output}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = cedric._temporary_output(output)
        print(f"[scan pid={os.getpid()}] {point_name(tbeta, td)} -> {output}",
              flush=True)
        try:
            cedric.save_root(
                args.datapath, args.ckpt, str(temporary),
                nstart=args.nstart, nend=args.nend, timingCut=args.timingCut,
                input_dim=args.input_dim, output_dim=args.output_dim,
                args=args, model=_NoopModel(), prediction_yielder=yielder)
            runtime.write_metadata(temporary, args, args.datapath)
            if not runtime.validate_existing(
                    temporary, args, args.datapath,
                    expected_counts=context.expected_counts, pid=False):
                raise RuntimeError(f"New scan ROOT output failed validation: {temporary}")
            os.replace(temporary, output)
            written.append(str(output))
        finally:
            temporary.unlink(missing_ok=True)
    return written


def run_scan(args):
    """Export one H5 across a grid; complete matching points avoid inference."""
    global _CONTEXT
    if args.pandora:
        raise ValueError("--beta-d-scan requires GNN clustering, not --pandora")
    if int(args.scan_workers) < 1:
        raise ValueError("--scan-workers must be at least 1")
    directory = Path(args.outfile).expanduser()
    if directory.suffix == ".root" or (directory.exists() and not directory.is_dir()):
        raise ValueError("--beta-d-scan outfile must be an output directory")
    beta_values, td_values = _threshold_values(args)
    # Completion metadata records the actual retained rows, including the
    # original unknown-truth and minimum-hit input selection.
    expected_counts = None

    pending = []
    skipped = 0
    protected = {Path(args.datapath).expanduser().resolve(),
                 Path(args.ckpt).expanduser().resolve()}
    for tbeta in beta_values:
        missing = []
        for td in td_values:
            output = point_output(args, tbeta, td)
            if output.resolve() in protected:
                raise ValueError(f"Scan output would replace an input: {output}")
            if output.is_dir():
                raise ValueError(f"Scan ROOT output is a directory: {output}")
            point_args = _point_args(args, tbeta, td)
            if _complete(output, point_args, expected_counts):
                skipped += 1
                print(f"[scan skip] {output}", flush=True)
            else:
                missing.append(float(td))
        if missing:
            pending.append((float(tbeta), tuple(missing)))
    if not pending:
        print(f"[scan] All {skipped} ROOT files are complete; "
              "skipping model loading and GNN inference.")
        return []

    cache = None
    try:
        cache, unused_event_energy = _build_prediction_cache(args)
        expected_counts = (len(cache), sum(len(event.x) for event, _ in cache),
                           sum(getattr(event, 'jet', None) is not None
                               for event, _ in cache))
        _CONTEXT = _ScanContext(args, cache, expected_counts)
        cedric._ensure_root_writer_loaded()
        workers = min(int(args.scan_workers), len(pending))
        print(f"[scan] Starting {workers} CPU threshold worker(s)")
        if workers == 1:
            groups = [_run_beta_group(job) for job in pending]
        else:
            # Copy-on-write shares the immutable CPU prediction cache.  No
            # child constructs a model or touches its parent's CUDA context.
            with mp.get_context("fork").Pool(processes=workers) as pool:
                groups = pool.map(_run_beta_group, pending)
        written = [path for group in groups for path in group]
        print(f"[scan] Wrote {len(written)} ROOT file(s), skipped {skipped}")
        return written
    finally:
        _CONTEXT = None
        cache = None
        gc.collect()
