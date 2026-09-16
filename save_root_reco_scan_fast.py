#!/usr/bin/env python3
"""Fast beta/distance scan with one GNN inference per event.

The physics/event filling code is delegated to save_root_reco_w_Cedric.py.
Only the scan orchestration, object-condensation clustering, and ROOT I/O are
replaced here:

* one inference cache is shared read-only with forked beta workers;
* beta ordering and distances are reused across the nine distance thresholds;
* independent beta groups run in CPU processes;
* legacy branch buffers are collected and sent to C++ in whole arrays;
* output files use ROOT LZ4 compression.

Scan output remains:
    <output-dir>/tbetaXXXtdXXX/<input-h5-stem>.root
"""

from __future__ import annotations

import argparse
import copy
import gc
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Avoid multiplying BLAS/OpenMP thread pools inside every threshold worker.
# These defaults can still be overridden explicitly before starting Python.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np  # noqa: E402
import ROOT  # noqa: E402
import torch  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import objectcondensation as _oc  # noqa: E402
import save_root_reco_w_Cedric as _legacy  # noqa: E402
from dataset import ILCDataset  # noqa: E402
from matching import make_matches  # noqa: E402
from model import get_model, get_model_branch  # noqa: E402
from test_yielder_edit import TestYielder  # noqa: E402


TBETA_VALUES = tuple(i / 10.0 for i in range(9, 0, -1))
TD_VALUES = tuple(i / 10.0 for i in range(9, 0, -1))
MAX_COMPOSITION = 64
ROOT_WRITER_SOURCE = PROJECT_DIR / "scan_fast_root_writer.cxx"

_WRITER_LOADED = False
_ACTIVE_BATCH_FILE = None
_PREDICTION_CACHE: Tuple = ()
_ACTIVE_CLUSTER_ENGINE = None
_GLOBAL_ARGS = None
_GLOBAL_EVENT_ENERGY = False


def _ensure_root_writer_loaded() -> None:
    global _WRITER_LOADED
    if _WRITER_LOADED:
        return
    source = ROOT_WRITER_SOURCE.read_text()
    if not ROOT.gInterpreter.Declare(source):
        raise RuntimeError(f"Failed to load {ROOT_WRITER_SOURCE}")
    _WRITER_LOADED = True


class _BatchTree:
    """Small TTree-compatible collector used by the unchanged legacy filler."""

    def __init__(self, name: str, title: str):
        if _ACTIVE_BATCH_FILE is None:
            raise RuntimeError("TTree was created without an active output file")
        self.name = str(name)
        self.title = str(title)
        self._sources: Dict[str, np.ndarray] = {}
        self._leaflists: Dict[str, str] = {}
        self._columns: Dict[str, List] = {}
        _ACTIVE_BATCH_FILE._add_tree(self)

    def Branch(self, name, source, leaflist):
        name = str(name)
        self._sources[name] = source
        self._leaflists[name] = str(leaflist)
        self._columns[name] = []
        return None

    def Fill(self):
        for name, source in self._sources.items():
            value = np.asarray(source)
            if "[" not in self._leaflists[name]:
                self._columns[name].append(value.reshape(-1)[0].item())
            else:
                self._columns[name].append(np.array(value, copy=True))
        return 1

    @property
    def entries(self) -> int:
        if not self._columns:
            return 0
        return len(next(iter(self._columns.values())))

    def scalar_matrix(self, names: Sequence[str], dtype) -> np.ndarray:
        nrows = self.entries
        if nrows == 0:
            return np.empty((0, len(names)), dtype=dtype)
        columns = [np.asarray(self._columns[name], dtype=dtype) for name in names]
        return np.ascontiguousarray(np.column_stack(columns), dtype=dtype)

    def array_blocks(
        self, names: Sequence[str], dtype, width: int
    ) -> np.ndarray:
        nrows = self.entries
        if nrows == 0:
            return np.empty((0, len(names) * width), dtype=dtype)
        blocks = []
        for name in names:
            block = np.asarray(self._columns[name], dtype=dtype).reshape(nrows, -1)
            if block.shape[1] < width:
                raise RuntimeError(
                    f"Branch {self.name}/{name} has width {block.shape[1]}, "
                    f"expected at least {width}"
                )
            blocks.append(block[:, :width])
        return np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=dtype)


class _BatchFile:
    """TFile-compatible collector that writes all collected rows in C++."""

    def __init__(self, filename, mode="RECREATE", *unused_args, **unused_kwargs):
        del unused_args, unused_kwargs
        global _ACTIVE_BATCH_FILE
        if str(mode).lower() not in ("recreate", "create", "new"):
            raise ValueError(f"Unsupported fast-writer mode: {mode}")
        self.filename = str(filename)
        self.trees: Dict[str, _BatchTree] = {}
        self._written = False
        _ACTIVE_BATCH_FILE = self

    def _add_tree(self, tree: _BatchTree) -> None:
        self.trees[tree.name] = tree

    def Write(self):
        if self._written:
            return 0
        _write_batch_file(self.filename, self.trees)
        self._written = True
        return 1

    def Close(self):
        return self.Write()


def _addresses(*arrays: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(array.ctypes.data) for array in arrays)


def _write_batch_file(filename: str, trees: Dict[str, _BatchTree]) -> None:
    _ensure_root_writer_loaded()
    expected = {"t", "reco", "prediction", "event", "jet"}
    missing = expected.difference(trees)
    if missing:
        raise RuntimeError(f"Missing trees for {filename}: {sorted(missing)}")

    level = int(os.environ.get("FAST_ROOT_LZ4_LEVEL", "1"))
    writer = ROOT.ScanFastRootWriter(filename, level)

    t = trees["t"]
    t_int = t.scalar_matrix(
        (
            "event",
            "hitid",
            "mcid",
            "truthid",
            "mcpdg",
            "mccharge",
            "mcstatus",
            "ncluster",
            "matched_ncluster",
            "matched_cluster",
            "cond_track",
        ),
        np.int32,
    )
    t_double = t.scalar_matrix(
        (
            "mcmass",
            "mcpx",
            "mcpy",
            "mcpz",
            "mcen",
            "edep",
            "edep_reco",
            "edep_match",
            "pred_edep",
            "pred_edep_cluster",
            "pred_edep_weight",
            "cond_beta",
            "sed_radius",
            "pred_photon_energy",
            "pred_charged_hadron_energy",
            "pred_neutral_hadron_energy",
            "pred_muon_energy",
            "pred_electron_energy",
        ),
        np.float64,
    )
    writer.append_t(t.entries, *_addresses(t_int, t_double))

    reco = trees["reco"]
    reco_int = reco.scalar_matrix(
        (
            "event",
            "cluster",
            "nhits",
            "mcid",
            "mcpdg",
            "mccharge",
            "mcstatus",
            "ntrack_hits",
            "cond_is_track",
            "matched_truth_pdgid",
            "npdg_comp",
        ),
        np.int32,
    )
    reco_double = reco.scalar_matrix(
        (
            "mcmass",
            "mcpx",
            "mcpy",
            "mcpz",
            "mcen",
            "edep_reco",
            "edep_mc",
            "edep_match",
            "pred_edep",
            "pred_edep_cluster",
            "cond_beta",
            "matched_truth_hit_frac",
            "matched_truth_edep_frac",
        ),
        np.float64,
    )
    reco_comp_int = reco.array_blocks(
        ("pdg_comp_ids", "pdg_comp_hits", "pdg_comp_track_hits"),
        np.int32,
        MAX_COMPOSITION,
    )
    reco_comp_double = reco.array_blocks(
        (
            "pdg_comp_hit_frac",
            "pdg_comp_edep_frac",
            "pdg_comp_edep",
            "pdg_comp_truth_edep",
        ),
        np.float64,
        MAX_COMPOSITION,
    )
    writer.append_reco(
        reco.entries,
        *_addresses(
            reco_int, reco_double, reco_comp_int, reco_comp_double
        ),
    )

    prediction = trees["prediction"]
    prediction_int = prediction.scalar_matrix(
        (
            "event",
            "hitid",
            "mcid",
            "truthid",
            "cluster",
            "mcpdg",
            "mccharge",
            "mcstatus",
            "pred_alpha",
            "trackness",
        ),
        np.int32,
    )
    prediction_double = prediction.scalar_matrix(
        (
            "mcmass",
            "mcpx",
            "mcpy",
            "mcpz",
            "mcen",
            "edep_mc",
            "pred_edep",
            "pred_edep_cluster",
            "pred_beta",
            "weight_photon",
            "weight_charged_hadron",
            "weight_neutral_hadron",
            "weight_muon",
            "weight_electron",
        ),
        np.float64,
    )
    writer.append_prediction(
        prediction.entries,
        *_addresses(prediction_int, prediction_double),
    )

    event = trees["event"]
    event_int = event.scalar_matrix(("event", "ncluster"), np.int32)
    event_double = event.scalar_matrix(
        (
            "MC_dijet_energy",
            "total_MC_energy_truth",
            "total_MC_energy_pred",
            "total_predicted_energy_truth",
            "total_predicted_energy_pred",
        ),
        np.float64,
    )
    writer.append_event(event.entries, *_addresses(event_int, event_double))

    jet = trees["jet"]
    jet_int = jet.scalar_matrix(("event", "n_jets"), np.int32)
    jet_double = jet.array_blocks(("jet_p4",), np.float64, 8)
    writer.append_jet(jet.entries, *_addresses(jet_int, jet_double))
    writer.close()


class _PreparedClustering:
    """One event/beta cache shared by all distance thresholds in a worker."""

    def __init__(self, prediction, tbeta: float):
        self.betas = np.asarray(prediction.pred_betas)
        self.x = np.asarray(prediction.pred_cluster_space_coords)
        self.charged_hits = np.asarray(prediction.charged_hits)
        self.n_points = self.betas.shape[0]

        selected = self.betas > tbeta
        selected_indices = np.nonzero(selected)[0]
        rejected_indices = np.nonzero(~selected)[0]
        self.high_beta = selected_indices[
            np.argsort(-self.betas[selected])
        ]
        self.low_beta = rejected_indices[
            np.argsort(-self.betas[~selected])
        ]

        self._nearest_high_index: Optional[np.ndarray] = None
        self._nearest_high_distance: Optional[np.ndarray] = None
        self._distance_from_low: Dict[int, np.ndarray] = {}

    def _prepare_nearest_high(self) -> None:
        if self._nearest_high_index is not None or self.high_beta.size == 0:
            return

        # Keep the same per-hit norm and candidate order as the legacy code.
        nearest_index = np.empty(self.n_points, dtype=np.int64)
        nearest_distance = np.empty(self.n_points, dtype=self.x.dtype)
        high_x = self.x[self.high_beta]
        for point in range(self.n_points):
            distances = np.linalg.norm(self.x[point] - high_x, axis=-1)
            argmin = np.argmin(distances, axis=-1)
            nearest_index[point] = self.high_beta[argmin]
            nearest_distance[point] = distances[argmin]
        self._nearest_high_index = nearest_index
        self._nearest_high_distance = nearest_distance

    def _low_seed_distances(self, seed: int) -> np.ndarray:
        key = int(seed)
        distances = self._distance_from_low.get(key)
        if distances is None:
            distances = np.linalg.norm(self.x - self.x[seed], axis=-1)
            self._distance_from_low[key] = distances
        return distances

    def cluster(self, td: float) -> Tuple[np.ndarray, np.ndarray]:
        condensation_points = np.zeros(self.n_points, dtype=np.int32)
        clustering = -1 * np.ones(self.n_points, dtype=np.int32)

        if self.high_beta.size > 0:
            self._prepare_nearest_high()
            assigned = self._nearest_high_distance < td
            clustering[assigned] = self._nearest_high_index[assigned]
            unassigned = np.nonzero(~assigned)[0]
        else:
            unassigned = np.arange(self.n_points)

        for seed in self.low_beta:
            if unassigned.size == 0:
                break
            distances = self._low_seed_distances(int(seed))[unassigned]
            selected = distances < td
            assigned_to_seed = unassigned[selected]
            clustering[assigned_to_seed] = seed
            unassigned = unassigned[~selected]

        clustering_indices, clustering_frequency = np.unique(
            clustering, return_counts=True
        )
        clustering_count = dict(zip(clustering_indices, clustering_frequency))
        charged_hits = self.charged_hits.astype(int)
        charged_index = clustering[charged_hits == 1]

        charged_is_cluster = np.isin(charged_index, clustering_indices)
        if not np.all(charged_is_cluster):
            print("charge index NOT found in cluster")

        remap_charged = {}
        for charged_i in charged_index:
            if charged_i == -1:
                print("track -1 skipped")
                continue
            if clustering_count[charged_i] == 1:
                charged_cluster_distance = {}
                for clustering_i in clustering_indices:
                    if clustering_i == charged_i:
                        continue
                    distance = np.linalg.norm(
                        self.x[charged_i] - self.x[clustering_i], axis=-1
                    )
                    charged_cluster_distance[clustering_i] = distance
                index_min = min(
                    charged_cluster_distance,
                    key=charged_cluster_distance.get,
                )
                remap_charged[charged_i] = index_min

        for old_id, new_id in remap_charged.items():
            clustering[clustering == old_id] = new_id

        _, condensation_indices = _oc.scatter_max(
            torch.from_numpy(self.betas.astype(np.int64)),
            torch.from_numpy(clustering.astype(np.int64)),
        )
        condensation_indices = condensation_indices.detach().numpy()
        valid = condensation_indices[condensation_indices < self.n_points]
        condensation_points[valid] = 1
        return clustering + 1, condensation_points


class _BetaClusterEngine:
    def __init__(self, cache: Sequence, tbeta: float):
        self.tbeta = float(tbeta)
        self.prepared = [
            None if prediction.pred_betas is None else _PreparedClustering(prediction, tbeta)
            for _, prediction in cache
        ]

    def cluster(self, event_index: int, td: float):
        prepared = self.prepared[event_index]
        if prepared is None:
            raise RuntimeError("Fast beta scan cannot cluster a Pandora prediction")
        return prepared.cluster(td)


class _CachedYielder:
    """Replacement resolved by the imported legacy writer in worker processes."""

    def __init__(self, *unused_args, **unused_kwargs):
        del unused_args, unused_kwargs

    def iter_matches(
        self,
        tbeta=0.7,
        td=0.5,
        nmax=None,
        energyRegression=False,
        energyRegressionCluster=False,
        clustering_td_momentum=False,
        energyRegressionWeight=False,
    ) -> Iterable:
        del energyRegression, energyRegressionCluster, energyRegressionWeight
        if clustering_td_momentum:
            raise RuntimeError("The fast cache does not support momentum-distance clustering")
        if _ACTIVE_CLUSTER_ENGINE is None:
            raise RuntimeError("No active beta clustering cache")
        if not np.isclose(float(tbeta), _ACTIVE_CLUSTER_ENGINE.tbeta):
            raise RuntimeError("Worker beta does not match its prepared cache")

        for event_index, (event, prediction) in enumerate(_PREDICTION_CACHE):
            if nmax is not None and event_index >= nmax:
                break
            if _GLOBAL_ARGS.pandora:
                clustering = np.array(event.pand[:, 0], dtype=int).flatten() + 1
                condensation_points = None
            else:
                clustering, condensation_points = _ACTIVE_CLUSTER_ENGINE.cluster(
                    event_index, td
                )
            matches = make_matches(
                event, prediction, clustering=clustering
            )
            yield event, prediction, clustering, matches, condensation_points


class _NoopModel:
    def to(self, unused_device):
        del unused_device
        return self


class _NoopDataset:
    def __init__(self, *unused_args, **unused_kwargs):
        del unused_args, unused_kwargs


def _install_worker_patches() -> None:
    _legacy.TFile = _BatchFile
    _legacy.TTree = _BatchTree
    _legacy.TestYielder = _CachedYielder
    _legacy.ILCDataset = _NoopDataset
    _legacy.get_model = lambda *args, **kwargs: _NoopModel()
    _legacy.get_model_branch = lambda *args, **kwargs: _NoopModel()
    _legacy.has_event_builder = lambda unused_path: _GLOBAL_EVENT_ENERGY


def _point_name(tbeta: float, td: float) -> str:
    return f"tbeta{round(tbeta * 100):03d}td{round(td * 100):03d}"


def _point_output(tbeta: float, td: float) -> Path:
    if _GLOBAL_ARGS.beta_d_scan:
        input_stem = Path(_GLOBAL_ARGS.datapath).stem
        return (
            Path(_GLOBAL_ARGS.outfile)
            / _point_name(tbeta, td)
            / f"{input_stem}.root"
        )
    return Path(_GLOBAL_ARGS.outfile)


def _run_beta_group(tbeta: float) -> List[str]:
    global _ACTIVE_CLUSTER_ENGINE
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _install_worker_patches()
    _ACTIVE_CLUSTER_ENGINE = _BetaClusterEngine(_PREDICTION_CACHE, tbeta)
    td_values = TD_VALUES if _GLOBAL_ARGS.beta_d_scan else (_GLOBAL_ARGS.td,)
    written = []

    worker_args = copy.copy(_GLOBAL_ARGS)
    worker_args.beta_d_scan = False
    # Predictions are already CPU NumPy arrays.  This also prevents a forked
    # worker from touching the parent's CUDA context.
    worker_args.device = "cpu"
    worker_args.tbeta = float(tbeta)

    for td in td_values:
        worker_args.td = float(td)
        output = _point_output(tbeta, td)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[fast-scan pid={os.getpid()}] "
            f"{_point_name(tbeta, td)} -> {output}",
            flush=True,
        )
        _legacy.save_root(
            _GLOBAL_ARGS.datapath,
            _GLOBAL_ARGS.ckpt,
            str(output),
            nstart=_GLOBAL_ARGS.nstart,
            nend=_GLOBAL_ARGS.nend,
            timingCut=_GLOBAL_ARGS.timingCut,
            input_dim=_GLOBAL_ARGS.input_dim,
            output_dim=_GLOBAL_ARGS.output_dim,
            args=worker_args,
        )
        written.append(str(output))

    _ACTIVE_CLUSTER_ENGINE = None
    gc.collect()
    return written


def _build_prediction_cache(args) -> Tuple[Tuple, bool]:
    event_energy = args.event_total_energy or _legacy.has_event_builder(args.datapath)
    if event_energy and not args.event_total_energy:
        print("Found event builder: storing truth q/qbar directions in the ROOT jet tree")

    input_dim = args.input_dim
    output_dim = args.output_dim
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
    if args.energy_branch:
        model = get_model_branch(
            args.ckpt, jit=False, input_dim=input_dim, output_dim=output_dim
        ).to(device)
    else:
        model = get_model(
            args.ckpt,
            jit=False,
            input_dim=input_dim,
            output_dim=output_dim,
            energy_regression=args.energy_regression,
            energy_regression_cluster=args.energy_regression_cluster,
            energy_regression_weight=args.energy_regression_weight,
            model_variant=args.model_variant,
        ).to(device)

    print(
        f"Loading data from {args.datapath} with "
        f"nstart={args.nstart}, nend={args.nend}, timingCut={args.timingCut}"
    )
    dataset = ILCDataset(
        args.datapath,
        timingCut=args.timingCut,
        thetaphi=thetaphi,
        test_mode=True,
        nstart=args.nstart,
        nend=args.nend,
        pandora=args.pandora,
        momentum=args.momentum,
        momentumAmp=args.momentum_amp,
        mctpe=args.mctpe,
        event_energy=event_energy,
    )
    yielder = TestYielder(
        model=model,
        dataset=dataset,
        device=device,
        pandora=args.pandora,
        event_energy=event_energy,
    )
    nmax = None if args.nend == -1 else args.nend - args.nstart + 1
    print("[fast-scan] Running exactly one GNN inference pass for this input...")
    cache = tuple(
        yielder.iter_pred(
            nmax=nmax,
            energyRegression=args.energy_regression,
            energyRegressionCluster=args.energy_regression_cluster,
            energyRegressionWeight=args.energy_regression_weight,
        )
    )
    print(f"[fast-scan] Cached {len(cache)} events in CPU memory")

    # Cached Event/Prediction objects contain the needed CPU arrays.  Drop the
    # model and loader before forking so workers do not retain GPU allocations.
    yielder.model = None
    yielder.loader = None
    del yielder, dataset, model
    gc.collect()
    if "cuda" in device:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return cache, event_energy


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fast one-inference beta/distance ROOT scan"
    )
    parser.add_argument("datapath")
    parser.add_argument("ckpt")
    parser.add_argument("outfile")
    parser.add_argument("nstart", type=int)
    parser.add_argument("nend", type=int)
    parser.add_argument("timingCut", type=lambda value: bool(_legacy.strtobool(value)))
    parser.add_argument("input_dim", type=int)
    parser.add_argument("output_dim", type=int)
    parser.add_argument("--pandora", action="store_true")
    parser.add_argument("--event-total-energy", action="store_true")
    parser.add_argument("--energy-regression", action="store_true")
    parser.add_argument("--energy-regression-cluster", action="store_true")
    parser.add_argument("--energy-regression-weight", action="store_true")
    parser.add_argument("-e", "--momentum", action="store_true")
    parser.add_argument("-ea", "--momentum-amp", action="store_true")
    parser.add_argument("--mctpe", action="store_true")
    parser.add_argument("-eb", "--energy-branch", action="store_true")
    parser.add_argument("--beta-d-scan", action="store_true")
    parser.add_argument("--tbeta", type=float, default=0.9)
    parser.add_argument("--td", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-variant",
        default="auto",
        choices=("auto", "legacy", "multihead"),
    )
    parser.add_argument("--truth-clustering", action="store_true")
    parser.add_argument("--1tomany-clustering", action="store_true")
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=int(os.environ.get("FAST_SCAN_WORKERS", "4")),
        help="CPU processes for independent beta groups (default: 4)",
    )
    args = parser.parse_args(argv)
    if args.scan_workers < 1:
        parser.error("--scan-workers must be at least 1")
    if args.beta_d_scan and str(args.outfile).endswith(".root"):
        parser.error("For --beta-d-scan, outfile must be an output directory")
    return args


def main(argv=None) -> None:
    global _PREDICTION_CACHE, _GLOBAL_ARGS, _GLOBAL_EVENT_ENERGY
    args = _parse_args(argv)
    if args.pandora and args.beta_d_scan:
        raise SystemExit(
            "--beta-d-scan is a GNN threshold scan and is not supported with --pandora"
        )

    _GLOBAL_ARGS = args
    _PREDICTION_CACHE, _GLOBAL_EVENT_ENERGY = _build_prediction_cache(args)
    _ensure_root_writer_loaded()

    beta_values = TBETA_VALUES if args.beta_d_scan else (args.tbeta,)
    workers = min(args.scan_workers, len(beta_values))
    print(
        f"[fast-scan] Starting {workers} CPU threshold worker(s); "
        "each beta worker reuses sorting/distances for all td values"
    )

    if workers == 1:
        results = [_run_beta_group(beta) for beta in beta_values]
    else:
        # Fork provides copy-on-write sharing of the CPU inference cache.  No
        # child calls CUDA; all GPU work was completed above.
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            results = pool.map(_run_beta_group, beta_values)

    count = sum(len(group) for group in results)
    print(f"[fast-scan] Completed {count} ROOT file(s)")


if __name__ == "__main__":
    main()
