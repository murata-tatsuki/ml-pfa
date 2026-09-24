import os
import sys
import glob
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import h5py
import numpy as np
from distutils.util import strtobool
#import evaluation_noNoise as ev
import awkward as ak
from model import get_model, get_model_branch
from cluster_energy import add_inference_arguments, inference_loader_kwargs, configure_inference
from dataset import ILCDataset
from test_yielder_edit import TestYielder
import ROOT
import argparse
import torch
from sed import minimum_enclosing_sphere
from clustering import cluster
from matching import make_matches, matching_1to1

## 1 to 1 match to reco-cluster and true cluster
## the largest edep_match reco-cluster is chosen

def to_numpy(x):
    """Convert torch/numpy-like arrays to numpy safely."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)

def has_event_builder(datapath):
    """Whether the input H5 data carries the small event-level builder."""
    filenames = [datapath] if datapath.endswith(".h5") else sorted(glob.glob(os.path.join(datapath, "*.h5")))
    if not filenames:
        return False
    try:
        with h5py.File(filenames[0], "r") as input_file:
            return "event" in input_file
    except OSError:
        return False


# Keep ROOT output memory bounded while removing one PyROOT call per entry.
# The existing C++ writer preserves the legacy five-tree schema and branch
# order.  Python buffers are flushed after a configurable number of events.
ROOT_WRITER_SOURCE = Path(__file__).resolve().parent / "scan_fast_root_writer.cxx"
MAX_COMPOSITION = 64
_ROOT_WRITER_LOADED = False
_ACTIVE_CHUNK_FILE = None
_ROOT_CHUNK_EVENTS = 10


def _ensure_root_writer_loaded():
    global _ROOT_WRITER_LOADED
    if _ROOT_WRITER_LOADED:
        return
    source = ROOT_WRITER_SOURCE.read_text()
    if not ROOT.gInterpreter.Declare(source):
        raise RuntimeError(f"Failed to load {ROOT_WRITER_SOURCE}")
    _ROOT_WRITER_LOADED = True


class _ChunkTree:
    """TTree-compatible Python collector flushed to ROOT in C++ chunks."""

    def __init__(self, name, title):
        if _ACTIVE_CHUNK_FILE is None:
            raise RuntimeError("TTree was created without an active output file")
        self.name = str(name)
        self.title = str(title)
        self._sources: Dict[str, np.ndarray] = {}
        self._leaflists: Dict[str, str] = {}
        self._columns: Dict[str, List] = {}
        self._column_chunks: Dict[str, List[np.ndarray]] = {}
        self._chunk_entries = 0
        _ACTIVE_CHUNK_FILE._add_tree(self)

    def Branch(self, name, source, leaflist):
        name = str(name)
        self._sources[name] = source
        self._leaflists[name] = str(leaflist)
        self._columns[name] = []
        self._column_chunks[name] = []
        return None

    def Fill(self):
        for name, source in self._sources.items():
            value = np.asarray(source)
            if "[" not in self._leaflists[name]:
                self._columns[name].append(value.reshape(-1)[0].item())
            else:
                self._columns[name].append(np.array(value, copy=True))
        return 1

    def extend_columns(self, columns: Dict[str, np.ndarray]):
        """Append a complete columnar block without a Python loop per hit."""
        expected = set(self._sources)
        actual = set(columns)
        if actual != expected:
            missing = sorted(expected.difference(actual))
            extra = sorted(actual.difference(expected))
            raise RuntimeError(
                f"Column mismatch for {self.name}: missing={missing}, extra={extra}"
            )
        if any(self._columns[name] for name in self._columns):
            raise RuntimeError(
                f"Cannot mix row Fill and column blocks in TTree {self.name}"
            )

        arrays = {name: np.asarray(value) for name, value in columns.items()}
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise RuntimeError(
                f"Column lengths differ for {self.name}: "
                f"{sorted((name, len(value)) for name, value in arrays.items())}"
            )
        nentries = lengths.pop() if lengths else 0
        for name in self._sources:
            # Own the memory until the next C++ append. This also ensures that
            # strided Torch/NumPy views do not leak into the writer boundary.
            self._column_chunks[name].append(np.array(arrays[name], copy=True))
        self._chunk_entries += nentries
        return nentries

    @property
    def entries(self):
        if not self._columns:
            return 0
        row_entries = len(next(iter(self._columns.values())))
        return row_entries + self._chunk_entries

    def clear(self):
        for column in self._columns.values():
            column.clear()
        for chunks in self._column_chunks.values():
            chunks.clear()
        self._chunk_entries = 0

    def _column_array(self, name, dtype):
        chunks = self._column_chunks[name]
        if chunks:
            if self._columns[name]:
                raise RuntimeError(
                    f"Mixed row and column storage in TTree {self.name}/{name}"
                )
            if len(chunks) == 1:
                return np.asarray(chunks[0], dtype=dtype)
            return np.concatenate(
                [np.asarray(chunk, dtype=dtype) for chunk in chunks]
            )
        return np.asarray(self._columns[name], dtype=dtype)

    def scalar_matrix(self, names: Sequence[str], dtype):
        if self.entries == 0:
            return np.empty((0, len(names)), dtype=dtype)
        columns = [self._column_array(name, dtype) for name in names]
        return np.ascontiguousarray(np.column_stack(columns), dtype=dtype)

    def array_blocks(self, names: Sequence[str], dtype, width):
        if self.entries == 0:
            return np.empty((0, len(names) * width), dtype=dtype)
        blocks = []
        for name in names:
            block = self._column_array(name, dtype).reshape(
                self.entries, -1
            )
            if block.shape[1] < width:
                raise RuntimeError(
                    f"Branch {self.name}/{name} has width {block.shape[1]}, "
                    f"expected at least {width}"
                )
            blocks.append(block[:, :width])
        return np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=dtype)


def _addresses(*arrays):
    return tuple(int(array.ctypes.data) for array in arrays)


class _ChunkFile:
    """TFile-compatible streaming writer with bounded Python buffers."""

    def __init__(self, filename, mode="RECREATE", *unused_args, **unused_kwargs):
        del unused_args, unused_kwargs
        global _ACTIVE_CHUNK_FILE
        if str(mode).lower() not in ("recreate", "create", "new"):
            raise ValueError(f"Unsupported chunk-writer mode: {mode}")
        _ensure_root_writer_loaded()
        self.filename = str(filename)
        self.trees: Dict[str, _ChunkTree] = {}
        self.events_in_chunk = 0
        self._closed = False
        compression_level = int(os.environ.get("CEDRIC_ROOT_LZ4_LEVEL", "1"))
        self.writer = ROOT.ScanFastRootWriter(
            self.filename, compression_level
        )
        _ACTIVE_CHUNK_FILE = self

    def _add_tree(self, tree):
        self.trees[tree.name] = tree

    def finish_event(self):
        self.events_in_chunk += 1
        if self.events_in_chunk >= _ROOT_CHUNK_EVENTS:
            self.flush()

    def flush(self):
        if self._closed:
            return
        if not self.trees or not any(tree.entries for tree in self.trees.values()):
            self.events_in_chunk = 0
            return

        expected = {"t", "reco", "prediction", "event", "jet"}
        missing = expected.difference(self.trees)
        if missing:
            raise RuntimeError(
                f"Missing trees for {self.filename}: {sorted(missing)}"
            )

        t = self.trees["t"]
        t_int = t.scalar_matrix(
            (
                "event", "hitid", "mcid", "truthid", "mcpdg", "mccharge",
                "mcstatus", "ncluster", "matched_ncluster", "matched_cluster",
                "cond_track",
            ),
            np.int32,
        )
        t_double = t.scalar_matrix(
            (
                "mcmass", "mcpx", "mcpy", "mcpz", "mcen", "edep",
                "edep_reco", "edep_match", "pred_edep", "pred_edep_cluster",
                "pred_edep_weight", "cond_beta", "sed_radius",
                "pred_photon_energy", "pred_charged_hadron_energy",
                "pred_neutral_hadron_energy", "pred_muon_energy",
                "pred_electron_energy",
            ),
            np.float64,
        )
        self.writer.append_t(t.entries, *_addresses(t_int, t_double))

        reco = self.trees["reco"]
        reco_int = reco.scalar_matrix(
            (
                "event", "cluster", "nhits", "mcid", "mcpdg", "mccharge",
                "mcstatus", "ntrack_hits", "cond_is_track",
                "matched_truth_pdgid", "npdg_comp",
            ),
            np.int32,
        )
        reco_double = reco.scalar_matrix(
            (
                "mcmass", "mcpx", "mcpy", "mcpz", "mcen", "edep_reco",
                "edep_mc", "edep_match", "pred_edep", "pred_edep_cluster",
                "cond_beta", "matched_truth_hit_frac",
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
                "pdg_comp_hit_frac", "pdg_comp_edep_frac", "pdg_comp_edep",
                "pdg_comp_truth_edep",
            ),
            np.float64,
            MAX_COMPOSITION,
        )
        self.writer.append_reco(
            reco.entries,
            *_addresses(
                reco_int, reco_double, reco_comp_int, reco_comp_double
            ),
        )

        prediction = self.trees["prediction"]
        prediction_int = prediction.scalar_matrix(
            (
                "event", "hitid", "mcid", "truthid", "cluster", "mcpdg",
                "mccharge", "mcstatus", "pred_alpha", "trackness",
            ),
            np.int32,
        )
        prediction_double = prediction.scalar_matrix(
            (
                "mcmass", "mcpx", "mcpy", "mcpz", "mcen", "edep_mc",
                "pred_edep", "pred_edep_cluster", "pred_beta",
                "weight_photon", "weight_charged_hadron",
                "weight_neutral_hadron", "weight_muon", "weight_electron",
            ),
            np.float64,
        )
        self.writer.append_prediction(
            prediction.entries,
            *_addresses(prediction_int, prediction_double),
        )

        event = self.trees["event"]
        event_int = event.scalar_matrix(("event", "ncluster"), np.int32)
        event_double = event.scalar_matrix(
            (
                "MC_dijet_energy", "total_MC_energy_truth",
                "total_MC_energy_pred", "total_predicted_energy_truth",
                "total_predicted_energy_pred",
            ),
            np.float64,
        )
        self.writer.append_event(
            event.entries, *_addresses(event_int, event_double)
        )

        jet = self.trees["jet"]
        jet_int = jet.scalar_matrix(("event", "n_jets"), np.int32)
        jet_double = jet.array_blocks(("jet_p4",), np.float64, 8)
        self.writer.append_jet(jet.entries, *_addresses(jet_int, jet_double))

        for tree in self.trees.values():
            tree.clear()
        self.events_in_chunk = 0

    def Write(self):
        if self._closed:
            return 0
        self.flush()
        self.writer.close()
        self._closed = True
        return 1

    def Close(self):
        return self.Write()


# The unchanged physics-filling section below uses these TFile/TTree names.
TFile = _ChunkFile
TTree = _ChunkTree


class _PandoraNoopModel:
    """Minimal model interface used because Pandora performs no inference."""

    def eval(self):
        return self


class _RootValidation(NamedTuple):
    valid: bool
    reason: str
    counts: Tuple[int, int, int]


_TREE_SCHEMAS = {
    "t": (
        ("event", "Int_t"), ("hitid", "Int_t"), ("mcid", "Int_t"),
        ("truthid", "Int_t"), ("mcpdg", "Int_t"),
        ("mccharge", "Int_t"), ("mcmass", "Double_t"),
        ("mcpx", "Double_t"), ("mcpy", "Double_t"),
        ("mcpz", "Double_t"), ("mcen", "Double_t"),
        ("mcstatus", "Int_t"), ("edep", "Double_t"),
        ("edep_reco", "Double_t"), ("edep_match", "Double_t"),
        ("ncluster", "Int_t"), ("matched_ncluster", "Int_t"),
        ("matched_cluster", "Int_t"), ("pred_edep", "Double_t"),
        ("pred_edep_cluster", "Double_t"),
        ("pred_edep_weight", "Double_t"), ("cond_beta", "Double_t"),
        ("cond_track", "Int_t"), ("sed_radius", "Double_t"),
        ("pred_photon_energy", "Double_t"),
        ("pred_charged_hadron_energy", "Double_t"),
        ("pred_neutral_hadron_energy", "Double_t"),
        ("pred_muon_energy", "Double_t"),
        ("pred_electron_energy", "Double_t"),
    ),
    "reco": (
        ("event", "Int_t"), ("cluster", "Int_t"), ("nhits", "Int_t"),
        ("mcid", "Int_t"), ("mcpdg", "Int_t"),
        ("mccharge", "Int_t"), ("mcmass", "Double_t"),
        ("mcpx", "Double_t"), ("mcpy", "Double_t"),
        ("mcpz", "Double_t"), ("mcen", "Double_t"),
        ("mcstatus", "Int_t"), ("edep_reco", "Double_t"),
        ("edep_mc", "Double_t"), ("edep_match", "Double_t"),
        ("pred_edep", "Double_t"), ("pred_edep_cluster", "Double_t"),
        ("ntrack_hits", "Int_t"), ("cond_beta", "Double_t"),
        ("cond_is_track", "Int_t"), ("matched_truth_pdgid", "Int_t"),
        ("matched_truth_hit_frac", "Double_t"),
        ("matched_truth_edep_frac", "Double_t"),
        ("npdg_comp", "Int_t"), ("pdg_comp_ids", "Int_t"),
        ("pdg_comp_hits", "Int_t"),
        ("pdg_comp_hit_frac", "Double_t"),
        ("pdg_comp_edep_frac", "Double_t"),
        ("pdg_comp_edep", "Double_t"),
        ("pdg_comp_truth_edep", "Double_t"),
        ("pdg_comp_track_hits", "Int_t"),
    ),
    "prediction": (
        ("event", "Int_t"), ("hitid", "Int_t"), ("mcid", "Int_t"),
        ("truthid", "Int_t"), ("cluster", "Int_t"),
        ("mcpdg", "Int_t"), ("mccharge", "Int_t"),
        ("mcmass", "Double_t"), ("mcpx", "Double_t"),
        ("mcpy", "Double_t"), ("mcpz", "Double_t"),
        ("mcen", "Double_t"), ("mcstatus", "Int_t"),
        ("edep_mc", "Double_t"), ("pred_edep", "Double_t"),
        ("pred_edep_cluster", "Double_t"), ("pred_beta", "Double_t"),
        ("pred_alpha", "Int_t"), ("trackness", "Int_t"),
        ("weight_photon", "Double_t"),
        ("weight_charged_hadron", "Double_t"),
        ("weight_neutral_hadron", "Double_t"),
        ("weight_muon", "Double_t"), ("weight_electron", "Double_t"),
    ),
    "event": (
        ("event", "Int_t"), ("ncluster", "Int_t"),
        ("MC_dijet_energy", "Double_t"),
        ("total_MC_energy_truth", "Double_t"),
        ("total_MC_energy_pred", "Double_t"),
        ("total_predicted_energy_truth", "Double_t"),
        ("total_predicted_energy_pred", "Double_t"),
    ),
    "jet": (
        ("event", "Int_t"), ("n_jets", "Int_t"),
        ("jet_p4", "Double_t"),
    ),
}


def _validate_root(
    path: Path,
    expected_counts: Optional[Tuple[int, int, int]] = None,
) -> _RootValidation:
    """Validate the five-tree schema and force-read each final basket."""
    try:
        if not path.is_file():
            return _RootValidation(False, "missing", (-1, -1, -1))
        if path.stat().st_size == 0:
            return _RootValidation(False, "empty file", (-1, -1, -1))
    except OSError as error:
        return _RootValidation(False, f"cannot stat: {error}", (-1, -1, -1))

    try:
        root_file = ROOT.TFile.Open(str(path), "READ")
    except (OSError, RuntimeError) as error:
        return _RootValidation(False, f"cannot open: {error}", (-1, -1, -1))
    if not root_file or root_file.IsZombie():
        if root_file:
            root_file.Close()
        return _RootValidation(False, "cannot open or zombie", (-1, -1, -1))

    try:
        recovered_bit = getattr(ROOT.TFile, "kRecovered", None)
        if recovered_bit is not None and root_file.TestBit(recovered_bit):
            return _RootValidation(
                False, "ROOT recovered file", (-1, -1, -1)
            )

        entries = {}
        for tree_name, schema in _TREE_SCHEMAS.items():
            tree = root_file.Get(tree_name)
            if not tree or not tree.InheritsFrom("TTree"):
                return _RootValidation(
                    False, f"missing TTree {tree_name}", (-1, -1, -1)
                )

            expected_names = [name for name, _ in schema]
            actual_names = [
                branch.GetName() for branch in tree.GetListOfBranches()
            ]
            if actual_names != expected_names:
                return _RootValidation(
                    False,
                    f"branch layout mismatch in {tree_name}",
                    (-1, -1, -1),
                )

            for branch_name, expected_type in schema:
                branch = tree.GetBranch(branch_name)
                leaf = branch.GetLeaf(branch_name) if branch else None
                if not leaf or leaf.GetTypeName() != expected_type:
                    return _RootValidation(
                        False,
                        f"type mismatch in {tree_name}/{branch_name}",
                        (-1, -1, -1),
                    )

            nentries = int(tree.GetEntries())
            entries[tree_name] = nentries
            if nentries > 0 and tree.GetEntry(nentries - 1) < 0:
                return _RootValidation(
                    False,
                    f"cannot read final entry of {tree_name}",
                    (-1, -1, -1),
                )

        counts = (
            entries["event"], entries["prediction"], entries["jet"]
        )
        if expected_counts is not None and counts != expected_counts:
            return _RootValidation(
                False,
                f"entry counts {counts} != expected {expected_counts}",
                counts,
            )
        if entries["event"] <= 0 or entries["prediction"] <= 0:
            return _RootValidation(False, "no event/hit entries", counts)
        return _RootValidation(True, "ok", counts)
    finally:
        root_file.Close()


def _expected_root_counts(
    datapath: str,
    nstart: int,
    nend: int,
    timing_cut: bool,
    event_energy: bool,
) -> Optional[Tuple[int, int, int]]:
    """Read only small event-offset arrays to predict invariant entries."""
    if timing_cut:
        # The timing/omega cut changes hit counts and currently requires the
        # full feature content.  Structural validation is still performed.
        return None

    path = Path(datapath).expanduser()
    filenames = [path] if path.suffix == ".h5" else sorted(path.glob("*.h5"))
    if not filenames:
        return None

    hit_counts = []
    try:
        for filename in filenames:
            with h5py.File(filename, "r") as input_file:
                feature = input_file["feature"]
                offsets = np.asarray(feature["node0-offsets"], dtype=np.int64)
                if offsets.ndim != 1 or offsets.size < 1:
                    return None
                hit_counts.append(np.diff(offsets))
    except (OSError, KeyError, ValueError):
        return None

    counts = np.concatenate(hit_counts) if hit_counts else np.empty(0, np.int64)
    total_events = len(counts)
    effective_nend = (
        total_events
        if nend < 0 or total_events < nend
        else nend
    )
    # This mirrors the existing ILCDataset slice exactly.
    selected = counts[nstart:nstart + effective_nend]
    # ILCDataset removes empty events; TestYielder additionally skips <50 hits.
    selected = selected[selected >= 50]
    event_entries = int(len(selected))
    prediction_entries = int(np.sum(selected, dtype=np.int64))
    jet_entries = event_entries if event_energy else 0
    return event_entries, prediction_entries, jet_entries


def _temporary_output(final_output: Path) -> Path:
    return final_output.with_name(
        f".{final_output.name}.part.{os.getpid()}.{time.time_ns()}.root"
    )


def _resolve_multi_h5(spec: str) -> Tuple[Path, ...]:
    """Resolve a directory, quoted glob, comma-list, or @text-file."""
    if spec.startswith("@"):
        list_path = Path(spec[1:]).expanduser()
        candidates = [
            Path(line.strip()).expanduser()
            for line in list_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    elif "," in spec:
        candidates = [
            Path(item.strip()).expanduser()
            for item in spec.split(",")
            if item.strip()
        ]
    else:
        expanded = Path(spec).expanduser()
        if expanded.is_dir():
            candidates = sorted(expanded.glob("*.h5"))
        elif glob.has_magic(str(expanded)):
            candidates = [Path(item) for item in sorted(glob.glob(str(expanded)))]
        else:
            candidates = [expanded]

    resolved = []
    seen = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"H5 input does not exist: {path}")
        if path.suffix != ".h5":
            raise ValueError(f"Input is not an .h5 file: {path}")
        seen.add(path)
        resolved.append(path)
    if not resolved:
        raise FileNotFoundError(f"No H5 input matched: {spec}")
    return tuple(resolved)


class _PrefetchFailure(NamedTuple):
    error: BaseException
    formatted_traceback: str


def _ordered_prefetch(iterable, capacity: int, device: str):
    """Run an iterator ahead in one producer thread while preserving order."""
    channel = queue.Queue(maxsize=max(1, int(capacity)))
    stop = threading.Event()
    completed = object()

    def put(value):
        while not stop.is_set():
            try:
                channel.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        try:
            if "cuda" in device:
                # The current CUDA device is thread-local in some PyTorch/CUDA
                # combinations, so initialise it explicitly in the producer.
                torch.cuda.set_device(device)
            for value in iterable:
                if not put(value):
                    return
        except BaseException as error:
            put(_PrefetchFailure(error, traceback.format_exc()))
        finally:
            put(completed)

    producer = threading.Thread(
        target=produce,
        name="cedric-gnn-prefetch",
        daemon=True,
    )
    producer.start()
    try:
        while True:
            value = channel.get()
            if value is completed:
                break
            if isinstance(value, _PrefetchFailure):
                raise RuntimeError(
                    "GNN prefetch producer failed:\n"
                    f"{value.formatted_traceback}"
                ) from value.error
            yield value
    finally:
        stop.set()


def _iter_analysis_inputs(
    yielder,
    tbeta,
    td,
    nmax,
    pandora,
    energy_regression,
    energy_regression_cluster,
    energy_regression_weight,
    pipeline_depth,
    device,
):
    """Overlap the next GNN batch with ordered CPU clustering/analysis."""
    predictions = yielder.iter_pred(
        nmax,
        energy_regression,
        energy_regression_cluster,
        energy_regression_weight,
    )
    if not pandora and "cuda" in device and pipeline_depth > 0:
        print(
            f"GNN/CPU ordered pipeline enabled: {pipeline_depth} "
            "prefetched event(s)"
        )
        predictions = _ordered_prefetch(predictions, pipeline_depth, device)

    for event, prediction in predictions:
        if not pandora:
            clustering, condensation_points = cluster(
                event, prediction, tbeta, td, False
            )
        else:
            clustering = (
                np.array(event.pand[:, 0], dtype=int).flatten() + 1
            )
            condensation_points = None
        matches = make_matches(
            event, prediction, clustering=clustering
        )
        yield event, prediction, clustering, matches, condensation_points


def _optional_numpy(value):
    return None if value is None else to_numpy(value)


class _EventNumpyCache:
    """One conversion and one reusable mask/statistics set per event."""

    def __init__(self, event, prediction, clustering):
        self.truth = np.asarray(to_numpy(event.y[:, 0]))
        self.hitid = np.asarray(to_numpy(event.hitid))
        self.mcid = np.asarray(to_numpy(event.mcid))
        self.feat = np.asarray(to_numpy(event.feat))
        self.label = np.asarray(to_numpy(event.label))
        self.clustering = np.asarray(to_numpy(clustering)).reshape(-1)
        self.edep = self.feat[:, 0].astype(np.float64)
        self.track_float = self.feat[:, 5].astype(np.float64)
        self.track_int = self.feat[:, 5].astype(np.int32)
        self.jet = _optional_numpy(event.jet)
        self.event = _optional_numpy(event.event)

        # Keep the exact legacy iteration construction. In particular, do not
        # silently change TTree row ordering by replacing set(...) here.
        self.truth_ids = list(set(np.unique(self.truth)))
        self.mcids = list(set(np.unique(self.mcid)))
        self.cluster_ids = list(set(np.unique(self.clustering)))
        self.truth_masks = {
            truth_id: self.truth == truth_id
            for truth_id in self.truth_ids
        }
        self.cluster_masks = {
            cluster_id: self.clustering == cluster_id
            for cluster_id in self.cluster_ids
        }
        self.truth_edep = {
            truth_id: np.sum(self.edep[mask])
            for truth_id, mask in self.truth_masks.items()
        }
        self.cluster_edep = {
            cluster_id: np.sum(self.edep[mask])
            for cluster_id, mask in self.cluster_masks.items()
        }
        self._pair_edep = {}
        self._beta_orders = {}

        self.pred_betas = _optional_numpy(prediction.pred_betas)
        self.pred_tracker_energy = _optional_numpy(
            prediction.pred_tracker_energy
        )
        self.pred_cluster_energy = _optional_numpy(
            prediction.pred_cluster_energy
        )
        self.pred_weight_photon = _optional_numpy(
            prediction.pred_weight_photon
        )
        self.pred_weight_charged_hadron = _optional_numpy(
            prediction.pred_weight_charged_hadron
        )
        self.pred_weight_neutral_hadron = _optional_numpy(
            prediction.pred_weight_neutral_hadron
        )
        self.pred_weight_muon = _optional_numpy(
            prediction.pred_weight_muon
        )
        self.pred_weight_electron = _optional_numpy(
            prediction.pred_weight_electron
        )

    def pair_edep(self, truth_id, cluster_id):
        key = (truth_id, cluster_id)
        if key not in self._pair_edep:
            mask = np.logical_and(
                self.truth_masks[truth_id],
                self.cluster_mask(cluster_id),
            )
            self._pair_edep[key] = np.sum(self.edep[mask])
        return self._pair_edep[key]

    def cluster_mask(self, cluster_id):
        # --truth-clustering keeps the legacy matches computed before the
        # clustering override. Some matched IDs can therefore be absent from
        # the overridden clustering; an all-false mask is the legacy result.
        if cluster_id not in self.cluster_masks:
            self.cluster_masks[cluster_id] = self.clustering == cluster_id
        return self.cluster_masks[cluster_id]

    def cluster_edep_sum(self, cluster_id):
        if cluster_id not in self.cluster_edep:
            self.cluster_edep[cluster_id] = np.sum(
                self.edep[self.cluster_mask(cluster_id)]
            )
        return self.cluster_edep[cluster_id]

    def beta_order(self, cluster_id):
        if cluster_id not in self._beta_orders:
            beta = self.pred_betas[self.cluster_mask(cluster_id)]
            self._beta_orders[cluster_id] = np.argsort(-beta)
        return self._beta_orders[cluster_id]


def _append_prediction_columns(
    tree,
    event_number,
    cache,
    condensation_points,
    pandora,
    energy_regression,
    energy_regression_cluster,
    energy_regression_weight,
):
    """Build the hit-level prediction tree as one columnar event block."""
    n_hits = len(cache.truth)
    label = cache.label
    mass = label[:, 4].astype(np.float64)
    px = label[:, 5].astype(np.float64)
    py = label[:, 6].astype(np.float64)
    pz = label[:, 7].astype(np.float64)
    energy = np.sqrt(mass**2 + px**2 + py**2 + pz**2)
    zeros_double = np.zeros(n_hits, dtype=np.float64)

    if pandora:
        tracker_energy = np.asarray(
            cache.pred_tracker_energy, dtype=np.float64
        )
        pred_edep = tracker_energy
        pred_edep_cluster = tracker_energy
        pred_beta = zeros_double
        pred_alpha = np.zeros(n_hits, dtype=np.int32)
    else:
        pred_edep = (
            np.asarray(cache.pred_tracker_energy, dtype=np.float64)
            if energy_regression and not energy_regression_weight
            else np.full(n_hits, -1.0, dtype=np.float64)
        )
        pred_edep_cluster = (
            np.asarray(cache.pred_cluster_energy, dtype=np.float64)
            if energy_regression and energy_regression_cluster
            else np.full(n_hits, -1.0, dtype=np.float64)
        )
        pred_beta = np.asarray(cache.pred_betas, dtype=np.float64)
        pred_alpha = np.asarray(condensation_points, dtype=np.int32)

    def weight(value):
        if energy_regression_weight and not pandora:
            return np.asarray(value, dtype=np.float64)
        return zeros_double

    tree.extend_columns(
        {
            "event": np.full(n_hits, event_number, dtype=np.int32),
            "hitid": cache.hitid.astype(np.int32),
            "mcid": cache.mcid.astype(np.int32),
            "truthid": cache.truth.astype(np.int32),
            "cluster": cache.clustering.astype(np.int32),
            "mcpdg": label[:, 2].astype(np.int32),
            "mccharge": label[:, 3].astype(np.int32),
            "mcmass": mass,
            "mcpx": px,
            "mcpy": py,
            "mcpz": pz,
            "mcen": energy,
            "mcstatus": label[:, 8].astype(np.int32),
            "edep_mc": cache.edep,
            "pred_edep": pred_edep,
            "pred_edep_cluster": pred_edep_cluster,
            "pred_beta": pred_beta,
            "pred_alpha": pred_alpha,
            "trackness": cache.feat[:, 5].astype(np.int32),
            "weight_photon": weight(cache.pred_weight_photon),
            "weight_charged_hadron": weight(
                cache.pred_weight_charged_hadron
            ),
            "weight_neutral_hadron": weight(
                cache.pred_weight_neutral_hadron
            ),
            "weight_muon": weight(cache.pred_weight_muon),
            "weight_electron": weight(cache.pred_weight_electron),
        }
    )

class Data:
    ''' TTree data for MCParticle
        to be used for evaluating the efficiency
    '''
    event = np.array([0], dtype=np.int32)
    hitid = np.array([0], dtype=np.int32)
    mcid = np.array([0], dtype=np.int32)
    truthid = np.array([0], dtype=np.int32)
    mcpdg = np.array([0], dtype=np.int32)
    mccharge = np.array([0], dtype=np.int32)
    mcmass = np.array([0], dtype=np.float64)
    mcpx = np.array([0], dtype=np.float64)
    mcpy = np.array([0], dtype=np.float64)
    mcpz = np.array([0], dtype=np.float64)
    mcen = np.array([0], dtype=np.float64)
    mcstatus = np.array([0], dtype=np.int32)
    edep = np.array([0], dtype=np.float64)
    edep_reco = np.array([0], dtype=np.float64)
    edep_match = np.array([0], dtype=np.float64)
    ncluster = np.array([0], dtype=np.int32)
    matched_ncluster = np.array([0], dtype=np.int32)
    matched_cluster = np.array([0], dtype=np.int32)
    pred_edep = np.array([0], dtype=np.float64)
    pred_edep_cluster = np.array([0], dtype=np.float64)
    pred_edep_weight = np.array([0], dtype=np.float64)
    cond_beta = np.array([0], dtype=np.float64)
    cond_track = np.array([0], dtype=np.int32)
    sed_radius = np.array([0], dtype=np.float64)    # smallest enclosing disk radius

    pred_photon_energy = np.array([0], dtype=np.float64)
    pred_charged_hadron_energy = np.array([0], dtype=np.float64)
    pred_neutral_hadron_energy = np.array([0], dtype=np.float64)
    pred_muon_energy = np.array([0], dtype=np.float64)
    pred_electron_energy = np.array([0], dtype=np.float64)

    def setup_branch(this,t):
        t.Branch("event",this.event,"event/I")
        t.Branch("hitid",this.hitid,"hitid/I")
        t.Branch("mcid",this.mcid,"mcid/I")
        t.Branch("truthid",this.truthid,"truthid/I")
        t.Branch("mcpdg",this.mcpdg,"mcpdg/I")
        t.Branch("mccharge",this.mccharge,"mccharge/I")
        t.Branch("mcmass",this.mcmass,"mcmass/D")
        t.Branch("mcpx",this.mcpx,"mcpx/D")
        t.Branch("mcpy",this.mcpy,"mcpy/D")
        t.Branch("mcpz",this.mcpz,"mcpz/D")
        t.Branch("mcen",this.mcen,"mcen/D")
        t.Branch("mcstatus",this.mcstatus,"mcstatus/I")
        t.Branch("edep",this.edep,"edep/D")
        t.Branch("edep_reco",this.edep_reco,"edep_reco/D")
        t.Branch("edep_match",this.edep_match,"edep_match/D")
        t.Branch("ncluster",this.matched_ncluster,"ncluster/I")
        t.Branch("matched_ncluster",this.matched_ncluster,"matched_ncluster/I")
        t.Branch("matched_cluster",this.matched_cluster,"matched_cluster/I")
        t.Branch("pred_edep",this.pred_edep,"pred_edep/D")
        t.Branch("pred_edep_cluster",this.pred_edep_cluster,"pred_edep_cluster/D")
        t.Branch("pred_edep_weight",this.pred_edep_weight,"pred_edep_weight/D")
        t.Branch("cond_beta",this.cond_beta,"cond_beta/D")
        t.Branch("cond_track",this.cond_track,"cond_track/I")
        t.Branch("sed_radius",this.sed_radius,"sed_radius/D")

        t.Branch("pred_photon_energy",this.pred_photon_energy,"pred_photon_energy/D")
        t.Branch("pred_charged_hadron_energy",this.pred_charged_hadron_energy,"pred_charged_hadron_energy/D")
        t.Branch("pred_neutral_hadron_energy",this.pred_neutral_hadron_energy,"pred_neutral_hadron_energy/D")
        t.Branch("pred_muon_energy",this.pred_muon_energy,"pred_muon_energy/D")
        t.Branch("pred_electron_energy",this.pred_electron_energy,"pred_electron_energy/D")

#   ak_feat: edep, x, y, z, time, track, charge, px, py, pz (atcalo)
#   ak_label: hitid, mcid, pdg, charge, mass, px, py, pz (of mcp), status

class RecoData:
    ''' TTree data for reconstructed cluster
        to be used for evaluating the purity
    '''
    event = np.array([0], dtype=np.int32)
    cluster = np.array([0], dtype=np.int32)
    nhits = np.array([0], dtype=np.int32)
    mcid = np.array([0], dtype=np.int32)
    mcpdg = np.array([0], dtype=np.int32)
    mccharge = np.array([0], dtype=np.int32)
    mcmass = np.array([0], dtype=np.float64)
    mcpx = np.array([0], dtype=np.float64)
    mcpy = np.array([0], dtype=np.float64)
    mcpz = np.array([0], dtype=np.float64)
    mcen = np.array([0], dtype=np.float64)
    mcstatus = np.array([0], dtype=np.int32)
    edep_reco = np.array([0], dtype=np.float64)
    edep_mc = np.array([0], dtype=np.float64)
    edep_match = np.array([0], dtype=np.float64)
    pred_edep = np.array([0], dtype=np.float64)
    pred_edep_cluster = np.array([0], dtype=np.float64)
    ntrack_hits = np.array([0], dtype=np.int32)
    cond_beta = np.array([0], dtype=np.float64)
    cond_is_track = np.array([0], dtype=np.int32)
    matched_truth_pdgid = np.array([0], dtype=np.int32)
    matched_truth_hit_frac = np.array([0], dtype=np.float64)
    matched_truth_edep_frac = np.array([0], dtype=np.float64)
    npdg_comp = np.array([0], dtype=np.int32)
    pdg_comp_ids = np.zeros(64, dtype=np.int32)
    pdg_comp_hits = np.zeros(64, dtype=np.int32)
    pdg_comp_hit_frac = np.zeros(64, dtype=np.float64)
    pdg_comp_edep_frac = np.zeros(64, dtype=np.float64)
    pdg_comp_edep = np.zeros(64, dtype=np.float64)
    pdg_comp_truth_edep = np.zeros(64, dtype=np.float64)
    pdg_comp_track_hits = np.zeros(64, dtype=np.int32)

    def setup_branch(this,t):
        t.Branch("event",this.event,"event/I")
        t.Branch("cluster",this.cluster,"cluster/I")
        t.Branch("nhits",this.nhits,"nhits/I")
        t.Branch("mcid",this.mcid,"mcid/I")
        t.Branch("mcpdg",this.mcpdg,"mcpdg/I")
        t.Branch("mccharge",this.mccharge,"mccharge/I")
        t.Branch("mcmass",this.mcmass,"mcmass/D")
        t.Branch("mcpx",this.mcpx,"mcpx/D")
        t.Branch("mcpy",this.mcpy,"mcpy/D")
        t.Branch("mcpz",this.mcpz,"mcpz/D")
        t.Branch("mcen",this.mcen,"mcen/D")
        t.Branch("mcstatus",this.mcstatus,"mcstatus/I")
        t.Branch("edep_reco",this.edep_reco,"edep_reco/D")
        t.Branch("edep_mc",this.edep_mc,"edep_mc/D")
        t.Branch("edep_match",this.edep_match,"edep_match/D")
        t.Branch("pred_edep",this.pred_edep,"pred_edep/D")
        t.Branch("pred_edep_cluster",this.pred_edep_cluster,"pred_edep_cluster/D")
        t.Branch("ntrack_hits",this.ntrack_hits,"ntrack_hits/I")
        t.Branch("cond_beta",this.cond_beta,"cond_beta/D")
        t.Branch("cond_is_track",this.cond_is_track,"cond_is_track/I")
        t.Branch("matched_truth_pdgid",this.matched_truth_pdgid,"matched_truth_pdgid/I")
        t.Branch("matched_truth_hit_frac",this.matched_truth_hit_frac,"matched_truth_hit_frac/D")
        t.Branch("matched_truth_edep_frac",this.matched_truth_edep_frac,"matched_truth_edep_frac/D")
        t.Branch("npdg_comp",this.npdg_comp,"npdg_comp/I")
        t.Branch("pdg_comp_ids",this.pdg_comp_ids,"pdg_comp_ids[npdg_comp]/I")
        t.Branch("pdg_comp_hits",this.pdg_comp_hits,"pdg_comp_hits[npdg_comp]/I")
        t.Branch("pdg_comp_hit_frac",this.pdg_comp_hit_frac,"pdg_comp_hit_frac[npdg_comp]/D")
        t.Branch("pdg_comp_edep_frac",this.pdg_comp_edep_frac,"pdg_comp_edep_frac[npdg_comp]/D")
        t.Branch("pdg_comp_edep",this.pdg_comp_edep,"pdg_comp_edep[npdg_comp]/D")
        t.Branch("pdg_comp_truth_edep",this.pdg_comp_truth_edep,"pdg_comp_truth_edep[npdg_comp]/D")
        t.Branch("pdg_comp_track_hits",this.pdg_comp_track_hits,"pdg_comp_track_hits[npdg_comp]/I")

class PredData:
    ''' TTree data for predicted (output of GravNet)
        to be used for evaluating the purity
    '''
    event = np.array([0], dtype=np.int32)
    hitid = np.array([0], dtype=np.int32)
    mcid = np.array([0], dtype=np.int32)
    truthid = np.array([0], dtype=np.int32)
    cluster = np.array([0], dtype=np.int32)
    mcpdg = np.array([0], dtype=np.int32)
    mccharge = np.array([0], dtype=np.int32)
    mcmass = np.array([0], dtype=np.float64)
    mcpx = np.array([0], dtype=np.float64)
    mcpy = np.array([0], dtype=np.float64)
    mcpz = np.array([0], dtype=np.float64)
    mcen = np.array([0], dtype=np.float64)
    mcstatus = np.array([0], dtype=np.int32)
    edep_mc = np.array([0], dtype=np.float64)
    pred_edep = np.array([0], dtype=np.float64)
    pred_edep_cluster = np.array([0], dtype=np.float64)
    pred_beta = np.array([0], dtype=np.float64)
    pred_alpha = np.array([0], dtype=np.int32)
    trackness = np.array([0], dtype=np.int32)
    weight_photon = np.array([0], dtype=np.float64)
    weight_charged_hadron = np.array([0], dtype=np.float64)
    weight_neutral_hadron = np.array([0], dtype=np.float64)
    weight_muon = np.array([0], dtype=np.float64)
    weight_electron = np.array([0], dtype=np.float64)

    def setup_branch(this,t):
        t.Branch("event",this.event,"event/I")
        t.Branch("hitid",this.hitid,"hitid/I")
        t.Branch("mcid",this.mcid,"mcid/I")
        t.Branch("truthid",this.truthid,"truthid/I")
        t.Branch("cluster",this.cluster,"cluster/I")
        t.Branch("mcpdg",this.mcpdg,"mcpdg/I")
        t.Branch("mccharge",this.mccharge,"mccharge/I")
        t.Branch("mcmass",this.mcmass,"mcmass/D")
        t.Branch("mcpx",this.mcpx,"mcpx/D")
        t.Branch("mcpy",this.mcpy,"mcpy/D")
        t.Branch("mcpz",this.mcpz,"mcpz/D")
        t.Branch("mcen",this.mcen,"mcen/D")
        t.Branch("mcstatus",this.mcstatus,"mcstatus/I")
        t.Branch("edep_mc",this.edep_mc,"edep_mc/D")
        t.Branch("pred_edep",this.pred_edep,"pred_edep/D")
        t.Branch("pred_edep_cluster",this.pred_edep_cluster,"pred_edep_cluster/D")
        t.Branch("pred_beta",this.pred_beta,"pred_beta/D")
        t.Branch("pred_alpha",this.pred_alpha,"pred_alpha/I")
        t.Branch("trackness",this.trackness,"trackness/I")
        t.Branch("weight_photon",this.weight_photon,"weight_photon/D")
        t.Branch("weight_charged_hadron",this.weight_charged_hadron,"weight_charged_hadron/D")
        t.Branch("weight_neutral_hadron",this.weight_neutral_hadron,"weight_neutral_hadron/D")
        t.Branch("weight_muon",this.weight_muon,"weight_muon/D")
        t.Branch("weight_electron",this.weight_electron,"weight_electron/D")

class EventData:
    ''' TTree data for MCParticle
        to be used for evaluating the efficiency
    '''
    event = np.array([0], dtype=np.int32)
    ncluster = np.array([0], dtype=np.int32)
    MC_dijet_energy = np.array([0], dtype=np.float64)
    total_MC_energy_truth = np.array([0], dtype=np.float64)
    total_MC_energy_pred = np.array([0], dtype=np.float64)
    total_predicted_energy_truth = np.array([0], dtype=np.float64)
    total_predicted_energy_pred = np.array([0], dtype=np.float64)


    def setup_branch(this,t):
        t.Branch("event",this.event,"event/I")
        t.Branch("ncluster",this.ncluster,"ncluster/I")
        t.Branch("MC_dijet_energy",this.MC_dijet_energy,"MC_dijet_energy/D")
        t.Branch("total_MC_energy_truth",this.total_MC_energy_truth,"total_MC_energy_truth/D")
        t.Branch("total_MC_energy_pred",this.total_MC_energy_pred,"total_MC_energy_pred/D")
        t.Branch("total_predicted_energy_truth",this.total_predicted_energy_truth,"total_predicted_energy_truth/D")
        t.Branch("total_predicted_energy_pred",this.total_predicted_energy_pred,"total_predicted_energy_pred/D")

class JetData:
    ''' TTree data for MCParticle
        to be used for evaluating the efficiency
    '''
    # jet_p4[MAX_JETS][4]: 行=ジェット、(E, px, py, pz)。実ジェット数は n_jets（以降の行は 0 埋め）
    MAX_JETS = 2

    event = np.array([0], dtype=np.int32)
    n_jets = np.array([0], dtype=np.int32)
    jet_p4 = np.zeros((MAX_JETS, 4), dtype=np.float64)

    def setup_branch(this,t):
        t.Branch("event",this.event,"event/I")
        t.Branch("n_jets",this.n_jets,"n_jets/I")
        t.Branch("jet_p4",this.jet_p4,f"jet_p4[{JetData.MAX_JETS}][4]/D")


def calc_origin_quark(quarks: np.array, clusters: np.array):    # calcurated from closest angles
    quarks = quarks / np.linalg.norm(quarks, axis=1, keepdims=True)
    clusters = clusters / np.linalg.norm(clusters, axis=1, keepdims=True)
    cos_angles = clusters @ quarks.T
    nearest_indices = np.argmax(cos_angles, axis=1)
    return nearest_indices

def calc_pred_jet_energy(energies: np.array, nearest_indices: np.array):    # calcurated jet energy from closest angle
    assert(not (energies.shape != nearest_indices.shape))
    return np.array([np.sum(energies[nearest_indices==0]), np.sum(energies[nearest_indices==1])])

# def save_root(datapath, ckpt, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, pandora=False, energyRegression=False, momentum=False, momentumAmp=False, mctpe=False):
def save_root(
    datapath,
    ckpt,
    outfile,
    nstart=0,
    nend=-1,
    timingCut=False,
    input_dim=5,
    output_dim=3,
    args={},
    model=None,
):
    global _ROOT_CHUNK_EVENTS
    debug = False
    pandora=args.pandora
    # Direction metadata is useful independently of the clustering choice.
    # Load it automatically when present, so --pandora only selects Pandora
    # clustering and does not control availability of the theta plot.
    event_energy = args.event_total_energy or has_event_builder(datapath)
    if event_energy and not args.event_total_energy:
        print("Found event builder: storing truth q/qbar directions in the ROOT jet tree")
    energyRegression=args.energy_regression
    energyRegressionCluster=args.energy_regression_cluster
    energyRegressionWeight=args.energy_regression_weight
    momentum=args.momentum
    momentumAmp=args.momentum_amp
    mctpe=args.mctpe
    energy_branch=args.energy_branch
    _ROOT_CHUNK_EVENTS = int(getattr(args, "root_chunk_events", 10))
    if _ROOT_CHUNK_EVENTS <= 0:
        raise ValueError("--root-chunk-events must be a positive integer")
    pipeline_depth = int(getattr(args, "pipeline_depth", 40))
    if pipeline_depth < 0:
        raise ValueError("--pipeline-depth must be zero or a positive integer")

    # Pandora clustering is already stored in the H5 file.  Keeping it on the
    # CPU avoids checkpoint loading, model construction, CUDA initialisation,
    # and unnecessary host/GPU transfers.
    device = "cpu" if pandora else args.device
    if not pandora and 'cuda' in device:
        torch.cuda.set_device(device)
    assert(not (args.beta_d_scan and (".root" in outfile)))

    thetaphi = True if input_dim == 7 else False
    if momentum:
        input_dim += 3 
        if momentumAmp:
            input_dim += 1
    if energyRegression:
        output_dim += 1
        if energyRegressionCluster:
            output_dim += 1
    if energyRegressionWeight:
            output_dim += 5
    if pandora:
        if model is None:
            model = _PandoraNoopModel()
        print("Pandora mode: skipping checkpoint and GNN model loading")
    else:
        if model is not None:
            print("Reusing previously loaded GNN model")
        else:
            print(f"Loading model from checkpoint {ckpt}")
            if energy_branch:
                model = get_model_branch(ckpt, jit=False, input_dim=input_dim,output_dim=output_dim).to(device)
            else:
                model = get_model(ckpt, jit=False, input_dim=input_dim, output_dim=output_dim, energy_regression=energyRegression, energy_regression_cluster=energyRegressionCluster, energy_regression_weight=energyRegressionWeight, model_variant=args.model_variant, **inference_loader_kwargs(args)).to(device)
    print(f"Loading data from {datapath} with {nstart=}, {nend=}, {timingCut=}")
    dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend, pandora=pandora,momentum=momentum,momentumAmp=momentumAmp, mctpe=mctpe,event_energy=event_energy)
    yielder = TestYielder(model=model, dataset=dataset, device=device, pandora=pandora, event_energy=event_energy)

    nmax = None if nend==-1 else nend-nstart+1
    print("number of entry : ", nmax)

    """
    ak_feat: edep, x, y, z, time, track, charge, px, py, pz (atcalo)
        --> save edep, drop others
    ak_label: hitid, mcid, pdg, charge, mass, px, py, pz (of mcp), status
        --> save all labels
    """
    
    outfileDir = outfile
    tbeta_list = [args.tbeta]
    td_list = [args.td]

    if args.beta_d_scan:
        tbeta_list = [i/10.0 for i in range(9,0,-1)]
        td_list = [i/10.0 for i in range(9,0,-1)]
        # tbeta_list = [0.9+i/100.0 for i in range(9,0,-1)]
        # td_list = [i/10.0 for i in range(9,0,-1)]

    print(tbeta_list)
    print(td_list)


    for tbeta in tbeta_list:
        for td in td_list:
            configure_inference(model, tbeta=tbeta, td=td)
            tbeta_now = round(tbeta * 100)
            td_now = round(td * 100)
            # outfile = outfileDir + '/tbeta' + format(tbeta_now, '02') + '0td' + format(td_now, '02') + '0.root'
            outfile = outfile if not args.beta_d_scan else outfileDir + '/tbeta' + format(tbeta_now, '03') + 'td' + format(td_now, '03') + '.root'

            print("")
            print(f"save_root()...  {outfile}")
            print("")
            outdir = os.path.dirname(outfile)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            file = TFile(outfile,"recreate")
            print(
                f"ROOT chunk writer: {_ROOT_CHUNK_EVENTS} event(s) per flush, "
                "LZ4 compression"
            )
            

            t = TTree("t","tree for MCParticle")
            d = Data()
            d.setup_branch(t)

            t2 = TTree("reco","tree for reconstructed clusters")
            d2 = RecoData()
            d2.setup_branch(t2)

            t3 = TTree("prediction","tree for model output")
            d3 = PredData()
            d3.setup_branch(t3)

            t4 = TTree("event","tree for event")
            d4 = EventData()
            d4.setup_branch(t4)

            t5 = TTree("jet","tree for jets")
            d5 = JetData()
            d5.setup_branch(t5)

            analysis_inputs = _iter_analysis_inputs(
                yielder,
                tbeta,
                td,
                nmax,
                pandora,
                energyRegression,
                energyRegressionCluster,
                energyRegressionWeight,
                pipeline_depth,
                device,
            )
            for i, (event, prediction, clustering, matches, condensation_points) in enumerate(analysis_inputs):
                if i == nmax: break
                if i < 10 or i%100 == 0:
                    print("Event", i, "processing...")

                matches12, matches21 = matches
                # print("matches12", matches12)
                # print("matches21", matches21)
                # print(clustering)
                if args.truth_clustering: clustering = event.y[:,0]
                cache = _EventNumpyCache(event, prediction, clustering)
                if (debug):
                    print(f"=== reco --> mc ===")
                    for k,v in matches12.items():
                        print(f"{k}-->{v}")
                    print(f"=== mc --> reco ===")
                    for k,v in matches21.items():
                        print(f"{k}-->{v}")

                all_truth_ids = cache.truth_ids
                # hitid/mcid are kept as integer sidecar tensors by dataset.py;
                # reading them from the mixed float32 label tensor can round
                # global IDs above 2**24.
                all_mcid = cache.mcids
                all_cluster_ids = cache.cluster_ids
                assert( len(all_truth_ids) == len(all_mcid) )
                assert( len(cache.truth) == len(cache.label[:,1]) )

                if (debug):
                    print(f"{all_truth_ids=}")
                    #print(f"{all_mcid=}")
                    print(f"{all_cluster_ids=}")

                # matched_reco_clusterIds = matching_1to1(event, clustering, matches12)
                # print(matched_reco_clusterIds)

                # Legacy code accumulated this value as a Torch float32 scalar
                # (Python float + the first Torch scalar). Keep that rounding
                # exactly while doing the remaining event work in NumPy.
                total_MC_energy = np.float32(0.)
                total_predicted_energy = 0.
                total_MC_energy_ = 0.
                total_predicted_energy_ = 0.
                MC_dijet_energy = cache.event[1] if event_energy else 0


                # iterate over all mcid
                for id in all_truth_ids:
                
                    ''' Get energy in three different ways.
                        - edep:       sum the hits that come from the MC particle (Perfect PFA)
                        - edep_reco:  find the matching cluster and sum all the hits
                                      (including those that do and do not come from the MC particle)
                        - edep_match: find the matching cluster and sum those that come from the MC particle
                    '''
                    ncluster = 0
                    matched_ncluster = 0
                    matched_cluster = -1

                    pattern_mcid = cache.truth_masks[id]
                    match_label = cache.label[pattern_mcid]
                    match_hitid = cache.hitid[pattern_mcid]
                    match_mcid = cache.mcid[pattern_mcid]
                    match_edep = cache.edep[pattern_mcid]
                    # match_track = match_feat[:,5].detach().numpy().astype(np.int32)
                    # match_pdg = match_label[:,2].detach().numpy().astype(np.int32)
                    # print("pdg", match_pdg)
                    # print("track", match_track)
                    edep_sum = cache.truth_edep[id]
                    ncluster = len(pattern_mcid)
                    
                    edep_reco = 0
                    edep_match = 0
                    cluster_match = []

                    if (id in matches12.keys()):
                        reco_match = matches12[id]
                        for rid in reco_match:
                            edep_reco = 0
                            edep_match = 0
                            # if (matched_cluster == -1):
                            matched_cluster = rid
                            matched_ncluster += 1
                            edep_reco += cache.cluster_edep_sum(rid)
                            edep_match += cache.pair_edep(id, rid)
                            cluster_match.append([edep_reco, edep_match, rid])
                    
                    cluster_match = np.array(cluster_match)
                    if cluster_match.shape[0]==0:
                        continue
                    cluster_match_ = cluster_match[np.argsort(cluster_match[:, 1])]
                    # print(cluster_match, cluster_match_)
                    edep_reco = np.sum(match_edep) if args.truth_clustering else cluster_match_[-1,0]
                    edep_match = np.sum(match_edep) if args.truth_clustering else cluster_match_[-1,1]
                    selected_cluster = (
                        id if args.truth_clustering else cluster_match_[-1,2]
                    )
                    pattern_cluster = cache.cluster_mask(selected_cluster)
                    match_track = cache.track_float[pattern_cluster]


                    if not pandora:
                        predicted_beta = cache.pred_betas[pattern_cluster]
                        beta_order = cache.beta_order(selected_cluster)
                        edeps = cache.edep[pattern_cluster]
                        predicted_energy = np.zeros(1)
                        predicted_energy_cluster = np.zeros(1)
                        predicted_energy_weight = np.zeros(1)
                        pred_photon_energy = np.zeros(1)
                        pred_charged_hadron_energy = np.zeros(1)
                        pred_neutral_hadron_energy = np.zeros(1)
                        pred_muon_energy = np.zeros(1)
                        pred_electron_energy = np.zeros(1)
                        if energyRegressionWeight and not energyRegression:
                            pred_weight_photon = cache.pred_weight_photon[pattern_cluster]
                            pred_weight_charged_hadron = cache.pred_weight_charged_hadron[pattern_cluster]
                            pred_weight_neutral_hadron = cache.pred_weight_neutral_hadron[pattern_cluster]
                            pred_weight_muon = cache.pred_weight_muon[pattern_cluster]
                            pred_weight_electron = cache.pred_weight_electron[pattern_cluster]
                            pred_weights = pred_weight_photon + pred_weight_charged_hadron + pred_weight_neutral_hadron + pred_weight_muon + pred_weight_electron
                            predicted_energy_weight = np.array([np.sum(edeps * pred_weights)])
                            pred_photon_energy = np.array([np.sum(edeps * pred_weight_photon)])
                            pred_charged_hadron_energy = np.array([np.sum(edeps * pred_weight_charged_hadron)])
                            pred_neutral_hadron_energy = np.array([np.sum(edeps * pred_weight_neutral_hadron)])
                            pred_muon_energy = np.array([np.sum(edeps * pred_weight_muon)])
                            pred_electron_energy = np.array([np.sum(edeps * pred_weight_electron)])
                            predicted_energy = predicted_energy_weight
                        if energyRegression and not energyRegressionWeight:
                            predicted_energy = cache.pred_tracker_energy[pattern_cluster]
                            predicted_energy = predicted_energy[beta_order]
                            predicted_energy_cluster = cache.pred_cluster_energy[pattern_cluster] if energyRegressionCluster else -np.ones(1)
                        if energyRegression and energyRegressionWeight:
                            predicted_energy = cache.pred_tracker_energy[pattern_cluster]
                            predicted_energy = predicted_energy[beta_order]
                            predicted_energy_cluster = cache.pred_cluster_energy[pattern_cluster] if energyRegressionCluster else -np.ones(1)
                            pred_weight_photon = cache.pred_weight_photon[pattern_cluster]
                            pred_weight_charged_hadron = cache.pred_weight_charged_hadron[pattern_cluster]
                            pred_weight_neutral_hadron = cache.pred_weight_neutral_hadron[pattern_cluster]
                            pred_weight_muon = cache.pred_weight_muon[pattern_cluster]
                            pred_weight_electron = cache.pred_weight_electron[pattern_cluster]
                            pred_weights = pred_weight_photon + pred_weight_charged_hadron + pred_weight_neutral_hadron + pred_weight_muon + pred_weight_electron
                            predicted_energy_weight = np.array([np.sum(edeps * pred_weights)])
                            pred_photon_energy = np.array([np.sum(edeps * pred_weight_photon)])
                            pred_charged_hadron_energy = np.array([np.sum(edeps * pred_weight_charged_hadron)])
                            pred_neutral_hadron_energy = np.array([np.sum(edeps * pred_weight_neutral_hadron)])
                            pred_muon_energy = np.array([np.sum(edeps * pred_weight_muon)])
                            pred_electron_energy = np.array([np.sum(edeps * pred_weight_electron)])
                        cond_tracknesses = match_track[beta_order]
                        cond_trackness = cond_tracknesses[0]
                        predicted_beta = predicted_beta[beta_order]
                        # print(predicted_beta[0], cond_trackness)
                    else:
                        predicted_energy = cache.pred_tracker_energy[pattern_cluster]
                        predicted_energy_weight = np.zeros(1)
                        pred_photon_energy = np.zeros(1)
                        pred_charged_hadron_energy = np.zeros(1)
                        pred_neutral_hadron_energy = np.zeros(1)
                        pred_muon_energy = np.zeros(1)
                        pred_electron_energy = np.zeros(1)
                        cond_trackness = 0
                        predicted_beta = np.zeros(1)

                    # for MC particle, take any element from the match because they should be the same
                    my_label = match_label[0]
                    my_hitid = match_hitid[0]
                    my_mcid = match_mcid[0]
                    if pandora:
                        # The legacy Pandora columns store the PFO energy on
                        # calorimeter-hit rows, while the track row is zero.
                        # Prefer the first non-track row; keep the existing
                        # zero value for a track-only PFO.
                        nontrack_indices = np.flatnonzero(match_track == 0)
                        energy_index = nontrack_indices[0] if len(nontrack_indices) > 0 else 0
                        pred_edep = predicted_energy[energy_index]
                    else:
                        pred_edep = predicted_energy[0]                              ## alpha
                    pred_edep_cluster = np.sum(predicted_energy_cluster) if not pandora else 0
                    # pred_edep = np.sum(predicted_energy) / np.sum(predicted_beta)      ## betaE
                    sed_radius=0
                    # pred_cood = prediction.pred_cluster_spsace_coords[pattern_mcid]
                    # print(type(pred_cood))
                    # sed_center, sed_radius = minimum_enclosing_sphere(prediction.pred_cluster_space_coords[pattern_mcid])
                    # print(prediction.pred_cluster_space_coords[pattern_mcid].shape, sed_center, _sed_radius, type(_sed_radius), _sed_radius.shape)

                    # Set values for TTree and fill
                    d.event[0] = i
                    d.hitid[0] = my_hitid
                    d.mcid[0] = my_mcid
                    d.truthid[0] = id
                    d.mcpdg[0] = my_label[2]
                    d.mccharge[0] = my_label[3]
                    d.mcmass[0] = my_label[4]
                    d.mcpx[0] = my_label[5]
                    d.mcpy[0] = my_label[6]
                    d.mcpz[0] = my_label[7]
                    d.mcen[0] = np.sqrt(d.mcmass[0]**2 + d.mcpx[0]**2 + d.mcpy[0]**2 + d.mcpz[0]**2)
                    d.mcstatus[0] = my_label[8]
                    d.edep[0] = edep_sum
                    d.edep_reco[0] = edep_reco
                    d.edep_match[0] = edep_match
                    d.ncluster[0] = ncluster
                    d.matched_ncluster[0] = matched_ncluster
                    d.matched_cluster[0] = matched_cluster
                    d.pred_edep[0] = pred_edep
                    d.pred_edep_cluster[0] = pred_edep_cluster
                    d.pred_edep_weight[0] = predicted_energy_weight
                    d.cond_beta[0] = predicted_beta[0]
                    d.cond_track[0] = cond_trackness
                    d.sed_radius[0] = sed_radius
                    d.pred_photon_energy[0] = pred_photon_energy
                    d.pred_charged_hadron_energy[0] = pred_charged_hadron_energy
                    d.pred_neutral_hadron_energy[0] = pred_neutral_hadron_energy
                    d.pred_muon_energy[0] = pred_muon_energy
                    d.pred_electron_energy[0] = pred_electron_energy

                    # if(sed_radius>3):
                    #     print(i,sed_center, sed_radius, my_label[2],my_label[5],my_label[6],my_label[7],np.sqrt(d.mcmass[0]**2 + d.mcpx[0]**2 + d.mcpy[0]**2 + d.mcpz[0]**2))

                    if (not d.mcid[0] == -1): # skip if track does not have hit
                        t.Fill()

                    total_MC_energy_ += d.mcen[0]
                    total_predicted_energy_ += pred_edep if cond_trackness!=0 else pred_edep_cluster

                # Iterate over reconstructed clusters
                event_truth_ids_all = cache.truth.astype(np.int32)
                truth_total_hits = {}
                truth_total_edep = {}
                for tid_all in np.unique(event_truth_ids_all):
                    mask_all = cache.truth_masks[tid_all]
                    tid_key = int(tid_all)
                    truth_total_hits[tid_key] = int(np.count_nonzero(mask_all))
                    truth_total_edep[tid_key] = float(cache.truth_edep[tid_all])

                for cl in all_cluster_ids:
                    pattern_cluster = cache.cluster_mask(cl)
                    match_label_ = cache.label[pattern_cluster]
                    match_feat_ = cache.feat[pattern_cluster]
                    match_track_ = cache.track_int[pattern_cluster]
                    cluster_truth_ids = cache.truth[pattern_cluster].astype(np.int32)
                    cluster_edeps = cache.edep[pattern_cluster]
                    nhits_cluster = len(match_label_)
                    ntrack_hits_cluster = int(np.count_nonzero(match_track_))
                    edep_reco_cluster = float(cache.cluster_edep_sum(cl))
                    if not pandora:
                        predicted_beta_ = cache.pred_betas[pattern_cluster]
                        beta_order = cache.beta_order(cl)
                        edeps = cluster_edeps
                        if energyRegression:
                            if not energyRegressionWeight:
                                predicted_energy_ = cache.pred_tracker_energy[pattern_cluster]
                                predicted_energy_ = predicted_energy_[beta_order]
                                predicted_energy_cluster_ = cache.pred_cluster_energy[pattern_cluster] if energyRegressionCluster else -np.ones(1)
                            else:
                                predicted_energy_ = cache.pred_tracker_energy[pattern_cluster]
                                predicted_energy_ = predicted_energy_[beta_order]
                                pred_weights = (cache.pred_weight_photon + cache.pred_weight_charged_hadron + cache.pred_weight_neutral_hadron + cache.pred_weight_muon + cache.pred_weight_electron)[pattern_cluster]
                                predicted_energy_weight = np.array([np.sum(edeps * pred_weights)])
                                predicted_energy_cluster_ = np.zeros(1)
                            match_label_ = match_label_[beta_order]
                            match_feat_ = match_feat_[beta_order]
                        else:
                            predicted_energy_ = np.zeros(1)
                            predicted_energy_cluster_ = np.zeros(1)
                        cond_tracknesses_ = match_track_[beta_order]
                        cond_trackness_ = cond_tracknesses_[0]
                        predicted_beta_ = predicted_beta_[beta_order]
                        # print(predicted_beta[0], cond_trackness)
                    else:
                        predicted_energy_ = cache.pred_tracker_energy[pattern_cluster]
                        cond_trackness_ = 0
                        predicted_beta_ = np.zeros(1)

                    # for MC particle, take any element from the match because they should be the same
                    if pandora:
                        # As above, avoid choosing the zero-energy track row
                        # when this PFO also contains calorimeter hits.
                        nontrack_indices_ = np.flatnonzero(match_track_ == 0)
                        energy_index_ = nontrack_indices_[0] if len(nontrack_indices_) > 0 else 0
                        pred_edep_ = predicted_energy_[energy_index_]
                    else:
                        pred_edep_ = predicted_energy_[0]                             ## alpha
                    pred_edep_cluster_ = np.sum(predicted_energy_cluster_) if not pandora else 0
                    # print(pred_edep_, pred_edep_cluster_, predicted_beta_, cond_tracknesses_)
                    oc_label = match_label_[0]
                    total_MC_energy += np.sqrt(
                        oc_label[4] * oc_label[4]
                        + oc_label[5] * oc_label[5]
                        + oc_label[6] * oc_label[6]
                        + oc_label[7] * oc_label[7]
                    )
                    total_predicted_energy += pred_edep_ if cond_trackness_!=0 else pred_edep_cluster_

                    # Compose truth contributions in this reconstructed cluster
                    unique_truth = np.unique(cluster_truth_ids)
                    contrib = []
                    for tid in unique_truth:
                        mask_tid = (cluster_truth_ids == tid)
                        if not np.any(mask_tid):
                            continue
                        hit_count_tid = int(np.count_nonzero(mask_tid))
                        edep_tid = float(np.sum(cluster_edeps[mask_tid]))
                        track_hit_count_tid = int(np.count_nonzero(match_track_[mask_tid]))
                        total_hit_tid = int(truth_total_hits.get(int(tid), 0))
                        total_edep_tid = float(truth_total_edep.get(int(tid), 0.0))
                        tid_labels = match_label_[mask_tid]
                        pdg_tid = int(tid_labels[0][2]) if len(tid_labels) > 0 else -1
                        contrib.append((int(tid), pdg_tid, hit_count_tid, edep_tid, track_hit_count_tid, total_hit_tid, total_edep_tid))

                    dominant_truth_id = -1
                    dominant_truth_pdg = -1
                    dominant_mcid = -1
                    dominant_hit_frac = 0.0
                    dominant_edep_frac = 0.0
                    edep_match_cluster = 0.0
                    mcp_label = None
                    edep_mc = -1.0

                    if len(contrib) > 0:
                        # "割合が一番多い truth" is defined by hit fraction (tie-break by edep fraction)
                        contrib_sorted = sorted(contrib, key=lambda x: (x[2], x[3]), reverse=True)
                        dominant_truth_id, dominant_truth_pdg, dominant_hit_count, dominant_edep, _, _, _ = contrib_sorted[0]
                        dominant_hit_frac = float(dominant_hit_count) / float(max(nhits_cluster, 1))
                        dominant_edep_frac = float(dominant_edep) / float(max(edep_reco_cluster, 1e-12))
                        edep_match_cluster = dominant_edep

                        pattern_mcp = cache.truth_masks[dominant_truth_id]
                        edep_mc = float(cache.truth_edep[dominant_truth_id])

                        pattern_mcp_match_label = cache.label[pattern_mcp].astype(np.float64)
                        pattern_mcp_mcid = cache.mcid[pattern_mcp]
                        if len(pattern_mcp_match_label) > 0:
                            mcp_label = pattern_mcp_match_label[0]
                            dominant_mcid = int(pattern_mcp_mcid[0])

                    # Fill composition arrays per truth PDG id (aligned index among 3 arrays)
                    contrib_sorted_all = sorted(contrib, key=lambda x: (x[2], x[3]), reverse=True)
                    ncomp = min(len(contrib_sorted_all), len(d2.pdg_comp_ids))
                    d2.pdg_comp_ids[:] = 0
                    d2.pdg_comp_hits[:] = 0
                    d2.pdg_comp_hit_frac[:] = 0.0
                    d2.pdg_comp_edep_frac[:] = 0.0
                    d2.pdg_comp_edep[:] = 0.0
                    d2.pdg_comp_truth_edep[:] = 0.0
                    d2.pdg_comp_track_hits[:] = 0
                    for ip in range(ncomp):
                        _, pdg_tid, hit_count_tid, edep_tid, track_hit_count_tid, total_hit_tid, total_edep_tid = contrib_sorted_all[ip]
                        d2.pdg_comp_ids[ip] = int(pdg_tid)
                        d2.pdg_comp_hits[ip] = int(hit_count_tid)
                        d2.pdg_comp_hit_frac[ip] = float(hit_count_tid) / float(max(total_hit_tid, 1))
                        d2.pdg_comp_edep_frac[ip] = float(edep_tid) / float(max(total_edep_tid, 1e-12))
                        d2.pdg_comp_edep[ip] = float(edep_tid)
                        d2.pdg_comp_truth_edep[ip] = float(total_edep_tid)
                        d2.pdg_comp_track_hits[ip] = int(track_hit_count_tid)

                    d2.event[0] = i
                    d2.cluster[0] = cl
                    d2.nhits[0] = nhits_cluster
                    d2.ntrack_hits[0] = ntrack_hits_cluster
                    d2.cond_beta[0] = float(predicted_beta_[0]) if len(predicted_beta_) > 0 else 0.0
                    d2.cond_is_track[0] = 1 if int(cond_trackness_) != 0 else 0
                    d2.matched_truth_pdgid[0] = dominant_truth_pdg
                    d2.matched_truth_hit_frac[0] = dominant_hit_frac
                    d2.matched_truth_edep_frac[0] = dominant_edep_frac
                    d2.npdg_comp[0] = ncomp

                    d2.mcid[0] = dominant_mcid
                    d2.mcpdg[0] = -1
                    d2.mccharge[0] = -1
                    d2.mcmass[0] = -1
                    d2.mcpx[0] = -1
                    d2.mcpy[0] = -1
                    d2.mcpz[0] = -1
                    d2.mcen[0] = -1
                    d2.mcstatus[0] = -1
                    d2.pred_edep[0] = pred_edep_
                    d2.pred_edep_cluster[0] = pred_edep_cluster_

                    if mcp_label is not None:
                        d2.mcpdg[0] = int(mcp_label[2])
                        d2.mccharge[0] = int(mcp_label[3])
                        d2.mcmass[0] = float(mcp_label[4])
                        d2.mcpx[0] = float(mcp_label[5])
                        d2.mcpy[0] = float(mcp_label[6])
                        d2.mcpz[0] = float(mcp_label[7])
                        d2.mcen[0] = np.sqrt(d2.mcmass[0]**2 + d2.mcpx[0]**2 + d2.mcpy[0]**2 + d2.mcpz[0]**2)
                        d2.mcstatus[0] = int(mcp_label[8])

                    d2.edep_reco[0] = edep_reco_cluster
                    d2.edep_mc[0] = edep_mc
                    d2.edep_match[0] = edep_match_cluster
                    t2.Fill()


                # Save the hit-level truth/reco association used by the Cedric
                # metrics.  This information is available for both GravNet and
                # Pandora; only the model-specific output fields need defaults
                # in the Pandora case.
                _append_prediction_columns(
                    t3,
                    i,
                    cache,
                    condensation_points,
                    pandora,
                    energyRegression,
                    energyRegressionCluster,
                    energyRegressionWeight,
                )

                # for cl in all_cluster_ids:
                #     pattern_cluster = (clustering==cl)
                #     pattern_cluster_feat = event.feat[pattern_cluster]
                #     pattern_cluster_edep = pattern_cluster_feat[:,0].detach().numpy().astype(np.float64)
                #     edep_reco = np.sum(pattern_cluster_edep)
                #     if not pandora:
                #         predicted_beta = prediction.pred_betas[pattern_cluster]
                #         if energyRegression:
                #             predicted_energy = prediction.pred_tracker_energy[pattern_cluster]
                #             predicted_energy = predicted_energy[np.argsort(-predicted_beta)]
                #             predicted_energy_cluster = prediction.pred_cluster_energy[pattern_cluster] if energyRegressionCluster else -np.ones(1)
                #         else:
                #             predicted_energy = np.zeros(1)
                #             predicted_energy_cluster = np.zeros(1)
                #         predicted_beta = -np.sort(-predicted_beta)
                #         # print(predicted_beta[0], cond_trackness)
                #     else:
                #         predicted_energy = prediction.pred_tracker_energy[pattern_cluster]
                #         cond_trackness = 0
                #         predicted_beta = np.zeros(1)
                #     pred_edep = predicted_energy[0]                                  ## alpha
                #     pred_edep_cluster = np.sum(predicted_energy_cluster) if not pandora else 0

                d4.event[0] = i
                d4.ncluster[0] = len(all_cluster_ids)
                d4.MC_dijet_energy[0] = MC_dijet_energy
                d4.total_MC_energy_truth[0] = total_MC_energy_
                d4.total_MC_energy_pred[0] = total_MC_energy
                d4.total_predicted_energy_truth[0] = total_predicted_energy_
                d4.total_predicted_energy_pred[0] = total_predicted_energy
                t4.Fill()

                if event.jet is not None:
                    # q_en_truthBase = calc_pred_jet_energy(np.array(pred_energy_truthBase) , calc_origin_quark(np.array(jet_momentum), np.array(reco_momentum_truthBase)))
                    # q_en_predBase = calc_pred_jet_energy(np.array(pred_energy_predBase) , calc_origin_quark(np.array(jet_momentum), np.array(reco_momentum_predBase)))
                    # print(q_en_truthBase, q_en_predBase, MC_jet_energies)
                    jet_np = cache.jet
                    nj = int(min(jet_np.shape[0], JetData.MAX_JETS))
                    d5.n_jets[0] = nj
                    d5.jet_p4.fill(0.0)
                    if nj > 0:
                        n4 = min(4, jet_np.shape[1])
                        d5.jet_p4[:nj, :n4] = jet_np[:nj, :n4]
                    d5.event[0] = i
                    # d5.MC_jet_energy[0] = MC_jet_energies[0]
                    # d5.total_predicted_energy_truthBase[0] = q_en_truthBase[0]
                    # d5.total_predicted_energy_predBase[0] = q_en_predBase[0]
                    t5.Fill()
                # d5.event[0] = i
                # d5.MC_jet_energy[0] = MC_jet_energies[1]
                # d5.total_predicted_energy_truthBase[0] = q_en_truthBase[1]
                # d5.total_predicted_energy_predBase[0] = q_en_predBase[1]
                # t5.Fill()

                # All five logical trees for this event have now been filled.
                # Flush only at event boundaries so each tree keeps the legacy
                # entry order while the in-memory buffer remains bounded.
                file.finish_event()

            print(f"Saving to {outfile}")
            file.Write()

    return model

def main():
    print(sys.argv)
    if (len(sys.argv) < 9):
        print("Usage: save_root.py datapath ckpt outfile nstart nend timingCut input_dim output_dim pandora energyRegression momentum momentumAmp MCTpe")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('datapath')
    parser.add_argument('ckpt')
    parser.add_argument('outfile')
    parser.add_argument('nstart', type=int)
    parser.add_argument('nend', type=int)
    parser.add_argument('timingCut')
    parser.add_argument('input_dim', type=int)
    parser.add_argument('output_dim', type=int)
    parser.add_argument('--pandora', action='store_true', help='Use PandoraPFA result')
    parser.add_argument('--event-total-energy', action='store_true', help='Use event visible energy')
    parser.add_argument('--energy-regression', action='store_true', help='Turn on energy regression term on loss function and output')
    parser.add_argument('--energy-regression-cluster', action='store_true', help='Turn on energy regression term on loss function and output (regression for neutral particle)')
    parser.add_argument('--energy-regression-weight', action='store_true', help='Turn on energy regression term on loss function and output (enegy weight loss)')
    parser.add_argument('-e','--momentum', action='store_true', help='Add momentum to GNN input')
    parser.add_argument('-ea','--momentum-amp', action='store_true', help='Add absoute momentum to GNN input')
    parser.add_argument('--mctpe', action='store_true', help='Use MC truth momentum and energy for virtual hits')
    parser.add_argument('-eb','--energy-branch', action='store_true', help='Change GNN model to bypass energy')
    parser.add_argument('--beta-d-scan', action='store_true', help='Turn on beta and diameter scan')
    parser.add_argument('--tbeta', type=float, default=0.9)
    parser.add_argument('--td', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu', help='Specify calculation device')
    parser.add_argument('--model-variant', type=str, default='auto', choices=['auto', 'legacy', 'multihead'], help='Select model loader: auto-detect, force legacy GravnetModel, or force multihead loader')
    parser.add_argument('--truth-clustering', action='store_true', help='Turn on MC truth clustering')
    parser.add_argument('--1tomany-clustering', action='store_true', help='Turn on combining reco-clusters')
    parser.add_argument(
        '--multi-h5',
        action='store_true',
        help=(
            'Process each H5 separately while reusing one model. datapath may '
            'be a directory, quoted glob, comma-separated list, or @list.txt; '
            'outfile is treated as an output directory.'
        ),
    )
    parser.add_argument(
        '--root-chunk-events',
        type=int,
        default=int(os.environ.get('CEDRIC_ROOT_CHUNK_EVENTS', '10')),
        help='Number of completed events buffered before one C++ ROOT append (default: 10)',
    )
    parser.add_argument(
        '--pipeline-depth',
        type=int,
        default=int(os.environ.get('CEDRIC_PIPELINE_DEPTH', '40')),
        help=(
            'Number of ordered GNN events prefetched while CPU analysis runs '
            '(CUDA only; 0 disables, default: 40)'
        ),
    )

    add_inference_arguments(parser)
    args = parser.parse_args()
    timing_cut = bool(strtobool(args.timingCut))

    if args.beta_d_scan:
        if args.multi_h5:
            parser.error(
                '--multi-h5 cannot be combined with --beta-d-scan; use '
                'save_root_reco_scan_fast_resume.py for threshold scans'
            )
        # Preserve the original threshold-scan path.  The dedicated scan
        # resume program should normally be used instead.
        save_root(
            args.datapath,
            args.ckpt,
            args.outfile,
            nstart=args.nstart,
            nend=args.nend,
            timingCut=timing_cut,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            args=args,
        )
        return

    if args.multi_h5:
        inputs = _resolve_multi_h5(args.datapath)
        output_directory = Path(args.outfile).expanduser()
        if output_directory.exists() and not output_directory.is_dir():
            parser.error(
                f'--multi-h5 outfile must be a directory: {output_directory}'
            )
        output_directory.mkdir(parents=True, exist_ok=True)
        jobs = [
            (str(input_path), output_directory / f'{input_path.stem}.root')
            for input_path in inputs
        ]
        output_names = [str(output) for _, output in jobs]
        if len(output_names) != len(set(output_names)):
            parser.error(
                'Multiple inputs have the same H5 stem and would overwrite '
                'the same ROOT output'
            )
    else:
        jobs = [(args.datapath, Path(args.outfile).expanduser())]

    pending = []
    reused_count = 0
    print(f'[resume] Checking {len(jobs)} requested ROOT output(s)')
    for datapath, final_output in jobs:
        event_energy = args.event_total_energy or has_event_builder(datapath)
        expected_counts = _expected_root_counts(
            datapath,
            args.nstart,
            args.nend,
            timing_cut,
            event_energy,
        )
        validation = _validate_root(final_output, expected_counts)
        if validation.valid:
            reused_count += 1
            print(
                f'[resume] skip completed {final_output} '
                f'(event, prediction, jet entries = {validation.counts})'
            )
        else:
            pending.append((datapath, final_output, expected_counts))
            if final_output.exists():
                print(
                    f'[resume] regenerate invalid {final_output}: '
                    f'{validation.reason}'
                )
            else:
                print(f'[resume] generate missing {final_output}')

    if not pending:
        print('[resume] All requested ROOT files are complete; model loading skipped.')
        return

    shared_model = None
    generated_count = 0
    for job_index, (datapath, final_output, expected_counts) in enumerate(
        pending, start=1
    ):
        final_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = _temporary_output(final_output)
        print(
            f'[resume] Processing {job_index}/{len(pending)}: {datapath} '
            f'-> {final_output}'
        )

        shared_model = save_root(
            datapath,
            args.ckpt,
            str(temporary_output),
            nstart=args.nstart,
            nend=args.nend,
            timingCut=timing_cut,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            args=args,
            model=shared_model,
        )

        validation = _validate_root(temporary_output, expected_counts)
        if not validation.valid:
            raise RuntimeError(
                f'Generated ROOT failed validation: {temporary_output}: '
                f'{validation.reason}. The temporary file was kept.'
            )
        os.replace(temporary_output, final_output)
        generated_count += 1
        print(f'[resume] completed -> {final_output}')

    print(
        f'[resume] Completed {generated_count} new ROOT file(s); '
        f'reused {reused_count} existing file(s)'
    )

if __name__=='__main__':
    main()
    
