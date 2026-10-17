#!/usr/bin/env python3
"""Resume-capable wrapper for save_root_reco_scan_fast.py.

Existing, structurally complete ROOT files are skipped.  Missing or invalid
scan points are regenerated using the same one-inference cache and beta-worker
parallelism as the fast implementation.  New files are first written to a
temporary path and atomically moved into place after validation.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import ROOT

import save_root_reco_scan_fast as _fast


class Validation(NamedTuple):
    valid: bool
    reason: str
    counts: Tuple[int, int, int]


TREE_SCHEMAS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "t": (
        ("event", "Int_t"),
        ("hitid", "Int_t"),
        ("mcid", "Int_t"),
        ("truthid", "Int_t"),
        ("mcpdg", "Int_t"),
        ("mccharge", "Int_t"),
        ("mcmass", "Double_t"),
        ("mcpx", "Double_t"),
        ("mcpy", "Double_t"),
        ("mcpz", "Double_t"),
        ("mcen", "Double_t"),
        ("mcstatus", "Int_t"),
        ("edep", "Double_t"),
        ("edep_reco", "Double_t"),
        ("edep_match", "Double_t"),
        ("ncluster", "Int_t"),
        ("matched_ncluster", "Int_t"),
        ("matched_cluster", "Int_t"),
        ("pred_edep", "Double_t"),
        ("pred_edep_cluster", "Double_t"),
        ("pred_edep_weight", "Double_t"),
        ("cond_beta", "Double_t"),
        ("cond_track", "Int_t"),
        ("sed_radius", "Double_t"),
        ("pred_photon_energy", "Double_t"),
        ("pred_charged_hadron_energy", "Double_t"),
        ("pred_neutral_hadron_energy", "Double_t"),
        ("pred_muon_energy", "Double_t"),
        ("pred_electron_energy", "Double_t"),
    ),
    "reco": (
        ("event", "Int_t"),
        ("cluster", "Int_t"),
        ("nhits", "Int_t"),
        ("mcid", "Int_t"),
        ("mcpdg", "Int_t"),
        ("mccharge", "Int_t"),
        ("mcmass", "Double_t"),
        ("mcpx", "Double_t"),
        ("mcpy", "Double_t"),
        ("mcpz", "Double_t"),
        ("mcen", "Double_t"),
        ("mcstatus", "Int_t"),
        ("edep_reco", "Double_t"),
        ("edep_mc", "Double_t"),
        ("edep_match", "Double_t"),
        ("pred_edep", "Double_t"),
        ("pred_edep_cluster", "Double_t"),
        ("ntrack_hits", "Int_t"),
        ("cond_beta", "Double_t"),
        ("cond_is_track", "Int_t"),
        ("matched_truth_pdgid", "Int_t"),
        ("matched_truth_hit_frac", "Double_t"),
        ("matched_truth_edep_frac", "Double_t"),
        ("npdg_comp", "Int_t"),
        ("pdg_comp_ids", "Int_t"),
        ("pdg_comp_hits", "Int_t"),
        ("pdg_comp_hit_frac", "Double_t"),
        ("pdg_comp_edep_frac", "Double_t"),
        ("pdg_comp_edep", "Double_t"),
        ("pdg_comp_truth_edep", "Double_t"),
        ("pdg_comp_track_hits", "Int_t"),
    ),
    "prediction": (
        ("event", "Int_t"),
        ("hitid", "Int_t"),
        ("mcid", "Int_t"),
        ("truthid", "Int_t"),
        ("cluster", "Int_t"),
        ("mcpdg", "Int_t"),
        ("mccharge", "Int_t"),
        ("mcmass", "Double_t"),
        ("mcpx", "Double_t"),
        ("mcpy", "Double_t"),
        ("mcpz", "Double_t"),
        ("mcen", "Double_t"),
        ("mcstatus", "Int_t"),
        ("edep_mc", "Double_t"),
        ("pred_edep", "Double_t"),
        ("pred_edep_cluster", "Double_t"),
        ("pred_beta", "Double_t"),
        ("pred_alpha", "Int_t"),
        ("trackness", "Int_t"),
        ("weight_photon", "Double_t"),
        ("weight_charged_hadron", "Double_t"),
        ("weight_neutral_hadron", "Double_t"),
        ("weight_muon", "Double_t"),
        ("weight_electron", "Double_t"),
    ),
    "event": (
        ("event", "Int_t"),
        ("ncluster", "Int_t"),
        ("MC_dijet_energy", "Double_t"),
        ("total_MC_energy_truth", "Double_t"),
        ("total_MC_energy_pred", "Double_t"),
        ("total_predicted_energy_truth", "Double_t"),
        ("total_predicted_energy_pred", "Double_t"),
    ),
    "jet": (
        ("event", "Int_t"),
        ("n_jets", "Int_t"),
        ("jet_p4", "Double_t"),
    ),
}


_PENDING_TD: Dict[float, Tuple[float, ...]] = {}
_EXPECTED_COUNTS: Optional[Tuple[int, int, int]] = None


def _validate_root(
    path: Path,
    expected_counts: Optional[Tuple[int, int, int]] = None,
) -> Validation:
    """Check readability, exact tree schema, and invariant entry counts."""
    try:
        if not path.is_file():
            return Validation(False, "missing", (-1, -1, -1))
        if path.stat().st_size == 0:
            return Validation(False, "empty file", (-1, -1, -1))
    except OSError as error:
        return Validation(False, f"cannot stat: {error}", (-1, -1, -1))

    try:
        root_file = ROOT.TFile.Open(str(path), "READ")
    except (OSError, RuntimeError) as error:
        return Validation(False, f"cannot open: {error}", (-1, -1, -1))
    if not root_file or root_file.IsZombie():
        if root_file:
            root_file.Close()
        return Validation(False, "cannot open or zombie", (-1, -1, -1))

    try:
        recovered_bit = getattr(ROOT.TFile, "kRecovered", None)
        if recovered_bit is not None and root_file.TestBit(recovered_bit):
            return Validation(False, "ROOT recovered file", (-1, -1, -1))

        entries: Dict[str, int] = {}
        for tree_name, schema in TREE_SCHEMAS.items():
            tree = root_file.Get(tree_name)
            if not tree or not tree.InheritsFrom("TTree"):
                return Validation(
                    False,
                    f"missing TTree {tree_name}",
                    (-1, -1, -1),
                )

            expected_names = [name for name, _ in schema]
            actual_names = [
                branch.GetName() for branch in tree.GetListOfBranches()
            ]
            if actual_names != expected_names:
                return Validation(
                    False,
                    f"branch layout mismatch in {tree_name}",
                    (-1, -1, -1),
                )

            for branch_name, type_name in schema:
                branch = tree.GetBranch(branch_name)
                leaf = branch.GetLeaf(branch_name) if branch else None
                if not leaf or leaf.GetTypeName() != type_name:
                    return Validation(
                        False,
                        f"type mismatch in {tree_name}/{branch_name}",
                        (-1, -1, -1),
                    )

            nentries = int(tree.GetEntries())
            entries[tree_name] = nentries
            # Reading the final entry also forces ROOT to read the final basket.
            if nentries > 0 and tree.GetEntry(nentries - 1) < 0:
                return Validation(
                    False,
                    f"cannot read final entry of {tree_name}",
                    (-1, -1, -1),
                )

        counts = (
            entries["event"],
            entries["prediction"],
            entries["jet"],
        )
        if expected_counts is not None:
            if counts != expected_counts:
                return Validation(
                    False,
                    f"entry counts {counts} != expected {expected_counts}",
                    counts,
                )
        elif entries["event"] <= 0 or entries["prediction"] <= 0:
            # Before inference there is no exact expected count.  Do not call
            # an empty output complete; it will be checked exactly afterward.
            return Validation(False, "no event/hit entries", counts)

        return Validation(True, "ok", counts)
    finally:
        root_file.Close()


def _points(args) -> Tuple[Tuple[float, float, Path], ...]:
    beta_values = _fast.TBETA_VALUES if args.beta_d_scan else (args.tbeta,)
    td_values = _fast.TD_VALUES if args.beta_d_scan else (args.td,)
    points = []
    for tbeta in beta_values:
        for td in td_values:
            if args.beta_d_scan:
                output = (
                    Path(args.outfile)
                    / _fast._point_name(tbeta, td)
                    / f"{Path(args.datapath).stem}.root"
                )
            else:
                output = Path(args.outfile)
            points.append((float(tbeta), float(td), output))
    return tuple(points)


def _consistent_complete_without_inference(
    validations: Sequence[Validation],
) -> bool:
    if not validations or not all(result.valid for result in validations):
        return False
    return len({result.counts for result in validations}) == 1


def _expected_counts(cache: Sequence) -> Tuple[int, int, int]:
    event_entries = len(cache)
    prediction_entries = sum(len(event.label[:, 1]) for event, _ in cache)
    jet_entries = sum(event.jet is not None for event, _ in cache)
    return event_entries, prediction_entries, jet_entries


def _temporary_output(final_output: Path) -> Path:
    return final_output.with_name(
        f".{final_output.name}.part.{os.getpid()}.{time.time_ns()}.root"
    )


def _run_beta_group_resume(tbeta: float) -> List[str]:
    """Process only missing td points for one beta and commit atomically."""
    try:
        _fast.torch.set_num_threads(1)
        _fast.torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _fast._install_worker_patches()
    _fast._ACTIVE_CLUSTER_ENGINE = _fast._BetaClusterEngine(
        _fast._PREDICTION_CACHE, tbeta
    )

    worker_args = _fast.copy.copy(_fast._GLOBAL_ARGS)
    worker_args.beta_d_scan = False
    worker_args.device = "cpu"
    worker_args.tbeta = float(tbeta)
    written: List[str] = []

    try:
        for td in _PENDING_TD[float(tbeta)]:
            final_output = _fast._point_output(tbeta, td)

            # Recheck immediately before work in case another process completed
            # the same point after the parent preflight.
            current = _validate_root(final_output, _EXPECTED_COUNTS)
            if current.valid:
                print(
                    f"[resume pid={os.getpid()}] skip completed "
                    f"{_fast._point_name(tbeta, td)} -> {final_output}",
                    flush=True,
                )
                continue

            final_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = _temporary_output(final_output)
            worker_args.td = float(td)
            print(
                f"[fast-resume pid={os.getpid()}] regenerate "
                f"{_fast._point_name(tbeta, td)} "
                f"({current.reason}) -> {final_output}",
                flush=True,
            )

            _fast._legacy.save_root(
                _fast._GLOBAL_ARGS.datapath,
                _fast._GLOBAL_ARGS.ckpt,
                str(temporary),
                nstart=_fast._GLOBAL_ARGS.nstart,
                nend=_fast._GLOBAL_ARGS.nend,
                timingCut=_fast._GLOBAL_ARGS.timingCut,
                input_dim=_fast._GLOBAL_ARGS.input_dim,
                output_dim=_fast._GLOBAL_ARGS.output_dim,
                args=worker_args,
            )

            generated = _validate_root(temporary, _EXPECTED_COUNTS)
            if not generated.valid:
                raise RuntimeError(
                    f"Generated ROOT failed validation: {temporary}: "
                    f"{generated.reason}. The temporary file was kept."
                )

            os.replace(temporary, final_output)
            written.append(str(final_output))
            print(
                f"[fast-resume pid={os.getpid()}] completed -> {final_output}",
                flush=True,
            )
    finally:
        _fast._ACTIVE_CLUSTER_ENGINE = None
        gc.collect()

    return written


def main(argv=None) -> None:
    global _PENDING_TD, _EXPECTED_COUNTS

    args = _fast._parse_args(argv)
    if args.pandora and args.beta_d_scan:
        raise SystemExit(
            "--beta-d-scan is a GNN threshold scan and is not supported with --pandora"
        )

    _fast._GLOBAL_ARGS = args
    points = _points(args)
    preflight = [_validate_root(output) for _, _, output in points]
    initially_valid = sum(result.valid for result in preflight)
    print(
        f"[resume] Preflight: {initially_valid}/{len(points)} "
        "ROOT file(s) are structurally complete"
    )

    if _consistent_complete_without_inference(preflight):
        counts = preflight[0].counts
        print(
            f"[resume] All {len(points)} points are complete and consistent "
            f"(event, prediction, jet entries = {counts})."
        )
        print("[resume] Skipping data loading and GNN inference.")
        return

    _fast._PREDICTION_CACHE, _fast._GLOBAL_EVENT_ENERGY = (
        _fast._build_prediction_cache(args)
    )
    _EXPECTED_COUNTS = _expected_counts(_fast._PREDICTION_CACHE)

    pending: Dict[float, List[float]] = {}
    valid_count = 0
    for tbeta, td, output in points:
        result = _validate_root(output, _EXPECTED_COUNTS)
        if result.valid:
            valid_count += 1
            print(
                f"[resume] skip completed {_fast._point_name(tbeta, td)} "
                f"-> {output}"
            )
        else:
            pending.setdefault(tbeta, []).append(td)
            if output.exists():
                print(
                    f"[resume] existing file is invalid and will be replaced: "
                    f"{output} ({result.reason})"
                )

    print(
        f"[resume] Exact check: {valid_count}/{len(points)} complete, "
        f"{len(points) - valid_count} pending"
    )
    if not pending:
        print("[resume] Nothing to regenerate.")
        return

    _PENDING_TD = {
        float(tbeta): tuple(td_values)
        for tbeta, td_values in pending.items()
    }
    _fast._ensure_root_writer_loaded()

    beta_values = tuple(_PENDING_TD)
    workers = min(args.scan_workers, len(beta_values))
    print(
        f"[fast-resume] Starting {workers} CPU worker(s) for "
        f"{len(beta_values)} incomplete beta group(s)"
    )

    if workers == 1:
        results = [_run_beta_group_resume(beta) for beta in beta_values]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            results = list(
                pool.imap_unordered(
                    _run_beta_group_resume,
                    beta_values,
                    chunksize=1,
                )
            )

    generated_count = sum(len(group) for group in results)
    print(
        f"[fast-resume] Completed {generated_count} new ROOT file(s); "
        f"reused {valid_count} existing file(s)"
    )


if __name__ == "__main__":
    main()
