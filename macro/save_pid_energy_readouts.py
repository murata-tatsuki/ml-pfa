#!/usr/bin/env python3
"""Save paired PID jet-energy summaries from frozen ROOT predictions.

The source summary defines the files, events, beta cut, truth energies and
angles. No inference, reclustering, MC momentum substitution or source-file
update is performed. Images use FORMAT/raw_head/ and FORMAT/track_replaced/;
All ROOT data and text tables are consolidated into
jet_energy_resolution_summary.root and .txt in the output base.
Older ROOT files recover measured track momentum from their
identified original H5, with event and input-row correspondence checked.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import awkward as ak
import h5py
import numpy as np
import uproot


SPECIES = ("charged_hadron", "electron", "muon", "photon", "neutral_hadron")
MASSES = np.array((0.13957039, 0.00051099895, 0.1056583755))
MODES = ("raw_head", "track_replaced")
SUMMARY_NAME = "jet_energy_resolution_summary.root"
COMPACT_SCHEMA = "pid-analysis-summary-2"
TEXT_NAME = "jet_energy_resolution_summary.txt"
MACRO_DIRECTORY = Path(__file__).resolve().parent


def _json_object(root_file, name, required=True):
    if name not in root_file:
        if required:
            raise ValueError(f"{root_file.file_path}: missing {name}")
        return {}
    value = json.loads(str(root_file[name]))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _source_path(stored, metadata, summary_path):
    path = Path(stored).expanduser()
    candidates = [path] if path.is_absolute() else [
        MACRO_DIRECTORY / path, Path.cwd() / path, summary_path.parent / path]
    outfile = metadata.get("arguments", {}).get("outfile")
    if outfile:
        candidates.append(Path(outfile).expanduser())
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find source ROOT: {stored}")


def _arrays(tree, names):
    missing = set(names).difference(tree.keys())
    if missing:
        raise ValueError(f"{tree.object_path}: missing branches {sorted(missing)}")
    return tree.arrays(list(names), library="np")


def _measured_policy(metadata):
    policy = metadata.get("single_track_energy")
    if not isinstance(policy, dict):
        return False
    description = str(policy.get("momentum", "")).lower()
    if not all(word in description for word in ("measured", "calorimeter", "7:10")):
        return False
    masses = policy.get("masses_GeV", {})
    if not isinstance(masses, dict) or any(
            key not in masses or not np.isclose(float(masses[key]), mass, rtol=0, atol=1e-12)
            for key, mass in zip(SPECIES[:3], MASSES)):
        raise ValueError("Stored single-track mass hypotheses do not match pion/electron/muon")
    return True


def _builder(handle, name):
    group = handle[name]
    return ak.from_buffers(json.loads(group.attrs["form"]), int(group.attrs["length"]),
                           {key: value[:] for key, value in group.items()})


def _matrix_event(handle, name, event, width):
    """Read one standard list-of-regular-rows builder without loading all hits."""
    group = handle[name]
    form = json.loads(group.attrs["form"])
    content = form.get("content", {})
    if (form.get("class") != "ListOffsetArray" or
            content.get("class") != "RegularArray" or content.get("size") != width):
        raise ValueError(f"{handle.filename}: unsupported {name} builder; refusing an unverified row mapping")
    offset_key = form["form_key"] + "-offsets"
    data_key = content["content"]["form_key"] + "-data"
    start, stop = map(int, group[offset_key][event:event + 2])
    return np.asarray(group[data_key][start * width:stop * width]).reshape(-1, width)


def _thrust(flat):
    if len(flat) < 8:
        return float("nan")
    p4 = np.asarray(flat[:8], dtype=np.float32).reshape(2, 4)
    energy = np.float32(0)
    weighted = np.float32(0)
    for e, px, py, pz in p4:
        momentum = np.sqrt(np.float32(px * px + py * py + pz * pz))
        if not (e > 0 and momentum > 0 and np.isfinite(e) and np.isfinite(momentum)):
            return float("nan")
        energy = np.float32(energy + e)
        weighted = np.float32(weighted + np.float32(np.abs(pz) / momentum) * e)
    angle = np.float32(weighted / energy)
    return float(angle) if np.isfinite(angle) and 0 <= angle <= 1 else float("nan")


def _event_key(truth, thrust):
    # The ROOT event builder and upstream angular calculation use float32.
    return (np.float32(truth).tobytes(),
            np.float32(thrust).tobytes() if np.isfinite(thrust) else None)


def _saved_flag(settings, name):
    """Decode booleans from both current and historical writer metadata."""
    value = settings.get(name, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("false", "true", "0", "1"):
        return value.strip().lower() in ("true", "1")
    raise ValueError(f"Unrecognized saved {name} flag: {value!r}")


class MeasuredMomentum:
    """Strict original-H5 recovery for historical ROOTs lacking measured energy."""

    def __init__(self, root_file, metadata):
        signature = _json_object(root_file, "save_root_metadata", required=False).get("signature", {})
        identities = signature.get("inputs") or metadata.get("resume", {}).get("inputs")
        if not identities:
            raise ValueError("Old ROOT has no original-H5 input identity; regenerate it to save both readouts")
        settings = signature.get("settings") or metadata.get("arguments", {})
        if _saved_flag(settings, "timingCut") or _saved_flag(settings, "mctpe"):
            raise ValueError("Old ROOT with on-the-fly timing cuts or MC track momentum is unsupported")
        self.stack = ExitStack()
        self.handles = []
        self.candidates = {}
        self.last_global_index = -1
        self.cache = {}
        self.mapped = 0
        try:
            start = int(settings.get("nstart", 0))
            count = int(settings.get("nend", -1))
            stop = start + count if count >= 0 else None
            if start < 0 or count < -1:
                raise ValueError("Invalid saved H5 event range")
            global_index = 0
            for identity in identities:
                path = Path(identity["path"]).expanduser()
                stat = path.stat()
                modified = identity.get("modified_ns", identity.get("mtime_ns"))
                if modified is None or "size" not in identity:
                    raise ValueError(f"Missing saved H5 size/mtime identity: {path}")
                if stat.st_size != int(identity["size"]) or stat.st_mtime_ns != int(modified):
                    raise ValueError(f"Original H5 identity changed: {path}")
                handle = self.stack.enter_context(h5py.File(path, "r"))
                self.handles.append(handle)
                if "event" not in handle or "feature" not in handle:
                    raise ValueError(f"{path}: original H5 needs event and feature builders")
                rows = _builder(handle, "event")
                for local, row in enumerate(rows):
                    current = global_index
                    global_index += 1
                    if current < start or (stop is not None and current >= stop):
                        continue
                    flat = np.asarray(ak.to_numpy(ak.flatten(row, axis=None)), dtype=np.float64).reshape(-1)
                    if flat.size <= 2 and flat.size:
                        truth = flat[-1]
                    elif flat.size >= 9:
                        truth = flat[9] if flat.size >= 10 else flat[8]
                    else:
                        raise ValueError(f"{path}: unsupported event payload")
                    key = _event_key(truth, _thrust(flat))
                    self.candidates.setdefault(key, []).append((current, handle, local))
            self.prediction = _arrays(root_file["prediction"],
                ("event", "cluster", "input_row", "hitid", "trackness"))
            events = self.prediction["event"]
            if np.any(events[1:] < events[:-1]):
                order = np.argsort(events, kind="stable")
                self.prediction = {name: values[order] for name, values in self.prediction.items()}
                events = self.prediction["event"]
            unique, begins, counts = np.unique(events, return_index=True, return_counts=True)
            self.slices = {int(event): slice(int(begin), int(begin + length))
                           for event, begin, length in zip(unique, begins, counts)}
        except BaseException:
            self.stack.close()
            raise

    def close(self):
        self.stack.close()

    def event(self, event, truth, thrust):
        if event in self.cache:
            return self.cache[event]
        if event not in self.slices:
            raise ValueError(f"ROOT event {event}: no prediction inputs for H5 correspondence")
        part = self.slices[event]
        prediction = {name: values[part] for name, values in self.prediction.items()}
        input_rows = prediction["input_row"].astype(np.int64)
        if (input_rows < 0).any() or len(np.unique(input_rows)) != len(input_rows):
            raise ValueError(f"ROOT event {event}: invalid or duplicate saved input rows")
        matches = []
        for current, handle, local in self.candidates.get(_event_key(truth, thrust), []):
            if current <= self.last_global_index:
                continue
            features = _matrix_event(handle, "feature", local, 13)
            if len(input_rows) == 0 or input_rows.max() >= len(features):
                continue
            if "row_info" in handle:
                info = _matrix_event(handle, "row_info", local, 11)
                if len(info) != len(features):
                    raise ValueError(f"{handle.filename}: row_info and feature lengths differ")
                hit_ids = info[input_rows, 9]
            elif "label" in handle:
                labels = _matrix_event(handle, "label", local, 9)
                hit_ids = labels[input_rows, 0]
            else:
                raise ValueError(f"{handle.filename}: cannot validate original input identifiers")
            if (np.array_equal(hit_ids, prediction["hitid"]) and
                    np.array_equal(features[input_rows, 5], prediction["trackness"])):
                matches.append((current, features, prediction))
        if len(matches) != 1:
            raise ValueError(f"ROOT event {event}: expected one validated original H5 event, found {len(matches)}; regenerate ROOT instead of assuming event-number correspondence")
        current, features, prediction = matches[0]
        self.last_global_index = current
        self.mapped += 1
        # Only the current event is kept: full-H5 hit data is never cached.
        self.cache = {event: (features, prediction)}
        return self.cache[event]

    def energies(self, event, truth, thrust, clusters, seed_rows, heads):
        features, prediction = self.event(event, truth, thrust)
        track_mask = prediction["trackness"] != 0
        track_clusters = prediction["cluster"][track_mask]
        track_rows = prediction["input_row"][track_mask]
        for cluster, seed in zip(clusters, seed_rows):
            member_tracks = track_rows[track_clusters == cluster]
            if len(member_tracks) != 1 or int(member_tracks[0]) != int(seed):
                raise ValueError(f"ROOT event {event}, cluster {cluster}: single-track seed does not match saved prediction membership")
        seed_rows = np.asarray(seed_rows, dtype=np.int64)
        if (seed_rows < 0).any() or (seed_rows >= len(features)).any() or not np.all(features[seed_rows, 5] > .5):
            raise ValueError(f"ROOT event {event}: measured-momentum row is not a valid track")
        # Event.feat is float32 in the production writer, then decoded as double.
        momenta = features[seed_rows, 7:10].astype(np.float32).astype(np.float64)
        if not np.isfinite(momenta).all():
            raise ValueError(f"ROOT event {event}: non-finite measured track momentum")
        return np.sqrt(np.sum(momenta ** 2, axis=1) + MASSES[heads] ** 2)


def _totals(events, energies, selected, target_events):
    unique = np.sort(np.asarray(target_events, dtype=np.int64))
    positions = np.searchsorted(unique, events)
    inside = positions < len(unique)
    inside[inside] &= unique[positions[inside]] == events[inside]
    keep = inside & selected
    sums = np.bincount(positions[keep], weights=energies[keep], minlength=len(unique))
    return {int(event): float(energy) for event, energy in zip(unique, sums)}


def _file_readouts(root_path, saved_metadata, event_rows, beta, source_readout="saved", cluster_sink=None):
    with uproot.open(root_path) as source:
        metadata = _json_object(source, "particle_eval_metadata")
        if metadata != saved_metadata:
            raise ValueError(f"{root_path}: particle metadata differs from source summary; re-run efficiency/purity analysis")
        names = ("event", "cluster", "energy_pid", "pid", "energy_head", "ntrack_hits", "cond_beta")
        columns = _arrays(source["reco"], names + tuple("energy_" + name for name in SPECIES))
        if "cluster_accepted" in source["reco"].keys():
            columns.update(_arrays(source["reco"], ("cluster_accepted",)))
        saved = columns["energy_pid"].astype(np.float64)
        heads = columns["energy_head"]
        if (not np.isfinite(saved).all() or (heads < -1).any() or (heads > 4).any() or
                (columns["ntrack_hits"] < 0).any()):
            raise ValueError(f"{root_path}: invalid saved PID energy/head/track count")
        eligible = (heads >= 0) & (heads < 3) & (columns["ntrack_hits"] == 1)
        if "cluster_accepted" in columns:
            eligible &= columns["cluster_accepted"] != 0
        if np.any(columns["pid"][eligible] != heads[eligible]):
            raise ValueError(f"{root_path}: charged energy_head and predicted PID disagree")
        selected = ~(columns["cond_beta"] < beta)  # Match the native macro, including its NaN comparison.
        event_ids = np.array([int(row["event"]) for row in event_rows], dtype=np.int64)
        if len(np.unique(event_ids)) != len(event_ids):
            raise ValueError(f"{root_path}: duplicate summary events")
        builder = _arrays(source["event"], ("event", "MC_dijet_energy"))
        builder_truth = {int(event): float(truth) for event, truth in
                         zip(builder["event"], builder["MC_dijet_energy"])}
        if len(builder_truth) != len(builder["event"]):
            raise ValueError(f"{root_path}: duplicate source event IDs")
        if cluster_sink is not None:
            valid_events = {event for event, truth in builder_truth.items()
                            if truth > 0 and np.isfinite(truth)}
            if set(event_ids) != valid_events:
                raise ValueError(f"{root_path}: native energy diagnostics require the complete valid-event selection from this source ROOT")
        for row in event_rows:
            event = int(row["event"])
            if event not in builder_truth or builder_truth[event] != float(row["truth_energy"]):
                raise ValueError(f"{root_path}: event {event} truth does not match source summary")
        original = _totals(columns["event"], saved, selected, event_ids)
        eligible &= np.isin(columns["event"], event_ids)
        if cluster_sink is None:
            eligible &= selected
        raw, track = saved.copy(), saved.copy()
        for head, name in enumerate(SPECIES[:3]):
            mask = eligible & (heads == head)
            raw[mask] = columns["energy_" + name][mask]
        if not np.isfinite(raw).all():
            raise ValueError(f"{root_path}: selected raw charged head is unavailable")
        recovered = 0
        if eligible.any() and not _measured_policy(metadata):
            if not np.allclose(saved[eligible], raw[eligible], rtol=1e-12, atol=1e-9):
                raise ValueError(f"{root_path}: adopted energy differs from raw head without an explicit measured-track policy")
            seed = _arrays(source["reco"], ("pid_seed_input_row",))["pid_seed_input_row"]
            recovery = MeasuredMomentum(source, metadata)
            try:
                for row in sorted(event_rows, key=lambda item: int(item["event"])):
                    event = int(row["event"])
                    mask = eligible & (columns["event"] == event)
                    if not mask.any():
                        continue
                    track[mask] = recovery.energies(event, float(row["truth_energy"]), float(row["thrust"]),
                        columns["cluster"][mask], seed[mask], heads[mask])
                    recovered += int(mask.sum())
            finally:
                recovery.close()
        result = {"raw_head": _totals(columns["event"], raw, selected, event_ids),
                  "track_replaced": _totals(columns["event"], track, selected, event_ids)}
        expected = original if source_readout == "saved" else result[source_readout]
        for row in event_rows:
            event = int(row["event"])
            if not np.isclose(expected[event], row["reco_energy"], rtol=1e-12, atol=1e-7, equal_nan=True):
                raise ValueError(f"{root_path}: event {event} {source_readout} energy sum differs from source summary")
        if cluster_sink is not None:
            identities = np.rec.fromarrays((columns["event"], columns["cluster"]))
            if len(np.unique(identities)) != len(identities):
                raise ValueError(f"{root_path}: duplicate reco event/cluster identity")
            cluster_sink(dict(event=columns["event"].astype(np.int32),
                cluster=columns["cluster"].astype(np.int32), saved=saved,
                raw_head=raw, track_replaced=track))
        return result, int(eligible.sum()), recovered


def _publish(staged, output, include_common_images=False):
    """Publish the compact pair and images, preserving unrelated output files."""
    token = uuid.uuid4().hex
    prepared = []
    replaced = []
    with tempfile.TemporaryDirectory(prefix="pid-readout-rollback-") as rollback:
        try:
            groups = [[staged / SUMMARY_NAME, staged / TEXT_NAME]]
            groups.extend(sorted(path for mode in MODES
                                 for path in (staged / section / mode).rglob("*"))
                          for section in ("png", "pdf", "eps"))
            if include_common_images:
                groups.extend(sorted(path for path in (staged / section).glob("*")
                                     if path.is_file())
                              for section in ("png", "pdf", "eps"))
            for paths in groups:
                for path in paths:
                    if not path.is_file():
                        continue
                    relative = path.relative_to(staged)
                    destination = output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    pending = destination.with_name(f".{destination.name}.{token}.pending")
                    if destination.is_symlink():
                        raise ValueError(f"Refusing to overwrite a symlink: {destination}")
                    backup = Path(rollback) / relative
                    existed = destination.exists()
                    if existed:
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(destination, backup)
                    prepared.append((pending, destination, backup, existed))
                    shutil.copy2(path, pending)
            for pending, destination, backup, existed in prepared:
                os.replace(pending, destination)
                replaced.append((destination, backup, existed))
        except BaseException:
            for destination, backup, existed in reversed(replaced):
                if existed:
                    restore = destination.with_name(f".{destination.name}.{token}.restore")
                    try:
                        shutil.copy2(backup, restore)
                        os.replace(restore, destination)
                    finally:
                        restore.unlink(missing_ok=True)
                else:
                    destination.unlink(missing_ok=True)
            raise
        finally:
            for pending, _, _, _ in prepared:
                pending.unlink(missing_ok=True)


def _finish_cache(cache_path, input_rows, ROOT, energy, beta, method):
    result = ROOT.TFile.Open(str(cache_path), "UPDATE")
    if not result or result.IsZombie():
        raise RuntimeError("Cannot finalize the temporary cluster readout cache")
    try:
        ROOT.TNamed("energy_readout_cache_schema", "pid-energy-readout-cache-1").Write()
        ROOT.TNamed("method", method).Write()
        ROOT.TParameter(int)("beam_energy_GeV", energy).Write()
        ROOT.TParameter("double")("beta_threshold", beta).Write()
        tree = ROOT.TTree("inputs", "Verified source ROOT identities for energy-only analysis")
        file_index = np.zeros(1, dtype=np.int32)
        path, metadata = ROOT.std.string(), ROOT.std.string()
        tree.Branch("file_index", file_index, "file_index/I")
        tree.Branch("path", path)
        tree.Branch("metadata", metadata)
        for index, source, metadata_text in input_rows:
            file_index[0] = index
            path.assign(source)
            metadata.assign(metadata_text)
            tree.Fill()
        tree.Write()
    finally:
        result.Close()


def _copy_directory(source, destination, ROOT):
    """Copy the latest ROOT keys, including complete trees and histogram stats."""
    for name in dict.fromkeys(key.GetName() for key in source.GetListOfKeys()):
        value = source.Get(name)
        if not value:
            raise RuntimeError(f"Cannot copy ROOT object {source.GetPath()}/{name}")
        if value.InheritsFrom("TDirectory"):
            _copy_directory(value, destination.mkdir(name), ROOT)
        else:
            destination.cd()
            if value.InheritsFrom("TTree"):
                value = value.CloneTree(-1, "fast")
                if not value:
                    raise RuntimeError(f"Cannot copy ROOT tree {name}")
            if value.Write(name, ROOT.TObject.kOverwrite) <= 0:
                raise RuntimeError(f"Cannot write ROOT object {name}")


def _compatible_container(path, staged):
    """Reuse diagnostics only when both complete event/input selections match."""
    if not path.is_file():
        return False
    try:
        with uproot.open(path) as previous:
            if previous["schema"].member("fTitle") != COMPACT_SCHEMA:
                return False
            for mode in MODES:
                old = previous[mode]
                with uproot.open(staged / mode / SUMMARY_NAME) as current:
                    for name in ("method", "energy_readout"):
                        if old[name].member("fTitle") != current[name].member("fTitle"):
                            return False
                    for name in ("beam_energy_GeV", "beta_threshold"):
                        if old[name].member("fVal") != current[name].member("fVal"):
                            return False
                    for table, fields in (("inputs", ("file_index", "path", "metadata", "pid_input")),
                                          ("events", ("file_index", "event", "thrust", "reco_energy", "truth_energy"))):
                        a, b = _arrays(old[table], fields), _arrays(current[table], fields)
                        for field in fields:
                            equal = (np.array_equal(a[field], b[field], equal_nan=True)
                                     if a[field].dtype.kind in "fc" else np.array_equal(a[field], b[field]))
                            if not equal:
                                return False
        return True
    except (KeyError, ValueError, OSError):
        return False


def _validate_group(group, energy, method, beta, schema=None, readout=None):
    required = {"method": method}
    if schema is not None:
        required["schema"] = schema
    if readout is not None:
        required["energy_readout"] = readout
    for name, expected in required.items():
        value = group.Get(name)
        if not value or value.GetTitle() != expected:
            raise ValueError(f"{group.GetPath()}: expected {name}={expected}")
    for name, expected in (("beam_energy_GeV", energy), ("beta_threshold", beta)):
        value = group.Get(name)
        if not value or value.GetVal() != expected:
            raise ValueError(f"{group.GetPath()}: incompatible {name}")


def _clustering_text(group):
    particles = ("electron", "pion", "photon", "neutron", "K0", "muon")
    definitions = ("original", "cedric")
    lines = ["[clustering]",
             "# original: " + group.Get("original_definition").GetTitle(),
             "# cedric: " + group.Get("cedric_definition").GetTitle(),
             "definition particle efficiency purity eff_num eff_den pur_num pur_den eff_entries pur_entries"]
    tree = group.Get("clustering")
    if not tree:
        raise ValueError("Common clustering group has no clustering table")
    for row in tree:
        efficiency = row.eff_num / row.eff_den if row.eff_den > 0 else float("nan")
        purity = row.pur_num / row.pur_den if row.pur_den > 0 else float("nan")
        lines.append(f"{definitions[int(row.definition)]} {particles[int(row.particle)]} "
            f"{efficiency:.17g} {purity:.17g} {row.eff_num:.17g} {row.eff_den:.17g} "
            f"{row.pur_num:.17g} {row.pur_den:.17g} {int(row.eff_entries)} {int(row.pur_entries)}")
    return "\n".join(lines) + "\n"


def _write_compact_summary(staged, output, source_summary, source_base, common_directory,
                           ROOT, energy, method, beta, preserve_previous=True):
    handles = []
    def open_root(path, mode="READ"):
        value = ROOT.TFile.Open(str(path), mode)
        if not value or value.IsZombie():
            raise RuntimeError(f"Cannot open {path}")
        handles.append(value)
        return value
    try:
        previous = None
        if preserve_previous:
            for path in dict.fromkeys((output / SUMMARY_NAME, source_summary)):
                if _compatible_container(path, staged):
                    previous = open_root(path)
                    break
        result = open_root(staged / SUMMARY_NAME, "RECREATE")
        result.cd()
        ROOT.TNamed("schema", COMPACT_SCHEMA).Write()
        ROOT.TNamed("method", method).Write()
        ROOT.TNamed("energy_readouts", ",".join(MODES)).Write()
        ROOT.TParameter(int)("beam_energy_GeV", energy).Write()
        ROOT.TParameter("double")("beta_threshold", beta).Write()
        common = None
        if common_directory is not None:
            common = open_root(common_directory / "clustering_summary.root")
        elif previous:
            common = previous.GetDirectory("clustering")
        if not common:
            for base in dict.fromkeys((source_base, output)):
                path = base / "clustering_summary.root"
                if path.is_file():
                    common = open_root(path)
                    break
        text = [f"# {COMPACT_SCHEMA} beam_energy_GeV={energy} method={method}\n"]
        if common:
            _validate_group(common, energy, method, beta, "pid-clustering-summary-1")
            common_target = result.mkdir("clustering")
            _copy_directory(common, common_target, ROOT)
            if common_directory is not None:
                diagnostics = open_root(common_directory / "efficiency_purity_result.root")
                _validate_group(diagnostics, energy, method, beta, readout="common")
                _copy_directory(diagnostics, common_target.mkdir("diagnostics"), ROOT)
            text.append(_clustering_text(common))
        else:
            text.append("[clustering]\n# No common clustering table supplied.\n")
        for mode in MODES:
            source = open_root(staged / mode / SUMMARY_NAME)
            target = result.mkdir(mode)
            _copy_directory(source, target, ROOT)
            diagnostics_path = staged / mode / "energy_regression_result.root"
            if diagnostics_path.is_file():
                diagnostics = open_root(diagnostics_path)
                _validate_group(diagnostics, energy, method, beta, readout=mode)
            else:
                diagnostics = previous.GetDirectory(mode + "/diagnostics") if previous else None
            if diagnostics:
                _copy_directory(diagnostics, target.mkdir("diagnostics"), ROOT)
            text.append(f"\n[{mode}]\n" + (staged / mode / TEXT_NAME).read_text())
        result.cd()
        ROOT.TNamed("clustering_available", "yes" if common else "no").Write()
        (staged / TEXT_NAME).write_text("\n".join(text))
    finally:
        for handle in reversed(handles):
            handle.Close()


def _fresh_pair_settings(staged):
    """Validate the native one-pass products without opening prediction inputs."""
    reference = None
    for mode in MODES:
        directory = staged / mode
        for name in (SUMMARY_NAME, TEXT_NAME, "energy_regression_result.root"):
            if not (directory / name).is_file():
                raise ValueError(f"Missing fresh {mode} output: {directory / name}")
        with uproot.open(directory / SUMMARY_NAME) as source:
            if source["schema"].member("fTitle") != "pid-jet-resolution-1":
                raise ValueError(f"{directory}: unsupported jet-energy summary schema")
            method = source["method"].member("fTitle")
            energy = int(source["beam_energy_GeV"].member("fVal"))
            beta = float(source["beta_threshold"].member("fVal"))
            if (source["energy_readout"].member("fTitle") != mode or
                    method == "pandora" or not np.isfinite(beta)):
                raise ValueError(f"{directory}: incompatible native energy readout")
            inputs = _arrays(source["inputs"], ("file_index", "path", "metadata", "pid_input"))
            events = _arrays(source["events"],
                ("file_index", "event", "thrust", "reco_energy", "truth_energy"))
        if not len(inputs["file_index"]) or not np.all(inputs["pid_input"]):
            raise ValueError(f"{directory}: paired native summaries require PID inputs")
        if len(np.unique(inputs["file_index"])) != len(inputs["file_index"]):
            raise ValueError(f"{directory}: duplicate input file indices")
        if not set(events["file_index"]).issubset(set(inputs["file_index"])):
            raise ValueError(f"{directory}: event rows refer to unknown input files")
        identities = np.rec.fromarrays((events["file_index"], events["event"]))
        if len(np.unique(identities)) != len(identities):
            raise ValueError(f"{directory}: duplicate file/event identities")
        if (not np.isfinite(events["truth_energy"]).all() or
                not np.all(events["truth_energy"] > 0) or
                not np.isfinite(events["reco_energy"]).all()):
            raise ValueError(f"{directory}: invalid native event energies")
        for value in inputs["metadata"]:
            if not isinstance(json.loads(str(value)), dict):
                raise ValueError(f"{directory}: input metadata must contain a JSON object")
        with uproot.open(directory / "energy_regression_result.root") as diagnostics:
            for name, expected in (("method", method), ("energy_readout", mode)):
                if diagnostics[name].member("fTitle") != expected:
                    raise ValueError(f"{directory}: incompatible diagnostic {name}")
            if (int(diagnostics["beam_energy_GeV"].member("fVal")) != energy or
                    float(diagnostics["beta_threshold"].member("fVal")) != beta):
                raise ValueError(f"{directory}: incompatible diagnostic energy/beta")
        settings = (energy, method, beta)
        if reference is None:
            reference = (settings, inputs, events)
        else:
            if settings != reference[0]:
                raise ValueError("Fresh native readouts have different analysis settings")
            for label, values, original in (("inputs", inputs, reference[1]),
                                            ("events", events, reference[2])):
                for name in values:
                    if label == "events" and name == "reco_energy":
                        continue
                    a, b = values[name], original[name]
                    same = (np.array_equal(a, b, equal_nan=True)
                            if a.dtype.kind in "fc" else np.array_equal(a, b))
                    if not same:
                        raise ValueError(f"Fresh native readouts differ in {label}/{name}")
    return reference[0], len(reference[1]["file_index"]), len(reference[2]["event"])


def compact_native_outputs(staged_directory, output_directory, common_directory):
    """Publish complete native paired outputs; never reopen their source ROOTs."""
    staged = Path(staged_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    common = Path(common_directory).expanduser().resolve()
    if output == staged or staged in output.parents:
        raise ValueError("The final output must be outside the temporary staging directory")
    for name in ("clustering_summary.root", "efficiency_purity_result.root"):
        if not (common / name).is_file():
            raise ValueError(f"Missing fresh common output: {common / name}")
    (energy, method, beta), files, events = _fresh_pair_settings(staged)
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    _write_compact_summary(staged, output, staged / SUMMARY_NAME, staged, common,
                           ROOT, energy, method, beta, preserve_previous=False)
    _publish(staged, output, include_common_images=True)
    print(f"Published native paired PID analysis: {files} input files, {events} events; {output}",
          flush=True)


def recover_measured_bridge(bridge_path, output_path):
    """Recover historical measured energies from a native snapshot and its H5."""
    bridge_path = Path(bridge_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path == bridge_path:
        raise ValueError("The measured-energy output must differ from its input bridge")
    with uproot.open(bridge_path) as source:
        metadata = _json_object(source, "particle_eval_metadata")
        if _measured_policy(metadata):
            raise ValueError("Native measured-energy recovery is only for historical raw-head ROOTs")
        rows = _arrays(source["reco"], ("event", "cluster", "pid_seed_input_row", "energy_head"))
        events = _arrays(source["event"], ("event", "MC_dijet_energy", "thrust"))
        for name, values in rows.items():
            if values.dtype.kind not in "iu":
                raise ValueError(f"Bridge reco/{name} must contain integer values")
        for name in ("event", "cluster"):
            if ((rows[name] < np.iinfo(np.int32).min).any() or
                    (rows[name] > np.iinfo(np.int32).max).any()):
                raise ValueError(f"Bridge reco/{name} is outside int32 range")
        identities = np.rec.fromarrays((rows["event"], rows["cluster"]))
        if len(np.unique(identities)) != len(identities):
            raise ValueError("Measured-energy bridge has duplicate event/cluster identities")
        if ((rows["energy_head"] < 0).any() or (rows["energy_head"] >= 3).any() or
                (rows["pid_seed_input_row"] < 0).any()):
            raise ValueError("Measured-energy bridge contains an invalid charged head or seed")
        event_ids = events["event"]
        if len(np.unique(event_ids)) != len(event_ids):
            raise ValueError("Measured-energy bridge has duplicate event IDs")
        if not set(rows["event"]).issubset(set(event_ids)):
            raise ValueError("Measured-energy bridge is missing event truth/angles")
        positions = {int(event): index for index, event in enumerate(event_ids)}
        measured = np.zeros(len(rows["event"]), dtype=np.float64)
        if len(measured):
            recovery = MeasuredMomentum(source, metadata)
            try:
                for event in sorted(set(map(int, rows["event"]))):
                    index = positions[event]
                    truth = float(events["MC_dijet_energy"][index])
                    if not np.isfinite(truth) or truth <= 0:
                        raise ValueError(f"Measured-energy bridge event {event} has invalid truth energy")
                    selected = rows["event"] == event
                    measured[selected] = recovery.energies(event, truth, float(events["thrust"][index]),
                        rows["cluster"][selected], rows["pid_seed_input_row"][selected],
                        rows["energy_head"][selected])
            finally:
                recovery.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pid-measured-", suffix=".root",
                                            dir=output_path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        with uproot.recreate(temporary) as target:
            tree = target.mktree("track_replacements", dict(event="int32", cluster="int32",
                                                           track_replaced="float64"))
            if len(measured):
                tree.extend(dict(event=rows["event"].astype(np.int32),
                                 cluster=rows["cluster"].astype(np.int32),
                                 track_replaced=measured))
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Recovered {len(measured)} measured-track energies from the native input snapshot",
          flush=True)


def save_readouts(summary_path, output_directory=None, draw=False, analysis_executable=None,
                  common_directory=None):
    summary_path = Path(summary_path).expanduser().resolve()
    with uproot.open(summary_path) as source_file:
        compact_source = source_file["schema"].member("fTitle") == COMPACT_SCHEMA
        source = source_file["raw_head"] if compact_source else source_file
        if source["schema"].member("fTitle") != "pid-jet-resolution-1":
            raise ValueError("Unsupported source jet-energy-summary schema")
        method = source["method"].member("fTitle")
        if method == "pandora":
            raise ValueError("Paired GNN energy readouts do not apply to Pandora")
        source_readout = source["energy_readout"].member("fTitle") if "energy_readout" in source else "saved"
        if source_readout not in ("saved",) + MODES:
            raise ValueError(f"Unsupported source energy readout: {source_readout}")
        energy = int(source["beam_energy_GeV"].member("fVal"))
        beta = float(source["beta_threshold"].member("fVal"))
        inputs = _arrays(source["inputs"], ("file_index", "path", "metadata", "pid_input"))
        events = _arrays(source["events"], ("file_index", "event", "thrust", "reco_energy", "truth_energy"))
    if compact_source or source_readout == "saved":
        source_base = summary_path.parent
    elif summary_path.parent.parent.name == "summary":
        source_base = summary_path.parent.parent.parent
    else:
        source_base = summary_path.parent.parent
    output = Path(output_directory).expanduser().resolve() if output_directory else source_base
    common_directory = Path(common_directory).expanduser().resolve() if common_directory else None
    executable = Path(analysis_executable).expanduser().resolve() if analysis_executable else None
    if executable is not None and (not executable.is_file() or not os.access(executable, os.X_OK)):
        raise ValueError(f"Cannot execute the PID energy analysis: {executable}")
    if not len(inputs["file_index"]) or not np.all(inputs["pid_input"]):
        raise ValueError("Paired summaries require representative-PID ROOT inputs only")
    if len(np.unique(inputs["file_index"])) != len(inputs["file_index"]):
        raise ValueError("Source summary has duplicate file indices")
    if not set(events["file_index"]).issubset(set(inputs["file_index"])):
        raise ValueError("Source summary events refer to missing input files")
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.gInterpreter.AddIncludePath(str(MACRO_DIRECTORY / "src"))
    if not ROOT.gInterpreter.Declare('#include "pid_jet_resolution_summary.h"'):
        raise RuntimeError("Cannot load the shared jet-energy-summary implementation")
    summaries = {mode: ROOT.pid_jer.Summary(energy, method, beta, mode) for mode in MODES}
    paired = {mode: np.zeros(len(events["event"]), dtype=np.float64) for mode in MODES}
    with tempfile.TemporaryDirectory(prefix="pid-energy-readouts-") as temporary:
        staged = Path(temporary)
        cache_path = staged / "cluster_readouts.root"
        input_rows = []
        eligible = recovered = 0
        with ExitStack() as stack:
            cache = stack.enter_context(uproot.recreate(cache_path)) if executable is not None else None
            for position in range(len(inputs["file_index"])):
                file_index = int(inputs["file_index"][position])
                metadata_text = str(inputs["metadata"][position])
                metadata = json.loads(metadata_text)
                root_path = _source_path(str(inputs["path"][position]), metadata, summary_path)
                indices = np.flatnonzero(events["file_index"] == file_index)
                rows = [{name: values[index] for name, values in events.items()} for index in indices]
                def cluster_sink(columns):
                    tree = cache.mktree(f"readouts_file_{file_index}",
                                       {name: values.dtype for name, values in columns.items()})
                    tree.extend(columns)
                result, count, restored = _file_readouts(root_path, metadata, rows, beta,
                    source_readout, cluster_sink if cache is not None else None)
                input_rows.append((file_index, str(root_path), metadata_text))
                eligible += count
                recovered += restored
                for mode in MODES:
                    summaries[mode].addInput(file_index, str(root_path), True, metadata_text)
                    for index in indices:
                        paired[mode][index] = result[mode][int(events["event"][index])]
                if position == 0 or (position + 1) % 25 == 0 or position + 1 == len(inputs["file_index"]):
                    print(f"Paired PID readouts: {position + 1}/{len(inputs['file_index'])} source files", flush=True)
        if cache is not None:
            _finish_cache(cache_path, input_rows, ROOT, energy, beta, method)
        for index in range(len(events["event"])):
            for mode in MODES:
                summaries[mode].observe(int(events["file_index"][index]), int(events["event"][index]),
                    float(events["thrust"][index]), float(paired[mode][index]), float(events["truth_energy"][index]))
        for mode in MODES:
            data_directory = staged / mode
            summaries[mode].write(str(data_directory), draw, str(staged), mode)
            if executable is not None:
                subprocess.run([str(executable), "--energy-readout", mode,
                    "--readout-cache", str(cache_path), "--output-directory", str(data_directory),
                    "--image-directory", str(staged)],
                    check=True, cwd=MACRO_DIRECTORY)
            result = ROOT.TFile.Open(str(data_directory / SUMMARY_NAME), "UPDATE")
            if not result or result.IsZombie():
                raise RuntimeError("Cannot write readout recovery metadata")
            try:
                ROOT.TObjString(json.dumps(dict(source_summary=str(summary_path), energy_readout=mode,
                    single_track_charged_clusters=eligible, h5_recovered_clusters=recovered,
                    recovery=("For h5_recovered_clusters: original H5 measured feature columns 7:10; input identity, float32 event truth/angle and all saved input rows validated. Other eligible clusters use the explicitly recorded measured-track policy and mass hypotheses.")), sort_keys=True)).Write("readout_recovery")
            finally:
                result.Close()
        _write_compact_summary(staged, output, summary_path, source_base, common_directory,
                               ROOT, energy, method, beta)
        _publish(staged, output)
    print(f"Saved one ROOT/text pair for raw_head and track_replaced: {len(events['event'])} events, {eligible} single-track charged clusters ({recovered} recovered from original H5); {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, help="Saved-energy or paired jet_energy_resolution_summary.root")
    parser.add_argument("--compact-only", action="store_true",
        help="Publish fresh native paired summaries and diagnostics without reopening prediction ROOTs")
    parser.add_argument("--staged-directory", type=Path,
        help="Native staging base containing raw_head/ and track_replaced/ products")
    parser.add_argument("--recover-measured", type=Path,
        help="Recover historical track energies from a native bridge ROOT and its identified H5")
    parser.add_argument("--recovery-output", type=Path,
        help="Output ROOT containing track_replacements for --recover-measured")
    parser.add_argument("--output-directory", type=Path,
        help="Output base for FORMAT/MODE images and one combined ROOT/text pair")
    parser.add_argument("--draw", action="store_true", help="Also save the angular resolution canvas in PNG/PDF/EPS")
    parser.add_argument("--analysis-executable", type=Path,
        help="Also run the matching-energy PID executable to save every energy-dependent diagnostic")
    parser.add_argument("--common-directory", type=Path,
        help="Directory containing freshly computed common clustering and diagnostic ROOT files")
    args = parser.parse_args(argv)
    try:
        if sum((args.summary is not None, args.compact_only,
                args.recover_measured is not None)) != 1:
            raise ValueError("Choose exactly one of --summary, --compact-only or --recover-measured")
        if args.compact_only:
            if (args.staged_directory is None or args.output_directory is None or
                    args.common_directory is None):
                raise ValueError("--compact-only requires --staged-directory, --output-directory and --common-directory")
            if args.recovery_output is not None or args.analysis_executable is not None or args.draw:
                raise ValueError("Native compact publication requires already completed plots and diagnostics")
            compact_native_outputs(args.staged_directory, args.output_directory, args.common_directory)
        elif args.recover_measured is not None:
            if args.recovery_output is None:
                raise ValueError("--recover-measured requires --recovery-output")
            if (args.staged_directory is not None or args.output_directory is not None or
                    args.common_directory is not None or args.analysis_executable is not None or args.draw):
                raise ValueError("Measured-energy recovery accepts only bridge and recovery-output paths")
            recover_measured_bridge(args.recover_measured, args.recovery_output)
        else:
            if args.staged_directory is not None or args.recovery_output is not None:
                raise ValueError("Staging/recovery options require their corresponding native mode")
            save_readouts(args.summary, args.output_directory, args.draw, args.analysis_executable,
                          args.common_directory)
    except Exception as error:
        print(f"Cannot save paired PID readouts: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
