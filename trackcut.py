#!/usr/bin/env python3
"""Select track rows using explicit lcio-track-quality-2 cuts.

All known builders are retained. Dense input builders share a row mask; sparse
quality and PFO links are remapped. Reference targets and Pandora energies are
never filtered or recalibrated. The source H5 is never modified.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import awkward as ak
import h5py
import numpy as np


QUALITY_SCHEMA = "lcio-track-quality-2"
FILE_SCHEMAS = {"pandora-eval-1", "nnqq-2m-eval-1", "single-particle-eval-1"}
DENSE_GROUPS = {"feature", "label", "row_info", "pandora", "truth_assignment_500mm"}
KNOWN_GROUPS = DENSE_GROUPS | {
    "cluster", "pfo", "pfo_links", "truth_particles", "event", "event_eval",
    "collections", "source", "truth_targets_500mm", "track_quality",
}
REQUIRED_GROUPS = {"feature", "label", "row_info", "collections", "event", "track_quality"}


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _read_group(group):
    return ak.from_buffers(
        _text(group.attrs["form"]), int(group.attrs["length"]),
        {name: dataset[:] for name, dataset in group.items()})


def _load(path):
    with h5py.File(path, "r") as handle:
        names = set(handle)
        if names - KNOWN_GROUPS:
            raise ValueError(f"Unsupported builders; refusing to discard or guess their row axes: {sorted(names - KNOWN_GROUPS)}")
        if REQUIRED_GROUPS - names:
            raise ValueError(f"Missing required builders: {sorted(REQUIRED_GROUPS - names)}")
        if ("truth_targets_500mm" in names) != ("truth_assignment_500mm" in names):
            raise ValueError("Both 500 mm truth builders must be present together")
        if _text(handle.attrs.get("schema_version", "")) not in FILE_SCHEMAS:
            raise ValueError("A supported extended H5 file schema is required")
        if _text(handle["track_quality"].attrs.get("schema_version", "")) != QUALITY_SCHEMA:
            raise ValueError(f"track_quality must use {QUALITY_SCHEMA}; dense v1 is unsupported")
        arrays = {name: _read_group(group) for name, group in handle.items()}
        attrs = dict(handle.attrs)
        group_attrs = {name: dict(group.attrs) for name, group in handle.items()}
    n = len(arrays["feature"])
    if any(len(array) != n for array in arrays.values()):
        raise ValueError("Event counts differ between builders")
    return arrays, attrs, group_attrs


def _matrix(array, width, description):
    result = np.asarray(ak.to_numpy(array))
    if result.size == 0:
        return result.reshape(0, width)
    if result.ndim != 2 or result.shape[1] != width:
        raise ValueError(f"{description} must have width {width}")
    return result


def _integers(values, description, *, integer_dtype=False):
    values = np.asarray(values)
    if values.dtype.kind not in "iuf" or (integer_dtype and values.dtype.kind not in "iu"):
        raise ValueError(f"{description} must contain integer values")
    if values.dtype.kind == "f":
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{description} contains non-finite or fractional indices")
        if np.any(np.abs(values) > 2**53):
            raise ValueError(f"{description} exceeds exact float64 integer range")
    if values.size and (values.min() < np.iinfo(np.int64).min or values.max() > np.iinfo(np.int64).max):
        raise ValueError(f"{description} exceeds int64 range")
    return values.astype(np.int64)


def _flat_field(array, name):
    return np.asarray(ak.to_numpy(array[name]))


def _bool_field(array, name, description):
    value = _flat_field(array, name)
    if value.dtype.kind != "b":
        raise ValueError(f"{description}.{name} must be boolean")
    return value


def _validate_event(arrays, index):
    feature = _matrix(arrays["feature"][index], 13, "feature")
    label = _matrix(arrays["label"][index], 9, "label")
    info = _matrix(arrays["row_info"][index], 11, "row_info")
    n = len(feature)
    if len(label) != n or len(info) != n:
        raise ValueError(f"Event {index}: dense input row counts differ")
    kind = _integers(info[:, 0], "row_info.kind")
    if not np.isin(kind, [0, 1]).all() or not np.array_equal(feature[:, 5], kind):
        raise ValueError(f"Event {index}: feature and row_info track flags differ")
    for name in DENSE_GROUPS & arrays.keys():
        if len(arrays[name][index]) != n:
            raise ValueError(f"Event {index}: {name} is not feature-row aligned")
    q = arrays["track_quality"][index]
    for field in ("input_row", "collection", "element", "state_ip", "state_calo"):
        if field not in ak.fields(q):
            raise ValueError(f"Event {index}: missing track_quality.{field}")
    rows = _integers(_flat_field(q, "input_row"), "track_quality.input_row", integer_dtype=True)
    expected = np.flatnonzero(kind == 1)
    if not np.array_equal(rows, expected):
        raise ValueError(f"Event {index}: track_quality must cover all track rows exactly in feature order")
    for field, column in (("collection", 1), ("element", 2)):
        values = _integers(_flat_field(q, field), f"track_quality.{field}", integer_dtype=True)
        if not np.array_equal(values, _integers(info[rows, column], f"row_info.{field}")):
            raise ValueError(f"Event {index}: quality {field} differs from row_info")
    collection = _integers(info[:, 1], "row_info.collection")
    if np.any(collection < 0) or np.any(collection >= len(arrays["collections"][index])):
        raise ValueError(f"Event {index}: invalid collection index")
    for state_name in ("state_ip", "state_calo"):
        state = q[state_name]
        if not {"present", "r", "r_valid", "covariance_valid"}.issubset(ak.fields(state)):
            raise ValueError(f"Event {index}: missing fields in {state_name}")
        present = _bool_field(state, "present", state_name)
        valid = _bool_field(state, "r_valid", state_name)
        _bool_field(state, "covariance_valid", state_name)
        r = _flat_field(state, "r")
        if np.any(valid & (~present | ~np.isfinite(r) | (r < 0))):
            raise ValueError(f"Event {index}: r_valid contradicts {state_name} presence/value")
    if "pfo_links" in arrays:
        links = _matrix(arrays["pfo_links"][index], 5, "pfo_links")
        link_rows = _integers(links[:, 3], "pfo_links.input_row")
        if np.any(link_rows < 0) or np.any(link_rows >= n):
            raise ValueError(f"Event {index}: PFO link references a missing input row")
        if not np.array_equal(links[:, 1], kind[link_rows]):
            raise ValueError(f"Event {index}: PFO link kind differs from row_info")
        signed = np.where(kind[link_rows] == 1, -links[:, 2], links[:, 2])
        if not np.array_equal(signed, info[link_rows, 9]):
            raise ValueError(f"Event {index}: PFO link object differs from row_info")
    return feature, info, q, rows


def _ragged_vectors(events, dtype):
    counts = np.asarray([len(event) for event in events], dtype=np.int64)
    flat = np.concatenate(events).astype(dtype, copy=False) if counts.sum() else np.empty(0, dtype=dtype)
    return ak.unflatten(flat, counts)


def _remap_links(links, mappings):
    results = []
    for index, mapping in enumerate(mappings):
        values = _matrix(links[index], 5, "pfo_links")
        old_rows = _integers(values[:, 3], "pfo_links.input_row")
        selected = mapping[old_rows] >= 0
        result = values[selected].copy()
        result[:, 3] = mapping[old_rows[selected]]
        results.append(result)
    # Preserve fixed-width and numeric dtype even when every link is removed.
    dtype = ak.to_numpy(ak.flatten(links, axis=1)).dtype
    flat = np.concatenate(results, axis=0) if results else np.empty((0, 5), dtype=dtype)
    return ak.unflatten(ak.Array(flat), np.asarray([len(result) for result in results], dtype=np.int64))


def _write_atomic(destination, arrays, attrs, group_attrs, replace):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".trackcut-", suffix=".h5", dir=destination.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as handle:
            for key, value in attrs.items():
                handle.attrs[key] = value
            for name, array in arrays.items():
                group = handle.create_group(name)
                for key, value in group_attrs[name].items():
                    if key not in ("form", "length"):
                        group.attrs[key] = value
                # Packing twice avoids retained buffers in nested empty slices on Awkward 2.1.
                packed = ak.to_packed(ak.to_packed(array))
                form, length, buffers = ak.to_buffers(packed)
                group.attrs["form"] = form.to_json()
                group.attrs["length"] = json.dumps(length)
                for key, value in buffers.items():
                    group.create_dataset(key, data=value, compression="gzip", compression_opts=1, shuffle=True)
            handle.flush()
        # Serialized metadata and references must remain valid before publication.
        checked, _, _ = _load(temporary)
        for index in range(len(checked["feature"])):
            _validate_event(checked, index)
        del checked
        if replace:
            os.replace(temporary, destination)
        else:
            # Unlike exists()+replace(), this cannot overwrite a concurrently created file.
            os.link(temporary, destination)
            os.unlink(temporary)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def track_cut_file(input_path, output_path, *, max_r, invalid_r_policy,
                   state="calo", require_valid_covariance=False,
                   maximum_time=None, minimum_pt=None, nstart=0, nend=-1,
                   replace=False, dry_run=False):
    """Write selected inputs. Keep tracks with valid r strictly below max_r.

    Invalid r is explicitly kept, dropped, or rejected according to the policy.
    Requiring valid full covariance is an independent, opt-in additional cut.
    Optional timing and pT cuts use the legacy comparisons time < max and pT > min.
    """
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve() or (
            destination.exists() and os.path.samefile(source, destination)):
        raise ValueError("Input and output must be different files")
    if state not in ("ip", "calo"):
        raise ValueError("state must be ip or calo")
    if invalid_r_policy not in ("keep", "drop", "error"):
        raise ValueError("invalid_r_policy must be keep, drop or error")
    if not np.isfinite(max_r) or max_r <= 0:
        raise ValueError("max_r must be finite and positive")
    if maximum_time is not None and not np.isfinite(maximum_time):
        raise ValueError("maximum_time must be finite")
    if minimum_pt is not None and (not np.isfinite(minimum_pt) or minimum_pt < 0):
        raise ValueError("minimum_pt must be finite and nonnegative")
    if nstart < 0 or nend < -1 or (nend >= 0 and nend < nstart):
        raise ValueError("Require nstart >= 0 and nend == -1 or nend >= nstart")
    if os.path.lexists(destination) and not replace and not dry_run:
        raise FileExistsError(f"Output already exists: {destination}")
    arrays, attrs, group_attrs = _load(source)
    total = len(arrays["feature"])
    stop = total if nend == -1 else min(nend, total)
    arrays = {name: value[nstart:stop] for name, value in arrays.items()}
    masks, q_masks, q_rows, mappings = [], [], [], []
    counters = dict(rows_before=0, rows_after=0, tracks_before=0, tracks_after=0,
                    invalid_r_tracks=0, r_rejected_tracks=0,
                    covariance_rejected_tracks=0)
    for index in range(len(arrays["feature"])):
        feature, info, q, rows = _validate_event(arrays, index)
        chosen = q["state_" + state]
        valid = _flat_field(chosen, "r_valid")
        r = _flat_field(chosen, "r")
        invalid = ~valid
        if invalid_r_policy == "error" and np.any(invalid):
            raise ValueError(f"Event {nstart + index}: {int(invalid.sum())} tracks have invalid {state} r")
        keep_track = np.where(valid, r < max_r, invalid_r_policy == "keep")
        counters["invalid_r_tracks"] += int(invalid.sum())
        counters["r_rejected_tracks"] += int(np.sum(valid & (r >= max_r)))
        if require_valid_covariance:
            covariance_valid = _flat_field(chosen, "covariance_valid")
            counters["covariance_rejected_tracks"] += int(np.sum(keep_track & ~covariance_valid))
            keep_track &= covariance_valid
        mask = np.ones(len(feature), dtype=bool)
        mask[rows] = keep_track
        if maximum_time is not None:
            mask &= feature[:, 4] < maximum_time
        if minimum_pt is not None:
            pt = np.hypot(feature[:, 7], feature[:, 8])
            mask &= (info[:, 0] == 0) | (pt > minimum_pt)
        mapping = np.full(len(feature), -1, dtype=np.int64)
        mapping[mask] = np.arange(int(mask.sum()), dtype=np.int64)
        selected_quality = mask[rows]
        masks.append(mask)
        q_masks.append(selected_quality)
        q_rows.append(mapping[rows[selected_quality]])
        mappings.append(mapping)
        counters["rows_before"] += len(feature)
        counters["rows_after"] += int(mask.sum())
        counters["tracks_before"] += len(rows)
        counters["tracks_after"] += int(selected_quality.sum())
    mask = _ragged_vectors(masks, np.bool_)
    for name in DENSE_GROUPS & arrays.keys():
        arrays[name] = arrays[name][mask]
    q_fields = ak.fields(arrays["track_quality"])
    q = arrays["track_quality"][_ragged_vectors(q_masks, np.bool_)]
    q = ak.with_field(q, _ragged_vectors(q_rows, np.int64), "input_row")
    arrays["track_quality"] = q[q_fields]
    if "pfo_links" in arrays:
        arrays["pfo_links"] = _remap_links(arrays["pfo_links"], mappings)
    if "event_eval" in arrays:
        ev = np.asarray(ak.to_numpy(arrays["event_eval"])).copy()
        if ev.ndim != 2 or ev.shape[1] != 23:
            raise ValueError("event_eval must have width 23")
        ev[:, 19] = np.arange(len(ev))
        arrays["event_eval"] = ak.Array(ev)
    summary = dict(events=len(arrays["feature"]), **counters)
    summary["tracks_removed"] = summary["tracks_before"] - summary["tracks_after"]
    summary["rows_removed"] = summary["rows_before"] - summary["rows_after"]
    summary["groups"] = sorted(arrays)
    options = dict(max_r=float(max_r), r_comparison="<", invalid_r_policy=invalid_r_policy,
                   state=state, require_valid_covariance=bool(require_valid_covariance),
                   maximum_time=maximum_time, time_comparison="<", minimum_pt=minimum_pt,
                   pt_comparison=">", nstart=nstart, nend=stop)
    record = dict(input_path=str(source.resolve()), options=options, summary=summary,
        pfo_links="Only links to retained inputs; input_row remapped. PFO quantities unchanged.",
        event_eval_counts="Original legacy/added row counters retained as source provenance.",
        target_policy="Reference target tables and event/Pandora energies unchanged.")
    history = json.loads(_text(attrs.get("track_cut_history", "[]")))
    if not isinstance(history, list):
        raise ValueError("track_cut_history must be a JSON list")
    attrs["training_only"] = True
    attrs["track_cut"] = json.dumps(record, sort_keys=True)
    attrs["track_cut_history"] = json.dumps(history + [record], sort_keys=True)
    summary["dry_run"] = bool(dry_run)
    if not dry_run:
        _write_atomic(destination, arrays, attrs, group_attrs, replace)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--max-r", type=float, required=True, help="Keep valid r strictly below this value")
    parser.add_argument("--invalid-r-policy", choices=("keep", "drop", "error"), required=True)
    parser.add_argument("--state", choices=("ip", "calo"), default="calo")
    parser.add_argument("--require-valid-covariance", action="store_true")
    parser.add_argument("--maximumTime", "--maximum-time", dest="maximum_time", type=float)
    parser.add_argument("--minimumPt", "--minimum-pt", dest="minimum_pt", type=float)
    parser.add_argument("--nstart", type=int, default=0)
    parser.add_argument("--nend", type=int, default=-1, help="Exclusive event index; -1 means all")
    parser.add_argument("--replace", action="store_true", help="Explicitly allow replacing an existing output")
    parser.add_argument("--dry-run", action="store_true", help="Validate/count selections without writing output")
    args = parser.parse_args()
    result = track_cut_file(args.input, args.output, max_r=args.max_r,
        invalid_r_policy=args.invalid_r_policy, state=args.state,
        require_valid_covariance=args.require_valid_covariance,
        maximum_time=args.maximum_time, minimum_pt=args.minimum_pt,
        nstart=args.nstart, nend=args.nend, replace=args.replace, dry_run=args.dry_run)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
