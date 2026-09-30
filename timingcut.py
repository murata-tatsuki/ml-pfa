"""Write a compact training H5 with the legacy time and track-pT cuts.

Extended inputs retain feature/label/row_info/collections/event and file attributes.
Legacy inputs retain feature/label and event when available. Evaluation-only
builders are neither loaded nor written. The source H5 is never modified.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import awkward as ak
import h5py
import numpy as np

from extended_h5 import TRAINING_SCHEMAS, read_training_bundle, validate_rows


def _read_group(group):
    return ak.from_buffers(json.loads(group.attrs['form']), int(group.attrs['length']),
                           {key: value[:] for key, value in group.items()})


def timing_cut_file(input_path, output_path, maximum_time=14, minimum_pt=0.3,
                    nstart=0, nend=-1):
    """Apply time < maximum_time and (calo or track pT > minimum_pt).

    nend is an exclusive event index, matching the original CLI. Empty events,
    unknown truth rows and gap rows are retained if they pass these cuts; gap
    and feature-validity selection remain the training loader's responsibility.
    """
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve() or (
            destination.exists() and os.path.samefile(source, destination)):
        raise ValueError('Input and output must be different files')
    if not np.isfinite(maximum_time) or not np.isfinite(minimum_pt) or minimum_pt < 0:
        raise ValueError('Cut thresholds must be finite and minimumPt must be nonnegative')
    if nstart < 0 or nend < -1 or (nend >= 0 and nend < nstart):
        raise ValueError('Require nstart >= 0 and nend == -1 or nend >= nstart')

    with h5py.File(source, 'r') as handle:
        attrs = dict(handle.attrs)
        schema = attrs.get('schema_version')
        if schema is not None and schema not in TRAINING_SCHEMAS:
            raise ValueError(f'Unsupported H5 schema: {schema}')
        extended = schema in TRAINING_SCHEMAS
        if not extended:
            if 'row_info' in handle or 'collections' in handle:
                raise ValueError('Extended builders require a supported schema_version and metadata')
            names = ['feature', 'label'] + (['event'] if 'event' in handle else [])
            arrays = {name: _read_group(handle[name]) for name in names}
    if extended:
        arrays = read_training_bundle(source)
    total = len(arrays['feature'])
    if any(len(value) != total for value in arrays.values()):
        raise ValueError('Event counts differ between training builders')
    stop = total if nend == -1 else min(nend, total)
    arrays = {name: value[nstart:stop] for name, value in arrays.items()}
    feat = arrays['feature']
    if not ak.all(ak.num(feat, axis=1) == ak.num(arrays['label'], axis=1)):
        raise ValueError('feature and label row counts differ')
    if extended:
        for index in range(len(feat)):
            validate_rows(
                np.asarray(ak.to_numpy(feat[index])).reshape(-1, 13),
                np.asarray(ak.to_numpy(arrays['label'][index])).reshape(-1, 9),
                np.asarray(ak.to_numpy(arrays['row_info'][index])).reshape(-1, 11),
                ak.to_list(arrays['collections'][index]))

    before = int(ak.sum(ak.num(feat, axis=1)))
    # Do not inspect truth when selecting inputs. Match ILCDataset.timingCut.
    if before:
        pt = np.sqrt(feat[:, :, 7] ** 2 + feat[:, :, 8] ** 2)
        mask = (feat[:, :, 4] < maximum_time) & ((feat[:, :, 5] == 0) | (pt > minimum_pt))
        for name in ('feature', 'label', 'row_info'):
            if name in arrays:
                arrays[name] = arrays[name][mask]
    after = int(ak.sum(ak.num(arrays['feature'], axis=1)))
    attrs['training_only'] = True
    attrs['timing_cut'] = json.dumps(dict(
        input_path=str(source.resolve()), maximum_time=maximum_time,
        minimum_pt=minimum_pt, time_comparison='<', pt_comparison='>',
        nstart=nstart, nend=stop, rows_before=before, rows_after=after))

    # Packing is essential: otherwise Awkward buffers can retain removed rows.
    # Write atomically so failed conversion cannot leave a partial training file.
    fd, temporary = tempfile.mkstemp(prefix='.timingcut-', suffix='.h5',
                                     dir=destination.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, 'w') as handle:
            for key, value in attrs.items():
                handle.attrs[key] = value
            for name, array in arrays.items():
                group = handle.create_group(name)
                # Awkward 2.1 needs a second pass for nested empty slices.
                packed = ak.to_packed(ak.to_packed(array))
                form, length, buffers = ak.to_buffers(packed)
                group.attrs['form'] = form.to_json()
                group.attrs['length'] = json.dumps(length)
                for key, value in buffers.items():
                    group.create_dataset(key, data=value, compression='gzip',
                                         compression_opts=1, shuffle=True)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return dict(events=len(arrays['feature']), rows_before=before, rows_after=after,
                groups=list(arrays))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--maximumTime', type=float, default=14)
    parser.add_argument('--minimumPt', type=float, default=0.3)
    parser.add_argument('--nstart', type=int, default=0)
    parser.add_argument('--nend', type=int, default=-1, help='Exclusive event index; -1 means all')
    args = parser.parse_args()
    result = timing_cut_file(args.input, args.output, args.maximumTime,
                             args.minimumPt, args.nstart, args.nend)
    print(f'Saved {args.output}: {json.dumps(result)}')


if __name__ == '__main__':
    main()
