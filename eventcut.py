"""Keep or drop whole events, preserving H5 builders and metadata.

Keep events with at least two input rows and at least one label with mcid != -1,
matching ILCDataset.eventCut. No time, pT or other row cuts are applied here;
run timingcut.py or timingcut_single_particle.py first when needed.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import awkward as ak
import h5py
import numpy as np


def _read_group(group):
    return ak.from_buffers(json.loads(group.attrs['form']), int(group.attrs['length']),
                           {key: value[:] for key, value in group.items()})


def event_cut_file(input_path, output_path, nstart=0, nend=-1):
    """Apply one event mask to every Awkward builder; nend is exclusive."""
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve() or (
            destination.exists() and os.path.samefile(source, destination)):
        raise ValueError('Input and output must be different files')
    if nstart < 0 or nend < -1 or (nend >= 0 and nend < nstart):
        raise ValueError('Require nstart >= 0 and nend == -1 or nend >= nstart')

    with h5py.File(source, 'r') as src:
        # Include optional builders (e.g. truth_particles) as well as training
        # builders, so retaining a schema never leaves event tables misaligned.
        arrays = {name: _read_group(group) for name, group in src.items()
                  if isinstance(group, h5py.Group)
                  and 'form' in group.attrs and 'length' in group.attrs}
        if not {'feature', 'label'} <= arrays.keys():
            raise ValueError('Input requires feature and label Awkward builders')
        total = len(arrays['feature'])
        if any(len(array) != total for array in arrays.values()):
            raise ValueError('Event counts differ between H5 builders')
        stop = total if nend == -1 else min(nend, total)
        arrays = {name: array[nstart:stop] for name, array in arrays.items()}
        counts = ak.num(arrays['feature'], axis=1)
        if not ak.all(counts == ak.num(arrays['label'], axis=1)):
            raise ValueError('feature and label row counts differ')
        if 'row_info' in arrays and not ak.all(counts == ak.num(arrays['row_info'], axis=1)):
            raise ValueError('feature and row_info row counts differ')

        keep = np.zeros(len(counts), dtype=bool)
        for index in np.flatnonzero(ak.to_numpy(counts >= 2)):
            # The legacy cluster remapping reserves zero for noise; after
            # removing mcid == -1, any remaining label makes the event valid.
            keep[index] = bool(ak.any(arrays['label'][index][:, 1] != -1))
        report = dict(events_before=len(keep), events=int(keep.sum()),
                      events_skipped=int((~keep).sum()), groups=list(arrays))

        fd, temporary = tempfile.mkstemp(prefix='.eventcut-', suffix='.h5',
                                         dir=destination.parent)
        os.close(fd)
        try:
            with h5py.File(temporary, 'w') as dst:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                dst.attrs['event_cut'] = json.dumps(dict(
                    input_path=str(source.resolve()), nstart=nstart, nend=stop,
                    minimum_rows=2, require_non_noise_label=True, **report))
                for name, obj in src.items():
                    if name not in arrays:
                        src.copy(obj, dst, name=name)
                        continue
                    group = dst.create_group(name)
                    for key, value in obj.attrs.items():
                        group.attrs[key] = value
                    # Repack twice for nested empty slices on Awkward 2.1.
                    packed = ak.to_packed(ak.to_packed(arrays[name][keep]))
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
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--nstart', type=int, default=0)
    parser.add_argument('--nend', type=int, default=-1, help='Exclusive event index; -1 means all')
    args = parser.parse_args()
    result = event_cut_file(args.input, args.output, args.nstart, args.nend)
    print(f'Saved {args.output}: {json.dumps(result)}')


if __name__ == '__main__':
    main()
