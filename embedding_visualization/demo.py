#!/usr/bin/env python3
"""Create explicitly synthetic examples with aligned and deliberately swapped tracks."""
import argparse
from pathlib import Path
import numpy as np


def make_event(swapped=False):
    rng = np.random.default_rng(7)
    centers = np.array([[-3, 0, 0, 0], [3, 0, 0, 0], [0, 5, 0, 0]], dtype=float)
    hits = np.concatenate([c + rng.normal(0, .25, size=(60, 4)) for c in centers])
    tracks = centers.copy()
    if swapped:
        tracks[[0, 1]] = tracks[[1, 0]]
    z = np.concatenate([hits, tracks])
    ids = np.r_[np.repeat([1, 2, 3], 60), [1, 2, 3]].astype(np.int64)
    return dict(embedding=z, truth_id=ids, is_track=np.arange(len(z)) >= len(hits),
                truth_valid=np.ones(len(z), dtype=bool), beta=rng.uniform(.1, .9, len(z)),
                metadata_json=np.array('{"synthetic": true}'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=False)
    for i in range(2):
        np.savez_compressed(path/f'event_{i:06d}.npz', **make_event(bool(i)))
    print(f'SYNTHETIC data only (not checkpoint predictions): {path}')


if __name__ == '__main__':
    main()
