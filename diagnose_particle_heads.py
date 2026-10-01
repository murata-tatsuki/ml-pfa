#!/usr/bin/env python3
"""Truth-grouped, truth-PID energy diagnostic. This is not normal reconstruction."""
import argparse
import csv
import os
from pathlib import Path
import tempfile

import torch
from dataset_ilc_sharded import ILCDatasetSharded
from model import get_model
from particle_heads import raw_model, truth_objects, object_energies, SPECIES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', help='Training H5 file or directory')
    p.add_argument('output', help='New CSV file; existing files are not overwritten')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--thetaphi', action='store_true')
    p.add_argument('--momentum', action='store_true')
    p.add_argument('--momentum-amp', action='store_true')
    p.add_argument('--extended-h5-input', action='store_true')
    p.add_argument('--exclude-gap-hits', action='store_true')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--stop', type=int, default=-1)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.exclude_gap_hits and not args.extended_h5_input:
        p.error('--exclude-gap-hits requires --extended-h5-input')
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    dataset = ILCDatasetSharded(args.input, test_mode=True,
        thetaphi=args.thetaphi, momentum=args.momentum, momentumAmp=args.momentum_amp,
        extended_h5_input=args.extended_h5_input, exclude_gap_hits=args.exclude_gap_hits)
    stop = len(dataset) if args.stop < 0 else min(len(dataset), args.stop)
    if not 0 <= args.start < stop:
        p.error('Requested event range is empty or invalid')
    first = dataset[args.start]
    model = get_model(args.checkpoint, jit=False, input_dim=first.x.shape[1],
                      energy_regression=True, energy_regression_cluster=True).to(args.device).eval()
    if not raw_model(model).five_particle_energy_heads:
        p.error('This diagnostic requires a five-particle-energy-heads checkpoint')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix='.'+output.name, suffix='.tmp', dir=output.parent)
    fields = ('event', 'truth_id', 'truth_pid', 'truth_species', 'pid_at_alpha', 'alpha_input_row',
              'n_tracks', 'n_calo', 'energy_valid', 'missing_track', 'truth_energy', 'predicted_energy', 'response')
    count = 0
    try:
        with os.fdopen(fd, 'w', newline='') as f, torch.no_grad():
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for event in range(args.start, stop):
                data = dataset[event].to(args.device)
                data.batch = torch.zeros(len(data.x), dtype=torch.long, device=args.device)
                result = model(data.x, data.batch, return_dict=True)
                o = truth_objects(data, result['clustering'][:, 0].sigmoid())
                energy = object_energies(result['regressions'], o)
                for j in range(len(o['alpha'])):
                    species = int(o['species'][j]); alpha = int(o['alpha'][j])
                    valid = bool(o['energy_valid'][j]) and bool(torch.isfinite(energy[j]))
                    target = float(o['target'][j]); predicted = float(energy[j]) if valid else float('nan')
                    pid = int(result['pid_logits'][alpha].argmax()) if 'pid_logits' in result else -1
                    rows = getattr(data, 'input_row', None)
                    writer.writerow(dict(event=event, truth_id=int(data.y[o['first'][j], 0]),
                        truth_pid=species, truth_species=SPECIES[species] if species >= 0 else 'unsupported',
                        pid_at_alpha=pid, alpha_input_row=int(rows[alpha]) if rows is not None else alpha,
                        n_tracks=int(o['track_count'][j]), n_calo=int(o['calo_count'][j]), energy_valid=int(valid),
                        missing_track=int(0 <= species < 3 and o['track_count'][j] == 0),
                        truth_energy=target, predicted_energy=predicted, response=predicted/target if valid else float('nan')))
                    count += 1
        os.link(staging, output)
    finally:
        os.unlink(staging)
    print(f'Wrote {count} truth objects to {output}; energy uses truth grouping and truth PID')


if __name__ == '__main__':
    main()
