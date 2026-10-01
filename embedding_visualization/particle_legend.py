#!/usr/bin/env python3
"""Separate particle legend images, using the EXACT colors saved with each plot."""
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import numpy as np

from core import display_ids
from visual_style import save_figure

# PDG Monte Carlo convention: https://pdg.lbl.gov/2026/mcdata/mc_particle_id_contents.html
PARTICLES = {11: 'electron', -11: 'positron', 13: 'mu-', -13: 'mu+',
    15: 'tau-', -15: 'tau+', 12: 'nu_e', -12: 'anti-nu_e',
    14: 'nu_mu', -14: 'anti-nu_mu', 16: 'nu_tau', -16: 'anti-nu_tau',
    22: 'photon', 111: 'pi0', 211: 'pi+', -211: 'pi-',
    130: 'K0_L', 310: 'K0_S', 311: 'K0', -311: 'anti-K0', 321: 'K+', -321: 'K-',
    2112: 'neutron', -2112: 'anti-neutron', 2212: 'proton', -2212: 'anti-proton',
    3122: 'Lambda0', -3122: 'anti-Lambda0'}


def truth_arrays(labels, known):
    """Use the same mass and MC momentum columns as display/display_h5.py."""
    labels = np.asarray(labels, dtype=np.float64)
    known = np.asarray(known, dtype=bool)
    if labels.ndim != 2 or labels.shape[1] < 8 or known.shape != (len(labels),):
        raise ValueError('Expected row-aligned labels with PDG/mass/MC momentum columns')
    pdg = labels[known, 2]
    if not np.isfinite(pdg).all() or not np.equal(pdg, np.rint(pdg)).all():
        raise ValueError('Invalid known-truth PDG code')
    kinematics = labels[known, 4:8]
    if not np.isfinite(kinematics).all():
        raise ValueError('Non-finite known-truth mass or MC momentum')
    result = dict(truth_pdg_id=np.zeros(len(labels), dtype=np.int64),
                  truth_energy=np.full(len(labels), np.nan, dtype=np.float64))
    result['truth_pdg_id'][known] = pdg.astype(np.int64)
    result['truth_energy'][known] = np.linalg.norm(kinematics, axis=1)
    return result


@lru_cache(maxsize=1)
def read_labels(path):
    """Old NPZ compatibility: read the label group only; one H5 cached at a time."""
    import h5py
    import awkward as ak
    with h5py.File(path, 'r') as h5:
        group = h5['label']
        return ak.from_buffers(json.loads(group.attrs['form']), int(group.attrs['length']),
                               {key: value[:] for key, value in group.items()})


def legend_rows(event, colors, selected):
    ids = display_ids(event)
    metadata = json.loads(str(event['metadata_json'])) if 'metadata_json' in event else {}
    raw = None
    if 'truth_pdg_id' in event and 'truth_energy' in event:
        if event['truth_pdg_id'].shape != ids.shape or event['truth_energy'].shape != ids.shape:
            raise ValueError('Particle metadata must align with saved event rows')
        source = 'saved NPZ truth metadata'
    elif 'source_file' in metadata and 'source_event' in metadata and 'mc_id' in event:
        import awkward as ak
        labels = read_labels(metadata['source_file'])
        raw_labels = np.asarray(ak.to_numpy(labels[int(metadata['source_event'])]), dtype=np.float64)
        raw = truth_arrays(raw_labels, raw_labels[:, 1] >= 0)
        raw_ids = raw_labels[:, 1]
        if not np.isfinite(raw_ids).all() or not np.equal(raw_ids, np.rint(raw_ids)).all():
            raise ValueError('Invalid MC IDs in source H5')
        raw['mc_id'] = raw_ids.astype(np.int64)
        source = 'original H5 labels matched by exact MC ID'
    else:
        source = 'unavailable (no truth metadata or source H5 reference)'
    rows = []
    for label in sorted(int(key) for key in colors):
        mask = ids == label
        if not mask.any():
            continue
        row = dict(truth_id=label, mc_id='', particle='Unknown/noise' if label == 0 else 'Unavailable',
                   pdg='', energy_min_GeV=None, energy_max_GeV=None, energy_GeV='N/A',
                   color_hex=to_hex(colors[str(label)]), rgba=list(colors[str(label)]),
                   shown_points=int(np.sum(ids[selected] == label)), total_points=int(mask.sum()))
        if label > 0:
            if 'mc_id' in event:
                mcids = np.unique(event['mc_id'][mask])
                if len(mcids) != 1:
                    raise ValueError(f'Truth ID {label} corresponds to multiple MC IDs')
                row['mc_id'] = str(int(mcids[0]))
            if raw is not None:
                selected_truth = raw['mc_id'] == int(row['mc_id'])
                if not selected_truth.any():
                    raise ValueError(f'MC ID {row["mc_id"]} is missing in source H5')
                codes = raw['truth_pdg_id'][selected_truth]
                energy = raw['truth_energy'][selected_truth]
            elif 'truth_pdg_id' in event and 'truth_energy' in event:
                codes, energy = event['truth_pdg_id'][mask], event['truth_energy'][mask]
            else:
                codes, energy = np.array([]), np.array([])
            if len(codes):
                if not np.isfinite(codes).all() or not np.equal(codes, np.rint(codes)).all():
                    raise ValueError(f'Invalid PDG metadata for truth ID {label}')
                codes = np.unique(codes.astype(np.int64))
                row['pdg'] = ', '.join(str(code) for code in codes)
                row['particle'] = ' / '.join(PARTICLES.get(int(code), f'PDG {code}') for code in codes)
            if len(energy) and np.isfinite(energy).all():
                low, high = float(np.min(energy)), float(np.max(energy))
                row['energy_min_GeV'], row['energy_max_GeV'] = low, high
                row['energy_GeV'] = f'{np.mean(energy):.5g}' if np.isclose(low, high, rtol=1e-5, atol=1e-7) else f'{low:.5g} - {high:.5g}'
        rows.append(row)
    return rows, source, metadata


def write_legend(event, colors, selected, output, dpi=180, rows_per_page=35, overwrite=False, styles=None):
    if rows_per_page < 1 or dpi < 1:
        raise ValueError('rows-per-page and dpi must be positive')
    output = Path(output)
    rows, source, metadata = legend_rows(event, colors, selected)
    modern = styles is not None
    styles = styles or {key: dict(mpl_marker='o', shape_name='circle', plotly_symbol='circle') for key in colors}
    for row in rows:
        row.update({key: styles[str(row['truth_id'])][key] for key in ('mpl_marker', 'shape_name', 'plotly_symbol')})
    if not rows:
        raise ValueError('No particle rows to draw')
    pages = [rows[i:i+rows_per_page] for i in range(0, len(rows), rows_per_page)]
    names = ['particle_legend.png'] + [f'particle_legend_{i+1:02d}.png' for i in range(1, len(pages))]
    pdf_names = [str(Path(name).with_suffix('.pdf')) for name in names]
    targets = [output/name for name in names + pdf_names] + [output/'particle_legend.csv', output/'particle_legend.json']
    if not overwrite:
        for target in targets:
            if target.exists():
                raise FileExistsError(target)
    output.mkdir(parents=True, exist_ok=True)
    source_name = Path(metadata.get('source_file', '')).name or 'event'
    title = f'{source_name} | event {metadata.get("source_event", "?")} | Particle legend'
    for page_index, page in enumerate(pages):
        fig, ax = plt.subplots(figsize=(12.8, 2.2 + .32*len(page)))
        ax.axis('off')
        headers = ['Symbol', 'Truth ID', 'MC ID', 'Particle', 'PDG', 'MC energy [GeV]', 'Shown / all']
        cells = [['', row['truth_id'], row['mc_id'], row['particle'], row['pdg'], row['energy_GeV'],
                  f'{row["shown_points"]} / {row["total_points"]}'] for row in page]
        table = ax.table(cellText=cells, colLabels=headers, cellLoc='center', loc='center',
                         colWidths=[.06, .08, .14, .23, .10, .17, .12], bbox=[0, .02, 1, .96])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#d1d8e0')
            cell.set_linewidth(.5)
            if r == 0:
                cell.set_facecolor('#eaf0f7')
                cell.set_text_props(weight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#f7f9fc')
        fig.suptitle(title + (f' ({page_index+1}/{len(pages)})' if len(pages)>1 else ''), fontsize=13, y=.97)
        base = Line2D([], [], marker='o', color='#555555', markersize=4, linestyle='')
        if modern:
            track = (Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor='#555555', markersize=6, linestyle=''),
                     Line2D([], [], marker='+', color='black', markersize=4, markeredgewidth=.8, linestyle=''))
            highest = (base, Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor='black', markersize=7, markeredgewidth=.8, linestyle=''))
        else:
            track = Line2D([], [], marker='*', color='black', markersize=6, linestyle='')
            highest = Line2D([], [], marker='s', markerfacecolor='none', color='black', markersize=6, linestyle='')
        fig.legend([base, track, highest], ['Calo hit: cluster color + shape',
                   'Track: outline + plus' if modern else 'Track: star',
                   'Highest beta calo hit: black ring' if modern else 'Highest beta calo hit: square'],
                   handler_map={tuple: HandlerTuple(ndivide=1)}, ncol=3, frameon=False, fontsize=9,
                   loc='upper center', bbox_to_anchor=(.5, 1-.65/fig.get_figheight()))
        fig.text(.04, .025, 'MC energy = sqrt(m^2 + px^2 + py^2 + pz^2); not deposited/predicted energy.\n'
                 'Highest beta: max FINAL beta among calo hits per truth cluster (marked only if displayed).\n'
                 'Same colors/shapes in all layers. Shown / all = displayed / saved points. '
                 + ('Particle metadata unavailable.' if source.startswith('unavailable') else ''), fontsize=9)
        fig.subplots_adjust(top=1-1.15/fig.get_figheight(), bottom=.85/fig.get_figheight(), left=.035, right=.985)
        fig.canvas.draw()
        for i, row in enumerate(page, start=1):
            cell = table[i, 0]
            ax.scatter([cell.get_x()+cell.get_width()/2], [cell.get_y()+cell.get_height()/2],
                       marker=row['mpl_marker'], s=45, c=[row['rgba']], linewidths=0,
                       transform=ax.transAxes, zorder=10, clip_on=False)
        save_figure(fig, output/names[page_index], dpi)
        plt.close(fig)
    fields = [key for key in rows[0] if key != 'rgba']
    with (output/'particle_legend.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    result = dict(metadata_source=source, event_metadata=metadata, images=names, pdfs=pdf_names,
                  role_markers=dict(track='outlined cluster shape + small plus' if modern else 'star',
                                    highest_beta='black ring' if modern else 'open square',
                                    highest_beta_definition='Maximum final beta among calo hits in each truth cluster; only if sampled'),
                  energy_definition='MC total energy sqrt(m^2+px^2+py^2+pz^2), GeV', particles=rows)
    (output/'particle_legend.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if overwrite:
        for suffix in ('png', 'pdf'):
            for old in output.glob(f'particle_legend_[0-9][0-9].{suffix}'):
                if old.name not in names + pdf_names:
                    old.unlink()
    return targets[0]


def add_legends(root, output_root=None, dpi=180, rows_per_page=35, overwrite=False):
    root = Path(root).resolve()
    groups = {}
    for path in sorted(root.rglob('report.json')):
        report = json.loads(path.read_text())
        if 'truth_colors' not in report or 'embedding_key' not in report:
            continue
        event_dir = path.parent.parent if path.parent.name == report['embedding_key'] else path.parent
        selected = np.load(path.parent/'selected_rows.npy', allow_pickle=False)
        if event_dir in groups:
            previous, previous_rows = groups[event_dir]
            if (report['source'] != previous['source'] or report['truth_colors'] != previous['truth_colors']
                    or report.get('truth_styles') != previous.get('truth_styles')
                    or not np.array_equal(selected, previous_rows)):
                raise ValueError(f'{event_dir}: layer colors, source, or displayed points differ')
        groups[event_dir] = (report, selected)
    if not groups:
        raise ValueError(f'No plot reports found: {root}')
    written = []
    for event_dir, (report, selected) in groups.items():
        with np.load(report['source'], allow_pickle=False) as saved:
            event = {key: saved[key] for key in saved.files}
        destination = event_dir if output_root is None else Path(output_root)/event_dir.relative_to(root)
        written.append(write_legend(event, report['truth_colors'], selected, destination, dpi, rows_per_page, overwrite,
                                    styles=report.get('truth_styles')))
        print(f'Particle legend: {written[-1]}', flush=True)
    return written


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots', required=True, help='Existing plots directory or parent containing multiple runs')
    parser.add_argument('--output', help='Optional separate output root; default: beside each event display')
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--rows-per-page', type=int, default=35)
    parser.add_argument('--overwrite', action='store_true', help='Replace only legend outputs')
    args = parser.parse_args()
    add_legends(args.plots, args.output, args.dpi, args.rows_per_page, args.overwrite)
