#!/usr/bin/env python3
"""Redraw existing PNG/HTML/legends from saved projections; no inference or fitting."""
import argparse
import json
from pathlib import Path
import numpy as np

from core import display_ids
from export_html import load_space, export_directory
from particle_legend import add_legends
from plot import render_scatter_outputs
from visual_style import styles_for_ids, STYLE_VERSION, DEFAULT_MARKER_SCALE, DEFAULT_HTML_MARKER_SCALE, marker_scale


def restyle(root, scale=DEFAULT_MARKER_SCALE, html_scale=None):
    scale = marker_scale(scale)
    if html_scale is not None:
        html_scale = marker_scale(html_scale)
    root = Path(root).resolve()
    reports = sorted(root.rglob('report.json'))
    changed = 0
    for path in reports:
        candidate = json.loads(path.read_text())
        if 'projections' not in candidate or 'truth_colors' not in candidate:
            continue
        report, event, selected, projections = load_space(path.parent)
        styles = styles_for_ids(display_ids(event))
        report['truth_styles'] = styles
        report['truth_colors'] = {key: value['color'] for key, value in styles.items()}
        report['style_version'] = STYLE_VERSION
        # Preserve the existing interactive point size, including older shared settings.
        previous_html_scale = report['settings'].get('html_marker_scale',
                              report['settings'].get('marker_scale', DEFAULT_HTML_MARKER_SCALE))
        report['settings']['html_marker_scale'] = previous_html_scale if html_scale is None else html_scale
        report['settings']['marker_scale'] = scale
        with np.load(path.parent/'distances.npz', allow_pickle=False) as d:
            metrics = dict(summary=report['metrics'], edges=d['histogram_edges'],
                           same_hist=d['same_hist'], other_hist=d['other_hist'])
        render_scatter_outputs(event, selected, projections, metrics, report, path.parent)
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        changed += 1
        print(f'Restyled {path.parent.relative_to(root)}', flush=True)
    if not changed:
        raise ValueError(f'No plot reports found: {root}')
    add_legends(root, overwrite=True)
    export_directory(root, overwrite=True)
    print(f'Restyled {changed} spaces using existing coordinates and display samples.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots', required=True, help='Existing plot directory or parent; replaces rendered views only')
    parser.add_argument('--marker-scale', type=marker_scale, default=DEFAULT_MARKER_SCALE,
                        help='PNG/PDF marker diameter multiplier (default: 1.0)')
    parser.add_argument('--html-marker-scale', type=marker_scale,
                        help='HTML marker diameter multiplier (default: preserve existing size)')
    args = parser.parse_args()
    restyle(args.plots, args.marker_scale, args.html_marker_scale)
