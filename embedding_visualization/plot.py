#!/usr/bin/env python3
"""Plot saved event embeddings and evaluate distances without rerunning a checkpoint."""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from core import validate_event, display_ids, sample_indices, project, distance_metrics
from visual_style import styles_for_ids, draw_points, STYLE_VERSION, DEFAULT_MARKER_SCALE, DEFAULT_HTML_MARKER_SCALE, marker_scale, save_figure


def palette(ids):
    return {int(key): value['color'] for key, value in styles_for_ids(ids).items()}


def scatter(ax, xy, event, selected, colors, title, focus=None, scale=DEFAULT_MARKER_SCALE):
    styles = styles_for_ids(display_ids(event))
    for key in styles:
        styles[key]['color'] = colors[int(key)]
    draw_points(ax, xy, event, selected, styles, focus, scale)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('coordinate 1')
    ax.set_ylabel('coordinate 2')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.15)


def distance_plot(ax, metrics):
    for key, color, label in [('same', 'tab:blue', 'Same truth particle'), ('other', 'tab:orange', 'Other truth particles')]:
        if metrics['summary']['histogram_track_counts'][key]:
            ax.stairs(metrics[f'{key}_hist'], metrics['edges'], color=color, label=label)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.set_xlabel('Euclidean distance in ORIGINAL embedding')
    ax.set_ylabel('Probability per bin (equal track weight)')
    ax.set_title('All labelled hits; no display sampling')


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_scatter_outputs(event, selected, results, metrics, report, output):
    """Render only: reused for restyling saved coordinates without refitting projections."""
    colors = {int(k): v for k, v in report['truth_colors'].items()}
    dpi = report['settings'].get('dpi', 160)
    focus = report['settings'].get('focus_truth')
    scale = marker_scale(report['settings'].get('marker_scale', DEFAULT_MARKER_SCALE))
    infos = {p['name']: p for p in report['projections']}
    for name, xy in results.items():
        info = infos[name]
        title = info['method'] if name == 'baseline' else name
        if name == 'mds' and 'relative_distance_error' in info:
            title += f" | relative distance error={info['relative_distance_error']:.3f}"
        fig, ax = plt.subplots(figsize=(8, 7))
        scatter(ax, xy, event, selected, colors, title, focus, scale)
        fig.tight_layout()
        save_figure(fig, output/f'{name}.png', dpi)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    scatter(axes[0, 0], results['baseline'], event, selected, colors, 'Original 1D/2D or PCA', focus, scale)
    scatter(axes[0, 1], results['mds'], event, selected, colors, 'Metric MDS', focus, scale)
    tsne = next((key for key in results if key.startswith('tsne')), None)
    if tsne:
        scatter(axes[1, 0], results[tsne], event, selected, colors, tsne, focus, scale)
    else:
        axes[1, 0].text(.5, .5, 't-SNE skipped: too few points', ha='center')
        axes[1, 0].set_axis_off()
    distance_plot(axes[1, 1], metrics)
    prefix = 'SYNTHETIC DEMO | ' if report.get('event_metadata', {}).get('synthetic') else ''
    fig.suptitle(f'{prefix}{Path(report["source"]).stem} | {report["embedding_key"]} '
                 f'D={event["embedding"].shape[1]} | {len(selected)}/{len(event["embedding"])} displayed\n'
                 'Color + shape: truth cluster | outlined shape with +: track | black ring: highest-beta calo hit')
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_figure(fig, output/'overview.png', dpi)
    plt.close(fig)
    if event['embedding'].shape[1] == 3:
        z = event['embedding'][selected]
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
        draw_points(ax, z, event, selected, report['truth_styles'], focus, scale)
        ranges = np.ptp(z, axis=0)
        ax.set_box_aspect(np.maximum(ranges, max(float(ranges.max()), 1e-12)*1e-3))
        ax.set(xlabel='z1', ylabel='z2', zlabel='z3', title='Original 3D coordinates')
        save_figure(fig, output/'original_3d.png', dpi)
        plt.close(fig)


def plot_event(path, output, args):
    with np.load(path, allow_pickle=False) as saved:
        event = {key: saved[key] for key in saved.files}
    if args.embedding_key not in event:
        raise ValueError(f'{path}: missing embedding key {args.embedding_key}')
    event['embedding'] = event[args.embedding_key]
    validate_event(event)
    event['is_track'] = np.asarray(event['is_track'], dtype=bool)
    event['truth_valid'] = np.asarray(event['truth_valid'], dtype=bool)
    metadata = json.loads(str(event['metadata_json'])) if 'metadata_json' in event else {}
    selected = sample_indices(event, args.max_points, args.sample_seed)
    z = event['embedding'][selected]
    colors = palette(display_ids(event))
    metrics = distance_metrics(event, k=args.k)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output/'selected_rows.npy', selected)
    np.savez_compressed(output/'distances.npz', track_rows=metrics['track_rows'],
                        cluster_ids=metrics['cluster_ids'], median_distance=metrics['heatmap'],
                        histogram_edges=metrics['edges'], same_hist=metrics['same_hist'], other_hist=metrics['other_hist'])
    fields = ['track_row', 'truth_id', 'same_hits', 'other_hits', 'k_effective',
              'same_median', 'other_median', 'nearest_correct', 'knn_purity', 'first_correct_rank']
    write_csv(output/'per_track.csv', metrics['rows'], fields)
    with (output/'distance_heatmap.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['track_row', 'track_truth_id'] + [f'calo_truth_{i}' for i in metrics['cluster_ids']])
        for i, row in enumerate(metrics['rows']):
            writer.writerow([row['track_row'], row['truth_id']] + metrics['heatmap'][i].tolist())
    report = dict(source=str(path.resolve()), event_metadata=metadata, embedding_key=args.embedding_key,
                  sampled_points=len(selected), sample_seed=args.sample_seed,
                  versions=dict(numpy=np.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__),
                  settings=vars(args), metrics=metrics['summary'], projections=[], skipped=[])
    results = {}
    specifications = [('baseline', args.seeds[0], None), ('mds', args.seeds[0], None)]
    specifications += [('tsne', seed, p) for seed in args.seeds for p in args.perplexities]
    for method, seed, perplexity in specifications:
        name = method if method != 'tsne' else f'tsne_p{perplexity:g}_seed{seed}'
        if method == 'tsne' and (len(z) < 3 or perplexity >= len(z)):
            report['skipped'].append(dict(name=name, reason='Too few sampled points for this perplexity'))
            continue
        print(f'{path.name}: {name} ({len(z)} displayed points)', flush=True)
        xy, info = project(z, method, seed, perplexity or 30, args.iterations)
        results[name] = xy
        report['projections'].append(dict(name=name, **info))
        np.save(output/f'{name}.npy', xy)
    matrix = metrics['heatmap']
    if matrix.size:
        # Keep full matrix in NPZ/CSV; cap only the rendered image.
        nr, nc = min(40, matrix.shape[0]), min(40, matrix.shape[1])
        fig, ax = plt.subplots(figsize=(max(7, nc*.23), max(4, nr*.22)))
        plotted = ax.imshow(matrix[:nr, :nc], aspect='auto', cmap='viridis_r')
        ax.set_xticks(np.arange(nc))
        ax.set_xticklabels(metrics['cluster_ids'][:nc], rotation=90, fontsize=7)
        ax.set_yticks(np.arange(nr))
        ax.set_yticklabels([f"{r['track_row']} / {r['truth_id']}" for r in metrics['rows'][:nr]], fontsize=7)
        for i, row in enumerate(metrics['rows'][:nr]):
            for j in np.flatnonzero(metrics['cluster_ids'][:nc] == row['truth_id']):
                ax.plot(j, i, 's', markerfacecolor='none', markeredgecolor='red', markersize=10)
        ax.set_xlabel('Calo truth ID (red square: corresponding truth)')
        ax.set_ylabel('Track row / truth ID')
        ax.set_title(f'Median distance in original space; showing {nr}/{matrix.shape[0]} tracks, {nc}/{matrix.shape[1]} clusters')
        fig.colorbar(plotted, ax=ax, label='Median Euclidean distance')
        fig.tight_layout()
        fig.savefig(output/'distance_heatmap.png', dpi=args.dpi)
        plt.close(fig)
    report['truth_colors'] = {str(k): list(v) for k, v in colors.items()}
    report['truth_styles'] = styles_for_ids(display_ids(event))
    report['style_version'] = STYLE_VERSION
    render_scatter_outputs(event, selected, results, metrics, report, output)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return dict(event=path.stem, **metrics['summary'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='NPZ file or extraction directory')
    p.add_argument('--output', required=True, help='New plot output directory')
    p.add_argument('--embedding-key', default='embedding', help='NPZ space key, e.g. embedding, intermediate, gravnet_1_coords')
    p.add_argument('--all-coordinates', action='store_true', help='Plot all saved GravNet coordinate spaces plus final embedding')
    p.add_argument('--html', action='store_true', help='Also export interactive offline HTML with space/projection selection')
    p.add_argument('--max-points', type=int, default=1500, help='Display cap only; every track retained')
    p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--perplexities', nargs='+', type=float, default=[10, 30, 50])
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--focus-truth', type=int, help='Highlight one truth ID without refitting on a subset')
    p.add_argument('--dpi', type=int, default=160)
    p.add_argument('--marker-scale', type=marker_scale, default=DEFAULT_MARKER_SCALE,
                   help='PNG/PDF marker diameter multiplier (default: 1.0)')
    p.add_argument('--html-marker-scale', type=marker_scale, default=DEFAULT_HTML_MARKER_SCALE,
                   help='HTML marker diameter multiplier (default: 0.6)')
    p.add_argument('--threads', type=int, default=1, help='Limit BLAS/OpenMP threads during projection')
    args = p.parse_args(argv)
    if args.iterations < 250 or args.max_points < 1 or args.k < 1 or args.threads < 1 or args.dpi < 1:
        p.error('Require iterations >= 250 and positive max-points, k, threads, dpi')
    if not all(np.isfinite(x) and x > 0 for x in args.perplexities):
        p.error('perplexities must be finite and positive')
    if not re.fullmatch(r'[A-Za-z0-9_]+', args.embedding_key):
        p.error('embedding-key must be an NPZ key containing only letters, digits, underscores')
    if args.html:
        import plotly  # fail before computation if this optional dependency is missing
    source = Path(args.input).expanduser()
    paths = sorted(source.glob('event_*.npz')) if source.is_dir() else [source]
    if not paths:
        p.error('No event NPZ files found')
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=False)
    from threadpoolctl import threadpool_limits
    summaries = []
    with threadpool_limits(limits=args.threads):
        for path in paths:
            if args.all_coordinates:
                with np.load(path, allow_pickle=False) as event:
                    keys = sorted((key for key in event.files if re.fullmatch(r'gravnet_\d+_coords', key)),
                                  key=lambda key: int(key.split('_')[1]))
                if not keys:
                    p.error(f'{path}: no block coordinates; extract with --gravnet-coordinates first')
                for key in keys + ['embedding']:
                    settings = argparse.Namespace(**vars(args))
                    settings.embedding_key = key
                    summary = plot_event(path, output/path.stem/key, settings)
                    summaries.append(dict(summary, space=key))
            else:
                summaries.append(plot_event(path, output/path.stem, args))
    (output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
    from particle_legend import add_legends
    add_legends(output, dpi=args.dpi)
    if args.html:
        from export_html import export_directory
        export_directory(output)
    print(f'Saved plots and metrics to {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
