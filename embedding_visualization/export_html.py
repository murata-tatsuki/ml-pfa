#!/usr/bin/env python3
"""Export existing projections as offline HTML, without inference or dimensionality reduction."""
import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core import display_ids, validate_event
from visual_style import highest_beta_rows, DEFAULT_HTML_MARKER_SCALE, marker_scale


def load_space(folder):
    report = json.loads((folder/'report.json').read_text())
    source = Path(report['source'])
    with np.load(source, allow_pickle=False) as saved:
        event = {key: saved[key] for key in saved.files}
    key = report['embedding_key']
    event['embedding'] = event[key]
    validate_event(event)
    selected = np.load(folder/'selected_rows.npy', allow_pickle=False)
    if (selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0) or np.any(selected >= len(event['embedding']))
            or len(np.unique(selected)) != len(selected)):
        raise ValueError(f'{folder}: invalid selected_rows')
    projections = {}
    for entry in report['projections']:
        name = entry['name']
        if Path(name).name != name:
            raise ValueError('Invalid projection name')
        xy = np.load(folder/f'{name}.npy', allow_pickle=False)
        if xy.shape != (len(selected), 2) or not np.isfinite(xy).all():
            raise ValueError(f'{folder}: invalid projection {name}')
        projections[name] = xy
    if not projections:
        raise ValueError(f'{folder}: no projections')
    return report, event, selected, projections


def scatter_figure(report, event, selected, projections):
    scale = marker_scale(report['settings'].get('html_marker_scale',
                         report['settings'].get('marker_scale', DEFAULT_HTML_MARKER_SCALE)))
    ids = display_ids(event)[selected]
    tracks = np.asarray(event['is_track'], dtype=bool)[selected]
    colors = report['truth_colors']
    styles = report.get('truth_styles', {})
    highest = np.isin(selected, highest_beta_rows(event))
    texts = []
    for row in selected:
        text = [f'row: {row}', 'track' if event['is_track'][row] else 'calorimeter hit',
                f'truth ID: {int(event["truth_id"][row])}',
                f'truth valid: {bool(event["truth_valid"][row])}', f'beta (final): {event["beta"][row]:.4f}']
        for key in ('hit_id', 'mc_id'):
            if key in event:
                text.append(f'{key}: {int(event[key][row])}')
        if 'detected_energy' in event:
            text.append(f'detected energy: {event["detected_energy"][row]:.5g}')
        text.append('original coordinates: ' + ', '.join(f'{v:.5g}' for v in event['embedding'][row]))
        texts.append('<br>'.join(text))
    texts = np.array(texts)
    fig = go.Figure()
    masks = []
    initial = next(iter(projections))
    for label in np.unique(ids):
        color = colors[str(label)]
        css = 'rgba(%d,%d,%d,%.3f)' % (round(color[0]*255), round(color[1]*255), round(color[2]*255), color[3])
        first = True
        symbol = styles.get(str(label), {}).get('plotly_symbol', 'circle')
        for track in (False, True):
            mask = (ids == label) & (tracks == track)
            if not mask.any():
                continue
            masks.append(mask)
            focus = report['settings'].get('focus_truth')
            fig.add_trace(go.Scatter(x=projections[initial][mask, 0], y=projections[initial][mask, 1],
                text=texts[mask], mode='markers', name=f'Truth {label}' if label else 'Unknown/noise',
                legendgroup=str(label), showlegend=first,
                opacity=0.12 if focus is not None and label != focus else 1,
                marker=dict(color=css, symbol=symbol+'-open' if track else symbol, size=(7 if track else 5)*scale,
                            line=dict(color=css, width=.8*scale if track else 0)),
                hovertemplate='%{text}<br>2D: (%{x:.5g}, %{y:.5g})<extra></extra>'))
            first = False
        for mask, overlay, size, description in [
                ((ids == label) & tracks, 'cross-thin', 5, 'Track'),
                ((ids == label) & highest, 'circle-open', 8, 'Highest-beta calo hit (final beta)')]:
            if not mask.any():
                continue
            masks.append(mask)
            fig.add_trace(go.Scatter(x=projections[initial][mask, 0], y=projections[initial][mask, 1],
                text=texts[mask], mode='markers', name=description, showlegend=False, legendgroup=str(label),
                opacity=.12 if report['settings'].get('focus_truth') not in (None, label) else 1,
                marker=dict(color='black', symbol=overlay, size=size*scale, line=dict(color='black', width=.8*scale)),
                hovertemplate=description+'<br>%{text}<br>2D: (%{x:.5g}, %{y:.5g})<extra></extra>'))
    buttons = []
    for name, xy in projections.items():
        buttons.append(dict(label=name, method='update', args=[
            dict(x=[xy[mask, 0] for mask in masks], y=[xy[mask, 1] for mask in masks]),
            {'title.text': name, 'xaxis.autorange': True, 'yaxis.autorange': True}]))
    fig.update_layout(template='plotly_white', title=dict(text=initial, x=.5, xanchor='center'), height=650,
        margin=dict(t=105, b=55, l=65, r=25),
        updatemenus=[dict(buttons=buttons, x=0, y=1.16, xanchor='left', yanchor='top')],
        legend=dict(title='Truth ID (click to hide)', groupclick='togglegroup'),
        xaxis=dict(title='coordinate 1'), yaxis=dict(title='coordinate 2', scaleanchor='x', scaleratio=1))
    return fig


def diagnostic_figure(folder):
    with np.load(folder/'distances.npz', allow_pickle=False) as saved:
        d = {key: saved[key] for key in saved.files}
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Distance distribution (all labelled hits)', 'Track–calo cluster median distance'))
    for name, color in [('same', '#1f77b4'), ('other', '#ff7f0e')]:
        fig.add_trace(go.Scatter(x=d['histogram_edges'], y=np.r_[d[f'{name}_hist'], 0],
            mode='lines', line=dict(color=color, shape='hv'), name=name+' truth'), row=1, col=1)
    if d['median_distance'].size:
        fig.add_trace(go.Heatmap(z=d['median_distance'],
            x=[str(int(v)) for v in d['cluster_ids']], y=[str(int(v)) for v in d['track_rows']],
            colorscale='Viridis', reversescale=True, colorbar=dict(title='distance', len=.8),
            hovertemplate='track row: %{y}<br>calo truth: %{x}<br>median distance: %{z:.5g}<extra></extra>'), row=1, col=2)
    fig.update_xaxes(title_text='Original-space Euclidean distance', row=1, col=1)
    fig.update_yaxes(title_text='Probability/bin (equal track weight)', row=1, col=1)
    fig.update_xaxes(title_text='Calo truth ID', type='category', row=1, col=2)
    fig.update_yaxes(title_text='Track row', type='category', row=1, col=2)
    fig.update_layout(template='plotly_white', height=480, margin=dict(t=55, b=70),
                      legend=dict(orientation='h', y=-.25))
    return fig


def space_order(item):
    key = item[1]['embedding_key']
    if key.startswith('gravnet_'):
        return (0, int(key.split('_')[1]), key)
    return (1, 0, key)


def write_event_page(event_dir, entries, overwrite=False):
    entries = sorted(entries, key=space_order)
    options, sections = [], []
    javascript_included = False
    reference_rows = None
    config = dict(responsive=True, displaylogo=False, toImageButtonOptions=dict(format='png', scale=2))
    for i, (folder, _) in enumerate(entries):
        report, event, selected, projections = load_space(folder)
        if reference_rows is not None and not np.array_equal(reference_rows, selected):
            raise ValueError('Layer comparison requires identical sampled rows; use the same sample seed and max-points')
        reference_rows = selected
        key = report['embedding_key']
        label = 'Final clustering coordinates' if key == 'embedding' else key
        label += f' (D={event["embedding"].shape[1]})'
        options.append(f'<option value="space-{i}">{escape(label)}</option>')
        specs = json.loads(str(event['spaces_json'])) if 'spaces_json' in event else {}
        description = specs.get(key, {}).get('description', '')
        summary = report['metrics']
        def percentage(value):
            return 'N/A' if value is None else f'{100*value:.1f}%'
        info = (f'{len(selected)} / {len(event["embedding"])} points displayed · '
                f'Tracks: {summary["tracks"]} · '
                f'Nearest-hit accuracy: {percentage(summary["nearest_hit_accuracy"])} · '
                f'kNN purity: {percentage(summary["mean_knn_purity"])}')
        relative = folder.relative_to(event_dir)
        png = quote((relative/'overview.png').as_posix())
        pdf_links = ' / '.join(
            f'<a href="{quote((relative/page.name).as_posix())}">{escape(page.stem)} PDF</a>'
            for page in sorted(folder.glob('*.pdf')))
        content = [f'<section id="space-{i}" class="space"'+(' hidden' if i else '')+'>',
                   f'<h2>{escape(label)}</h2><p>{escape(description)}</p><p>{escape(info)}</p>',
                   f'<p><a href="{png}">PNG画像を開く</a> / {pdf_links}</p>']
        for fig in (scatter_figure(report, event, selected, projections), diagnostic_figure(folder)):
            content.append(fig.to_html(full_html=False, include_plotlyjs=not javascript_included, config=config))
            javascript_included = True
        content.append('</section>')
        sections.append('\n'.join(content))
    synthetic = any(e[1].get('event_metadata', {}).get('synthetic') for e in entries)
    title = ('SYNTHETIC DEMO · ' if synthetic else '') + event_dir.name
    html = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Embedding display</title>
<style>body{font:16px system-ui,sans-serif;margin:24px auto;max-width:1400px;padding:0 20px;color:#172332}
select{font:inherit;padding:8px;max-width:100%}p{line-height:1.6}a{color:#145cb3}.space[hidden]{display:none}
header{position:sticky;top:0;background:white;z-index:10;padding:12px 0;border-bottom:1px solid #ddd}</style>
<body>'''
    html += f'<h1>{escape(title)} — Embedding display</h1>'
    legend_pages = sorted(event_dir.glob('particle_legend*.png'))
    if legend_pages:
        html += '<p>粒子の種類・MC energy・色の凡例（別画像）： ' + ' / '.join(
            f'<a href="{quote(page.name)}" target="_blank">凡例 {i+1}</a>'
            for i, page in enumerate(legend_pages)) + '</p>'
    legend_pdfs = sorted(event_dir.glob('particle_legend*.pdf'))
    if legend_pdfs:
        html += '<p>拡大用の凡例PDF： ' + ' / '.join(
            f'<a href="{quote(page.name)}" target="_blank">凡例 {i+1} PDF</a>'
            for i, page in enumerate(legend_pdfs)) + '</p>'
    html += ('<p>色と形の組み合わせでtruth clusterを区別します。trackは中抜きの形＋小さな＋印、'
             'highest βのcalo hitは小さな黒いリングです（各truth clusterのcalo hit内で最大の最終β）。'
             '点にマウスを重ねるとID・β・座標を表示します。'
             'ドラッグで拡大、ダブルクリックで範囲を戻せます。凡例をクリックすると粒子ごとに表示を切り替えられます。'
             'HTMLのグラフ内ズームでは点の画面上の大きさを保って範囲を拡大します。'
             'PDFはベクター形式なので拡大してもぼやけません（点自体も拡大されます）。'
             'カメラボタンでPNGを保存できます。</p>'
             '<p>層内の座標は近傍探索用、最終座標はクラスタリング用です。各層の投影は独立です。'
             '図の位置・向きや距離の縮小だけで学習の改善とは判断せず、元空間の近傍指標も比較してください。</p>')
    html += '<header><label for="space-select">表示する層・空間： </label><select id="space-select">'+''.join(options)+'</select></header>'
    html += '\n'.join(sections)
    html += '''<script>
document.getElementById('space-select').addEventListener('change', function () {
  document.querySelectorAll('.space').forEach(s => { s.hidden = s.id !== this.value; });
  document.getElementById(this.value).querySelectorAll('.plotly-graph-div')
    .forEach(p => Plotly.relayout(p, {width: Math.max(320, p.parentElement.clientWidth), autosize: true}));
});
</script></body></html>'''
    target = event_dir/'display.html'
    with target.open('w' if overwrite else 'x', encoding='utf-8') as stream:
        stream.write(html)
    return target


def export_directory(root, overwrite=False):
    root = Path(root).resolve()
    groups = defaultdict(list)
    for path in sorted(root.rglob('report.json')):
        report = json.loads(path.read_text())
        if 'embedding_key' not in report or 'projections' not in report:
            continue
        # plot.py stores event/space/report.json or event/report.json.
        event_dir = path.parent.parent if path.parent.name == report['embedding_key'] else path.parent
        groups[event_dir].append((path.parent, report))
    if not groups:
        raise ValueError(f'No plot reports found in {root}')
    for event_dir, entries in groups.items():
        if len({entry[1]['source'] for entry in entries}) != 1:
            raise ValueError('Different events cannot share one layer comparison page')
        if not overwrite and (event_dir/'display.html').exists():
            raise FileExistsError(event_dir/'display.html')
    index = root/'index.html'
    if not overwrite and index.exists():
        raise FileExistsError(index)
    pages = [write_event_page(event_dir, entries, overwrite) for event_dir, entries in groups.items()]
    links = ''.join(f'<li><a href="{quote(p.relative_to(root).as_posix())}">{escape(p.parent.relative_to(root).as_posix())}</a></li>' for p in pages)
    with index.open('w' if overwrite else 'x', encoding='utf-8') as stream:
        stream.write('<!doctype html><html lang="ja"><meta charset="utf-8"><title>Embedding displays</title>'
                     '<body style="font:18px system-ui;padding:24px"><h1>Embedding displays</h1>'
                     '<p>イベントを選ぶと、層と投影手法を切り替えられます。HTMLはオフラインで開けます。</p><ul>'
                     + links + '</ul></body></html>')
    print(f'HTML index: {index}', flush=True)
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots', required=True, help='Existing plot directory (reports and projection NPY files)')
    parser.add_argument('--overwrite', action='store_true', help='Replace generated HTML only')
    args = parser.parse_args()
    export_directory(args.plots, args.overwrite)
