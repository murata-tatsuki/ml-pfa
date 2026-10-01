"""Shared color/shape identities and compact role markers for PNG, HTML, and legends."""
import numpy as np
from matplotlib.colors import to_rgba
from core import display_ids

STYLE_VERSION = 4
DEFAULT_MARKER_SCALE = 1.0
DEFAULT_HTML_MARKER_SCALE = .6


def marker_scale(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('marker-scale must be finite and positive')
    return value


def save_figure(fig, path, dpi):
    """Keep PNG previews and export vector geometry directly to PDF."""
    from pathlib import Path
    path = Path(path)
    fig.savefig(path.with_suffix('.png'), dpi=dpi)
    fig.savefig(path.with_suffix('.pdf'))
COLORS = ['#0072B2', '#E66100', '#00875A', '#B22222', '#7B2CBF',
          '#C2188B', '#008B9A', '#8C564B', '#555555', '#A18100']
SHAPES = [('o', 'circle', 'circle'), ('s', 'square', 'square'),
          ('^', 'triangle-up', 'triangle up'), ('v', 'triangle-down', 'triangle down'),
          ('D', 'diamond', 'diamond'), ('p', 'pentagon', 'pentagon'),
          ('h', 'hexagon', 'hexagon'), ('<', 'triangle-left', 'triangle left'),
          ('>', 'triangle-right', 'triangle right')]


def styles_for_ids(ids):
    result = {'0': dict(color=[.65, .65, .65, .4], mpl_marker='o', plotly_symbol='circle', shape_name='circle')}
    for i, label in enumerate(np.unique(np.asarray(ids)[np.asarray(ids) > 0])):
        marker, symbol, name = SHAPES[(i % len(COLORS) + i // len(COLORS)) % len(SHAPES)]
        rgba = list(to_rgba(COLORS[i % len(COLORS)]))
        # 90 distinct color/shape pairs before reusing shapes with darker tones.
        cycle = i // (len(COLORS)*len(SHAPES))
        rgba[:3] = [channel * .8**cycle for channel in rgba[:3]]
        result[str(int(label))] = dict(color=rgba, mpl_marker=marker, plotly_symbol=symbol, shape_name=name)
    return result


def highest_beta_rows(event):
    """Maximum FINAL beta among calo hits in each positive truth cluster; ties: first row."""
    ids = display_ids(event)
    calo = ~np.asarray(event['is_track'], dtype=bool)
    rows = []
    for label in np.unique(ids[ids > 0]):
        candidates = np.flatnonzero(calo & (ids == label))
        if len(candidates):
            rows.append(candidates[np.argmax(event['beta'][candidates])])
    return np.asarray(rows, dtype=np.int64)


def draw_points(ax, coords, event, selected, styles, focus=None, scale=DEFAULT_MARKER_SCALE):
    """Color + shape encode cluster. Outline/+ marks track; small black ring marks max beta."""
    scale = marker_scale(scale)
    ids = display_ids(event)[selected]
    track = np.asarray(event['is_track'], dtype=bool)[selected]
    for label in np.unique(ids):
        style = styles[str(int(label))]
        color = style['color']
        alpha = color[3] * (.12 if focus is not None and label != focus else 1.)
        normal = (ids == label) & ~track
        tracked = (ids == label) & track
        if normal.any():
            ax.scatter(*coords[normal].T, c=[color], marker=style['mpl_marker'], s=11*scale**2,
                       edgecolors='none', alpha=alpha, zorder=2)
        if tracked.any():
            ax.scatter(*coords[tracked].T, facecolors='none', edgecolors=[color],
                       marker=style['mpl_marker'], s=20*scale**2, linewidths=.8*scale, alpha=alpha, zorder=4)
            ax.scatter(*coords[tracked].T, c='black', marker='+', s=10*scale**2,
                       linewidths=.65*scale, alpha=alpha, zorder=5)
    highest = np.isin(selected, highest_beta_rows(event))
    if focus is not None:
        highest &= ids == focus
    if highest.any():
        ax.scatter(*coords[highest].T, marker='o', s=24*scale**2, facecolors='none',
                   edgecolors='black', linewidths=.65*scale, zorder=6)
