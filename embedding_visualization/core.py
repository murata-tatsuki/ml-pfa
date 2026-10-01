"""Embedding diagnostics. No torch dependency; distances always use original coordinates."""
import inspect
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import pairwise_distances


def validate_event(event):
    z = np.asarray(event['embedding'])
    if z.ndim != 2 or not len(z) or z.shape[1] < 1 or not np.isfinite(z).all():
        raise ValueError('embedding must be a finite, nonempty (N, D) array')
    for key in ('truth_id', 'is_track', 'truth_valid', 'beta'):
        if np.asarray(event[key]).shape != (len(z),):
            raise ValueError(f'{key} must have shape (N,) aligned with embedding')
    if not np.isfinite(event['beta']).all():
        raise ValueError('beta contains non-finite values')
    ids = np.asarray(event['truth_id'])
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('truth_id must have integer dtype')
    for key in ('is_track', 'truth_valid'):
        if not np.isin(event[key], [0, 1]).all():
            raise ValueError(f'{key} must be boolean or 0/1')


def display_ids(event):
    return np.where(event['truth_valid'] & (event['truth_id'] > 0), event['truth_id'], 0)


def sample_indices(event, max_points, seed):
    """Keep every track, then round-robin sample calo truth groups (including unknown)."""
    n = len(event['embedding'])
    if max_points < 1:
        raise ValueError('max_points must be positive')
    if n <= max_points:
        return np.arange(n)
    tracks = np.flatnonzero(event['is_track'])
    if len(tracks) > max_points:
        raise ValueError('max_points is smaller than track count; increase it to retain all tracks')
    rng = np.random.default_rng(seed)
    ids = display_ids(event)
    calo = np.flatnonzero(~event['is_track'].astype(bool))
    groups = [rng.permutation(calo[ids[calo] == label]).tolist() for label in np.unique(ids[calo])]
    rng.shuffle(groups)
    selected = tracks.tolist()
    while groups and len(selected) < max_points:
        remaining = []
        for group in groups:
            if len(selected) == max_points:
                break
            selected.append(group.pop())
            if group:
                remaining.append(group)
        groups = remaining
    return np.sort(selected)


def project(z, method, seed=42, perplexity=30, iterations=1000):
    """Joint projection of track + calorimeter rows, without scaling/normalization."""
    z = np.asarray(z, dtype=np.float64)
    info = dict(method=method, seed=int(seed), points=len(z), dimensions=z.shape[1])
    if len(z) < 2 or np.all(z == z[0]):
        info['note'] = 'Degenerate input: no meaningful projection'
        return np.zeros((len(z), 2)), info
    if method == 'baseline':
        if z.shape[1] <= 2:
            info['method'] = 'original coordinates'
            return np.pad(z, ((0, 0), (0, 2-z.shape[1]))), info
        pca = PCA(n_components=min(2, len(z), z.shape[1]))
        xy = pca.fit_transform(z)
        info['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
        return xy, info
    if method == 'mds':
        kwargs = dict(n_components=2, random_state=seed, n_init=4, max_iter=300, eps=1e-6)
        params = inspect.signature(MDS).parameters
        kwargs['metric_mds' if 'metric_mds' in params else 'metric'] = True
        if 'normalized_stress' in params:
            kwargs['normalized_stress'] = False
        estimator = MDS(**kwargs)
        xy = estimator.fit_transform(z)
        original = pairwise_distances(z)
        projected = pairwise_distances(xy)
        denom = np.sum(original ** 2)
        info['relative_distance_error'] = float(np.sqrt(np.sum((original-projected)**2) / denom))
        info['stress'] = float(estimator.stress_)
        return xy, info
    if method == 'tsne':
        if not 0 < perplexity < len(z):
            raise ValueError('t-SNE requires 0 < perplexity < number of plotted points')
        # Random initialization also supports 1D coordinates / small samples.
        kwargs = dict(n_components=2, random_state=seed, perplexity=perplexity,
                      init='random', learning_rate=200.0, metric='euclidean')
        key = 'max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
        kwargs[key] = iterations
        estimator = TSNE(**kwargs)
        xy = estimator.fit_transform(z)
        info.update(perplexity=float(perplexity), kl_divergence=float(estimator.kl_divergence_))
        return xy, info
    raise ValueError(f'Unknown projection method: {method}')


def distance_metrics(event, k=10, bins=50):
    """Exact metrics on ALL saved points; exclude unknown truth from labelled comparisons.

    At most one track-to-calo distance vector is allocated at a time.
    Histograms give equal weight to each track with a corresponding class.
    """
    validate_event(event)
    if k < 1 or bins < 2:
        raise ValueError('k must be positive and bins >= 2')
    z = np.asarray(event['embedding'], dtype=np.float64)
    ids = display_ids(event)
    tracks = np.flatnonzero(event['is_track'].astype(bool) & (ids > 0))
    hits = np.flatnonzero(~event['is_track'].astype(bool) & (ids > 0))
    clusters = np.unique(ids[hits])
    matrix = np.full((len(tracks), len(clusters)), np.nan)
    upper = max(float(np.linalg.norm(np.ptp(z, axis=0))), 1e-12)
    edges = np.linspace(0, upper * (1 + 1e-12), bins+1)
    hist = {'same': np.zeros(bins), 'other': np.zeros(bins)}
    counts = {'same': 0, 'other': 0}
    rows = []
    for row_index, t in enumerate(tracks):
        d = np.linalg.norm(z[hits] - z[t], axis=1)
        same = ids[hits] == ids[t]
        order = np.argsort(d, kind='stable')
        actual_k = min(k, len(hits))
        row = dict(track_row=int(t), truth_id=int(ids[t]), same_hits=int(same.sum()),
                   other_hits=int((~same).sum()), k_effective=actual_k,
                   same_median=None, other_median=None, nearest_correct=None,
                   knn_purity=None, first_correct_rank=None)
        for name, mask in [('same', same), ('other', ~same)]:
            values = d[mask]
            if len(values):
                row[f'{name}_median'] = float(np.median(values))
                hist[name] += np.histogram(values, bins=edges)[0] / len(values)
                counts[name] += 1
        # Tracks without any known matching hit do not have a retrieval target.
        if same.any():
            row['nearest_correct'] = bool(same[order[0]])
            row['knn_purity'] = float(same[order[:actual_k]].mean())
            row['first_correct_rank'] = int(np.flatnonzero(same[order])[0] + 1)
        for j, label in enumerate(clusters):
            matrix[row_index, j] = np.median(d[ids[hits] == label])
        rows.append(row)
    for name in hist:
        if counts[name]:
            hist[name] /= counts[name]
    eligible = [r for r in rows if r['knn_purity'] is not None]
    summary = dict(points=len(z), dimensions=z.shape[1], tracks=int(np.sum(event['is_track'])),
                   labelled_tracks=len(tracks), labelled_calo_hits=len(hits),
                   unknown_truth_points=int(np.sum(ids == 0)), eligible_tracks=len(eligible),
                   tracks_without_matching_hits=len(tracks)-len(eligible), k_requested=k,
                   mean_knn_purity=float(np.mean([r['knn_purity'] for r in eligible])) if eligible else None,
                   nearest_hit_accuracy=float(np.mean([r['nearest_correct'] for r in eligible])) if eligible else None,
                   histogram_track_counts=counts,
                   distance='Euclidean in original embedding; no normalization',
                   retrieval_candidates='All calorimeter hits with valid positive truth ID; no display sampling')
    return dict(summary=summary, rows=rows, track_rows=tracks, cluster_ids=clusters,
                heatmap=matrix, edges=edges, same_hist=hist['same'], other_hist=hist['other'])
