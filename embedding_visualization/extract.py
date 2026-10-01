#!/usr/bin/env python3
"""Extract event-aligned embeddings using the existing master model/data implementations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True, help='One H5 file, or directory of H5 files')
    p.add_argument('--output', required=True, help='New output directory')
    p.add_argument('--embedding-dim', required=True, type=int, help='Coordinate dimensions ONLY, excluding beta/regression')
    p.add_argument('--start', type=int, default=0, help='Global raw H5 event index, including empty events')
    p.add_argument('--num-events', type=int, default=3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--thetaphi', action='store_true')
    p.add_argument('--momentum', action='store_true')
    p.add_argument('--momentum-amp', action='store_true')
    p.add_argument('--mctpe', action='store_true')
    p.add_argument('--extended-h5-input', action='store_true')
    p.add_argument('--exclude-gap-hits', action='store_true')
    p.add_argument('--layer', help='Also save an intermediate module output, e.g. postgn_dense or model.head_postgn_dense.0')
    p.add_argument('--gravnet-coordinates', action='store_true', help='Save actual learned kNN coordinates from EVERY GravNet block')
    p.add_argument('--gravnet-features', action='store_true', help='Also save unmodified output features at the end of EVERY block')
    p.add_argument('--master-dir', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=1, help='PyTorch CPU thread count')
    return p


def checkpoint_layout(state, embedding_dim):
    """Infer architecture widths; require the semantic coordinate width from the user."""
    if embedding_dim < 1:
        raise ValueError('embedding-dim must be positive')
    input_dim = int(state['batchnorm1.running_mean'].shape[0])
    if 'head_output.0.4.weight' in state:
        clustering_dim = int(state['head_output.0.4.weight'].shape[0])
        extra = clustering_dim - embedding_dim
        if extra not in (1, 2):
            raise ValueError('Multihead clustering output must contain beta, optional charge, and embedding-dim coordinates')
        energy = 'head_output.1.4.weight' in state
        cluster = 'head_output.2.4.weight' in state
        return dict(input_dim=input_dim, output_dim=clustering_dim+int(energy)+int(cluster),
                    use_charge_track_likeness=(extra == 2), energy_regression=energy,
                    energy_regression_cluster=cluster, model_variant='multihead')
    if 'output.4.weight' not in state:
        raise ValueError('Supported checkpoints: legacy GravnetModel or GravNetModelMultiHead (not JIT/energy-branch)')
    width = int(state['output.4.weight'].shape[0])
    if width <= embedding_dim:
        raise ValueError('embedding-dim must leave at least the beta output column')
    return dict(input_dim=input_dim, output_dim=width, model_variant='legacy')


def main(argv=None):
    args = parser().parse_args(argv)
    if args.start < 0 or args.num_events < 1 or args.embedding_dim < 1 or args.threads < 1:
        raise ValueError('Require start >= 0, num-events >= 1, embedding-dim >= 1, threads >= 1')
    if args.momentum_amp and not args.momentum:
        raise ValueError('--momentum-amp requires --momentum')
    sys.path.insert(0, str(args.master_dir.resolve()))
    import h5py
    import numpy as np
    import torch
    from torch_geometric.data import Batch
    from model import get_model, _extract_state_dict
    from dataset_ilc_sharded import ILCDatasetSharded
    from cluster_energy import inference_forward
    from core import validate_event
    from capture import SpaceCapture
    from particle_legend import truth_arrays

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    state = _extract_state_dict(checkpoint)
    layout = checkpoint_layout(state, args.embedding_dim)
    pooling_config = checkpoint.get('cluster_energy_config', {})
    if pooling_config and pooling_config.get('coordinate_start', 1) != 1 + int(layout.get('use_charge_track_likeness', False)):
        raise ValueError('embedding-dim disagrees with checkpoint cluster_energy_config.coordinate_start')
    input_dim = 5 + 2*args.thetaphi + 3*args.momentum + args.momentum_amp
    if input_dim != layout['input_dim']:
        raise ValueError(f"Checkpoint input width {layout['input_dim']} != configured {input_dim}; check --thetaphi/--momentum/--momentum-amp")
    config = checkpoint.get('training_input_config', {})
    if config and (not args.extended_h5_input or bool(config.get('exclude_gap_hits')) != args.exclude_gap_hits):
        raise ValueError('Match checkpoint training_input_config with --extended-h5-input / --exclude-gap-hits')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('Requested CUDA but CUDA is unavailable')
    model = get_model(ckpt=str(checkpoint_path), jit=False, **layout).to(args.device).eval()
    source = Path(args.input).expanduser().resolve()
    paths = sorted(source.glob('*.h5')) if source.is_dir() else [source]
    if not paths:
        raise ValueError('No H5 input files found')
    entries, event_ids, offset = [], [], 0
    for path in paths:
        with h5py.File(path, 'r') as h5:
            count = int(h5['feature'].attrs['length'])
            extended = h5.attrs.get('schema_version') in ('pandora-eval-1', 'nnqq-2m-eval-1')
            if extended != args.extended_h5_input:
                raise ValueError(f'{path}: H5 schema and --extended-h5-input disagree')
        for local in range(max(0, args.start-offset), min(count, args.start+args.num_events-offset)):
            entries.append((str(path), local))
            event_ids.append(offset+local)
        offset += count
        if offset >= args.start+args.num_events:
            break
    if not entries:
        raise ValueError('Requested event range is outside input')
    dataset = ILCDatasetSharded(str(source), test_mode=True, _entries=entries,
        thetaphi=args.thetaphi, momentum=args.momentum, momentumAmp=args.momentum_amp,
        mctpe=args.mctpe, extended_h5_input=args.extended_h5_input,
        exclude_gap_hits=args.exclude_gap_hits, file_cache_size=1)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with checkpoint_path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    manifest = dict(status='incomplete', checkpoint=str(checkpoint_path), checkpoint_sha256=digest.hexdigest(),
                    settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    layout=layout, versions=dict(torch=torch.__version__, numpy=np.__version__),
                    events=[], coordinate_start=layout['output_dim']-args.embedding_dim)
    capture = SpaceCapture(model, args.gravnet_coordinates, args.gravnet_features, args.layer)
    manifest['spaces'] = capture.specs
    try:
        for index, (event_id, (path, local)) in enumerate(zip(event_ids, entries)):
            data = dataset[index]
            record = dict(event_id=event_id, source_file=path, source_event=local)
            if len(data.x) == 0:
                record['skipped'] = 'No rows remain after existing input preprocessing'
                manifest['events'].append(record)
                continue
            batch = Batch.from_data_list([data]).to(args.device)
            capture.clear()
            with torch.no_grad():
                prediction = inference_forward(model, batch)
            if prediction.ndim != 2 or prediction.shape != (len(data.x), layout['output_dim']):
                raise ValueError('Unexpected output shape; check checkpoint architecture/layout')
            def cpu(t):
                return t.detach().cpu().numpy()
            ids = cpu(data.y[:, 0]).astype(np.int64)
            event = dict(embedding=cpu(prediction[:, -args.embedding_dim:]),
                         beta=cpu(prediction[:, 0].sigmoid()), truth_id=ids,
                         is_track=cpu(data.x[:, 4]) > 0.5,
                         truth_valid=cpu(data.truth_valid).astype(bool) if hasattr(data, 'truth_valid') else ids > 0,
                         hit_id=cpu(data.hitid), mc_id=cpu(data.mcid),
                         input_row=cpu(data.input_row) if hasattr(data, 'input_row') else np.arange(len(ids)),
                         detected_energy=cpu(data.feat[:, 0]))
            event.update(truth_arrays(cpu(data.label), event['truth_valid'] & (ids > 0)))
            event.update(capture.collect(len(ids)))
            event['spaces_json'] = np.array(json.dumps(capture.specs))
            validate_event(event)
            record.update(file=f'event_{event_id:06d}.npz', points=len(ids), dimensions=args.embedding_dim)
            event['metadata_json'] = np.array(json.dumps(record))
            np.savez_compressed(output/record['file'], **event)
            manifest['events'].append(record)
            print(f"Saved {record['file']}: {len(ids)} points, {args.embedding_dim} coordinates", flush=True)
        manifest['status'] = 'complete'
    finally:
        capture.close()
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
