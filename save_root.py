#!/usr/bin/env python3
"""Export Cedric ROOT trees with optional PID and two/five energy heads.

The positional arguments and physics options of the former Cedric fast, PID
and resume-scan commands are accepted here. The shared writer always stores
t/reco/prediction/event/jet, uses C++ chunk filling, and publishes each file
only after validation. --energy-heads selects a compatible trained checkpoint;
it does not change the checkpoint's network or invent untrained heads.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path


# Set these before importing NumPy/Torch or starting scan worker processes.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')


def _boolean(value):
    value = str(value).lower()
    if value in ('true', '1', 'yes', 'on'):
        return True
    if value in ('false', '0', 'no', 'off'):
        return False
    raise argparse.ArgumentTypeError('Use True or False for timingCut')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datapath', help='Input H5/directory; use --multi-h5 for multiple glob/list matches')
    parser.add_argument('ckpt', help='Trained checkpoint (unused with --pandora)')
    parser.add_argument('outfile', help='ROOT path; directory for scans or --multi-h5')
    parser.add_argument('nstart', type=int, nargs='?', default=0)
    parser.add_argument('nend', type=int, nargs='?', default=-1, help='Original positional event count; -1 reads to the end')
    parser.add_argument('timingCut', type=_boolean, nargs='?', default=False)
    parser.add_argument('input_dim', type=int, nargs='?', default=7, help='Base input dimension before momentum/one-hot columns')
    parser.add_argument('output_dim', type=int, nargs='?', default=5, help='Clustering output dimension, including beta')

    model = parser.add_argument_group('Trained model and energy/PID readout')
    model.add_argument('--energy-heads', choices=('auto', '2', '5'), default='auto', help='auto detects checkpoint; 2=track/calo, 5=species energy heads')
    model.add_argument('--pid-readout', choices=('auto', 'legacy', 'none', 'representative'), default='auto', help='auto enables PID when trained; legacy/none disables it')
    model.add_argument('--pid', dest='pid_readout', action='store_const', const='representative', help='Alias for --pid-readout representative')
    model.add_argument('--no-pid', dest='pid_readout', action='store_const', const='legacy', help='Alias for --pid-readout legacy (two energy heads only)')
    model.add_argument('--model-variant', choices=('auto', 'legacy', 'multihead'), default='auto')
    model.add_argument('--energy-regression', action='store_true', help='Accepted for existing commands; implied by --energy-heads')
    model.add_argument('--energy-regression-cluster', action='store_true', help='Accepted for existing commands; implied by two/five energy heads')
    model.add_argument('-eb', '--energy-branch', action='store_true')
    model.add_argument('--cluster-energy-source', choices=('truth', 'predicted'), default=None, help='Membership used by a trained cluster-pooling energy head')
    model.add_argument('--energy-regression-weight', action='store_true', help=argparse.SUPPRESS)
    model.add_argument('--particle-cluster-score-threshold', type=float, default=None,
                       help='Reject trackless candidates below this trained additional-particle score; '
                            'default 0.5 for score checkpoints, inactive for legacy checkpoints')

    inputs = parser.add_argument_group('Inputs and evaluation')
    inputs.add_argument('-e', '--momentum', action='store_true')
    inputs.add_argument('-ea', '--momentum-amp', action='store_true')
    inputs.add_argument('--mctpe', action='store_true')
    inputs.add_argument('--event-total-energy', action='store_true')
    inputs.add_argument('--pandora', action='store_true', help='Use the legacy stored Pandora clustering')
    inputs.add_argument('--truth-clustering', action='store_true')
    inputs.add_argument('--exclude-gap-hits', action='store_true')
    inputs.add_argument('--detector-one-hot', action='store_true')
    inputs.add_argument('--pid-input-format', choices=('auto', 'extended', 'legacy'), default='auto')
    inputs.add_argument('--pid-detail', choices=('full',), default='full')
    inputs.add_argument('--multi-h5', action='store_true', help='One ROOT per H5, sharing the model')
    inputs.add_argument('--1tomany-clustering', action='store_true', help=argparse.SUPPRESS)

    scan = parser.add_argument_group('Thresholds and scans')
    scan.add_argument('--tbeta', type=float, default=0.9)
    scan.add_argument('--td', type=float, default=0.5)
    scan.add_argument('--beta-d-scan', '--scan', dest='beta_d_scan', action='store_true', help='Scan both thresholds; default grid is 0.9 through 0.1 on each axis')
    scan.add_argument('--scan-tbeta', type=float, nargs='+')
    scan.add_argument('--scan-td', type=float, nargs='+')
    scan.add_argument('--scan-workers', type=int, default=int(os.environ.get('FAST_SCAN_WORKERS', '4')))

    execution = parser.add_argument_group('Fast output and resume')
    execution.add_argument('--device', default='cpu')
    execution.add_argument('--pid-threads', type=int, default=4)
    execution.add_argument('--root-chunk-events', type=int, default=int(os.environ.get('CEDRIC_ROOT_CHUNK_EVENTS', '10')))
    execution.add_argument('--pipeline-depth', type=int, default=int(os.environ.get('CEDRIC_PIPELINE_DEPTH', '40')))
    execution.add_argument('--replace', '--no-resume', dest='replace', action='store_true', help='Regenerate outputs; default skips complete identical results')
    execution.add_argument('--skip-existing', dest='replace', action='store_false', help='Reuse only outputs with matching input/checkpoint/physics identity')
    parser.set_defaults(replace=False)
    return parser


def _configure_model(args, parser):
    """Validate explicit selectors against trained weights before any output."""
    if args.pandora:
        if args.particle_cluster_score_threshold is not None:
            parser.error('Stored Pandora clustering has no trained additional-particle score')
        if args.pid_readout == 'representative' or args.energy_heads == '5':
            parser.error('Stored Pandora clustering does not use GNN PID/species heads')
        args.energy_heads, args.pid_readout = 2, 'legacy'
        return

    import torch
    from model import (_extract_state_dict, _strip_module_prefix,
                       _is_multihead_state_dict, _infer_multihead_config)

    checkpoint = torch.load(str(Path(args.ckpt).expanduser()), map_location='cpu')
    state = _strip_module_prefix(_extract_state_dict(checkpoint))
    config = checkpoint.get('particle_heads_config') or {}
    cluster_config = config.get('particle_cluster_config') or {}
    has_particle_cluster = any(name.startswith('particle_cluster_head.') for name in state)
    if has_particle_cluster and args.pid_readout in ('legacy', 'none'):
        parser.error('Particle-cluster checkpoints require representative PID readout')
    if args.particle_cluster_score_threshold is not None and not cluster_config.get('score', False):
        parser.error('--particle-cluster-score-threshold requires a trained particle-cluster score head')
    has_pid = any(name.startswith('pid_output.') for name in state)
    if config and bool(config.get('pid_head')) != has_pid:
        parser.error('Checkpoint PID metadata and trained weights disagree')
    if _is_multihead_state_dict(state):
        _, clustering_dim, energy_dims, _ = _infer_multihead_config(state)
        count = len(energy_dims)
        if any(dim != 1 for dim in energy_dims):
            parser.error('Each energy regression head must have one output')
        if clustering_dim != args.output_dim:
            parser.error(f'Checkpoint clustering output has {clustering_dim} columns, not {args.output_dim}')
    else:
        output = state.get('output.4.weight')
        if output is None:
            output = state.get('output.weight')
        if output is None:
            parser.error('Cannot identify the checkpoint energy heads')
        count = int(output.shape[0]) - args.output_dim
    if count not in (2, 5):
        parser.error(f'This exporter requires two or five trained energy heads; found {count}')
    if args.energy_heads != 'auto' and int(args.energy_heads) != count:
        parser.error(f'--energy-heads {args.energy_heads} disagrees with the {count}-head checkpoint')
    if count == 5 and not config.get('five_particle_energy_heads'):
        parser.error('Five species energy heads require matching particle_heads_config metadata')
    if args.pid_readout == 'auto':
        args.pid_readout = 'representative' if has_pid else 'legacy'
    elif args.pid_readout == 'none':
        args.pid_readout = 'legacy'
    if args.pid_readout == 'representative' and not has_pid:
        parser.error('Representative PID requires a trained PID head in the checkpoint')
    if count == 5 and args.pid_readout != 'representative':
        parser.error('Normal five-head energy readout requires predicted PID; enable --pid')
    args.energy_heads = count
    args.energy_regression = True
    args.energy_regression_cluster = True
    print(f'[model] {count} energy heads; PID readout: {args.pid_readout}')


def _validate_options(args, parser):
    if args.timingCut:
        parser.error('Use timingCut=False and pre-cut H5 inputs (timingcut.py)')
    if args.nstart < 0 or args.nend == 0 or args.nend < -1:
        parser.error('Require nstart >= 0 and nend > 0 or -1')
    if args.root_chunk_events < 1 or args.pid_threads < 1 or args.scan_workers < 1:
        parser.error('Chunk size, PID threads and scan workers must be positive')
    if args.pipeline_depth < 0:
        parser.error('--pipeline-depth must be zero or positive')
    if not math.isfinite(args.tbeta) or not 0 < args.tbeta < 1:
        parser.error('--tbeta must be finite and between zero and one')
    if not math.isfinite(args.td) or args.td <= 0:
        parser.error('--td must be finite and positive')
    if args.particle_cluster_score_threshold is not None and (
            not math.isfinite(args.particle_cluster_score_threshold)
            or not 0 <= args.particle_cluster_score_threshold <= 1):
        parser.error('--particle-cluster-score-threshold must be finite and in [0, 1]')
    if (args.scan_tbeta or args.scan_td) and not args.beta_d_scan:
        parser.error('--scan-tbeta/--scan-td require --beta-d-scan')
    if args.energy_regression_weight or getattr(args, '1tomany_clustering'):
        parser.error('Legacy weighted-coefficient/1tomany modes are outside this Cedric integration')
    args.output_schema = 'cedric'
    args.outfile = str(Path(args.outfile).expanduser())
    if not args.pandora:
        args.ckpt = str(Path(args.ckpt).expanduser().resolve())


def _normalize_inputs(args, parser):
    """Give loaders literal paths; keep the old directory-as-one-dataset mode."""
    from root_export.cedric import _resolve_multi_h5

    paths = _resolve_multi_h5(args.datapath)
    if args.multi_h5 or args.beta_d_scan:
        return
    if len(paths) == 1:
        args.datapath = str(paths[0].resolve())
    elif Path(args.datapath).expanduser().is_dir():
        args.datapath = str(Path(args.datapath).expanduser().resolve())
    else:
        parser.error('Multiple glob/list matches require --multi-h5; '
                     'a literal directory can be combined into one output')


def _write_legacy_point(args, output, model=None):
    from root_export import cedric, runtime

    output = Path(output).expanduser()
    if not args.replace and runtime.validate_existing(output, args, args.datapath):
        print(f'[skip] complete identical ROOT: {output}')
        return model
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = cedric._temporary_output(output)
    try:
        model = cedric.save_root(
            args.datapath, args.ckpt, str(temporary), nstart=args.nstart,
            nend=args.nend, timingCut=args.timingCut, input_dim=args.input_dim,
            output_dim=args.output_dim, args=args, model=model)
        runtime.write_metadata(temporary, args, args.datapath)
        if not runtime.validate_existing(temporary, args, args.datapath):
            raise RuntimeError('Generated ROOT failed identity/schema validation: ' + str(temporary))
        os.replace(temporary, output)
        print(f'[complete] {output}')
        return model
    finally:
        temporary.unlink(missing_ok=True)


def _run_legacy(args, parser):
    import h5py
    from root_export import cedric

    if args.exclude_gap_hits or args.detector_one_hot:
        parser.error('Gap/one-hot input options currently require representative PID input')
    inputs = cedric._resolve_multi_h5(args.datapath)
    for path in inputs:
        with h5py.File(path, 'r') as handle:
            if handle.attrs.get('schema_version') == 'pandora-eval-1':
                parser.error('Full pandora-eval-1 comparison is a separate writer; this command uses Cedric output')
    if args.multi_h5 or args.beta_d_scan:
        if len({path.stem for path in inputs}) != len(inputs):
            parser.error('Inputs with identical H5 stems would overwrite each other')
        jobs = [(str(path), Path(args.outfile) / (path.stem + '.root')) for path in inputs]
    else:
        jobs = [(args.datapath, Path(args.outfile))]
    protected = {path.resolve() for path in inputs}
    if not args.pandora:
        protected.add(Path(args.ckpt).expanduser().resolve())
    model = None
    for datapath, output in jobs:
        job = copy.copy(args)
        job.datapath = datapath
        if args.beta_d_scan:
            from root_export.scan import run_scan
            run_scan(job)
            continue
        if output.expanduser().resolve() in protected:
            parser.error('Output must differ from the input H5 and checkpoint')
        if output.is_dir():
            parser.error('ROOT output is a directory: ' + str(output))
        model = _write_legacy_point(job, output, model)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_options(args, parser)
    _configure_model(args, parser)
    _normalize_inputs(args, parser)
    if args.pid_readout == 'representative':
        from root_export.cedric import _run_representative_pid
        _run_representative_pid(args, args.timingCut, parser)
    else:
        _run_legacy(args, parser)


if __name__ == '__main__':
    main()
