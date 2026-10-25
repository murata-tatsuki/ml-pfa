#!/usr/bin/env python3
"""Write one scan ROOT for an epoch while retaining the public save_root CLI.

H5s stream through the existing writer in input-list order. Each H5 receives
the original nstart/nend selection independently, as in --multi-h5 exports.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from epoch_root import input_identity, validate_epoch_output, write_epoch_metadata


class _InputYielder:
    def __init__(self, model, args, inputs, training):
        self.model, self.args, self.inputs = model, args, inputs
        self.training_inputs = training
        self.pid_mode = args.pid_readout == 'representative'
        self.provenance, self.source_ranges = {}, []

    def iter_pred(self, nmax=None, *physics):
        from root_export import cedric
        from dataset import ILCDataset
        from test_yielder_edit import TestYielder

        offset = 0
        for identity in self.inputs:
            path = identity['path']
            if self.training_inputs:
                current = copy.copy(self.args)
                current.datapath = path
                yielder = cedric._PIDYielder(
                    self.model, current, path, self.args.nstart, self.args.nend)
            else:
                dataset = ILCDataset(path, timingCut=False, thetaphi=self.args.input_dim == 7,
                    test_mode=True, nstart=self.args.nstart, nend=self.args.nend,
                    pandora=self.args.pandora, momentum=self.args.momentum,
                    momentumAmp=self.args.momentum_amp, mctpe=self.args.mctpe,
                    event_energy=self.args.event_total_energy or cedric.has_event_builder(path))
                yielder = TestYielder(model=self.model, dataset=dataset, device=self.args.device,
                    pandora=self.args.pandora,
                    event_energy=self.args.event_total_energy or cedric.has_event_builder(path))
            count = 0
            limit = None if self.args.nend == -1 else self.args.nend
            for event, prediction in yielder.iter_pred(limit, *physics):
                count += 1
                if offset + count > 2147483647:
                    raise ValueError('Epoch event IDs exceed the Cedric Int_t schema')
                yield event, prediction
            if self.training_inputs:
                self.provenance.update(yielder.provenance)
            self.source_ranges.append(dict(input=identity, event_offset=offset, event_count=count))
            offset += count
            del yielder


def _validate_inputs(args, paths, parser, training):
    import h5py
    from extended_h5 import validate_training_schema

    if training:
        for flag in ('pandora', '1tomany_clustering', 'energy_regression_weight', 'mctpe', 'energy_branch'):
            if getattr(args, flag):
                parser.error('Shared training-input export cannot use --' + flag.replace('_', '-'))
        if args.cluster_energy_source not in (None, 'predicted'):
            parser.error('Use --truth-clustering to select truth membership')
    elif args.exclude_gap_hits or args.detector_one_hot:
        parser.error('Gap/one-hot options require current training inputs')
    for path in paths:
        with h5py.File(path, 'r') as handle:
            extended = (handle.attrs.get('schema_version') is not None
                        or 'row_info' in handle or 'collections' in handle)
            if training:
                if args.pid_input_format == 'legacy' and extended:
                    parser.error(str(path) + ': --pid-input-format legacy requires an old H5')
                if args.pid_input_format == 'extended' or extended:
                    validate_training_schema(handle)
                elif args.exclude_gap_hits or args.detector_one_hot:
                    parser.error(str(path) + ': legacy H5 lacks detector collections')
            if 'feature' not in handle or 'label' not in handle:
                parser.error(str(path) + ': H5 requires feature and label builders')
            if args.event_total_energy and 'event' not in handle:
                parser.error(str(path) + ': --event-total-energy requires an event builder')


def _model(args, training):
    from root_export import cedric

    if args.pandora:
        return cedric._PandoraNoopModel()
    input_dim = args.input_dim + (3 + int(args.momentum_amp) if args.momentum else 0)
    if args.detector_one_hot:
        from extended_h5 import DETECTOR_ONE_HOT_DIM
        input_dim += DETECTOR_ONE_HOT_DIM
    if training:
        model = cedric.get_model(args.ckpt, jit=False, input_dim=input_dim,
            output_dim=args.output_dim + 2, energy_regression=True, energy_regression_cluster=True,
            model_variant=args.model_variant, detector_one_hot=args.detector_one_hot,
            cluster_energy_source='predicted', cluster_energy_tbeta=args.tbeta,
            cluster_energy_td=args.td, clustering_bn_eps=args.clustering_bn_eps)
    elif args.energy_branch:
        model = cedric.get_model_branch(args.ckpt, jit=False,
            input_dim=input_dim, output_dim=args.output_dim + 2)
    else:
        model = cedric.get_model(args.ckpt, jit=False, input_dim=input_dim,
            output_dim=args.output_dim + 2, energy_regression=args.energy_regression,
            energy_regression_cluster=args.energy_regression_cluster,
            energy_regression_weight=args.energy_regression_weight, model_variant=args.model_variant,
            clustering_bn_eps=args.clustering_bn_eps, **cedric.inference_loader_kwargs(args))
    model = model.to(args.device).eval()
    if args.pid_readout == 'representative':
        from particle_heads import resolve_pid_readout
        resolve_pid_readout(model, 'representative')
    return model


def main(argv=None):
    import save_root
    from root_export import cedric, runtime

    parser = save_root.build_parser()
    parser.description = __doc__
    args = parser.parse_args(argv)
    save_root._validate_options(args, parser)
    if args.multi_h5 or args.beta_d_scan:
        parser.error('Epoch export writes one ROOT; omit --multi-h5 and threshold scan options')
    paths = cedric._resolve_multi_h5(args.datapath)
    output = Path(args.outfile).absolute()
    if output.suffix != '.root' or output.is_symlink() or output.is_dir():
        parser.error('Epoch outfile must be a regular .root file path')
    if output.resolve() in set(paths) | {Path(args.ckpt).resolve()}:
        parser.error('Output must differ from input H5 and checkpoint')
    save_root._configure_model(args, parser)
    training = cedric._uses_training_inputs(args)
    _validate_inputs(args, paths, parser, training)
    identities = []
    for path in paths:
        stat = path.stat()
        identities.append(input_identity(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)))
    if not args.replace and validate_epoch_output(output, args, identities):
        print('[ROOT epoch resume] skip complete identical ' + str(output), flush=True)
        return 0
    import torch
    torch.set_num_threads(args.pid_threads)
    model = _model(args, training)
    yielder = _InputYielder(model, args, identities, training)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = cedric._temporary_output(output)
    try:
        # Child yielders own each H5 range. The outer writer must not truncate
        # the concatenation to one H5's nend; metadata retains the original args.
        cedric.save_root(args.datapath, args.ckpt, str(staging), nstart=args.nstart,
            nend=-1, timingCut=False, input_dim=args.input_dim, output_dim=args.output_dim,
            args=args, model=model, prediction_yielder=yielder)
        runtime.write_metadata(staging, args, args.datapath)
        write_epoch_metadata(staging, yielder.source_ranges)
        if not validate_epoch_output(staging, args, identities):
            raise RuntimeError('Single-epoch ROOT failed completion/identity validation')
        os.replace(staging, output)
        print('[ROOT epoch] completed -> ' + str(output), flush=True)
    finally:
        if cedric._ACTIVE_CHUNK_FILE is not None:
            cedric._ACTIVE_CHUNK_FILE.abort()
        staging.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
