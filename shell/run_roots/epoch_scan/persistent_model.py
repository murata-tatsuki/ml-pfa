"""Reuse one eager inference model inside an epoch-scan worker.

This adapter is installed only in the dedicated worker process. The public
``save_root.py`` command and its normal model factory remain unchanged.
"""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path


_CONFIG_NAMES = ('particle_heads_config', 'cluster_energy_config',
                 'training_input_config')


def _file_identity(path):
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _metadata_signature(metadata):
    settings = copy.deepcopy(metadata)
    for name in ('particle_heads_config', 'cluster_energy_config'):
        if isinstance(settings.get(name), dict):
            settings[name].pop('inference_epoch', None)
    return json.dumps(settings, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


class ModelCache:
    """Callable with the original ``get_model`` signature; cache at most one model.

    Compatible checkpoints replace every trained parameter and buffer strictly.
    Constructor options or checkpoint schemas that differ rebuild the model.
    TorchScript, DDP and implicit checkpoint selection use the original factory
    directly. Device placement and eval mode stay with the existing caller.
    """

    def __init__(self, original_factory):
        self.original_factory = original_factory
        self._factory_signature = inspect.signature(original_factory)
        self._counts = dict(builds=0, reloads=0, reuses=0)
        self.invalidate()

    @property
    def stats(self):
        return dict(self._counts)

    def invalidate(self):
        """Drop all cached state, including after a partially failed weight load."""
        self._model = None
        self._signature = None
        self._identity = None
        self._metadata = None
        self._validation_state = None
        self._state_signature = None

    @staticmethod
    def _validate(metadata, validation_state, options):
        from extended_h5 import validate_one_hot_checkpoint
        from particle_heads import validate_checkpoint_config

        validate_one_hot_checkpoint(metadata, options.get('detector_one_hot', False))
        validate_checkpoint_config(validation_state,
                                   metadata.get('particle_heads_config'))

    @staticmethod
    def _restore_runtime(model, metadata, options):
        from particle_heads import raw_model

        raw = raw_model(model)
        if getattr(raw, 'cluster_energy_pooling', False):
            config = metadata.get('cluster_energy_config') or {}
            raw.cluster_energy_source = options.get('cluster_energy_source') or 'predicted'
            for name, default in (('tbeta', .7), ('td', .5)):
                override = options.get('cluster_energy_' + name)
                setattr(raw, 'cluster_energy_' + name,
                        config.get(name, default) if override is None else override)
            raw.interaction_start_epoch = config.get('interaction_start_epoch', 0)
            raw.cluster_energy_coordinate_start = config.get(
                'coordinate_start', 2 if options.get('use_charge_track_likeness') else 1)
            raw.cluster_energy_inference_epoch = config.get(
                'inference_epoch', raw.interaction_start_epoch)
        particle = metadata.get('particle_heads_config') or {}
        if particle:
            raw.interaction_start_epoch = particle['interaction_start_epoch']
            raw.particle_heads_inference_epoch = particle['inference_epoch']
            raw.cluster_energy_coordinate_start = particle.get('coordinate_start', 1)

    def __call__(self, *args, **kwargs):
        try:
            return self._get(*args, **kwargs)
        except BaseException:
            # load_state_dict can fail after copying some tensors. Such a model
            # must never be offered to another job as a valid cached checkpoint.
            self.invalidate()
            raise

    def _get(self, *args, **kwargs):
        bound = self._factory_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        options = dict(bound.arguments)
        if (options.get('ckpt') is None or options.get('jit', True)
                or options.get('ddp', False)):
            self.invalidate()
            model = self.original_factory(*args, **kwargs)
            self._counts['builds'] += 1
            return model

        import torch
        from model import _extract_state_dict
        from particle_heads import raw_model

        path = Path(options['ckpt']).expanduser().resolve()
        identity = _file_identity(path)
        state = None
        if identity == self._identity:
            metadata = self._metadata
            validation_state = self._validation_state
            state_signature = self._state_signature
        else:
            checkpoint = torch.load(str(path), map_location='cpu')
            if _file_identity(path) != identity:
                raise RuntimeError('Checkpoint changed while loading: ' + str(path))
            state = _extract_state_dict(checkpoint)
            metadata = {name: copy.deepcopy(checkpoint[name]) for name in _CONFIG_NAMES
                        if name in checkpoint}
            state_signature = tuple(sorted(
                (name, tuple(value.shape), str(value.dtype))
                for name, value in state.items()))
            # Validation needs the names of all weights and only one scalar
            # buffer, rather than retaining checkpoint/optimizer tensors in RAM.
            validation_state = dict.fromkeys(state)
            if '_particle_energy_version' in state:
                validation_state['_particle_energy_version'] = int(
                    state['_particle_energy_version'])

        self._validate(metadata, validation_state, options)
        factory_options = {name: value for name, value in options.items() if name != 'ckpt'}
        signature = (json.dumps(factory_options, sort_keys=True, default=str,
                                separators=(',', ':'), allow_nan=False),
                     state_signature, _metadata_signature(metadata))
        if self._model is not None and signature == self._signature:
            model = self._model
            if identity == self._identity:
                action = 'reuses'
            else:
                raw = raw_model(model)
                raw.load_state_dict(state, strict=True)
                raw.__dict__.pop('_particle_cluster_inference_head', None)
                action = 'reloads'
        else:
            # Release the previous model before allocating an incompatible one.
            self.invalidate()
            model = self.original_factory(*args, **kwargs)
            action = 'builds'

        self._restore_runtime(model, metadata, options)
        if _file_identity(path) != identity:
            raise RuntimeError('Checkpoint changed while preparing model: ' + str(path))
        self._model = model
        self._signature = signature
        self._identity = identity
        self._metadata = metadata
        self._validation_state = validation_state
        self._state_signature = state_signature
        self._counts[action] += 1
        return model
