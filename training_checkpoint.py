"""Epoch-boundary training checkpoints, including each DDP rank's state."""
import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


# These options affect only the launch, output, or total stopping epoch.
_RUNTIME_OPTIONS = {
    'resume', 'model_ckpt', 'epochs', 'ckptdir', 'ddp_log_dir', 'log_path',
    'pretraining_metrics_log_path', 'master_addr', 'master_port', 'gpus', 'cuda',
    'progress_rank', 'progress_mininterval', 'rank_log_interval', 'verbose',
    'epoch_budget_deadline', 'epoch_budget_stop_file',
}


def _resume_config(config):
    """Interpret checkpoints predating the opt-in particle heads as disabled.

    Only these known additions have legacy equivalents. All other missing or
    changed options still fail the comparison; enabling a head is not a resume.
    Do not mutate the saved configuration or relax model state validation.
    """
    return dict(dict(pid_head=False, five_particle_energy_heads=False,
                     pid_loss_weight=1., epochs_noPID=-1), **config)


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def _kind(obj):
    return None if obj is None else type(obj).__module__ + '.' + type(obj).__qualname__


class TrainingCheckpoint:
    def __init__(self, model, optimizer, scheduler, scaler, args, payload_fn):
        self.model = model.module if hasattr(model, 'module') else model
        self.optimizer, self.scheduler, self.scaler = optimizer, scheduler, scaler
        self.args, self.payload_fn = args, payload_fn
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.config = {k: v for k, v in vars(args).items() if k not in _RUNTIME_OPTIONS}

    def _local_state(self):
        return _cpu_copy(dict(
            optimizer=self.optimizer.state_dict(),
            # The custom cosineReduce scheduler's inherited state_dict includes
            # t_epoch, restart_period, restarts, eta_max, counters and callbacks.
            scheduler=self.scheduler.state_dict() if self.scheduler is not None else None,
            scheduler_type=_kind(self.scheduler),
            scaler=self.scaler.state_dict() if self.scaler is not None else None,
            rng=_rng_state(), buffers=dict(self.model.named_buffers()),
        ))

    def save(self, directory, completed_epoch, history):
        local = self._local_state()
        states = [None] * self.world_size
        if self.distributed:
            dist.all_gather_object(states, local)
        else:
            states[0] = local
        error = [None]
        if self.rank == 0:
            try:
                directory = Path(directory)
                directory.mkdir(parents=True, exist_ok=True)
                name = ('ckpt_initial.pth.tar' if completed_epoch < 0 else
                        f'ckpt_{completed_epoch}_1.pth.tar')
                path = directory / name
                payload = self.payload_fn(self.model, epoch=completed_epoch, args=self.args)
                payload['training_state'] = dict(
                    version=1, next_epoch=completed_epoch + 1,
                    world_size=self.world_size, config=self.config,
                    ranks=states, history=history,
                )
                # A killed writer leaves the previous last.pth.tar usable.
                with open(str(path) + '.tmp', 'wb') as stream:
                    torch.save(payload, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(str(path) + '.tmp', path)
                latest_tmp = directory / 'last.pth.tar.tmp'
                if os.path.lexists(latest_tmp):
                    latest_tmp.unlink()
                latest_tmp.symlink_to(name)
                os.replace(latest_tmp, directory / 'last.pth.tar')
                print(f'CHECKPOINT completed_epoch={completed_epoch} '
                      f'next_epoch={completed_epoch + 1} path={directory / "last.pth.tar"}', flush=True)
            except Exception as exc:
                error[0] = f'Checkpoint save failed: {exc}'
        if self.distributed:
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])

    def load(self, path):
        # Only load trusted training checkpoints; optimizer/RNG/custom scheduler
        # states include Python objects, not just tensor weights.
        payload = torch.load(path, map_location='cpu', weights_only=False)
        state = payload.get('training_state')
        if state is None or state.get('version') != 1:
            raise ValueError('Full resume requires a new training checkpoint; '
                             'old weights-only files support --model-ckpt only.')
        if state['world_size'] != self.world_size:
            raise ValueError('Resume requires the same GPU/rank count as the checkpoint.')
        current_config = _resume_config(self.config)
        saved_config = _resume_config(state['config'])
        differences = [key for key in sorted(set(current_config) | set(saved_config))
                       if current_config.get(key) != saved_config.get(key)]
        if differences:
            raise ValueError('Resume configuration differs: ' + ', '.join(differences))
        local = state['ranks'][self.rank]
        if local['scheduler_type'] != _kind(self.scheduler):
            raise ValueError('Resume scheduler type differs from the checkpoint.')
        if (local['scaler'] is None) != (self.scaler is None):
            raise ValueError('Resume AMP scaler configuration differs.')
        if self.scheduler is not None:
            for key in ('epoch_size', 'batch_size'):
                if key in local['scheduler'] and getattr(self.scheduler, key) != local['scheduler'][key]:
                    raise ValueError(f'Resume scheduler {key} differs; keep the same data and batch size.')
        self.model.load_state_dict(payload['model'], strict=True)
        # BatchNorm buffers can differ across ranks even with synchronized weights.
        with torch.no_grad():
            for name, buffer in self.model.named_buffers():
                buffer.copy_(local['buffers'][name])
        self.optimizer.load_state_dict(local['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(local['scheduler'])
        if self.scaler is not None:
            self.scaler.load_state_dict(local['scaler'])
        _restore_rng(local['rng'])
        print(f'RESUME rank={self.rank} next_epoch={state["next_epoch"]} '
              f'lr={self.optimizer.param_groups[0]["lr"]:.12g} path={path}', flush=True)
        return state['next_epoch'], state['history']
