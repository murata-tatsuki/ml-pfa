"""Training / eval batch steps shared by single-GPU and DDP (optional DDP sync helpers)."""

from __future__ import annotations
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn.utils as utils
from typing import Optional
from torch.cuda.amp import GradScaler, autocast
from training.loss import loss_fn

def amp_autocast(args):
    """No-op context when AMP is off; otherwise CUDA autocast with requested dtype."""
    if not getattr(args, "amp", False):
        return nullcontext()
    dt = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return autocast(enabled=True, dtype=dt)


def amp_grad_scaler(args):
    """GradScaler only for fp16 AMP; None when AMP off or bf16."""
    if not getattr(args, "amp", False):
        return None
    if args.amp_dtype == "fp16":
        return GradScaler()
    return None


def forward_training_loss(
    model,
    data,
    device,
    args,
    qmin: float,
    loss_offset: float,
    epoch: int,
    ):
    """Forward pass under autocast and compute train loss (no backward)."""
    data = data.to(device)
    with amp_autocast(args):
        result = model(data.x, data.batch)
        if args.jit:
            raise NotImplementedError("--jit training path is not wired")
        loss, components = loss_fn(
            result,
            data,
            args,
            qmin,
            loss_offset,
            i_epoch=epoch,
            use_charge_track_likeness=args.use_charged_cluster_loss,
        )
    return loss, components, result


def backward_with_optimizer_step(
    loss,
    model,
    optimizer,
    scaler: Optional[GradScaler],
    args,
    scheduler=None,
    ):
    """Backward, optional grad clipping, optimizer step, optional cyclic LR batch_step."""
    if scaler is not None:
        scaler.scale(loss).backward()
        if not args.no_clipping:
            scaler.unscale_(optimizer)
            utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if not args.no_clipping:
            utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)
        optimizer.step()
    if (
        scheduler is not None
        and not args.settings_Sep01
        and not args.ReduceLROnPlateau
    ):
        scheduler.batch_step()


def training_batch_step(
    model,
    data,
    device,
    optimizer,
    scaler: Optional[GradScaler],
    args,
    qmin: float,
    loss_offset: float,
    epoch: int,
    scheduler=None,
    ):
    """
    One training iteration: zero_grad, forward + loss (autocast), backward, clip, optimizer step,
    and cyclic LR batch_step when applicable.

    Returns
    -------
    loss : Tensor
        Scalar loss tensor (with grad) before .item() for logging.
    components : dict
        Second return from ``loss_fn`` (loss term breakdown).
    """
    optimizer.zero_grad()
    loss, components, _result = forward_training_loss(
        model, data, device, args, qmin, loss_offset, epoch
    )
    backward_with_optimizer_step(loss, model, optimizer, scaler, args, scheduler)
    return loss, components


def eval_batch_loss_components(
    model,
    data,
    device,
    args,
    qmin: float,
    loss_offset: float,
    epoch: int,
    ):
    """
    Forward + loss with ``return_components=True`` (validation). Caller should wrap the loop in
    ``torch.no_grad()`` and set ``model.eval()`` as appropriate.
    """
    data = data.to(device)
    with amp_autocast(args):
        result = model(data.x, data.batch)
        if args.jit:
            raise NotImplementedError("--jit eval path is not wired")
        return loss_fn(
            result,
            data,
            args,
            qmin,
            loss_offset,
            i_epoch=epoch,
            return_components=True,
            use_charge_track_likeness=args.use_charged_cluster_loss,
        )


def ddp_all_reduce_loss_totals(
    loss_components: dict,
    n_local_batches: int,
    device: torch.device,
    ) -> int:
    """
    Sum loss component tensors across ranks and divide by total batch count across all ranks.

    Returns total batch count across all ranks (for logging if needed).
    """
    nb = torch.tensor([n_local_batches], device=device, dtype=torch.long)
    dist.all_reduce(nb, op=dist.ReduceOp.SUM)
    total_batches = nb.item()
    for key in loss_components:
        dist.all_reduce(loss_components[key], op=dist.ReduceOp.SUM)
        loss_components[key] /= total_batches
    return total_batches
