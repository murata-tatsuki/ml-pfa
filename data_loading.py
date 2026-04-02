"""Build ILC training datasets (shared by single-GPU and DDP code paths)."""

from __future__ import annotations

from dataset import ILCDataset
from dataset_ilc_sharded import ILCDatasetSharded


def make_ilc_dataset(args, inputdir):
    """Select ILCDataset vs ILCDatasetSharded from args (default: ILCDataset)."""
    common = dict(
        timingCut=args.timing_cut,
        thetaphi=args.thetaphi,
        test_mode=True,
        momentum=args.momentum,
        momentumAmp=args.momentum_amp,
        mctpe=args.mctpe,
    )
    if getattr(args, "ilc_sharded", False):
        print(
            "Using ILCDatasetSharded (per-file load, no concatenate). "
            f"file_cache_size={getattr(args, 'ilc_file_cache', 2)}"
        )
        return ILCDatasetSharded(
            inputdir,
            **common,
            file_cache_size=getattr(args, "ilc_file_cache", 2),
        )
    return ILCDataset(inputdir, **common)


def prepare_train_val_datasets(args, batch_size):
    """
    Load the dataset from the input directory and apply reduce_noise, dry subsampling, and train/validation splitting.

    Parameters
    ----------
    args : Namespace
        Must expose timing_cut, thetaphi, dry, no_split, reduce_noise, inputdir,
        inputdir_validate, and related fields used by ``make_ilc_dataset``.
    batch_size : int
        Batch size already adjusted by the caller for DataParallel / DDP if applicable.

    Returns
    -------
    train_dataset, test_dataset, batch_size
        When reduce_noise is enabled, ``batch_size`` is updated as in the original behavior.
    """
    print(f'thetaphi at main: {args.thetaphi}')
    print("Loading dataset...")

    dataset = make_ilc_dataset(args, args.inputdir)

    if args.reduce_noise:
        dataset.reduce_noise = 0.70
        multiply_batch_size = 1
        print(f'Throwing away {dataset.reduce_noise * 100:.0f}% of noise (good for testing ideas, not for final results)')
        print(f'Batch size: {batch_size} --> {multiply_batch_size * batch_size}')
        batch_size *= multiply_batch_size

    if args.dry:
        keep = 0.005
        print(f'Keeping only {100.0 * keep:.1f}% of events for debugging')
        dataset, _ = dataset.split(keep)

    if args.no_split:
        train_dataset = dataset
        test_dataset = make_ilc_dataset(args, args.inputdir_validate)
    else:
        train_dataset, test_dataset = dataset.split(0.8)

    return train_dataset, test_dataset, batch_size
