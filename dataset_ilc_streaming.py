from __future__ import annotations

import glob
import json
from typing import Iterable

import awkward as ak
import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from dataset import ILCDataset
import tools.load_awkward as la


class ILCStreamingDataset(IterableDataset):
    """
    Stream ILC HDF5 samples file-by-file without concatenating all awkward arrays.

    Shuffle is intentionally done inside the dataset because IterableDataset cannot
    use DataLoader(shuffle=True) or DistributedSampler.
    """

    def __init__(
        self,
        path,
        flip=True,
        reduce_noise=None,
        para_tanh=True,
        recreate=False,
        timingCut=False,
        thetaphi=False,
        test_mode=False,
        nstart=0,
        nend=-1,
        pandora=False,
        momentum=False,
        momentumAmp=False,
        mctpe=False,
        event_energy=False,
        seed=1001,
        shuffle=True,
        shuffle_buffer_size=256,
        files_per_chunk=1,
        pad_to_equal_workers=True,
        _files=None,
        _event_counts=None,
    ):
        super().__init__()
        self.root = path
        self.flip = flip
        self.reduce_noise = reduce_noise
        self.noise_index = -1
        self.noise_mask_cache = {}
        self.thetaphi = thetaphi
        self.test_mode = test_mode
        self.pandora = pandora
        self.event_energy = event_energy
        self.momentum = momentum
        self.momentumAmp = momentumAmp
        self.max_momentum = 3.0 if momentum else 1.0
        self.mctpe = mctpe
        self.timingCut = timingCut
        self.seed = int(seed)
        self.epoch = 0
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        self.files_per_chunk = max(1, int(files_per_chunk))
        self.pad_to_equal_workers = bool(pad_to_equal_workers)
        self._para_tanh = para_tanh
        self._recreate = recreate

        if _files is None:
            if path.endswith(".h5"):
                files = [path]
            else:
                files = list(sorted(glob.iglob(path + "/*.h5")))
        else:
            files = list(_files)
        if not files:
            raise ValueError(f"No .h5 files found in {path}")

        self.files = files
        if _event_counts is None:
            event_counts = [self._read_event_count(fp) for fp in self.files]
        else:
            event_counts = list(_event_counts)

        n_tot = sum(event_counts)
        nend_eff = n_tot if nend < 0 or n_tot < nend else nend
        if nstart > 0 or nend_eff != n_tot:
            self.files, event_counts = self._slice_files_by_event_range(
                self.files, event_counts, nstart, nstart + nend_eff
            )

        self.event_counts = event_counts
        self.total_events = int(sum(self.event_counts))
        self._stream_kw = dict(
            flip=self.flip,
            reduce_noise=self.reduce_noise,
            para_tanh=self._para_tanh,
            recreate=self._recreate,
            timingCut=self.timingCut,
            thetaphi=self.thetaphi,
            test_mode=self.test_mode,
            nstart=0,
            nend=-1,
            pandora=self.pandora,
            momentum=self.momentum,
            momentumAmp=self.momentumAmp,
            mctpe=self.mctpe,
            event_energy=self.event_energy,
            seed=self.seed,
            shuffle=self.shuffle,
            shuffle_buffer_size=self.shuffle_buffer_size,
            files_per_chunk=self.files_per_chunk,
            pad_to_equal_workers=self.pad_to_equal_workers,
        )

        print(
            "ILCStreamingDataset: "
            f"path={path}, files={len(self.files)}, events~={self.total_events}, "
            f"shuffle_buffer={self.shuffle_buffer_size}, files_per_chunk={self.files_per_chunk}"
        )

    @staticmethod
    def _read_event_count(path):
        with h5py.File(path, "r") as f:
            return int(json.loads(f["feature"].attrs["length"]))

    @staticmethod
    def _slice_files_by_event_range(files, counts, start, stop):
        selected_files = []
        selected_counts = []
        cursor = 0
        for fp, count in zip(files, counts):
            next_cursor = cursor + count
            if next_cursor > start and cursor < stop:
                selected_files.append(fp)
                selected_counts.append(count)
            cursor = next_cursor
            if cursor >= stop:
                break
        return selected_files, selected_counts

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_shuffle(self, shuffle):
        self.shuffle = bool(shuffle)
        self._stream_kw["shuffle"] = self.shuffle

    def set_shuffle_buffer_size(self, shuffle_buffer_size):
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        self._stream_kw["shuffle_buffer_size"] = self.shuffle_buffer_size

    def set_files_per_chunk(self, files_per_chunk):
        self.files_per_chunk = max(1, int(files_per_chunk))
        self._stream_kw["files_per_chunk"] = self.files_per_chunk

    def set_pad_to_equal_workers(self, pad_to_equal_workers):
        self.pad_to_equal_workers = bool(pad_to_equal_workers)
        self._stream_kw["pad_to_equal_workers"] = self.pad_to_equal_workers

    def shaper_tanh(self, x, a=1.0, b=1.0, c=0.0, d=0.0):
        return a * np.tanh(b * (x - c)) + d

    def __len__(self):
        return self.total_events

    def len(self):
        return self.total_events

    def split(self, fraction):
        split_index = max(1, min(len(self.files) - 1, int(fraction * len(self.files))))
        if len(self.files) == 1:
            raise ValueError("ILCStreamingDataset cannot split a single .h5 file by event yet; use --no-split with validation input.")
        print("ILCStreamingDataset split at file index = ", split_index)
        left = ILCStreamingDataset(
            self.root,
            _files=self.files[:split_index],
            _event_counts=self.event_counts[:split_index],
            **self._stream_kw,
        )
        right = ILCStreamingDataset(
            self.root,
            _files=self.files[split_index:],
            _event_counts=self.event_counts[split_index:],
            **self._stream_kw,
        )
        return left, right

    def _distributed_worker_info(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        global_worker_id = rank * num_workers + worker_id
        num_global_workers = world_size * num_workers
        return rank, world_size, worker_id, num_workers, global_worker_id, num_global_workers

    def _shuffled_file_indices(self, rng):
        indices = np.arange(len(self.files))
        if self.shuffle:
            rng.shuffle(indices)
        return indices.tolist()

    def _assign_files(self, file_indices, num_global_workers):
        assignments = [[] for _ in range(num_global_workers)]
        loads = [0 for _ in range(num_global_workers)]

        for file_idx in file_indices:
            target = int(np.argmin(loads))
            assignments[target].append(file_idx)
            loads[target] += self.event_counts[file_idx]

        return assignments, loads

    def _chunked_file_indices(self, file_indices: Iterable[int]):
        file_indices = list(file_indices)
        for start in range(0, len(file_indices), self.files_per_chunk):
            yield file_indices[start : start + self.files_per_chunk]

    def _iter_file_events(self, file_indices: Iterable[int], rng):
        for chunk_file_indices in self._chunked_file_indices(file_indices):
            feat_chunks = []
            label_chunks = []
            pand_chunks = [] if self.pandora else None
            event_chunks = [] if self.event_energy else None
            event_sources = []

            for file_idx in chunk_file_indices:
                path = self.files[file_idx]
                feat_ak, label_ak, _, _, pand_ak, _, event_ak = la.load_awkward2(path)
                if self.timingCut:
                    n_ev = int(ak.num(feat_ak, axis=0))
                    feat_ak, label_ak = ILCDataset.timingCut(
                        feat_ak, label_ak, cutoff_time=14, nstart=0, nend=n_ev
                    )
                if self.pandora and pand_ak is None:
                    raise ValueError(f"Pandora requested but missing in {path}")
                if self.event_energy and event_ak is None:
                    raise ValueError(f"event group missing in {path}")

                n_events = int(ak.num(feat_ak, axis=0))
                feat_chunks.append(feat_ak)
                label_chunks.append(label_ak)
                if self.pandora:
                    pand_chunks.append(pand_ak)
                if self.event_energy:
                    event_chunks.append(event_ak)
                event_sources.extend((path, local_i) for local_i in range(n_events))

            if not feat_chunks:
                continue

            feat_ak = feat_chunks[0] if len(feat_chunks) == 1 else ak.concatenate(feat_chunks, axis=0)
            label_ak = label_chunks[0] if len(label_chunks) == 1 else ak.concatenate(label_chunks, axis=0)
            pand_ak = None
            event_ak = None
            if self.pandora:
                pand_ak = pand_chunks[0] if len(pand_chunks) == 1 else ak.concatenate(pand_chunks, axis=0)
            if self.event_energy:
                event_ak = event_chunks[0] if len(event_chunks) == 1 else ak.concatenate(event_chunks, axis=0)

            n_events = int(ak.num(feat_ak, axis=0))
            nhits = ak.num(feat_ak, axis=1)
            event_indices = np.arange(n_events)
            if self.shuffle:
                rng.shuffle(event_indices)

            for chunk_i in event_indices.tolist():
                if int(nhits[chunk_i]) <= 0:
                    continue
                feat = np.copy(ak.to_numpy(feat_ak[chunk_i]))
                label = np.copy(ak.to_numpy(label_ak[chunk_i]))
                pand = None
                eventE = None
                jetE = None
                path, local_i = event_sources[chunk_i]
                if self.pandora:
                    pand = np.copy(ak.to_numpy(pand_ak[chunk_i]))
                if self.event_energy:
                    row = event_ak[chunk_i]
                    eventE = np.copy(ak.to_numpy(row[2]))
                    jetE = np.copy(ak.to_numpy(row[:2]))

                yield ILCDataset.featurize_from_numpy(
                    feat, label, pand, eventE, jetE, f"{path}:{local_i}", self
                )

    def _yield_with_buffer(self, stream, rng):
        if self.shuffle_buffer_size <= 1 or not self.shuffle:
            yield from stream
            return

        buffer = []
        for item in stream:
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer_size:
                idx = int(rng.integers(len(buffer)))
                yield buffer.pop(idx)

        while buffer:
            idx = int(rng.integers(len(buffer)))
            yield buffer.pop(idx)

    def __iter__(self):
        _, world_size, _, _, global_worker_id, num_global_workers = self._distributed_worker_info()
        rng = np.random.default_rng(self.seed + self.epoch)
        file_indices = self._shuffled_file_indices(rng)
        assignments, loads = self._assign_files(file_indices, num_global_workers)

        my_files = assignments[global_worker_id]
        target_events = None
        if self.pad_to_equal_workers and world_size > 1 and self.shuffle:
            target_events = max(loads) if loads else None

        def base_stream():
            yielded = 0
            while True:
                yielded_before = yielded
                replay_rng = np.random.default_rng(self.seed + self.epoch + 1000003 + yielded)
                for item in self._iter_file_events(my_files, replay_rng):
                    yield item
                    yielded += 1
                    if target_events is not None and yielded >= target_events:
                        return
                if target_events is None or yielded >= target_events or not my_files:
                    return
                if yielded == yielded_before:
                    raise RuntimeError(
                        "Streaming worker had assigned files but produced no valid events. "
                        "Check the input files or reduce the number of DataLoader workers."
                    )

        yield from self._yield_with_buffer(base_stream(), rng)
