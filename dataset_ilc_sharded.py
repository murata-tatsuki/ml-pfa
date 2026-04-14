"""
ILCDataset のメモリ節約版: HDF5 をファイル単位で読み込み、連結しない。
既存の ILCDataset は変更せず、--ilc-sharded で train.py から切り替え可能。

前提: 単一ファイル全体はメモリに載るが、全ファイルを一度に連結すると載らない場合向け。
"""
from __future__ import annotations

from collections import OrderedDict

import awkward as ak
import numpy as np
from torch_geometric.data import Dataset

import glob
import tools.load_awkward as la

from dataset import ILCDataset


class ILCDatasetSharded(Dataset):
    """
    各 .h5 を順にスキャンして (path, local_event_index) のリストを構築し、
    get(i) で該当ファイルだけ load_awkward2 して1イベントを取り出す。
    直近 file_cache_size 本まで LRU キャッシュする。
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
        file_cache_size=2,
        _entries=None,
        timing=False,
    ):
        super().__init__(path)
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
        self.file_cache_size = max(1, int(file_cache_size))
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self.timing = timing

        if _entries is not None:
            self.entries = list(_entries)
        else:
            if path.endswith(".h5"):
                filenames = [path]
            else:
                filenames = list(sorted(glob.iglob(path + "/*.h5")))
            print(f"ILCDatasetSharded: {path=} ({len(filenames)} files)")
            entries = []
            for fp in filenames:
                bundle = la.load_awkward2(fp)
                feat, label = bundle[0], bundle[1]
                if timingCut:
                    n_ev = int(ak.num(feat, axis=0))
                    feat, label = ILCDataset.timingCut(
                        feat, label, cutoff_time=14, nstart=0, nend=n_ev
                    )
                nh = ak.num(feat, axis=1)
                nloc = int(ak.num(feat, axis=0))
                for local_i in range(nloc):
                    if int(nh[local_i]) > 0:
                        entries.append((fp, local_i))
                del bundle, feat, label

            n_tot = len(entries)
            nend_eff = n_tot if nend < 0 or n_tot < nend else nend
            self.entries = entries[nstart : nstart + nend_eff]

        self._sharded_kw = dict(
            flip=self.flip,
            reduce_noise=self.reduce_noise,
            para_tanh=para_tanh,
            timingCut=False,
            thetaphi=self.thetaphi,
            test_mode=self.test_mode,
            nstart=0,
            nend=-1,
            pandora=self.pandora,
            momentum=self.momentum,
            momentumAmp=self.momentumAmp,
            mctpe=self.mctpe,
            event_energy=self.event_energy,
            file_cache_size=self.file_cache_size,
            timing=self.timing,
        )

    def shaper_tanh(self, x, a=1.0, b=1.0, c=0.0, d=0.0):
        return a * np.tanh(b * (x - c)) + d

    def _get_bundle(self, path: str) -> tuple:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        self._cache[path] = la.load_awkward2(path)
        self._cache.move_to_end(path)
        while len(self._cache) > self.file_cache_size:
            self._cache.popitem(last=False)
        return self._cache[path]

    def get(self, idx):
        path, local_i = self.entries[idx]
        bundle = self._get_bundle(path)
        feat_ak, label_ak = bundle[0], bundle[1]
        feat_t = ak.to_numpy(feat_ak[local_i])
        label_t = ak.to_numpy(label_ak[local_i])
        feat = np.copy(feat_t)
        label = np.copy(label_t)
        pand = None
        eventE = None
        jetE = None
        if self.pandora:
            pand_ak = bundle[4]
            if pand_ak is None:
                raise ValueError(f"Pandora requested but missing in {path}")
            pand = np.copy(ak.to_numpy(pand_ak[local_i]))
        if self.event_energy:
            event_ak = bundle[6]
            if event_ak is None:
                raise ValueError(f"event group missing in {path}")
            row = event_ak[local_i]
            eventE = np.copy(ak.to_numpy(row[2]))
            jetE = np.copy(ak.to_numpy(row[:2]))

        return ILCDataset.featurize_from_numpy(
            feat, label, pand, eventE, jetE, idx, self
        )

    def __len__(self):
        return len(self.entries)

    def len(self):
        return len(self.entries)

    def split(self, fraction):
        n = int(fraction * len(self))
        print("ILCDatasetSharded split at index = ", n)
        left = ILCDatasetSharded(self.root, _entries=self.entries[:n], **self._sharded_kw)
        right = ILCDatasetSharded(self.root, _entries=self.entries[n:], **self._sharded_kw)
        return left, right
