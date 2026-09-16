#!/usr/bin/env python3
"""Run the existing beta/distance ROOT scan with one GNN inference pass.

This is a thin wrapper around ``save_root_reco_w_Cedric.py``. The existing
ROOT-writing and scan code is reused, while the yielder is replaced by a
caching variant. Predictions are materialized once for the input dataset and
reused for every (tbeta, td) point. Scan outputs are arranged as
``tbetaXXXtdXXX/<input HDF5 stem>.root``. The contents of each ROOT file,
including the ``t``, ``reco``, ``prediction``, ``event`` and ``jet`` trees,
remain delegated to the Cedric-compatible writer. In particular, the
hit-level ``prediction`` tree includes the reconstructed ``cluster`` ID and
correct ``hitid``/``truthid`` branch bindings.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterator


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import save_root_reco_w_Cedric as _root_writer  # noqa: E402
from test_yielder_edit import TestYielder as _BaseTestYielder  # noqa: E402


SCAN_ROOT_NAME = re.compile(r"^(tbeta\d{3}td\d{3})\.root$")
SCAN_ROOT_PATH_IN_MESSAGE = re.compile(
    r"(?P<path>(?:[^\s]+/)?tbeta\d{3}td\d{3}\.root)"
)


class SingleInferenceTestYielder(_BaseTestYielder):
    """Cache ``iter_pred`` output and replay it for later scan points."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prediction_cache = None
        self._prediction_cache_key = None
        self._reuse_announced = False

    @staticmethod
    def _cache_key(
        nmax,
        energyRegression,
        energyRegressionCluster,
        energyRegressionWeight,
    ):
        return (
            nmax,
            bool(energyRegression),
            bool(energyRegressionCluster),
            bool(energyRegressionWeight),
        )

    def iter_pred(
        self,
        nmax=None,
        energyRegression=False,
        energyRegressionCluster=False,
        energyRegressionWeight=False,
    ) -> Iterator:
        key = self._cache_key(
            nmax,
            energyRegression,
            energyRegressionCluster,
            energyRegressionWeight,
        )

        if self._prediction_cache is None:
            print(
                "[scan-cache] Running the GNN inference pass once and "
                "caching this HDF5 file in memory..."
            )
            self._prediction_cache = tuple(
                super().iter_pred(
                    nmax=nmax,
                    energyRegression=energyRegression,
                    energyRegressionCluster=energyRegressionCluster,
                    energyRegressionWeight=energyRegressionWeight,
                )
            )
            self._prediction_cache_key = key
            print(
                f"[scan-cache] Cached {len(self._prediction_cache)} events. "
                "Later beta/distance points will not run the GNN again."
            )
        elif key != self._prediction_cache_key:
            raise RuntimeError(
                "The prediction cache was requested with different inference "
                "options. Create a new yielder/process for that configuration."
            )
        elif not self._reuse_announced:
            print(
                "[scan-cache] Reusing cached predictions for the remaining "
                "scan points."
            )
            self._reuse_announced = True

        yield from self._prediction_cache


def install_scan_output_router(datapath: str) -> None:
    """Route each scan ROOT file and its log message to the actual path."""
    original_tfile = _root_writer.TFile
    original_print = print
    input_name = Path(datapath).stem
    root_name = f"{input_name}.root"

    def routed_path(filename) -> Path:
        output_path = Path(str(filename))
        match = SCAN_ROOT_NAME.fullmatch(output_path.name)
        if match is not None:
            output_path = output_path.parent / match.group(1) / root_name
        return output_path

    def routed_tfile(filename, *args, **kwargs):
        output_path = routed_path(filename)
        if output_path != Path(str(filename)):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[scan-output] Writing {output_path}")
        return original_tfile(str(output_path), *args, **kwargs)

    def routed_writer_print(*values, **kwargs):
        def replace_path(match):
            return str(routed_path(match.group("path")))

        routed_values = tuple(
            SCAN_ROOT_PATH_IN_MESSAGE.sub(replace_path, value)
            if isinstance(value, str)
            else value
            for value in values
        )
        original_print(*routed_values, **kwargs)

    _root_writer.TFile = routed_tfile
    # The delegated writer prints its pre-routing filename after file.Write().
    # Route strings printed by that module as well, so the log always names the
    # file that was actually opened and written.
    _root_writer.print = routed_writer_print


def main():
    # save_root_reco_w_Cedric.save_root resolves TestYielder from its module globals.
    # Replacing that reference changes only this process and leaves the existing
    # source files untouched.
    _root_writer.TestYielder = SingleInferenceTestYielder
    if "--beta-d-scan" in sys.argv and len(sys.argv) >= 2:
        install_scan_output_router(sys.argv[1])
    _root_writer.main()


if __name__ == "__main__":
    main()
