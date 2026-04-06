# ml-pfa

Training and inference for ILC calorimeter hits using **GravNet-style graph neural networks** and **Object Condensation**. Built on PyTorch and PyTorch Geometric; clustering, energy regression, and related tasks are run from `train.py`.

## Requirements

- **Python 3.9.18 (see comments in `requirements.txt`)
- For **CUDA**, install PyTorch / PyG builds that match your environment (CPU-only runs are possible; GPU is recommended for training)
- Main dependencies: `torch==2.2.2`, `torch-geometric`, `numpy`, `h5py`, `awkward`, `scipy`, `scikit-learn`, `matplotlib`, `plotly`, `tqdm`

## Setup

Create a virtual environment and install dependencies:

```bash
cd /data/suehara/singularity
singularity shell --nv --bind /data/suehara --bind /data/murata pfa.sif
scl enable devtoolset-8 bash
export PATH=/opt/pyenv/bin:/opt/pyenv/shims:$PATH
```

PyTorch Geometric extensions (`torch_scatter`, etc.) must be installed from **wheels matching your PyTorch version and CPU/GPU**. The header comments in `requirements.txt` include an example for Intel Mac / Linux CPU (torch 2.2).

Pass the repository root as the first argument. If `.venv` exists, its `python` is prepended to `PATH`.

## Data

Place **HDF5 (`.h5`)** files under a directory and pass that path with `train.py` `-i` / `-ii`. For large datasets you do not want to load entirely into memory, use `--ilc-sharded` for per-file loading (`dataset_ilc_sharded`).

## Training

The entry point is **`train.py`**. The input directory `-i` is required.

```bash
.venv/bin/python train.py -i /path/to/train_data --epochs 20 --batch-size 100 --cuda cuda:0
```

Common options:

| Use case | Flags |
|----------|--------|
| Separate validation directory | `-ii /path/to/val_data` with `--no-split` |
| Timing window cut | `--timing-cut` |
| Add θ, φ to inputs | `--thetaphi` |
| Track β loss, etc. | `--beta-track`, `--force-track-alpha` |
| Energy regression | `--energy-regression` (and optionally `--energy-regression-cluster`, etc.) |
| Distributed training | `--ddp` (use `--gpus 0,1,...` to pick GPUs) |
| Mixed precision | `--amp` (`--amp-dtype bf16` or `fp16`) |

For the full list of arguments, see **`cli.py`**, function `build_train_argument_parser()`.

## Other scripts

| Path | Role |
|------|------|
| `train_clustering.py` / `train_clustering_ddp.py` | Alternative clustering training entry points |
| `prediction.py` | `Prediction` and related inference helpers |
| `save_root*.py` | Utilities for ROOT output |
| `tools/` | `h5` ↔ `npz` conversion, awkward loading, etc. |
| `display/` | Event display and plotting |
| `shell/` | Example / template shell commands (paths may need editing for your machine) |

Experimental and older copies live under `backups/` and `modifications/`. Prefer the top-level `train.py` and `training/` for the main workflow.

## License

If no LICENSE file is present in the repository, confirm terms of use and redistribution with the maintainers.
