# ml-pfa

Training and inference for ILC calorimeter hits using **GravNet-style graph neural networks** and **Object Condensation**. Built on PyTorch and PyTorch Geometric; clustering, energy regression, and related tasks are run from `train.py`.

## Requirements

- **Python 3.9.18 (see comments in `requirements.txt`)
- For **CUDA**, install PyTorch / PyG builds that match your environment (CPU-only runs are possible; GPU is recommended for training)
- Main dependencies: `torch==2.2.2`, `torch-geometric`, `numpy`, `h5py`, `awkward`, `scipy`, `scikit-learn`, `matplotlib`, `plotly`, `tqdm`

## 🗂️ Core Components

This repository contains scripts for the following purposes. 

> **⚠️ IMPORTANT:** The evaluation scripts (`test.py` and `test_cross_attn.py`) strictly require the **[ROOT](https://root.cern/)** framework (Data Analysis Framework for high-energy physics) to run.

| Component | Main Script(s) | Description |
| :--- | :--- | :--- |
| 🏋️ **GravNet Training** | `train.py` | Scripts for training the GravNet model. |
| 🧪 **GravNet Evaluation** | `save_root.py`<br>`macro/*.cc` | Performs GravNet model inference and generates `.root` files. <br>**⚠️ Requires ROOT.** |
| 📊 **Event Display** | `event_display.py` | Data visualization and event display tools. |
| 🧠 **Cross Attention Training** | 🚧 ~~`train_cross_attn.py`~~ | Training code for the model using Cross Attention. |
| 🎯 **Cross Attention Evaluation** | 🚧 ~~`test_cross_attn.py`~~ | Performs Cross Attention model inference and generates `.root` files. <br>**⚠️ Requires ROOT.** |

## ⚙️ Testing & Analysis Workflow

The evaluation process is strictly divided into two steps:

**1. Inference & ROOT File Generation**

These scripts handle the testing phase by running the model inference and are solely responsible for saving the outputs as `.root` files:
* **`save_root.py`** : For the GravNet model.
* **🚧 ~~`test_cross_attn.py`~~** : For the Cross Attention model.

**2. Subsequent Analysis (`macro/`)**

The actual physical analysis of the generated `.root` files is performed using the macros located in the `macro/` directory.
> 🚧 **Note:** The `macro/` directory is currently **under active development (WIP)**.

## Setup (Proposal)
Environment Requirements
The setup requirements vary depending on the GPU cluster you are using, primarily due to the compilation environment needed for GravNet.

1. On ```bepp-gpu```

  You can run the training directly on the host system. No additional container setup is required.

2. On ```iutgpu```

  A Singularity container is required to provide the necessary environment for GravNet compilation. Follow these steps to enter the environment:

```Bash
# Navigate to the singularity directory
cd /data/suehara/singularity

# Launch the Singularity shell with necessary bindings
singularity shell --nv --bind /data/suehara --bind /data/murata pfa.sif

# Inside the container, enable the development toolset and set paths
scl enable devtoolset-8 bash
export PATH=/opt/pyenv/bin:/opt/pyenv/shims:$PATH
```
[!NOTE]
Exceptions for GravNet-less Training > If you are running training sessions that do not involve GravNet (e.g., downstream tasks using Cross-Attention), you can execute the code directly on the host even on iutgpu, as the specific compilation environment is not required.

## Data

The dataset is located at:
```/data/suehara/mldata/pfa/murata/data```

- `tc`  : Simulation samples with a timing cut applied (late-time hits have been removed).
- `raw` : Raw simulation data without any cuts applied.

The dataset contains two types of simulation samples:

Place **HDF5 (`.h5`)** files under a directory and pass that path with `train.py` `-i` / `-ii`. For large datasets you do not want to load entirely into memory, use `--ilc-sharded` for per-file loading (`dataset_ilc_sharded`).
```
python train.py -i /path/to/your/training_samples -ii /path/to/your/validation_samples --ilc-sharded
```

## Training

The entry point is **`train.py`**. The input directory `-i` is required.

```bash
python train.py -i /data/suehara/mldata/pfa/murata/data/tc/eventCut/tc_nnqq_brems/train -ii /data/suehara/mldata/pfa/murata/data/tc/eventCut/tc_nnqq_brems/validation --no-split --thetaphi --cuda cuda:${ncuda} --epochs 500 --beta-track --force-track-alpha --batch-size 20 --output-dimension ${outputD} --ckptdir checkpoint/energy_regression/ckpts_gravnet_new02_${DATE}_outputD${outputD} --energy-regression --energy-regression-cluster --LE-track ${alphbeta} --LE-cluster ${sumdis} --momentum --momentum-amp --qmin 0.2 --learning-rate 1.25e-4 --lr-policy cosineReduce --clip-value 10 --ddp > log/energy_regression/tc_nnqq_brems_timingcut_forcealpha_thetaphi_outputD${outputD}_${DATE}_${alphbeta}.log
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
| `save_root*.py` | Utilities for ROOT output (see `docs/save_root.md`) |
| `tools/` | `h5` ↔ `npz` conversion, awkward loading, etc. |
| `display/` | Event display and plotting |
| `shell/` | Example / template shell commands (paths may need editing for your machine) |

Experimental and older copies live under `backups/` and `modifications/`. Prefer the top-level `train.py` and `training/` for the main workflow.

## License

If no LICENSE file is present in the repository, confirm terms of use and redistribution with the maintainers.
