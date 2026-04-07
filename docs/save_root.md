# save_root.py

`save_root.py` is a utility script that runs inference and exports event-level and cluster-level outputs to a ROOT file.

## Purpose

- Load a trained checkpoint and input dataset
- Run prediction with the same model family used in training
- Save reconstructed information into ROOT TTrees for downstream analysis

## Basic usage

```bash
python save_root.py <datapath> <ckpt> <outfile> <nstart> <nend> <timingCut> <input_dim> <output_dim> [options]
```

Positional arguments:

- `datapath`: input dataset directory or file path
- `ckpt`: checkpoint path
- `outfile`: output ROOT file path
- `nstart`: first event index
- `nend`: last event index (`-1` for all)
- `timingCut`: `true`/`false`
- `input_dim`: model input dimension
- `output_dim`: model output dimension

## Example

```bash
python save_root.py \
  /data/suehara/mldata/pfa/murata/data/tc/eventCut/tc_nnqq_brems/validation \
  checkpoint/energy_regression/ckpts_xxx/best_epoch_state.pt \
  output/pred_validation.root \
  0 -1 true 5 3 \
  --energy-regression --energy-regression-cluster --momentum --device cuda:0
```

## Frequently used options

- `--pandora`: use PandoraPFA result
- `--event-total-energy`: enable event visible energy mode
- `--energy-regression`: use energy regression output
- `--energy-regression-cluster`: neutral-particle energy regression branch
- `--energy-regression-weight`: energy weight loss branch
- `--momentum` / `--momentum-amp`: include momentum features
- `--mctpe`: use MC truth momentum/energy for virtual hits
- `--energy-branch`: bypass energy branch in model
- `--beta-d-scan` with `--tbeta`, `--td`: beta and diameter scan
- `--device`: device selection (default: `cpu`)
- `--truth-clustering`: use MC truth clustering
- `--1tomany-clustering`: combine reco-clusters

## Notes

- Keep `input_dim`/`output_dim` consistent with the checkpoint configuration.
- For GPU inference, specify `--device cuda:<id>`.
- ROOT and project dependencies must be available in your runtime environment.
