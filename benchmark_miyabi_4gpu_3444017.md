# Miyabi 4 GPU benchmark — job 3444017

Updated: 2026-09-29T09:49:25+09:00
PBS state: F

Data: 1,960,000 training events; 20,000 validation events.
4 nodes × 1 GPU; 32 events/GPU/batch; FP32; full beta and energy losses.
Loader sweep (chunk/buffer/workers): 32/256/2, 8/256/2, 8/256/4, 8/128/4.
Each configuration/phase: 2 warm-up batches + 10 measured batches per rank.
Each configuration resets model, optimizer and Torch RNG; fixed learning rate for timing.
No training checkpoints are saved by this benchmark.

estimated.start_time: Tue Sep 29 09:45:50 2026
stime: Tue Sep 29 09:45:51 2026
resources_used.walltime: 00:03:26
Exit_status: 0

Completed configurations: 4/4.

| Chunk/buffer/workers | Train s/batch (slowest) | Warm-up train s (slowest) | Host tree RSS GiB (max) | Estimated epoch h | 500 epochs days |
|---|---:|---:|---:|---:|---:|
| 32/256/2 | 0.777 | 52.3 | 54.54 | 3.339 | 69.56 |
| 8/256/2 | 0.726 | 15.8 | 16.81 | 3.122 | 65.04 |
| 8/256/4 | 0.729 | 16.2 | 31.28 | 3.160 | 65.83 |
| 8/128/4 | 0.731 | 15.8 | 30.88 | 3.166 | 65.95 |

Fastest measured candidate (provisional): **8/256/2**.

Host tree RSS sums parent/worker RSS and may double-count shared pages. Warm-up includes loading and two batches.

This is a small-sample projection, not a completed-epoch measurement. It excludes queue wait, initial loading, checkpoint I/O and later data-chunk loading. Later candidates benefit from warmed filesystem caches. Event order/mixing changes with loader settings; a longer chunk-boundary test and training-quality check are needed before calling any setting optimal. The projection uses the slowest rank per phase and its full batch count.

PBS output: `/home/w25002/ml-pfa/pfa-bench4.o3444017`
Rank logs: `/work/gw25/w25002/checkpoint/energy_4gpu_3444017.opbs/ranks`
