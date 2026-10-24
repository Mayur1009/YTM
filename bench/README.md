# Bench

## Running the scrips

From the repo root, in the `bench` pixi environment:

```bash
pixi run -e bench python -m bench.ytm.mnist --patch none                 # MNIST flat, coalesced
pixi run -e bench python -m bench.ytm.mnist --patch 10                   # MNIST conv 10x10, coalesced
pixi run -e bench python -m bench.ytm.mnist --patch none --coalesced 0   # non-coalesced
```

Each run writes `bench/results/<expname>/<libname>_<expname>_<date>.json`.

(YTM / TMU)

## Non-coalesced
### CPU

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak RAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 1.32s / 2.21s | 0.77s / 1.92s | 0.05s / 0.58s | 376 / 740 | 92.35% / 92.12% |
| MNIST conv | 2.91s / 9.45s | 3.06s / 4.88s | 1.75s / 17.15s | 381 / 1509 | 96.50% / 96.56% |
| FMNIST conv | 16.78s / 83.06s | 16.59s / 72.09s | 2.05s / 64.95s | 717 / 3691 | 87.15% / 86.24% |
| IMDb | 285.02s / 86.91min* | 94.89s / - | 16.88s / 4.14s* | 2003 / 3680* | 86.24% / - |

\* Ran only for 1 epoch.

### CUDA

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak VRAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 3.11s / 20.85s | 3.00s / 16.25s | 0.01s / 3.70s | 606 / 4702 | 92.44% / 92.13% |
| MNIST conv | 3.45s / 61.68s | 2.99s / 38.71s | 0.02s / 19.87s | 606 / 4702 | 96.51% / 96.46% |
| FMNIST conv | 3.99s / 107.32s | 3.57s / 71.19s | 0.05s / 37.48s | 638 / 4702 | 87.05% / 86.54% |
| CIFAR-10 conv | 5.38s / 147.47s | 5.21s / 106.47s | 0.45s / 43.88s | 789 / 4702 | 59.08% / 59.72% |
| IMDb | 10.51s / 108.71s | 4.36s / 49.12s | 0.51s / 20.39s | 1825 / 4702 | 86.22% / 86.18% |
| AG News | 7.89s / 83.92s | 6.86s / 66.55s | 0.04s / 2.32s | 893 / 4702 | 89.67% / 89.36% |


## Coalesced
### CPU

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak RAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 10.88s / 7.87s | 4.78s / 5.42s | 0.17s / 0.21s | 390 / 765 | 94.80% / 95.41% |
| MNIST conv | 14.23s / 28.04s | 13.72s / 19.55s | 1.70s / 5.77s | 381 / 1531 | 97.61% / 97.77% |
| FMNIST conv | 32.83s / 448.25s | 26.97s / 353.52s | 1.28s / 82.87s | 720 / 3231 | 87.78% / 87.65% |

### CUDA

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak VRAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 4.67s / 25.40s | 4.49s / 14.44s | 0.01s / 1.77s | 606 / 610 | 95.07% / 95.06% |
| MNIST conv | 4.12s / 31.06s | 3.89s / 25.77s | 0.01s / 2.21s | 606 / 715 | 97.61% / 97.74% |
| FMNIST conv | 4.71s / 99.74s | 4.07s / 96.56s | 0.04s / 12.48s | 638 / 1495 | 87.69% / 87.52% |
| CIFAR-10 conv | 7.40s / 98.11s | 7.28s / 94.88s | 0.52s / 14.43s | 791 / 1632 | 62.23% / 62.29% |
| CelebA conv | 87.91s / | 86.03s / | 3.30s / | 3924 / | 55.59% / |
| AG News | 12.18s / 68.26s | 10.55s / 51.38s | 0.04s / 0.73s | 893 / 1116 | 89.55% / 89.41% |
| GoEmotions | 75.12s / 435.38s | 63.27s / 346.05s | 0.50s / 2.06s | 2345 / 8187 | 34.85% / 35.47% |


## Hyperparameters

| Experiment | Classes | Input | Patch | Coalesced clauses / T | Non-coalesced clauses per class / T | s | q |
|---|---|---|---|---|---|---|---|
| MNIST flat | 10 | >75 binary | - | 500 / 1000 | 50 / 100 | 10 | |
| MNIST conv | 10 | >75 binary | 10x10 | 500 / 1000 | 50 / 100 | 10 | |
| FMNIST conv | 10 | 8-level thermometer | 3x3 | 6000 / 10000 | 600 / 1000 | 10 | |
| CIFAR-10 conv | 10 | adaptive threshold RGB | 5x5 | 15000 / 20000 | 1500 / 2000 | 10 | |
| IMDb | 2 | 12000 chi2 1-2 grams | - | - | 5000 / 10000 | 1.1 | |
| AG News | 4 | 2000 chi2 lemmatized | - | 8000 / 40000 | 2000 / 10000 | 2.0 | |
| CelebA conv | 40 | 64x64 RGB 8-level thermometer | 3x3 | 80000 / 80000 | - | 30 | 10 |
| GoEmotions | 28 | 10000 1-2 grams | - | 22400 / 25000 | - | 2.0 | 1.5 |

All runs: 10 epochs, seed 10, weighted clauses.
Times: mean over all epochs, accuracy at the last epoch.
