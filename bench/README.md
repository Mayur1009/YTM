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

| Experiment | Fit (s/epoch)  | Predict (s) | Peak RAM (MB)  | Acc (%) |
|---|---|---|---|---|
| MNIST flat | 0.83 / 1.95 | 0.05 / 0.58 | 376 / 740 | 92.35 / 92.12 |
| MNIST conv | 3.05 / 5.34 | 1.75 / 17.15 | 381 / 1509 | 96.50 / 96.56 |
| FMNIST conv | 16.61 / 73.19 | 2.05 / 64.95 | 717 / 3691 | 87.15 / 86.24 |
| IMDb | | | | |

### CUDA

| Experiment | Fit (s/epoch)  | Predict (s) | Peak VRAM (MB)  | Acc (%) |
|---|---|---|---|---|
| MNIST flat | | | | |
| MNIST conv | | | | |
| FMNIST conv | | | | |
| CIFAR-10 conv | | | | |
| IMDb | | | | |
| AG News | | | | |


## Coalesced
### CPU

| Experiment | Fit (s/epoch)  | Predict (s)  | Peak RAM (MB)  | Acc (%)|
|---|---|---|---|---|
| MNIST flat | 5.39 / 5.66 | 0.17 / 0.21 | 390 / 765 | 94.80 / 95.41 |
| MNIST conv | 13.77 / 20.40 | 1.70 / 5.77 | 381 / 1531 | 97.61 / 97.77 |
| FMNIST conv | 27.55 / 362.99 | 1.28 / 82.87 | 720 / 3231 | 87.78 / 87.65 |

### CUDA

| Experiment | Fit (s/epoch)  | Predict (s) | Peak VRAM (MB)  | Acc (%) |
|---|---|---|---|---|
| MNIST flat | | | | |
| MNIST conv | | | | |
| FMNIST conv | | | | |
| CIFAR-10 conv | | | | |
| CelebA conv | | | | |
| AG News | | | | |
| GoEmotions | | | | |


## Hyperparameters

| Experiment | Classes | Input | Patch | Coalesced clauses / T | Non-coalesced clauses per class / T | s | q |
|---|---|---|---|---|---|---|---|
| MNIST flat | 10 | >75 binary | - | 500 / 1000 | 50 / 100 | 10 | |
| MNIST conv | 10 | >75 binary | 10x10 | 500 / 1000 | 50 / 100 | 10 | |
| FMNIST conv | 10 | 8-level thermometer | 3x3 | 6000 / 10000 | 600 / 1000 | 10 | |
| CIFAR-10 conv | 10 | adaptive threshold RGB | 5x5 | 15000 / 20000 | 1500 / 2000 | 10 | |
| IMDb | 2 | 12000 chi2 1-2 grams | - | - | 5000 / 10000 | 1.1 | |
| AG News | 4 | 2000 chi2 lemmatized | - | 8000 / 40000 | | 2.0 | |
| CelebA conv | 40 | 64x64 RGB 8-level thermometer | 3x3 | 80000 / 80000 | - | 30 | 10 |
| GoEmotions | 28 | 10000 1-2 grams | - | 22400 / 25000 | - | 2.0 | 1.5 |

All runs: 10 epochs, seed 10, weighted clauses.
Times: mean over all epochs, accuracy at the last epoch.
