# Bench

## Results

All runs: 10 epochs, weighted clauses.
Predict: mean sec over all epochs. 
Acc/metric: only last epoch.

CPU Runs: Apple M3, 16 GB RAM
CUDA Runs: Nvidia H100, 80GB VRAM

### Non-coalesced
#### CPU

(YTM / TMU)

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak RAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 1.32s / 2.21s | 0.77s / 1.92s | 0.05s / 0.58s | 376 / 740 | 92.35% / 92.12% |
| MNIST conv | 2.91s / 9.45s | 3.06s / 4.88s | 1.75s / 17.15s | 381 / 1509 | 96.50% / 96.56% |
| FMNIST conv | 16.78s / 83.06s | 16.59s / 72.09s | 2.05s / 64.95s | 717 / 3691 | 87.15% / 86.24% |
| IMDb | 285.02s / 86.91min* | 94.89s / - | 16.88s / 4.14s* | 2003 / 3680* | 86.24% / - |

\* Ran only for 1 epoch.

#### CUDA

(YTM / TMU)

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak VRAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 3.11s / 20.85s | 3.00s / 16.25s | 0.01s / 3.70s | 606 / 4702 | 92.44% / 92.13% |
| MNIST conv | 3.45s / 61.68s | 2.99s / 38.71s | 0.02s / 19.87s | 606 / 4702 | 96.51% / 96.46% |
| FMNIST conv | 3.99s / 107.32s | 3.57s / 71.19s | 0.05s / 37.48s | 638 / 4702 | 87.05% / 86.54% |
| IMDb | 10.51s / 108.71s | 4.36s / 49.12s | 0.51s / 20.39s | 1825 / 4702 | 86.22% / 86.18% |
| CIFAR-10 conv | 5.38s / 147.47s | 5.21s / 106.47s | 0.45s / 43.88s | 789 / 4702 | 59.08% / 59.72% |
| AG News | 7.89s / 83.92s | 6.86s / 66.55s | 0.04s / 2.32s | 893 / 4702 | 89.67% / 89.36% |


### Coalesced
#### CPU

(YTM / TMU)

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak RAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 10.88s / 7.87s | 4.78s / 5.42s | 0.17s / 0.21s | 390 / 765 | 94.80% / 95.41% |
| MNIST conv | 14.23s / 28.04s | 13.72s / 19.55s | 1.70s / 5.77s | 381 / 1531 | 97.61% / 97.77% |
| FMNIST conv | 32.83s / 448.25s | 26.97s / 353.52s | 1.28s / 82.87s | 720 / 3231 | 87.78% / 87.65% |


#### CUDA

(YTM / PySparseCoalescedTsetlinMachineCUDA)

| Experiment | Fit (epoch 1) | Fit (avg epoch 2-10) | Predict | Peak VRAM (MB) | Acc |
|---|---|---|---|---|---|
| MNIST flat | 4.67s / 25.40s | 4.49s / 14.44s | 0.01s / 1.77s | 606 / 610 | 95.07% / 95.06% |
| MNIST conv | 4.12s / 31.06s | 3.89s / 25.77s | 0.01s / 2.21s | 606 / 715 | 97.61% / 97.74% |
| FMNIST conv | 4.71s / 99.74s | 4.07s / 96.56s | 0.04s / 12.48s | 638 / 1495 | 87.69% / 87.52% |
| CIFAR-10 conv | 7.40s / 98.11s | 7.28s / 94.88s | 0.52s / 14.43s | 791 / 1632 | 62.23% / 62.29% |
| CelebA conv | 87.91s / >3.5hr* | 86.03s / - | 3.30s / - | 3924 / - | 55.59% / - |
| AG News | 12.18s / 68.26s | 10.55s / 51.38s | 0.04s / 0.73s | 893 / 1116 | 89.55% / 89.41% |
| GoEmotions | 75.12s / 435.38s | 63.27s / 346.05s | 0.50s / 2.06s | 2345 / 8187 | 34.85% / 35.47% |

\* Epoch 1 still running.

Note:
- CPU threads: YTM CPU uses 1 thread for MNIST and 8 threads (`cpu:8`) for FMNIST and IMDb.
- CIFAR-10, AG News, CelebA and GoEmotions are CUDA only.
- CelebA and GoEmotions are multi-label and coalesced only. Acc is that case is macro F1.
- PySparse has no seed argument, so results for it can vary.

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


## Commands

From the repo root. CPU runs use the `bench` environment, CUDA runs use `bench-cuda`.

- MNIST flat

  ```bash
  pixi run -e bench python -m bench.ytm.mnist --patch none --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --device cpu
  pixi run -e bench python -m bench.tmu.mnist --patch none --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench python -m bench.ytm.mnist --patch none --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --device cpu
  pixi run -e bench python -m bench.tmu.mnist --patch none --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench-cuda python -m bench.ytm.mnist --patch none --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.mnist --patch none --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --platform CUDA
  pixi run -e bench-cuda python -m bench.ytm.mnist --patch none --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.mnist --patch none --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --platform CUDA
  ```

- MNIST conv

  ```bash
  pixi run -e bench python -m bench.ytm.mnist --patch 10 --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --device cpu
  pixi run -e bench python -m bench.tmu.mnist --patch 10 --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench python -m bench.ytm.mnist --patch 10 --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --device cpu
  pixi run -e bench python -m bench.tmu.mnist --patch 10 --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench-cuda python -m bench.ytm.mnist --patch 10 --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.mnist --patch 10 --coalesced 1 --n_clauses 500 --T 1000 --s 10 --epochs 10 --seed 10 --platform CUDA
  pixi run -e bench-cuda python -m bench.ytm.mnist --patch 10 --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.mnist --patch 10 --coalesced 0 --n_clauses 50 --T 100 --s 10 --epochs 10 --seed 10 --platform CUDA
  ```

- FMNIST conv

  ```bash
  pixi run -e bench python -m bench.ytm.fmnist --patch 3 --coalesced 1 --n_clauses 6000 --T 10000 --s 10 --levels 8 --epochs 10 --seed 10 --device cpu:8
  pixi run -e bench python -m bench.tmu.fmnist --patch 3 --coalesced 1 --n_clauses 6000 --T 10000 --s 10 --levels 8 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench python -m bench.ytm.fmnist --patch 3 --coalesced 0 --n_clauses 600 --T 1000 --s 10 --levels 8 --epochs 10 --seed 10 --device cpu:8
  pixi run -e bench python -m bench.tmu.fmnist --patch 3 --coalesced 0 --n_clauses 600 --T 1000 --s 10 --levels 8 --epochs 10 --seed 10 --platform CPU
  pixi run -e bench-cuda python -m bench.ytm.fmnist --patch 3 --coalesced 1 --n_clauses 6000 --T 10000 --s 10 --levels 8 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.fmnist --patch 3 --coalesced 1 --n_clauses 6000 --T 10000 --s 10 --levels 8 --epochs 10 --seed 10 --platform CUDA
  pixi run -e bench-cuda python -m bench.ytm.fmnist --patch 3 --coalesced 0 --n_clauses 600 --T 1000 --s 10 --levels 8 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.fmnist --patch 3 --coalesced 0 --n_clauses 600 --T 1000 --s 10 --levels 8 --epochs 10 --seed 10 --platform CUDA
  ```

- IMDb

  ```bash
  pixi run -e bench python -m bench.ytm.imdb --max_features 12000 --coalesced 0 --n_clauses 5000 --T 10000 --s 1.1 --epochs 10 --seed 10 --device cpu:8
  pixi run -e bench python -m bench.tmu.imdb --max_features 12000 --coalesced 0 --n_clauses 5000 --T 10000 --s 1.1 --epochs 1 --seed 10 --platform CPU
  pixi run -e bench-cuda python -m bench.ytm.imdb --max_features 12000 --coalesced 0 --n_clauses 5000 --T 10000 --s 1.1 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.imdb --max_features 12000 --coalesced 0 --n_clauses 5000 --T 10000 --s 1.1 --epochs 10 --seed 10 --platform CUDA
  ```

- CIFAR-10 conv

  ```bash
  pixi run -e bench-cuda python -m bench.ytm.cifar10 --patch 5 --coalesced 1 --n_clauses 15000 --T 20000 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.cifar10 --patch 5 --coalesced 1 --n_clauses 15000 --T 20000 --s 10 --epochs 10 --seed 10 --platform CUDA
  pixi run -e bench-cuda python -m bench.ytm.cifar10 --patch 5 --coalesced 0 --n_clauses 1500 --T 2000 --s 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.cifar10 --patch 5 --coalesced 0 --n_clauses 1500 --T 2000 --s 10 --epochs 10 --seed 10 --platform CUDA
  ```

- AG News

  ```bash
  pixi run -e bench-cuda python -m bench.ytm.ag_news --max_features 2000 --coalesced 1 --n_clauses 8000 --T 40000 --s 2 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.ag_news --max_features 2000 --coalesced 1 --n_clauses 8000 --T 40000 --s 2 --epochs 10 --seed 10 --platform CUDA
  pixi run -e bench-cuda python -m bench.ytm.ag_news --max_features 2000 --coalesced 0 --n_clauses 2000 --T 10000 --s 2 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.ag_news --max_features 2000 --coalesced 0 --n_clauses 2000 --T 10000 --s 2 --epochs 10 --seed 10 --platform CUDA
  ```

- CelebA conv

  ```bash
  pixi run -e bench-cuda python -m bench.ytm.celeba --patch 3 --levels 8 --n_clauses 80000 --T 80000 --s 30 --q 10 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.celeba --patch 3 --levels 8 --n_clauses 80000 --T 80000 --s 30 --q 10 --epochs 10
  ```

- GoEmotions

  ```bash
  pixi run -e bench-cuda python -m bench.ytm.go_emotions --max_features 10000 --n_clauses 22400 --T 25000 --s 2 --q 1.5 --epochs 10 --seed 10 --device cuda
  pixi run -e bench-cuda python -m bench.tmu.go_emotions --max_features 10000 --n_clauses 22400 --T 25000 --s 2 --q 1.5 --epochs 10
  ```
