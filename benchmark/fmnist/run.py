import numpy as np
import wandb
from keras.datasets import fashion_mnist

from benchmark.fmnist import standard, discrete

EPOCHS = 10
N_CLAUSES = 6000
T = 10000
S = 10
PATCH_DIM = (3, 3)
N_THREADS = 8


def load_data():
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = np.copy(X_train)
    X_test = np.copy(X_test)
    return X_train, Y_train, X_test, Y_test


def main():
    run = wandb.init(group="fmnist", settings=wandb.Settings(quiet=True))
    config = run.config

    approach = config.approach
    seed = config.seed
    device = config.device
    bins = config.bins

    X_train, Y_train, X_test, Y_test = load_data()

    kwargs = dict(
        X_train_raw=X_train,
        Y_train=Y_train,
        X_test_raw=X_test,
        Y_test=Y_test,
        bins=bins,
        seed=seed,
        device=device,
        epochs=EPOCHS,
        n_clauses=N_CLAUSES,
        T=T,
        s=S,
        patch_dim=PATCH_DIM,
        n_threads=N_THREADS,
    )

    if approach == "standard":
        standard.run(**kwargs)
    elif approach == "discrete":
        discrete.run(**kwargs)

    wandb.finish()


if __name__ == "__main__":
    main()
