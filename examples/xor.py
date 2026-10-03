"""Train discrete or guided TM on noisy XOR (parity of --nbits relevant bits, plus --noisybits random bits).

Training labels are flipped with probability --noise. Test set is clean, so 100% test accuracy means every xor pattern was learned.

Usage:
    python xor.py {discrete,guided} [options]

Examples:
    python xor.py discrete                                     # 2-bit XOR, 10% label noise
    python xor.py discrete --noisybits 10                    # classic noisy XOR: 2 relevant + 10 random bits
    python xor.py discrete --nbits 3 --n_clauses 8 --s 8 --T 4 --save xor3.ytm
    python xor.py guided --noise 0.0

Options:
    - python xor.py <discrete/guided> --help shows all options.
    - Common:    --nbits --noisybits --noise --N --repeat_test --epochs --n_clauses --s --n_states --weighted 0|1 --coalesced 0|1
                 --boost_tp_inc 0|1 --boost_tp_dec 0|1 --seed --device cpu:N|cuda:N --save PATH --print
    - discrete:  --T
    - guided:    --lr --lambda
"""

import argparse
import pickle

import numpy as np

from ytm.discrete import BinaryTM as DiscreteTM
from ytm.guided import BinaryTM as GuidedTM


def make_xor(N: int, nbits: int, noisybits: int, noise: float, rng: np.random.Generator):
    X = rng.integers(0, 2, size=(N, nbits + noisybits), dtype=np.uint8)
    Y = X[:, :nbits].sum(axis=1) % 2
    flip = rng.random(N) < noise
    Y[flip] = 1 - Y[flip]
    return X, Y.astype(np.uint8)


def make_test(nbits: int, noisybits: int, repeat: int, rng: np.random.Generator):
    # All 2**nbits relevant patterns, each repeated with random noisy bits, clean labels.
    patterns = ((np.arange(2**nbits)[:, None] >> np.arange(nbits)) & 1).astype(np.uint8)
    X_rel = np.repeat(patterns, repeat, axis=0)
    X = np.concatenate([X_rel, rng.integers(0, 2, size=(len(X_rel), noisybits), dtype=np.uint8)], axis=1)
    return X, (X_rel.sum(axis=1) % 2).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TM noisy XOR")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--N", type=int, default=1000, help="training set size")
    common.add_argument("--repeat_test", type=int, default=1, help="test set = all 2^nbits patterns, each repeated this many times")

    common.add_argument("--nbits", type=int, default=2, help="relevant bits, label is their parity")
    common.add_argument("--noisybits", type=int, default=0, help="extra random bits")
    common.add_argument("--noise", type=float, default=0.1, help="add noise to training labels")

    common.add_argument("--epochs", type=int, default=1)

    common.add_argument("--n_clauses", type=int, default=4)
    common.add_argument("--s", type=float, default=4.0)
    common.add_argument("--n_states", type=int, default=10)
    common.add_argument("--weighted", type=lambda v: bool(int(v)), default=True, metavar="0|1")
    common.add_argument("--coalesced", type=lambda v: bool(int(v)), default=False, metavar="0|1")
    common.add_argument("--boost_tp_inc", type=lambda v: bool(int(v)), default=False, metavar="0|1")
    common.add_argument("--boost_tp_dec", type=lambda v: bool(int(v)), default=False, metavar="0|1")
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")
    common.add_argument("--save", type=str, default=None, help="pickle the trained model to this path, e.g. xor.ytm")
    common.add_argument("--print", dest="print_clauses", action="store_true", help="print the learned clauses after training")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=2)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=1.0)
    guided.add_argument("--lambda", dest="lambda_", type=float, default=1.0)

    # Parse args
    args = parser.parse_args()

    # Args to dict
    params = vars(args)
    model_name, epochs, save = params.pop("model"), params.pop("epochs"), params.pop("save")
    nbits, noisybits, noise, N = params.pop("nbits"), params.pop("noisybits"), params.pop("noise"), params.pop("N")
    repeat_test, show_clauses = params.pop("repeat_test"), params.pop("print_clauses")

    # Data: noisy training labels, clean test labels
    rng = np.random.default_rng(params["seed"] + 50)
    X_train, Y_train = make_xor(N, nbits, noisybits, noise, rng)
    X_test, Y_test = make_test(nbits, noisybits, repeat_test, rng)

    # Model Initialization
    TM = DiscreteTM if model_name == "discrete" else GuidedTM
    tm = TM(**params, dim=(nbits + noisybits, 1, 1))

    # Training
    for epoch in range(epochs):
        tm.fit(X_train, Y_train)
        train_acc = (tm.predict(X_train)[0].reshape(-1) == Y_train).mean()
        test_acc = (tm.predict(X_test)[0].reshape(-1) == Y_test).mean()
        print(f"{model_name} epoch {epoch + 1}/{epochs}  train acc {train_acc * 100:.2f}%  test acc {test_acc * 100:.2f}%")

    if show_clauses:
        # Relevant bits are x0, x1, ..., the random ones n0, n1, ...
        tm.print_clauses([f"x{i}" for i in range(nbits)] + [f"n{i}" for i in range(noisybits)], sort="w0,len")

    if save is not None:
        with open(save, "wb") as f:
            pickle.dump(tm, f)
        print(f"saved to {save}")

        # Loading it back (always lands on cpu:1, use .to() to move it):
        # with open(save, "rb") as f:
        #     tm = pickle.load(f)
        # tm.to("cuda:0")
        # pred, _ = tm.predict(X_test)
