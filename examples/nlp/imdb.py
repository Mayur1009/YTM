"""Train discrete or guided TM on IMDB movie reviews (binary sentiment).

Text -> binary bag-of-words (unigram + bigram, English stopwords removed), top --max_features terms by chi2.

Usage:
    python imdb.py {discrete,guided} [options]

Examples:
    python imdb.py discrete --device cuda:0
    python imdb.py guided --lr 0.005 --device cuda:0 --epochs 20
    python imdb.py discrete --max_features 2000 --n_clauses 1000 --T 2000

Options:
    - python imdb.py <discrete/guided> --help shows all options.
    - Common:    --epochs --clause_drop_p --max_features --n_clauses --s --max_includes --seed --device cpu:N|cuda:N
    - discrete:  --T
    - guided:    --lr --lambda
"""

import argparse

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from ytm.discrete import BinaryTM as DiscreteTM
from ytm.guided import BinaryTM as GuidedTM
from ytm.utils import Timer, print_table


def load_imdb(max_features: int):
    ds = load_dataset("stanfordnlp/imdb")
    Y_train, Y_test = (np.array(ds[split]["label"], dtype=np.uint8) for split in ("train", "test"))

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), binary=True)
    selector = SelectKBest(chi2, k=max_features)
    X_train = selector.fit_transform(vectorizer.fit_transform(ds["train"]["text"]), Y_train)
    X_test = selector.transform(vectorizer.transform(ds["test"]["text"]))
    return X_train.toarray().astype(np.uint8), Y_train, X_test.toarray().astype(np.uint8), Y_test


def train_model(tm, xtrain, ytrain, xtest, ytest, epochs: int, clause_drop_p: float, model_name: str):
    for epoch in range(epochs):
        with (fit_timer := Timer()):
            loss = tm.fit(xtrain, ytrain, clause_drop_p=clause_drop_p)

        with (test_timer := Timer()):
            test_pred, _ = tm.predict(xtest)

        with (train_timer := Timer()):
            train_pred, _ = tm.predict(xtrain)

        train_log = {
            "Acc": f"{(train_pred.reshape(-1) == ytrain).mean() * 100:.4f}%",
            "Eval Time": f"{train_timer.elapsed:.2f}s",
            "Fit Time": f"{fit_timer.elapsed:.2f}s",
        }
        test_log = {"Acc": f"{(test_pred.reshape(-1) == ytest).mean() * 100:.4f}%", "Eval Time": f"{test_timer.elapsed:.2f}s"}

        if loss is not None:  # Only loss-guided TM returns loss
            train_log["Loss"] = f"{loss:.4f}"

        print_table(
            f"{model_name} epoch {epoch + 1}/{epochs}",
            {
                "Train": train_log,
                "Test": test_log,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TM IMDB dataset")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--epochs", type=int, default=10)
    common.add_argument("--clause_drop_p", type=float, default=0.0)
    common.add_argument("--max_features", type=int, default=12000, help="terms kept by chi2")
    common.add_argument("--n_clauses", type=int, default=10000)
    common.add_argument("--s", type=float, default=1.1)
    common.add_argument("--max_includes", type=int, default=None, help="max literals per clause (default: no limit)")
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=10000)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=0.01)
    guided.add_argument("--lambda", dest="lambda_", type=float, default=1.0)

    # Parse args
    args = parser.parse_args()

    # Args to dict
    params = vars(args)
    model_name, epochs, clause_drop_p = params.pop("model"), params.pop("epochs"), params.pop("clause_drop_p")
    max_features = params.pop("max_features")

    # Model Initialization
    TM = DiscreteTM if model_name == "discrete" else GuidedTM
    tm = TM(
        **params,
        dim=(max_features, 1, 1),
    )

    # Training
    train_model(tm, *load_imdb(max_features), epochs=epochs, clause_drop_p=clause_drop_p, model_name=model_name)
