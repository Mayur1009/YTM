import argparse

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ytm.discrete import MultiClassTM
from ytm.utils import print_table


def imdb_dataset(max_ngram=1):
    imdb = load_dataset("stanfordnlp/imdb")
    ytrain, ytest = map(np.array, (imdb["train"]["label"], imdb["test"]["label"]))
    train_docs, test_docs = imdb["train"]["text"], imdb["test"]["text"]

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, max_ngram),
        max_df=0.95,
        min_df=2,
        binary=True,
    )
    X_train = vectorizer.fit_transform(train_docs).toarray()
    X_test = vectorizer.transform(test_docs).toarray()
    return X_train, ytrain, X_test, ytest


def metrics(ytrue, ypred):
    accuracy = accuracy_score(ytrue, ypred)
    precision, recall, f1, _ = precision_recall_fscore_support(ytrue, ypred, average="binary")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def cs_score(cs, T):
    return (cs + T) / (2 * T)


def train(tm: MultiClassTM, X_train, y_train, X_test, y_test, epochs=1):
    for epoch in range(epochs):
        tm.fit(X_train, y_train)

        train_preds, _ = tm.predict(X_train)
        test_preds, _ = tm.predict(X_test)

        print_table(
            f"Epoch {epoch + 1}/{epochs}",
            {
                "Train": {k: f"{v:.4f}" for k, v in metrics(y_train, train_preds).items()},
                "Test":  {k: f"{v:.4f}" for k, v in metrics(y_test, test_preds).items()},
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clauses", type=int, default=1000)
    parser.add_argument("--T", type=int, default=10000)
    parser.add_argument("--s", type=float, default=1.0)
    parser.add_argument("--max_ngram", type=int, default=1)
    parser.add_argument("--seed", type=lambda x: None if x == "None" else int(x), default=10)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = imdb_dataset(max_ngram=args.max_ngram)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    num_classes = int(np.max(y_train)) + 1

    tm = MultiClassTM(
        n_clauses=args.n_clauses,
        T=args.T,
        s=args.s,
        dim=(X_train.shape[1], 1, 1),
        n_classes=num_classes,
        seed=args.seed,
        device=args.device,
        n_threads=args.n_threads,
    )

    train(tm, X_train, y_train, X_test, y_test, epochs=args.epochs)
