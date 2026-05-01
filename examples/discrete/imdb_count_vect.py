import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ytm.discrete import MultiClassTM


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


def metrics(ytrue, ypred, yscore):
    accuracy = accuracy_score(ytrue, ypred)
    precision, recall, f1, _ = precision_recall_fscore_support(ytrue, ypred, average="binary")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def cs_score(cs, T):
    return (cs + T) / (2 * T)


def print_metrics(epoch, train_met: dict, test_met: dict):
    """Prints the training and testing metrics in a formatted table."""
    col_width = 9
    metrics = train_met.keys()
    header = f"| {'Epoch = ' + str(epoch):^{col_width}} |"
    for metric in metrics:
        header += f" {metric:>{col_width}} |"
    print(header)
    separator = "+" + "+".join(["-" * (col_width + 2)] * (len(metrics) + 1)) + "+"
    print(separator)
    for name, data in [("Train", train_met), ("Test", test_met)]:
        row = f"| {name:>{col_width}} |"
        for metric in metrics:
            row += f" {data[metric]:>{col_width}.4f} |"
        print(row)
    print(separator)


def train(tm: MultiClassTM, xtrain, ytrain, xtest, ytest, epochs):
    for epoch in range(epochs):
        tm.fit(xtrain, ytrain)

        train_preds, cs_train = tm.predict(xtrain)
        test_preds, cs_test = tm.predict(xtest)

        train_metrics = metrics(ytrain, train_preds, cs_score(cs_train, tm.args.T_max))
        test_metrics = metrics(ytest, test_preds, cs_score(cs_test, tm.args.T_max))

        print_metrics(epoch + 1, train_metrics, test_metrics)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = imdb_dataset(max_ngram=1)
    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)
    print("Number of training samples:", len(y_train))
    print("Number of test samples:", len(y_test))
    num_classes = np.max(y_train) + 1
    print("Number of classes:", num_classes)

    tm = MultiClassTM(
        n_clauses=1000,
        T=10000,
        s=1.0,
        dim=(X_train.shape[1], 1, 1),
        n_classes=num_classes,
        seed=10,
        device="cpu",
        n_threads=8,
    )

    train(tm, X_train, y_train, X_test, y_test, epochs=30)
