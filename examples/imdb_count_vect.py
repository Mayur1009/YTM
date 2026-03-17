import numpy as np
from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# from sklearn.feature_selection import SelectKBest, chi2
from ytm.tm import MultiClassTM


def load_dataset(num_words=10000, max_ngram=1, features=5000):
    (xtrain, ytrain), (xtest, ytest) = imdb.load_data(num_words=num_words)

    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}

    train_docs = [[id_to_word[word_id].lower() for word_id in doc] for doc in xtrain]
    test_docs = [[id_to_word[word_id].lower() for word_id in doc] for doc in xtest]

    vectorizer = CountVectorizer(tokenizer=lambda s: s, token_pattern=None, lowercase=False, ngram_range=(1, max_ngram), binary=True)

    X_train = vectorizer.fit_transform(train_docs).toarray()
    X_test = vectorizer.transform(test_docs).toarray()

    ytrain = np.array(ytrain, dtype=np.uint32)
    ytest = np.array(ytest, dtype=np.uint32)

    # Feature selection using chi-squared test
    # skb = SelectKBest(chi2, k=features)
    # skb.fit(X_train, ytrain)
    #
    # X_train = skb.transform(X_train).toarray().astype(np.uint32)  # pyright: ignore[reportAttributeAccessIssue]
    # X_test = skb.transform(X_test).toarray().astype(np.uint32)  # pyright: ignore[reportAttributeAccessIssue]

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
    separator = "+" + "+".join(["-" * (col_width+2)] * (len(metrics) + 1)) + "+"
    print(separator)
    for name, data in [("Train", train_met), ("Test", test_met)]:
        row = f"| {name:>{col_width}} |"
        for metric in metrics:
            row += f" {data[metric]:>{col_width}.4f} |"
        print(row)
    print(separator)



def train(tm: MultiClassTM, xtrain, ytrain, xtest, ytest, epochs):
    encoded_xtrain = tm.encode(xtrain)
    encoded_xtest = tm.encode(xtest)

    for epoch in range(epochs):
        tm.fit(encoded_xtrain, ytrain, is_X_encoded=True)

        train_preds, cs_train = tm.predict(encoded_xtrain, is_X_encoded=True)
        test_preds, cs_test = tm.predict(encoded_xtest, is_X_encoded=True)

        train_metrics = metrics(ytrain, train_preds, cs_score(cs_train, tm.T))
        test_metrics = metrics(ytest, test_preds, cs_score(cs_test, tm.T))

        print_metrics(epoch + 1, train_metrics, test_metrics)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset(num_words=5000, max_ngram=1, features=5000)
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

