"""Train discrete or guided TM on GoEmotions (28 emotion labels, multilabel).

Text -> tweet tokenizer (case kept, repeated chars collapsed), URLs replaced by a placeholder,
English stopwords removed, lemmatized, binary bag-of-words (unigram + bigram), top --max_features terms by frequency.

Usage:
    python go_emotions.py {discrete,guided} [options]

Examples:
    python go_emotions.py discrete --device cuda:0
    python go_emotions.py guided --device cuda:0                          # ASL, lr 0.05, lambda 6
    python go_emotions.py guided --act_loss bce --lr 0.5 --lambda 4 --device cuda:0

Options:
    - python go_emotions.py <discrete/guided> --help shows all options.
    - Common:    --epochs --clause_drop_p --max_features --n_clauses --s --max_includes --seed --device cpu:N|cuda:N
    - discrete:  --T --q
    - guided:    --lr --lambda --act_loss asl|bce
"""

import argparse
import re

import nltk
import numpy as np
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ytm.discrete import MultiOutputTM as DiscreteTM
from ytm.guided import MultiOutputTM as GuidedTM
from ytm.guided.backends.act_loss import ASL, SigmoidBCE
from ytm.utils import Timer, print_table

ACT_LOSSES = {
    "asl": lambda: ASL(gamma_pos=0.0, gamma_neg=4.0, clip=0.01),
    "bce": lambda: SigmoidBCE(),
}
N_LABELS = 28


def load_go_emotions(max_features: int):
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")

    # Label index lists -> multi-hot (samples, 28).
    Y_train, Y_test = (np.zeros((len(ds[split]), N_LABELS), dtype=np.uint8) for split in ("train", "test"))
    for split, Y in (("train", Y_train), ("test", Y_test)):
        for i, labels in enumerate(ds[split]["labels"]):
            Y[i, labels] = 1

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    stop = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True)
    url = re.compile(r"https?://\S+|www\.\S+")

    def analyze(doc):
        tokens = [lemmatizer.lemmatize(t) for t in tokenizer.tokenize(url.sub("URL", doc)) if t.lower() not in stop]
        return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]

    vectorizer = CountVectorizer(analyzer=analyze, binary=True, max_features=max_features)
    X_train = vectorizer.fit_transform(ds["train"]["text"])
    X_test = vectorizer.transform(ds["test"]["text"])
    return X_train.toarray().astype(np.uint8), Y_train, X_test.toarray().astype(np.uint8), Y_test


def multilabel_metrics(y, pred, scores):
    # Scores rank the samples per label, so raw votes (discrete) and probabilities (guided) both work for the AUCs.
    return {
        "Macro F1": f"{f1_score(y, pred, average='macro', zero_division=0) * 100:.2f}%",
        "Weighted F1": f"{f1_score(y, pred, average='weighted', zero_division=0) * 100:.2f}%",
        "ROC AUC": f"{roc_auc_score(y, scores, average='macro') * 100:.2f}%",
        "PR AUC": f"{average_precision_score(y, scores, average='macro') * 100:.2f}%",
    }


def train_model(tm, xtrain, ytrain, xtest, ytest, epochs: int, clause_drop_p: float, model_name: str):
    for epoch in range(epochs):
        with (fit_timer := Timer()):
            loss = tm.fit(xtrain, ytrain, clause_drop_p=clause_drop_p)

        with (test_timer := Timer()):
            test_pred, test_scores = tm.predict(xtest)

        with (train_timer := Timer()):
            train_pred, train_scores = tm.predict(xtrain)

        train_log = {
            **multilabel_metrics(ytrain, train_pred, tm.to_prob(train_scores)),
            "Eval Time": f"{train_timer.elapsed:.2f}s",
            "Fit Time": f"{fit_timer.elapsed:.2f}s",
        }
        test_log = {**multilabel_metrics(ytest, test_pred, tm.to_prob(test_scores)), "Eval Time": f"{test_timer.elapsed:.2f}s"}

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
    parser = argparse.ArgumentParser(description="TM GoEmotions dataset")
    # Subcommand to select discrete or guided
    models = parser.add_subparsers(dest="model", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--epochs", type=int, default=10)
    common.add_argument("--clause_drop_p", type=float, default=0.0)
    common.add_argument("--max_features", type=int, default=10000, help="most frequent terms kept")
    common.add_argument("--n_clauses", type=int, default=22400)
    common.add_argument("--s", type=float, default=2.0)
    common.add_argument("--max_includes", type=int, default=None, help="max literals per clause (default: no limit)")
    common.add_argument("--seed", type=int, default=10)
    common.add_argument("--device", type=str, default="cpu:1", help="cpu:N_THREADS or cuda:GPU_ID")

    # Discrete args
    discrete = models.add_parser("discrete", parents=[common])
    discrete.add_argument("--T", type=float, default=25000)
    discrete.add_argument("--q", type=float, default=1.5)

    # Guided args
    guided = models.add_parser("guided", parents=[common])
    guided.add_argument("--lr", type=float, default=0.05)
    guided.add_argument("--lambda", dest="lambda_", type=float, default=6.0)
    guided.add_argument("--act_loss", choices=list(ACT_LOSSES), default="asl")

    # Parse args
    args = parser.parse_args()

    # Args to dict
    params = vars(args)
    model_name, epochs, clause_drop_p = params.pop("model"), params.pop("epochs"), params.pop("clause_drop_p")
    max_features = params.pop("max_features")
    if "act_loss" in params:
        params["act_loss"] = ACT_LOSSES[params["act_loss"]]()

    # Model Initialization
    TM = DiscreteTM if model_name == "discrete" else GuidedTM
    tm = TM(
        **params,
        dim=(max_features, 1, 1),
        n_classes=N_LABELS,
    )

    # Training
    train_model(tm, *load_go_emotions(max_features), epochs=epochs, clause_drop_p=clause_drop_p, model_name=model_name)
