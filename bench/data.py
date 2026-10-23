import re
from io import BytesIO

import nltk
import numpy as np
import PIL.Image
from datasets import Image, load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from scipy.ndimage import gaussian_filter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2


def load_mnist():
    ds = load_dataset("ylecun/mnist")
    X_train, X_test = ((np.array(ds[split]["image"]) > 75).astype(np.uint8) for split in ("train", "test"))
    Y_train, Y_test = (np.array(ds[split]["label"], dtype=np.uint8) for split in ("train", "test"))
    return X_train, Y_train, X_test, Y_test


def load_fmnist(levels: int):
    ds = load_dataset("zalando-datasets/fashion_mnist")
    X_train, X_test = ((levels * np.array(ds[split]["image"], dtype=np.float32) / 255.0).astype(np.int32) for split in ("train", "test"))
    Y_train, Y_test = (np.array(ds[split]["label"], dtype=np.uint8) for split in ("train", "test"))
    return X_train, Y_train, X_test, Y_test


def thermometer(X, levels: int):
    return (X[..., None] >= np.arange(1, levels + 1)).astype(np.uint8)


def adaptive_threshold(X):
    # Same as cv2.adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=11, C=2) per channel:
    # pixel is 1 if above the Gaussian-weighted mean of its 11x11 neighbourhood minus 2.
    X = X.astype(np.float32)
    local_mean = gaussian_filter(X, sigma=(0, 2.0, 2.0, 0), truncate=2.5, mode="nearest")
    return (X > local_mean - 2).astype(np.uint8)


def load_cifar10():
    ds = load_dataset("uoft-cs/cifar10")
    X_train, X_test = (adaptive_threshold(np.array(ds[split]["img"])) for split in ("train", "test"))
    Y_train, Y_test = (np.array(ds[split]["label"], dtype=np.uint8) for split in ("train", "test"))
    return X_train, Y_train, X_test, Y_test


def load_celeba_split(split, levels: int):
    # Raw JPEG bytes, so the decoder can downscale while decoding (draft).
    split = split.select_columns("image").cast_column("image", Image(decode=False))
    X = np.empty((len(split), 64, 64, 3), dtype=np.uint8)
    for i, row in enumerate(split):
        img = PIL.Image.open(BytesIO(row["image"]["bytes"]))
        img.draft("RGB", (64, 64))
        X[i] = np.asarray(img.convert("RGB").resize((64, 64)))
    # Pixels 0-255 -> levels 0-levels, thermometer encoded by the TM.
    return (levels * X.astype(np.float32) / 255.0).astype(np.uint8)


def load_celeba(levels: int):
    ds = load_dataset("tpremoli/CelebA-attrs")
    attrs = [name for name in ds["train"].column_names if name not in ("image", "prompt_string")]
    # Attributes are -1/1 -> 0/1.
    Y_train, Y_test = ((np.stack([ds[split][a] for a in attrs], axis=1) + 1) // 2 for split in ("train", "test"))
    return (
        load_celeba_split(ds["train"], levels),
        Y_train.astype(np.uint8),
        load_celeba_split(ds["test"], levels),
        Y_test.astype(np.uint8),
    )


def load_imdb(max_features: int):
    ds = load_dataset("stanfordnlp/imdb")
    Y_train, Y_test = (np.array(ds[split]["label"], dtype=np.uint8) for split in ("train", "test"))

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), binary=True)
    selector = SelectKBest(chi2, k=max_features)
    X_train = selector.fit_transform(vectorizer.fit_transform(ds["train"]["text"]), Y_train)
    X_test = selector.transform(vectorizer.transform(ds["test"]["text"]))
    return X_train.toarray().astype(np.uint8), Y_train, X_test.toarray().astype(np.uint8), Y_test


def load_go_emotions(max_features: int):
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")

    # Label index lists -> multi-hot (samples, 28).
    Y_train, Y_test = (np.zeros((len(ds[split]), 28), dtype=np.uint8) for split in ("train", "test"))
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


def load_ag_news(max_features: int):
    ds = load_dataset("fancyzhx/ag_news")
    Y_train, Y_test = (np.array(ds[split]["label"], dtype=np.uint8) for split in ("train", "test"))

    nltk.download("wordnet", quiet=True)
    lemmatizer = WordNetLemmatizer()
    tokenize = CountVectorizer(stop_words="english").build_analyzer()
    vectorizer = CountVectorizer(analyzer=lambda doc: [lemmatizer.lemmatize(t) for t in tokenize(doc)], binary=True)
    selector = SelectKBest(chi2, k=max_features)
    X_train = selector.fit_transform(vectorizer.fit_transform(ds["train"]["text"]), Y_train)
    X_test = selector.transform(vectorizer.transform(ds["test"]["text"]))
    return X_train.toarray().astype(np.uint8), Y_train, X_test.toarray().astype(np.uint8), Y_test
