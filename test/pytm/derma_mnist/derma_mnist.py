from lzma import LZMAFile
import pickle
from medmnist.dataset import DermaMNIST
import numpy as np
from mltm.utils import Timer
from pytm.tm import MultiClassConvolutionalTsetlinMachine2D

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from skimage import exposure


def binarize_images(imgs, ch=8):
    centered = imgs.copy()
    # for i in range(imgs.shape[0]):
    #     img = imgs[i]
    #     for ch in range(3):
    #         p2, p98 = np.percentile(img[:, :, ch], (2, 98))
    #         centered[i, :, :, ch] = exposure.rescale_intensity(img[:, :, ch], in_range=(p2, p98))

    out = np.zeros((*imgs.shape[:-1], ch * 3), dtype=np.uint32)
    for i in range(3):
        for j in range(ch):
            t1 = (j + 1) * 255 / (ch + 1)
            out[:, :, :, (i * ch) + j] = (centered[..., i] >= t1) & 1

    return out.reshape(imgs.shape[0], -1).astype(np.uint32)


def load_dataset(ch=8):
    train = DermaMNIST(split="train", download=True)
    val = DermaMNIST(split="val", download=True)
    test = DermaMNIST(split="test", download=True)
    xtrain = binarize_images(train.imgs, ch)
    xval = binarize_images(val.imgs, ch)
    xtest = binarize_images(test.imgs, ch)
    return (
        (xtrain, train.labels.squeeze()),
        (xval, val.labels.squeeze()),
        (xtest, test.labels.squeeze()),
    )

def balance_by_undersampling(y):
    n_class = np.max(y) + 1
    samples_per_class = np.bincount(y)

    # Calculate amount to remove from each class
    min_samples = np.min(samples_per_class)
    balanced_indices = []
    for c in range(n_class):
        c_inds = np.where(y == c)[0]
        if len(c_inds) > 0:
            balanced_c_inds = np.random.choice(c_inds, size=min_samples, replace=False)
            balanced_indices.append(balanced_c_inds)
    balanced_indices = np.concatenate(balanced_indices)
    print(f'New Training size: {len(balanced_indices)=}')
    return balanced_indices

def balance_by_oversampling(y):
    n_class = np.max(y) + 1
    samples_per_class = np.bincount(y)

    # Calculate amount to duplicate each class
    max_samples = np.max(samples_per_class)
    multipliers = np.ceil(max_samples / samples_per_class).astype(int)
    balanced_indices = []
    for c in range(n_class):
        c_inds = np.where(y == c)[0]
        if len(c_inds) > 0:
            balanced_c_inds = np.random.choice(c_inds, size=multipliers[c] * len(c_inds), replace=True)
            balanced_indices.append(balanced_c_inds)

    balanced_indices = np.concatenate(balanced_indices)
    print(f'New Training size: {len(balanced_indices)=}')
    return balanced_indices


def train(tm: MultiClassConvolutionalTsetlinMachine2D, xtrain, ytrain, xval, yval, xtest, ytest, epochs=1):
    for epoch in range(epochs):
        # balanced_indices = balance_by_oversampling(ytrain)
        # xtrain = xtrain[balanced_indices]
        # ytrain = ytrain[balanced_indices]

        balanced_indices = balance_by_undersampling(ytrain)
        xtrain = xtrain[balanced_indices]
        ytrain = ytrain[balanced_indices]

        iota = np.arange(len(ytrain))
        np.random.shuffle(iota)
        xtrain = xtrain[iota]
        ytrain = ytrain[iota]

        train_timer = Timer()
        with train_timer:
            tm.fit(xtrain, ytrain, epochs=1, incremental=True)

        # Train
        preds_train, cs_train = tm.predict(xtrain, return_class_sums=True)
        prob_train = (np.clip(cs_train, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_train = prob_train / (np.sum(prob_train, axis=1, keepdims=True) + 1e-7)
        train_acc = accuracy_score(ytrain, preds_train)
        ytrain_bin = np.zeros((len(ytrain), np.max(ytrain) + 1))
        ytrain_bin[np.arange(len(ytrain)), ytrain] = 1
        auc_train = roc_auc_score(ytrain_bin, prob_train)
        cm_train = confusion_matrix(ytrain, preds_train)

        # Validation
        preds_val, cs_val = tm.predict(xval, return_class_sums=True)
        prob_val = (np.clip(cs_val, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_val = prob_val / (np.sum(prob_val, axis=1, keepdims=True) + 1e-7)
        acc_val = accuracy_score(yval, preds_val)
        yval_bin = np.zeros((len(yval), np.max(ytrain) + 1))
        yval_bin[np.arange(len(yval)), yval] = 1
        auc_val = roc_auc_score(yval_bin, prob_val)

        # Test
        preds, cs_test = tm.predict(xtest, return_class_sums=True)
        prob_test = (np.clip(cs_test, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_test = prob_test / (np.sum(prob_test, axis=1, keepdims=True) + 1e-7)
        acc_test = np.mean(preds == ytest)
        cm_test = confusion_matrix(ytest, preds)
        ytest_bin = np.zeros((len(ytest), np.max(ytrain) + 1))
        ytest_bin[np.arange(len(ytest)), ytest] = 1
        auc_test = roc_auc_score(ytest_bin, prob_test)

        print(
            f"Epoch {epoch + 1} | Time: {train_timer.elapsed():.4f}s | Train Acc: {train_acc}| Train AUC: {auc_train} | Val Acc: {acc_val} | Val AUC: {auc_val} | Test Acc: {acc_test} | AUC: {auc_test}"
        )
        print(f"Train Confusion Matrix:\n{cm_train}")
        print(f"Test Confusion Matrix:\n{cm_test}")
        print(f"Class sums:\n{cs_test}")
        # print(f'{tm.get_weights()=}')

if __name__ == "__main__":
    ch = 8
    (xtrain, ytrain), (xval, yval), (xtest, ytest) = load_dataset(ch)

    for c in range(7):
        print(f'Number of samples in class {c}: {np.sum(ytrain == c)}')

    params = {
        "number_of_clauses": 5000,
        "T": 4500,
        "s": 10,
        "q": 1,
        "dim": (28, 28, ch * 3),
        "patch_dim": (28, 28),
        "encode_loc": False,
        "seed": 10,
    }
    tm = MultiClassConvolutionalTsetlinMachine2D(**params)

    train(tm, xtrain, ytrain, xval, yval, xtest, ytest, epochs=120)

    with LZMAFile("./runs/derma_mnist.tm", "wb") as f:
        states = tm.save()
        pickle.dump(states, f)

