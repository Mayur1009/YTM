import io
import h5py
import numpy as np
import pandas as pd
from cutm import MultiOutputTM
from tqdm import tqdm
from PIL import Image
from tm_utils import Timer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from tm_utils.binarizer import ThermometerBinarizer

label_names = [
    "Attractive",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Smiling",
    "Wearing_Lipstick",
]


def load_celeba(dir, split):
    file = h5py.File(f"{dir}/img_align_celeba.h5", "r")
    attr = pd.read_csv(f"{dir}/list_attr_celeba.csv")
    attr = attr.replace(-1, 0)
    part = pd.read_csv(f"{dir}/list_eval_partition.txt", sep=" ", names=["image_id", "partition"])
    attr = attr.merge(part)
    attr = attr[["image_id", "partition", *label_names]]

    # Drop rows with all zeros
    attr = attr[(attr.loc[:, ~attr.columns.isin(["image_id", "partition"])] != 0).any(axis=1)]
    xset = attr.loc[attr.partition == split]
    ids = xset[["image_id"]].values.ravel()
    Y = xset.drop(columns=["image_id", "partition"]).to_numpy()
    return file, ids, Y


def load_image_batch(file, ids, ch=8):
    imgs = []
    for id in tqdm(ids, total=len(ids), desc="Loading images", leave=False, dynamic_ncols=True):
        img = Image.open(io.BytesIO(np.array(file[id])))
        img = img.convert("RGB").resize((64, 64))
        imgs.append(np.array(img).astype(np.float32))

    imgs = np.array(imgs)  # Shape: (N, 64, 64, 3)
    therm_bin = ThermometerBinarizer(ch=ch)
    out = therm_bin.binarize_rgb(imgs)  # Shape: (N, 64, 64, ch * 3)
    return out.reshape((len(ids), -1)).astype(np.uint32)


def metrics(true, preds, probs):
    n_class = true.shape[1]

    acc_per_class = np.mean(true == preds, axis=0)
    assert len(acc_per_class) == n_class, "Mismatch in number of classes"
    mean_acc = np.mean(acc_per_class)

    pre, rec, f1, _ = precision_recall_fscore_support(true, preds, average="macro")
    roc_auc = roc_auc_score(true, probs, average="macro", multi_class="ovr")

    return {
        "acc": mean_acc,
        "f1": f1,
        "precision": pre,
        "recall": rec,
        "roc_auc": roc_auc,
    }


def encode_data(tm, file, ids, ch):
    BATCH_SIZE = 90000
    encoded_X = []
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="Encoding", dynamic_ncols=True, leave=False):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_X = load_image_batch(file, batch_ids, ch)
        encoded_X.append(tm.encode(batch_X))
    return np.vstack(encoded_X)


def train(tm: MultiOutputTM, file, ids_train, Ytrain, ids_test, Ytest, ch, epochs=1):
    encoded_X_train = encode_data(tm, file, ids_train, ch)
    encoded_X_test = encode_data(tm, file, ids_test, ch)
    for epoch in range(epochs):
        iota = np.arange(len(ids_train))
        np.random.shuffle(iota)

        Xtrain_suf = encoded_X_train[iota]
        Ytrain_suf = Ytrain[iota]

        train_fit_timer = Timer()
        with train_fit_timer:
            tm.fit(Xtrain_suf, Ytrain_suf, is_X_encoded=True)
        train_time = train_fit_timer.elapsed()

        test_timer = Timer()
        with test_timer:
            test_preds, test_cs = tm.predict(encoded_X_test, is_X_encoded=True)
        test_time = test_timer.elapsed()

        test_prob = (np.clip(test_cs, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        test_metrics = metrics(Ytest, test_preds, test_prob)

        print(f"Epoch {epoch + 1} | Fit time: {train_time:.4f}s | Test time: {test_time:.4f}s")
        print(f"Test Metrics: {test_metrics}")


if __name__ == "__main__":
    file, ids_train, Y_train = load_celeba("./data/CelebA", 0)
    file, ids_test, Y_test = load_celeba("./data/CelebA", 2)

    ch = 8
    tm = MultiOutputTM(
        number_of_clauses_per_class=30000,
        T=60000,
        s=25,
        q=5,
        dim=(64, 64, ch * 3),
        n_classes=len(label_names),
        patch_dim=(3, 3),
        seed=10,
        block_size=128,
    )

    train(tm, file, ids_train, Y_train, ids_test, Y_test, ch, 50)
