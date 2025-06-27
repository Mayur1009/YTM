import io
import h5py
import numpy as np
import pandas as pd
from cutm import MultiOutputTM
from tqdm import tqdm
from PIL import Image
from tm_utils import Timer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

BATCH_SIZE = 25000
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

    imgs = np.array(imgs) # Shape: (N, 64, 64, 3)

    out = np.empty((len(ids), 64, 64, ch * 3))
    for i in range(3):
        for j in range(ch):
            t1 = (j + 1) * 255 / (ch + 1)
            out[:, :, :, i * ch + j] = (imgs[..., i] >= t1) & 1

    return out.reshape((len(ids), -1)).astype(np.uint32)


def metrics(true, preds):
    n_class = true.shape[1]

    acc_per_class = np.mean(true == preds, axis=0)
    assert len(acc_per_class) == n_class, "Mismatch in number of classes"
    mean_acc = np.mean(acc_per_class)

    pre, rec, f1, _ = precision_recall_fscore_support(true, preds, average="macro")
    roc_auc = roc_auc_score(true, preds, average="macro", multi_class="ovr")

    return {
        "acc": mean_acc,
        "f1": f1,
        "precision": pre,
        "recall": rec,
        "roc_auc": roc_auc,
    }


def train(tm: MultiOutputTM, file, ids_train, Ytrain, ids_test, Ytest, ch, epochs=1):
    for epoch in range(epochs):
        iota = np.arange(len(ids_train))
        np.random.shuffle(iota)
        ids_train = ids_train[iota]
        Ytrain = Ytrain[iota]
        train_fit_time = 0

        for i in (tfitbar := tqdm(range(0, len(ids_train), BATCH_SIZE), desc="Training", dynamic_ncols=True)):
            batch_ids = ids_train[i : i + BATCH_SIZE]
            batch_Y = Ytrain[i : i + BATCH_SIZE]
            batch_X = load_image_batch(file, batch_ids, ch)

            train_fit_timer = Timer()
            with train_fit_timer:
                tm.fit(batch_X, batch_Y)
            train_fit_time += train_fit_timer.elapsed()

            # train_pred_timer = Timer()
            # with train_pred_timer:
            #     bpred, _ = tm.predict(batch_X)
            # bmet = metrics(batch_Y, bpred)
            #
            # tfitbar.set_postfix_str(
            #     f"Time: Fit-{train_fit_time:.4f}s, pred-{train_pred_timer.elapsed():.4f} | {' | '.join([f'{k}: {v:.4f}' for k, v in bmet.items()])}"
            # )

        test_time = 0
        test_preds = []
        for i in tqdm(range(0, len(ids_test), BATCH_SIZE), desc="Testing", leave=False, dynamic_ncols=True):
            batch_ids = ids_test[i : i + BATCH_SIZE]
            batch_Y = Ytest[i : i + BATCH_SIZE]
            batch_X = load_image_batch(file, batch_ids, ch)
            test_timer = Timer()
            with test_timer:
                    test_pred, _ = tm.predict(batch_X)
            test_time += test_timer.elapsed()
            test_preds.append(test_pred)

        test_preds = np.vstack(test_preds)
        test_metrics = metrics(Ytest, test_preds)

        print(f"Epoch {epoch + 1} | Fit time: {train_fit_time:.4f}s | Test time: {test_time:.4f}s")
        print(f"Test Metrics: {test_metrics}")


if __name__ == "__main__":
    file, ids_train, Y_train = load_celeba("./data/CelebA", 0)
    file, ids_test, Y_test = load_celeba("./data/CelebA", 2)

    ch = 8
    tm = MultiOutputTM(
        number_of_clauses=25000,
        T=60000,
        s=25,
        q=4,
        dim=(64, 64, ch * 3),
        n_classes=len(label_names),
        patch_dim=(3, 3),
        seed=10,
        block_size=256,
    )

    train(tm, file, ids_train, Y_train, ids_test, Y_test, ch, 10)
