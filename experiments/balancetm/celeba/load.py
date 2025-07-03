import io

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from tm_utils.binarizer import ThermometerBinarizer

# label_names = [
#     "Attractive",
#     "Heavy_Makeup",
#     "High_Cheekbones",
#     "Male",
#     "Mouth_Slightly_Open",
#     "Smiling",
#     "Wearing_Lipstick",
# ]

label_names = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
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

