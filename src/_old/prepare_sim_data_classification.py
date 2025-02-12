# %%
import os
import torch
import numpy as np
import pandas as pd
import rootutils
import tqdm
import copick

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.czii_data_utils import (
    load_my_sim_data_classification,
    crop_center,
)
from src.czii_utils.blob_detector import process_all_runs

# %%
preprocess_version = 4.1
n_tomogram = 10
max_class_samples = 10000
use_class_idices = [4, 5, 6, 7, 8, 9, 10]
data_dir = "/workspace/data/my_sim_241221"

label_mapping = {label: i + 1 for i, label in enumerate(use_class_idices)}

# %%
crop_size = [15, 75, 75]
save_dir = os.path.join(
    data_dir, f"crop_classification-preV{preprocess_version}-n{max_class_samples}-250107-s15_75_75"
)
os.makedirs(save_dir, exist_ok=True)

# %%
data_dicts = load_my_sim_data_classification(
    data_dir="/workspace/data/my_sim_241221",
    num_classes=17,
    use_class_idices=np.arange(17),
    transform=None,
    preprocess_version=preprocess_version,
    max_data=n_tomogram,
)

# クロップ時に範囲外をゼロパディングするため、あらかじめ画像をパディングしておく
# パディングサイズはクロップサイズの半分
pad_size = [s // 2 for s in crop_size]
for data_dict in data_dicts:
    data_dict["image"] = np.pad(
        data_dict["image"],
        ((0, 0), (pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2])),
        mode="constant",
        constant_values=0,
    )

# %%
image_indices = np.concatenate(
    [[i] * len(data_dicts[i]["position_labels"]) for i in range(len(data_dicts))], axis=0
)
positions = np.concatenate([data_dict["positions"] for data_dict in data_dicts], axis=0)
position_labels = np.concatenate([data_dict["position_labels"] for data_dict in data_dicts], axis=0)

shuffle_indices = np.random.permutation(len(image_indices))
image_indices = image_indices[shuffle_indices]
positions = positions[shuffle_indices]
position_labels = position_labels[shuffle_indices]

# %%
# position_labelsからuse_class_idicesに含まれるクラスのサンプルをmax_class_samplesまで取得
use_sample_indices = []
for label in use_class_idices:
    label_indices = np.where(position_labels == label)[0]
    use_sample_indices.extend(label_indices[:max_class_samples])

# use_class_idices以外のクラスのサンプルをmax_class_samplesまで取得
neg_sample_indices = []
for i, label in enumerate(position_labels):
    if i in use_sample_indices:
        continue
    neg_sample_indices.append(i)
    if len(neg_sample_indices) >= max_class_samples:
        break

use_sample_indices = np.concatenate([use_sample_indices, neg_sample_indices], axis=0)

image_indices = image_indices[use_sample_indices]
positions = positions[use_sample_indices]
position_labels = position_labels[use_sample_indices]

# %%
images = []
labels = []
for image_idx, center, label in tqdm.tqdm(zip(image_indices, positions, position_labels)):
    data_dict = data_dicts[image_idx]
    center = (center[2], center[1], center[0])
    crop = crop_center(
        data_dict["image"],
        center=center,
        crop_size=crop_size,
    )
    if label in use_class_idices:
        label = label_mapping[label]
    else:
        label = 0
    images.append(crop)
    labels.append(label)

images = np.stack(images, axis=0)
labels = np.array(labels)

# %%
# labelの分布を確認
import matplotlib.pyplot as plt

plt.hist(labels, bins=np.arange(0, len(use_class_idices) + 2) - 0.5, rwidth=0.8)

# %% cropした画像を確認
import matplotlib.pyplot as plt

n_row = 3
n_col = 5

label = 0
i = 10

image_show = images[labels == label][i]

fig, axes = plt.subplots(n_row, n_col, figsize=(15, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(image_show[0, i], cmap="gray")
    ax.axis("off")


# %%
np.save(os.path.join(save_dir, "images.npy"), images)
np.save(os.path.join(save_dir, "labels.npy"), labels)

# %%
