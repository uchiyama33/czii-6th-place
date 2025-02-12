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
    load_data,
    load_picks,
    extract_particle_coordinates,
    crop_center,
)
from src.czii_utils.blob_detector import process_all_runs

# %%
preprocess_version = 4.1
copick_config_path = "/workspace/data/czii-cryo-et-object-identification/copick.config"

data_dicts, picks_radius_dict = load_data(
    copick_config_path,
    num_classes=7,
    transform=None,
    preprocess_version=preprocess_version,
    maps_name="gaussian_maps_r0.6",
)

picks_coord_dict = {}
for data_dict in data_dicts:
    picks, particle_names = load_picks(copick_config_path, data_dict["id"])
    particle_coordinates = extract_particle_coordinates(picks)
    picks_coord_dict[data_dict["id"]] = particle_coordinates

# %%
crop_size = [45, 75, 75]
save_dir = f"/workspace/data/czii-cryo-et-object-identification/crop_classification-preV{preprocess_version}-250122-s45_75_75"
os.makedirs(save_dir, exist_ok=True)
scale = 10.012444

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
# data_dictごと（IDごと）に、各粒子の座標を中心として、crop_sizeの範囲を切り出して保存
# /workspace/data/czii-cryo-et-object-identification/{data_dict["id"]}/以下に保存
# ファイル名は{particle_name}_{i}.npy


# %% クロップした画像を保存
for data_dict in tqdm.tqdm(data_dicts, desc="Processing data_dicts"):
    data_id = data_dict["id"]
    data_dir = os.path.join(save_dir, data_id)
    os.makedirs(data_dir, exist_ok=True)
    particle_coordinates = picks_coord_dict[data_id]
    for particle_name, particle_coords in tqdm.tqdm(
        particle_coordinates.items(), desc=f"Processing {data_id}"
    ):
        for i, particle_coord in enumerate(tqdm.tqdm(particle_coords, desc=f"Particles for {particle_name}")):
            center = (
                np.round(particle_coord[2] / scale).astype(int),
                np.round(particle_coord[1] / scale).astype(int),
                np.round(particle_coord[0] / scale).astype(int),
            )
            crop = crop_center(
                data_dict["image"],
                center=center,
                crop_size=crop_size,
            )
            np.save(os.path.join(data_dir, f"{particle_name}_{i}.npy"), crop)

# %% cropした画像を確認
import matplotlib.pyplot as plt

n_row = 3
n_col = 5

name_idx = 0
i = 5

particle_name = particle_names[name_idx]

crop = np.load(os.path.join(data_dir, f"{particle_name}_{i}.npy"))

fig, axes = plt.subplots(n_row, n_col, figsize=(15, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(crop[0, i], cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% ネガティブサンプルを作成
# ランダムにクロップして保存
# ただし、粒子の中心からpicks_radius_dictの値以上の距離であること

n_negative_samples = 1000

for data_dict in tqdm.tqdm(data_dicts, desc="Processing data_dicts"):
    data_id = data_dict["id"]
    data_dir = os.path.join(save_dir, data_id)
    particle_coordinates = picks_coord_dict[data_id]
    for i in range(n_negative_samples):
        while True:
            center = (
                np.random.randint(pad_size[0] + 1, data_dict["image"].shape[1] - pad_size[0] * 2),
                np.random.randint(pad_size[1] + 1, data_dict["image"].shape[2] - pad_size[1] * 2),
                np.random.randint(pad_size[2] + 1, data_dict["image"].shape[3] - pad_size[2] * 2),
            )
            flag = True
            for particle_name, particle_coords in particle_coordinates.items():
                distances = np.linalg.norm(
                    (np.array(particle_coords) / scale)[:, ::-1] - np.array(center)[None, :],
                    axis=1,
                )
                if np.any(distances <= picks_radius_dict[particle_name] / scale):
                    flag = False

            if flag:
                break

        crop = crop_center(
            data_dict["image"],
            center=center,
            crop_size=crop_size,
        )
        np.save(os.path.join(data_dir, f"negative_{i}.npy"), crop)


# %% ネガティブサンプルを確認
n_row = 3
n_col = 5

i = 8

crop = np.load(os.path.join(data_dir, f"negative_{i}.npy"))

fig, axes = plt.subplots(n_row, n_col, figsize=(15, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(crop[0, i], cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# blob detectorで粒子を検出、真の粒子以外をネガティブサンプルとして保存
# ただし、真の粒子の中心からpicks_radius_dictの値以上の距離であること

root = copick.from_file("/workspace/data/czii-cryo-et-object-identification/copick.config")
df_blob = process_all_runs(root=root, session_id="0", user_id="blobDetector", voxel_spacing=10)

for data_dict in tqdm.tqdm(data_dicts, desc="Processing data_dicts"):
    id = data_dict["id"]
    df_blob_id = df_blob[df_blob["experiment"] == id]
    data_dir = os.path.join(save_dir, id)

    particle_coordinates = []
    for particle_name in particle_names:
        df_blob_id_particle = df_blob_id[df_blob_id["particle_type"] == particle_name]
        _particle_coordinates = df_blob_id_particle[["x", "y", "z"]].values
        particle_coordinates.append(_particle_coordinates)
    particle_coordinates = np.concatenate(particle_coordinates, axis=0)
    # ほぼ同じ座標の粒子を削除
    distances = np.linalg.norm(particle_coordinates[:, None, :] - particle_coordinates[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    mask = np.any(distances < 10, axis=1)
    particle_coordinates = particle_coordinates[~mask]

    cnt = 0
    for particle_coord in tqdm.tqdm(particle_coordinates, desc=f"Particles for {particle_name}"):
        center = (
            np.round(particle_coord[2] / scale).astype(int),
            np.round(particle_coord[1] / scale).astype(int),
            np.round(particle_coord[0] / scale).astype(int),
        )
        flag = True
        for particle_name_true, particle_coords in picks_coord_dict[id].items():
            distances = np.linalg.norm(
                (np.array(particle_coords) / scale)[:, ::-1] - np.array(center)[None, :],
                axis=1,
            )
            if np.any(distances <= picks_radius_dict[particle_name_true] / scale):
                flag = False
        if flag:
            crop = crop_center(
                data_dict["image"],
                center=center,
                crop_size=crop_size,
            )
            np.save(os.path.join(data_dir, f"negative_blob_{cnt}.npy"), crop)
            cnt += 1
        else:
            print("skip: ", center)

# %% ネガティブサンプルを確認
i = 25

n_row = 3
n_col = 5

crop = np.load(os.path.join(data_dir, f"negative_blob_{i}.npy"))

fig, axes = plt.subplots(n_row, n_col, figsize=(15, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(crop[0, i], cmap="gray")
    ax.axis("off")

plt.tight_layout()

plt.show()

# %%
