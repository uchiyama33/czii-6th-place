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
)
from src.czii_utils.blob_detector import process_all_runs
from src.czii_utils.utils_data import extract_region_withCH

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
save_dir = f"/workspace/data/czii-cryo-et-object-identification/coord-2nd-stage"
os.makedirs(save_dir, exist_ok=True)
scale = 10.012444

# %%
coord_dict = {}
for data_dict in data_dicts:
    data_id = data_dict["id"]
    tmp_dict = {}
    for particle_name in particle_names + ["negative"]:
        tmp_dict[particle_name] = []
    coord_dict[data_id] = tmp_dict

# %% クロップした画像を保存
for data_dict in tqdm.tqdm(data_dicts, desc="Processing data_dicts"):
    data_id = data_dict["id"]
    particle_coordinates = picks_coord_dict[data_id]
    for particle_name, particle_coords in tqdm.tqdm(
        particle_coordinates.items(), desc=f"Processing {data_id}"
    ):
        for i, particle_coord in enumerate(tqdm.tqdm(particle_coords, desc=f"Particles for {particle_name}")):
            coord_dict[data_id][particle_name].append(particle_coord)

# %% ネガティブサンプルを作成
# ランダムにクロップして保存
# ただし、粒子の中心からpicks_radius_dictの値以上の距離であること

n_negative_samples = 1000

for data_dict in tqdm.tqdm(data_dicts, desc="Processing data_dicts"):
    data_id = data_dict["id"]
    particle_coordinates = picks_coord_dict[data_id]
    for i in range(n_negative_samples):
        while True:
            # FIXME
            center = (
                np.random.randint(0, data_dict["image"].shape[3] * scale),
                np.random.randint(0, data_dict["image"].shape[2] * scale),
                np.random.randint(0, data_dict["image"].shape[1] * scale),
            )
            flag = True
            for particle_name, particle_coords in particle_coordinates.items():
                distances = np.linalg.norm(
                    (np.array(particle_coords) / scale) - np.array(center)[None, :] / scale,
                    axis=1,
                )
                if np.any(distances <= picks_radius_dict[particle_name] / scale):
                    flag = False

            if flag:
                break

        coord_dict[data_id]["negative"].append(center)


# %%
# blob detectorで粒子を検出、真の粒子以外をネガティブサンプルとして保存
# ただし、真の粒子の中心からpicks_radius_dictの値以上の距離であること

root = copick.from_file("/workspace/data/czii-cryo-et-object-identification/copick.config")
df_blob = process_all_runs(root=root, session_id="0", user_id="blobDetector", voxel_spacing=10)

for data_dict in tqdm.tqdm(data_dicts, desc="Processing data_dicts"):
    id = data_dict["id"]
    df_blob_id = df_blob[df_blob["experiment"] == id]

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

    for particle_coord in tqdm.tqdm(particle_coordinates, desc=f"Particles for {particle_name}"):
        center = particle_coord
        flag = True
        for particle_name_true, particle_coords in picks_coord_dict[id].items():
            distances = np.linalg.norm(
                (np.array(particle_coords) / scale) - np.array(center)[None, :] / scale,
                axis=1,
            )
            if np.any(distances <= picks_radius_dict[particle_name_true] / scale):
                flag = False
        if flag:
            coord_dict[data_id]["negative"].append(center)
        else:
            print("skip: ", center)

# %% save coord_dict
save_path = os.path.join(save_dir, "coord_dict.npy")
np.save(save_path, coord_dict)

# %%
# load coord_dict
save_path = os.path.join(save_dir, "coord_dict.npy")
coord_dict = np.load(save_path, allow_pickle=True).item()

# %% サンプルを確認
import matplotlib.pyplot as plt

i = 0

n_row = 3
n_col = 5

data_id = data_dicts[i]["id"]
particle_name = "thyroglobulin"

for j in range(n_row * n_col):
    center = coord_dict[data_id][particle_name][j]
    crop = extract_region_withCH(
        data_dicts[i]["image"],
        coord_center=center,
        scale=scale,
        size=30,
    )
    plt.subplot(n_row, n_col, j + 1)
    plt.imshow(crop[0, 15, :, :], cmap="gray")

# %%
