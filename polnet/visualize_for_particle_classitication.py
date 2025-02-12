# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# %%
data_dir = "/home/tomo/kaggle/polnet/output/sim_241213/dataset"
tomo = np.load(f"{data_dir}/tomogram_0/tomo.npy")
label = np.load(f"{data_dir}/tomogram_0/label.npy")
motif_list = pd.read_csv(f"{data_dir}/tomogram_0/tomos_motif_list.csv", sep="\t")

# %% 可視化
z = 50

# トモグラム
plt.subplot(1, 2, 1)
plt.imshow(tomo[z], cmap="gray")
plt.title("Tomogram")

# ラベル
plt.subplot(1, 2, 2)
plt.imshow(label[z], cmap="gray")
plt.title("Label")

plt.show()

# %% 粒子位置でクロップ
particle = motif_list.iloc[868]
x, y, z = (particle[["X", "Y", "Z"]] / 10).astype(int)
print(x, y, z)

crop_size = 64
tomo_crop = tomo[z, y - crop_size // 2 : y + crop_size // 2, x - crop_size // 2 : x + crop_size // 2]

plt.imshow(tomo_crop, cmap="gray")
plt.title("Tomogram Crop")
plt.show()

# %%
# Polymer列でグルーピングして一つ一つの粒子を可視化
polymer_groups = motif_list.groupby(["Label", "Polymer"])

# %%
label = 5
polymer = 4
particles = polymer_groups.get_group((label, polymer)).reset_index(drop=True)

n_particles = len(particles)
n_cols = 5
n_rows = n_particles // n_cols + 1

plt.figure(figsize=(20, 4 * n_rows))
for i, particle in particles.iterrows():
    x, y, z = (particle[["X", "Y", "Z"]] / 10).astype(int)
    print(x, y, z)
    tomo_crop = tomo[z, y - crop_size // 2 : y + crop_size // 2, x - crop_size // 2 : x + crop_size // 2]

    plt.subplot(n_rows, n_cols, i + 1)
    plt.imshow(tomo_crop, cmap="gray")
    plt.title(f"Particle {i}")

# %%
z_list = np.arange(-20, 21, 1)
n_cols = 3
n_rows = len(z_list) // n_cols + 1
plt.figure(figsize=(20, 4 * n_rows))
for i, _z in enumerate(z_list):
    plt.subplot(n_rows, n_cols, i + 1)
    tomo_crop = tomo[z + _z, y - crop_size // 2 : y + crop_size // 2, x - crop_size // 2 : x + crop_size // 2]
    plt.imshow(tomo_crop, cmap="gray")
    plt.axis("off")
    plt.title(f"z={_z}")
# %%
"""
particle radius
1: not particle
2: 4
3: 6
4: 8
5: 4
6: 8
7: 16
8: 12
9: 10
10: 12
"""
