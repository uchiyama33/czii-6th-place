# %%
# Make a copick project
import os
import shutil

config_blob = """{
    "name": "czii_cryoet_mlchallenge_2024",
    "description": "2024 CZII CryoET ML Challenge training data.",
    "version": "1.0.0",

    "pickable_objects": [
        {
            "name": "apo-ferritin",
            "is_particle": true,
            "pdb_id": "4V1W",
            "label": 1,
            "color": [  0, 117, 220, 128],
            "radius": 60,
            "map_threshold": 0.0418
        },
        {
            "name": "beta-amylase",
            "is_particle": true,
            "pdb_id": "1FA2",
            "label": 2,
            "color": [0, 0, 255, 255],
            "radius": 65,
            "map_threshold": 0.035
        },
        {
            "name": "beta-galactosidase",
            "is_particle": true,
            "pdb_id": "6X1Q",
            "label": 3,
            "color": [ 76,   0,  92, 128],
            "radius": 90,
            "map_threshold": 0.0578
        },
        {
            "name": "ribosome",
            "is_particle": true,
            "pdb_id": "6EK0",
            "label": 4,
            "color": [  0,  92,  49, 128],
            "radius": 150,
            "map_threshold": 0.0374
        },
        {
            "name": "thyroglobulin",
            "is_particle": true,
            "pdb_id": "6SCJ",
            "label": 5,
            "color": [ 43, 206,  72, 128],
            "radius": 130,
            "map_threshold": 0.0278
        },
        {
            "name": "virus-like-particle",
            "is_particle": true,
            "label": 6,
            "color": [255, 204, 153, 128],
            "radius": 135,
            "map_threshold": 0.201
        },
        {
            "name": "membrane",
            "is_particle": false,
            "label": 8,
            "color": [100, 100, 100, 128]
        },
        {
            "name": "background",
            "is_particle": false,
            "label": 9,
            "color": [10, 150, 200, 128]
        }
    ],

    "overlay_root": "/workspace/data/czii-cryo-et-object-identification/train/overlay_preprocessed",

    "overlay_fs_args": {
        "auto_mkdir": true
    },

    "static_root": "/workspace/data/czii-cryo-et-object-identification/train/static"
}"""

copick_config_path = "/workspace/data/czii-cryo-et-object-identification/copick.config"
output_overlay = "/workspace/data/czii-cryo-et-object-identification/train/overlay_preprocessed"

with open(copick_config_path, "w") as f:
    f.write(config_blob)

# Update the overlay
# Define source and destination directories
source_dir = "/workspace/data/czii-cryo-et-object-identification/train/overlay"
destination_dir = "/workspace/data/czii-cryo-et-object-identification/train/overlay_preprocessed"

create = True

if create:
    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Create corresponding subdirectories in the destination
        relative_path = os.path.relpath(root, source_dir)
        target_dir = os.path.join(destination_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)

        # Copy and rename each file
        for file in files:
            if file.startswith("curation_0_"):
                new_filename = file
            else:
                new_filename = f"curation_0_{file}"

            # Define full paths for the source and destination files
            source_file = os.path.join(root, file)
            destination_file = os.path.join(target_dir, new_filename)

            # Copy the file with the new name
            shutil.copy2(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")

# %%
import os
import numpy as np
from pathlib import Path
import torch
import torchinfo
import zarr, copick
from tqdm import tqdm
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    AsDiscrete,
    RandFlipd,
    RandRotate90d,
    NormalizeIntensityd,
    RandCropByLabelClassesd,
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric

# %%
root = copick.from_file(copick_config_path)

copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 10
tomo_type = "denoised"

# %%
from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write
from collections import defaultdict

# Just do this once
generate_masks = True

if generate_masks:
    target_objects = defaultdict(dict)
    for object in root.pickable_objects:
        if object.is_particle:
            target_objects[object.name]["label"] = object.label
            target_objects[object.name]["radius"] = object.radius

    for run in tqdm(root.runs):
        tomo = run.get_voxel_spacing(10)
        tomo = tomo.get_tomogram(tomo_type).numpy()
        target = np.zeros(tomo.shape, dtype=np.uint8)
        for pickable_object in root.pickable_objects:
            pick = run.get_picks(object_name=pickable_object.name, user_id="curation")
            if len(pick):
                target = segmentation_from_picks.from_picks(
                    pick[0],
                    target,
                    target_objects[pickable_object.name]["radius"] * 0.8,
                    target_objects[pickable_object.name]["label"],
                )
        write.segmentation(run, target, copick_user_name, name=copick_segmentation_name)


# %%
data_dicts = []
for run in tqdm(root.runs):
    tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(tomo_type).numpy()
    segmentation = run.get_segmentations(
        name=copick_segmentation_name, user_id=copick_user_name, voxel_size=voxel_size, is_multilabel=True
    )[0].numpy()
    data_dicts.append({"image": tomogram, "label": segmentation, "id": run.name})

print(np.unique(data_dicts[0]["label"]))

# %%
import matplotlib.pyplot as plt

# Plot the images
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title("Tomogram")
plt.imshow(data_dicts[0]["image"][100], cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Painted Segmentation from Picks")
plt.imshow(data_dicts[0]["label"][100], cmap="viridis")
plt.axis("off")

plt.tight_layout()
plt.show()


# %%
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.czii_data_utils import load_data, load_picks, extract_particle_coordinates

# %%
data_dicts, picks_radius_dict = load_data(copick_config_path, 6, maps_name=None)

# %%
scale = 10.012444
radius_scale = 0.5

# %% coordを中心に半径radiusの範囲を1とするラベルマップを作成

output_dir = f"/workspace/data/czii-cryo-et-object-identification/train/particle_hard_masks_r{radius_scale}"
os.makedirs(output_dir, exist_ok=True)


for data_idx in tqdm(range(len(data_dicts))):
    picks, particle_names = load_picks(copick_config_path, data_dicts[data_idx]["id"])
    particle_coordinates = extract_particle_coordinates(picks)

    volume = data_dicts[data_idx]["image"][0]
    label_map_dict = {}

    for name in particle_names:
        label_map = torch.zeros(volume.shape, device="cuda")
        for coord in particle_coordinates[name]:
            x, y, z = coord
            x, y, z = int(round(x / scale)), int(round(y / scale)), int(round(z / scale))
            radius = picks_radius_dict[name] / scale * radius_scale

            zz, yy, xx = torch.meshgrid(
                torch.arange(volume.shape[0], device="cuda"),
                torch.arange(volume.shape[1], device="cuda"),
                torch.arange(volume.shape[2], device="cuda"),
                indexing="ij",
            )

            dist = torch.sqrt((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2)
            mask = dist <= radius
            label_map = torch.max(label_map, mask.float())

        label_map_dict[name] = label_map.cpu().numpy()

    # npyで保存
    os.makedirs(f"{output_dir}/{data_dicts[data_idx]['id']}", exist_ok=True)
    for i, name in enumerate(particle_names):
        np.save(f"{output_dir}/{data_dicts[data_idx]['id']}/{i}_{name}.npy", label_map_dict[name])

# %% 保存したラベルマップを読み込み、確認
import matplotlib.pyplot as plt

data_idx = 0
name_idx = 5
name = particle_names[name_idx]

volume = data_dicts[data_idx]["image"][0]
label_map = np.load(f"{output_dir}/{data_dicts[data_idx]['id']}/{name_idx}_{name}.npy")

z = 60
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Tomogram")
plt.imshow(volume[z], cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Hard Mask")
plt.imshow(label_map[z], cmap="viridis", vmin=0, vmax=1)

plt.subplot(1, 3, 3)
plt.title("Painted Segmentation from Picks")
plt.imshow(data_dicts[data_idx]["label"][0][z], cmap="viridis")

plt.tight_layout()
plt.show()


# %%
