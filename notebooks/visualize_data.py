# %%
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    OneOf,
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    RandRotated,
    RandFlipd,
    RandShiftIntensityd,
    RandAffined,
    RandRotate90d,
    NormalizeIntensityd,
    RandCropByLabelClassesd,
    RandSpatialCropSamplesd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Rand3DElasticd,
    RandAdjustContrastd,
    CutOutd,
    CutMixd,
    CutOut,
)
import torch
import torch.nn.functional as F
import rootutils
from copy import deepcopy
import albumentations as A

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.components.czii_data_utils import (
    load_data,
    load_picks,
    extract_particle_coordinates,
    load_my_sim_data,
)


# %%
def extract_region(data, center, size=32):
    half_size = size // 2
    x, y, z = center
    x_min, x_max = max(0, x - half_size), min(data.shape[0], x + half_size)
    y_min, y_max = max(0, y - half_size), min(data.shape[1], y + half_size)
    z_min, z_max = max(0, z - half_size), min(data.shape[2], z + half_size)

    cropped = np.zeros((size, size, size))
    cropped[
        half_size - (x - x_min) : half_size + (x_max - x),
        half_size - (y - y_min) : half_size + (y_max - y),
        half_size - (z - z_min) : half_size + (z_max - z),
    ] = data[x_min:x_max, y_min:y_max, z_min:z_max]

    return cropped


def visualize_particle(volume):
    size = volume.shape[0]
    half_size = size // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # X-Y plane
    axes[0].imshow(volume[:, :, half_size], cmap="gray")
    axes[0].set_title("X-Y plane")

    # X-Z plane
    axes[1].imshow(volume[:, half_size, :], cmap="gray")
    axes[1].set_title("X-Z plane")

    # Y-Z plane
    axes[2].imshow(volume[half_size, :, :], cmap="gray")
    axes[2].set_title("Y-Z plane")

    plt.show()


# # %%

# copick_config_path = "/workspace/data/czii-cryo-et-object-identification/copick.config"
# data_dicts, picks_radius_dict = load_data(copick_config_path, 7)

# for data_dict in data_dicts:
#     picks, particle_names = load_picks(copick_config_path, data_dict["id"])
#     particle_coordinates = extract_particle_coordinates(picks)

#     print(f"ID: {data_dict['id']}")
#     print(f"Particle names: {particle_names}")
#     print(f"Particle coordinates: {particle_coordinates}")


# # %%
# volume_spaccing = 10
# cropped_particle_volumes = {}
# for name in particle_names:
#     cropped_particle_volumes[name] = []
#     for data_dict in data_dicts:
#         data = data_dict["image"]
#         picks, particle_names = load_picks(copick_config_path, data_dict["id"])
#         particle_coordinates = extract_particle_coordinates(picks)
#         for coordinate in particle_coordinates[name]:
#             coordinate = np.round(np.array(coordinate) / volume_spaccing).astype(int)
#             coordinate = coordinate[::-1]  # zyx -> xyz
#             cropped_particle_volume = extract_region(data, coordinate, size=32)
#             cropped_particle_volumes[name].append(cropped_particle_volume)


# # %%
# for name in particle_names:
#     print(f"Number of cropped particle volumes: {len(cropped_particle_volumes[name])}")


# # %%
# # cropped_particle_volumesを使って、各粒子の3DボリュームをN個ずつ可視化する
# N = 5

# for name in particle_names:
#     print(f"Particle type: {name}")
#     for i, cropped_particle_volume in enumerate(cropped_particle_volumes[name][:N]):
#         print(f"Particle volume {i}")
#         visualize_particle(cropped_particle_volume)


# # %%
# d = {"image": np.copy(cropped_particle_volumes["virus-like-particle"][1])}
# d = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")(d)
# d = NormalizeIntensityd(keys="image")(d)
# # d = RandRotated(
# #     keys=["image"],
# #     range_x=90,
# #     range_y=0,
# #     range_z=0,
# #     prob=1,
# #     keep_size=True,
# # )(d)
# # d=RandRotate90d(keys=["image"], prob=1, spatial_axes=[1, 2])(d)
# # d=RandGaussianNoised(keys=["image"], prob=1, std=0.5, sample_std=True)(d)
# # d=RandGaussianSmoothd(keys=["image"], prob=1, sigma_x=(0.1,0.3))(d)
# # d=Rand3DElasticd(keys=["image"], prob=1, sigma_range=(5, 7), magnitude_range=(50, 100), spatial_size=(32, 32, 32))(d)
# d = RandAdjustContrastd(keys=["image"], prob=1, gamma=(0.5, 1.5))(d)

# visualize_particle(d["image"][0])


# %%

transforms_common = Compose(
    [
        # EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel", allow_missing_keys=True),
        Orientationd(keys=["image", "label", "maps"], axcodes="RAS", allow_missing_keys=True),
    ]
)
sim_data_dicts = load_my_sim_data(
    "/workspace/data/my_sim_241221", 17, transforms_common, preprocess_version=4, max_data=10
)

copick_config_path = "/workspace/data/czii-cryo-et-object-identification/copick.config"
data_dicts, picks_radius_dict = load_data(copick_config_path, 7, transforms_common, preprocess_version=4)

# %%

# def normalize_percentile(image, pmin=5, pmax=99):
#     imin, imax = np.percentile(image, (pmin, pmax))
#     return (image - imin) / (imax - imin)


def normalize_min_max(image):
    return (image - image.min()) / (image.max() - image.min())


def normalise_by_percentile(data, min=5, max=99):
    min = np.percentile(data, min)
    max = np.percentile(data, max)
    data = (data - min) / (max - min)
    return data


def normalise_by_percentile_2(data, min=5, max=99):
    min = np.percentile(data, min)
    max = np.percentile(data, max)
    data = (data - min) / (max - min)
    data = data - data.mean()
    data = data / data.std()
    return data


def normalize_max_percentile(image, pmax=99.5):
    imax = np.percentile(image, pmax)
    image = np.clip(image, -imax, imax)
    return image / imax


def clip_percentile(image, pmin, pmax):
    imin, imax = np.percentile(image, (pmin, pmax))
    image = np.clip(image, imin, imax)
    image = (image - imin) / (imax - imin)
    image = (image - 0.5) * 2
    return image


def clip_percentile_2(image, pmin, pmax):
    imin, imax = np.percentile(image, (pmin, pmax))
    image = np.clip(image, imin, imax)
    image = (image - imin) / (imax - imin)
    image = image - 1
    return image


def log_transform(image):
    image = -image
    image = np.clip(image, np.percentile(image, 0.001), None)
    image = np.log1p(image - image.min())
    image = image / image.mean()
    image = image - image.mean()
    return image


def normalize(image):
    image = image - image.mean()
    image = image / image.std()
    return image


# %%
trans = Compose(
    [
        RandCropByLabelClassesd(
            keys=["image", "maps"],
            label_key="label",
            spatial_size=(76, 160, 160),
            num_samples=1,
            indices_key="label_indices",
            allow_missing_keys=True,
        ),
        # RandAdjustContrastd(keys=["image"], prob=1, gamma=(1.5, 1.5)),
        # RandGaussianSmoothd(keys=["image"], prob=1, sigma_x=(0.1, 0.1)),
        # RandGaussianNoised(keys=["image"], prob=1, std=1),
    ]
)

# %%
i = 1
d = deepcopy(data_dicts[i])
# d["image"] = normalize(d["image"])
d = trans(d)

plt.subplot(121)
plt.imshow(d[0]["image"][0, 30, :, :], cmap="gray", vmin=0, vmax=1)
# plt.subplot(122)
# plt.hist(d[0]["image"].flatten(), bins=50)
# print(d[0]["image"].min(), d[0]["image"].max())


# %%
s = deepcopy(sim_data_dicts[i])
# s["image"] = normalise_by_percentile_2(s["image"], min=5, max=99)
s = trans(s)
# s = s[0]

# # mapのある位置でimageの画素値を減衰させる
# for map in s["maps"]:
#     map = map.to(torch.float32)
#     if map.sum() == 0:
#         continue
#     map[map > 0] = (
#         map[map > 0]
#         + rand_log_normal(map[map > 0].shape, mu=0.5, sigma=0.6)
#         # + torch.rand(map[map > 0].shape) * 0.5
#     )
#     map = map.unsqueeze(0).to(s["image"].dtype)
#     map = torch.clamp(map, 0, 8)
#     print(map[map > 0].max(), map[map > 0].mean())
#     s["image"][map > 0] = s["image"][map > 0] - map[map > 0]
#     # 0.9~1.1の乱数をかける
#     # s[0]["image"] = s[0]["image"] * (torch.rand(map.shape) * 0.2 + 0.9)

# trans2 = Compose(
#     [
#         RandGaussianSmoothd(keys=["image"], prob=1, sigma_x=(0.1, 0.1)),
#         RandGaussianNoised(keys=["image"], prob=1, std=1),
#     ]
# )
# s = trans2(s)
# %%
s3 = deepcopy(s)
trans3 = A.Compose(
    [
        # A.Blur(blur_limit=(3, 5), p=0.5),
        # A.GaussNoise(var_limit=(0.01, 0.1), p=0.5),
        # A.Downscale(scale_min=0.7, scale_max=0.9, p=0.5),
        # A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, -0.3), p=0.5),
        # A.RandomShadow(
        #     shadow_roi=(0.0, 0.0, 1.0, 1.0),
        #     shadow_dimension=5,
        #     p=0.5,
        # ),
        A.ElasticTransform(
            alpha=20,
            sigma=50,
            p=1,
        ),
    ]
)
s3 = s3[0]
img = s3["image"]
# img = (img - img.min()) / (img.max() - img.min())
C, D, H, W = img.shape
img = img.reshape(C * D, H, W).permute(1, 2, 0)
img = trans3(image=img.numpy().astype(np.float32))["image"]
img = torch.from_numpy(img).reshape(H, W, C * D).permute(2, 0, 1).reshape(C, D, H, W)


plt.subplot(121)
plt.imshow(img[0, 30, :, :], cmap="gray", vmin=0, vmax=1)
# plt.subplot(122)
# plt.hist(s[0]["image"].flatten(), bins=50)
# print(s[0]["image"].min(), s[0]["image"].max())

# %%
d2 = deepcopy(d)
trans2 = Compose(
    [
        # Rand3DElasticd(
        #     keys=["image", "label", "maps"],
        #     prob=1,
        #     sigma_range=(5, 10),
        #     magnitude_range=(100, 100),
        #     spatial_size=(64, 128, 128),
        #     mode="bilinear",
        #     padding_mode="zeros",
        #     allow_missing_keys=True,
        # ),
        RandAffined(
            keys=["image", "label", "maps"],
            spatial_size=(64, 128, 128),
            prob=1,
            rotate_range=(45, 0.2, 0.2),
            padding_mode="zeros",
        ),
        RandRotate90d(
            keys=["image", "label", "maps"],
            prob=0.5,
            spatial_axes=[1, 2],
            allow_missing_keys=True,
        ),
        RandFlipd(
            keys=["image", "label", "maps"],
            prob=0.5,
            spatial_axis=[0, 1, 2],
            allow_missing_keys=True,
        ),
    ]
)

d2 = trans2(d2)
# plt.subplot(121)
# plt.imshow(d2[0]["image"][0, 31, :, :], cmap="gray", vmin=0, vmax=1)

plt.figure(figsize=(6, 18))
size = d2[0]["image"][0].shape[0]
n_cols = 4
for i in range(size):
    plt.subplot(size // n_cols, n_cols, i + 1)
    plt.imshow(d2[0]["image"][0, i], vmin=0, vmax=1, cmap="gray")
    plt.axis("off")
    plt.title(f"z={i}")

# %%
d3 = deepcopy(d2)
trans3 = A.Compose(
    [
        # A.Blur(blur_limit=(3, 3), p=1),
        # A.GaussNoise(var_limit=(0.005, 0.01), p=1),
        A.Downscale(scale_min=0.9, scale_max=0.95, p=1),
        # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, -0.1), p=1),
        # A.RandomShadow(
        #     shadow_roi=(0.0, 0.0, 1.0, 1.0),
        #     shadow_dimension=5,
        #     p=0.5,
        # ),
        A.ElasticTransform(
            alpha=20,
            sigma=30,
            p=1,
        ),
    ]
)
d3 = d3[0]
img = d3["image"]
# img = (img - img.min()) / (img.max() - img.min())
C, D, H, W = img.shape
img = img.reshape(C * D, H, W).permute(1, 2, 0)
img = trans3(image=img.numpy().astype(np.float32))["image"]
img = torch.from_numpy(img).reshape(H, W, C * D).permute(2, 0, 1).reshape(C, D, H, W)


plt.subplot(121)
plt.imshow(img[0, 30, :, :], cmap="gray", vmin=0, vmax=1)
# plt.subplot(122)
# plt.hist(s[0]["image"].flatten(), bins=50)
# print(s[0]["image"].min(), s[0]["image"].max())

# %%
plt.hist(data_dicts[0]["image"][0, 10:12].flatten(), bins=100)
# %%
plt.hist(sim_data_dicts[0]["image"][0, 10:12].flatten(), bins=100)
# %%
s = deepcopy(sim_data_dicts[0])
s = s["image"]
s[s < 0] = s[s < 0] * 1.2
plt.hist(s[0, 10:12].flatten(), bins=100)
# %%
# ヒストグラムマッチング
from skimage.exposure import match_histograms

source = normalize_percentile(data_dicts[0]["image"].numpy(), pmax=99)
target = sim_data_dicts[0]["image"].numpy()
matched = match_histograms(target, source)

plt.subplot(231)
plt.imshow(source[0, 30, :, :], cmap="gray")
plt.subplot(232)
plt.imshow(target[0, 30, :, :], cmap="gray")
plt.subplot(233)
plt.imshow(matched[0, 30, :, :], cmap="gray")
plt.subplot(234)
plt.hist(source[0, 30, :].flatten(), bins=50)
plt.subplot(235)
plt.hist(target[0, 30, :].flatten(), bins=50)
plt.subplot(236)
plt.hist(matched[0, 30, :].flatten(), bins=50)


# %%
s = deepcopy(sim_data_dicts[0])
s = s["image"]

s1 = normalize_percentile(s, pmax=98)
s2 = clip_percentile(s, 0.01, 20)

plt.subplot(221)
plt.imshow(s1[0, 30, :, :], cmap="gray")
plt.subplot(222)
plt.imshow(s2[0, 30, :, :], cmap="gray")

plt.subplot(223)
plt.hist(s1[0, 30, :].flatten(), bins=50, log=True)
plt.subplot(224)
plt.hist(s2[0, 30, :].flatten(), bins=50, log=True)

print(s1.min(), s1.max())
print(s2.min(), s2.max())
# %%
s = deepcopy(data_dicts[0])
s = s["image"]

s1 = normalize_percentile(s, pmax=99)
s2 = clip_percentile(s, 0.03, 20)

plt.subplot(221)
plt.imshow(s1[0, 30, :, :], cmap="gray")
plt.subplot(222)
plt.imshow(s2[0, 30, :, :], cmap="gray")

plt.subplot(223)
plt.hist(s1[0, 30, :].flatten(), bins=50, log=True)
plt.subplot(224)
plt.hist(s2[0, 30, :].flatten(), bins=50, log=True)

print(s1.min(), s1.max())
print(s2.min(), s2.max())

# %%
s = deepcopy(sim_data_dicts[1])

s1 = normalize_percentile(s["image"], pmax=97)
s2 = clip_percentile(s["image"], 0.1, 40)
s3 = clip_percentile(s["image"], 0.01, 10)

s["image"] = torch.from_numpy(np.concatenate([s1, s2, s3], axis=0))

s = trans(s)

plt.subplot(231)
plt.imshow(s[0]["image"][0, 30, :, :], cmap="gray")
plt.subplot(232)
plt.imshow(s[0]["image"][1, 30, :, :], cmap="gray")
plt.subplot(233)
plt.imshow(s[0]["image"][2, 30, :, :], cmap="gray")
plt.subplot(234)
plt.hist(s[0]["image"][0][30].flatten(), bins=30)
plt.subplot(235)
plt.hist(s[0]["image"][1][30].flatten(), bins=30)
plt.subplot(236)
plt.hist(s[0]["image"][2][30].flatten(), bins=30)
print(s[0]["image"].min(), s[0]["image"].max())

# %%
s = deepcopy(data_dicts[0])

s1 = normalize_percentile(s["image"], pmax=99)
s2 = clip_percentile(s["image"], 0.03, 30)
s3 = clip_percentile(s["image"], 0.001, 10)

s["image"] = torch.from_numpy(np.concatenate([s1, s2, s3], axis=0))

s = trans(s)

plt.subplot(231)
plt.imshow(s[0]["image"][0, 30, :, :], cmap="gray")
plt.subplot(232)
plt.imshow(s[0]["image"][1, 30, :, :], cmap="gray")
plt.subplot(233)
plt.imshow(s[0]["image"][2, 30, :, :], cmap="gray")
plt.subplot(234)
plt.hist(s[0]["image"][0][30].flatten(), bins=30)
plt.subplot(235)
plt.hist(s[0]["image"][1][30].flatten(), bins=30)
plt.subplot(236)
plt.hist(s[0]["image"][2][30].flatten(), bins=30)
print(s[0]["image"].min(), s[0]["image"].max())


# %%
center = (32, 68, 60)

visualize_particle(extract_region(sim_data_dicts[0]["image"][0], center, size=32))
# %%
