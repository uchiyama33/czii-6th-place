import numpy as np
import torch
import matplotlib.pyplot as plt


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


def visualize_particle(volume, axes=None, vmin=None, vmax=None, cmap="gray", title=True):
    size = volume.shape[0]
    half_size = size // 2

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        show = True
    else:
        show = False

    # Y-Z plane
    axes[0].imshow(volume[:, :, half_size], cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        axes[0].set_title("Y-Z plane")

    # X-Z plane
    axes[1].imshow(volume[:, half_size, :], cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        axes[1].set_title("X-Z plane")

    # X-Y plane
    axes[2].imshow(volume[half_size, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        axes[2].set_title("X-Y plane")

    if show:
        plt.show()


def extract_region_withCH(data, coord_center, scale, size):
    if type(size) == int:
        size = [size, size, size]

    x, y, z = coord_center
    x = int(x / scale + 0.5)
    y = int(y / scale + 0.5)
    z = int(z / scale + 0.5)

    x_min = max(0, x - size[2] // 2)
    x_max = min(data.shape[3], x + size[2] // 2)
    y_min = max(0, y - size[1] // 2)
    y_max = min(data.shape[2], y + size[1] // 2)
    z_min = max(0, z - size[0] // 2)
    z_max = min(data.shape[1], z + size[0] // 2)

    if type(data) == np.ndarray:
        cropped = np.zeros((data.shape[0], size[0], size[1], size[2]))
    elif type(data) == torch.Tensor:
        cropped = torch.zeros((data.shape[0], size[0], size[1], size[2])).to(data.device)
    else:
        raise ValueError("data should be numpy.ndarray or torch.Tensor")
    cropped[
        :,
        size[0] // 2 - (z - z_min) : size[0] // 2 + (z_max - z),
        size[1] // 2 - (y - y_min) : size[1] // 2 + (y_max - y),
        size[2] // 2 - (x - x_min) : size[2] // 2 + (x_max - x),
    ] = data[:, z_min:z_max, y_min:y_max, x_min:x_max]

    return cropped
