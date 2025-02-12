# https://www.kaggle.com/code/hengck23/speed-up-connected-component-analysis-with-pytorch/notebook

import numpy as np
import torch
import torch.nn.functional as F

# https://github.com/kornia/kornia/blob/9ccae8c297a00a35d811b5a6e4f468a1d54d17f4/kornia/contrib/connected_components.py#L7
# https://stackoverflow.com/questions/46840707/efficiently-find-centroid-of-labelled-image-regions


@torch.no_grad()
def find_connected_component(probability, threshold, max_radius=10):
    device = probability.device
    probability = probability.detach().half()
    num_particle_type, D, H, W = probability.shape
    mask = probability > torch.tensor(threshold, device=device).reshape(num_particle_type, 1, 1, 1)

    # allocate the output tensors for labels
    out = (torch.arange(D * H * W, device=device, dtype=torch.float32) + 1).reshape(1, D, H, W)
    out = out.repeat(num_particle_type, 1, 1, 1)
    out[~mask] = 0

    out = out.reshape(num_particle_type, 1, D, H, W)
    mask = mask.reshape(num_particle_type, 1, D, H, W)
    for _ in range(max_radius):
        out = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)
        out = torch.mul(out, mask)  # mask using element-wise multiplication
    out = out.reshape(num_particle_type, D, H, W)
    out = out.long()
    component = []
    for i in range(num_particle_type):
        u, inverse = torch.unique(out[i], sorted=True, return_inverse=True)
        component.append(inverse)
    component = torch.stack(component)
    # plt.imshow(component[1].data.cpu().numpy().max(0))
    return component


@torch.no_grad()
def find_centroid(component):
    device = component.device
    num_particle_type, D, H, W = component.shape

    assert num_particle_type == 1, "Only support one particle type"

    count = component.flatten(1).max(-1)[0] + 1
    cumcount = torch.zeros(num_particle_type + 1, dtype=torch.int32, device=device)
    cumcount[1:] = torch.cumsum(count, 0)
    component = component + cumcount[:-1].reshape(num_particle_type, 1, 1, 1)

    gridz = torch.arange(0, D, device=device).reshape(1, D, 1, 1).expand(num_particle_type, -1, H, W)
    gridy = torch.arange(0, H, device=device).reshape(1, 1, H, 1).expand(num_particle_type, D, -1, W)
    gridx = torch.arange(0, W, device=device).reshape(1, 1, 1, W).expand(num_particle_type, D, H, -1)
    n = torch.bincount(component.flatten())
    nx = torch.bincount(component.flatten(), weights=gridx.flatten())
    ny = torch.bincount(component.flatten(), weights=gridy.flatten())
    nz = torch.bincount(component.flatten(), weights=gridz.flatten())

    x = nx / n
    y = ny / n
    z = nz / n
    xyz = torch.stack([x, y, z], 1).float()
    xyz = torch.split(xyz, count.tolist(), dim=0)
    centroid = [xxyyzz[1:] for xxyyzz in xyz]
    centroid = torch.stack(centroid).squeeze(0).cpu().numpy()
    return centroid
