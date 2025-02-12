# %%
import hydra
import rootutils
import joblib
import os
from lightning import LightningDataModule, LightningModule
from hydra import compose, initialize
from pathlib import Path
from glob import glob
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# os.chdir("/kaggle/working/src")

rootutils.setup_root(Path().resolve(), indicator=".project-root", pythonpath=True)

root = os.environ["PROJECT_ROOT"]

# %%
experiment = "241205-classmap_dice-pretrained_my_sim_241205_499_wHead-monai_unet_d32_512_res1_head1-s64_128-lr1e-3-bs8_1-ep300-transV1-preV1"

fp16_mode = False

output_root = "/workspace/logs/compiled_trt/"

batch_size_pred = 4
n_fold = 1

ckpt_type = "last"  # "last" or "best"

# %%
import copick
from monai.data import MetaTensor


def get_ckpt_name(ckpt_type, fold):
    if ckpt_type == "last":
        if fold == 0:
            ckpt_name = "last.ckpt"
        else:
            ckpt_name = f"last-v{fold}.ckpt"
    elif ckpt_type == "best":
        ckpt_name = f"fold{fold}_epoch_*.ckpt"
    else:
        assert False, f"unknown ckpt_type: {ckpt_type}"

    return ckpt_name


# %%
from copy import deepcopy
import torch_tensorrt

torch_tensorrt.runtime.set_multi_device_safe_mode(True)

print(f"experiment: {experiment}")

output_dir = os.path.join(output_root, experiment)
os.makedirs(output_dir, exist_ok=True)

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="train",
        overrides=[f"experiment={experiment}"],
        return_hydra_config=True,
    )
    cfg.paths.output_dir = "${hydra.runtime.output_dir}"
    cfg.paths.work_dir = "${hydra.runtime.cwd}"
    cfg.hydra.run.dir = cfg.log_dir
    cfg.hydra.runtime.output_dir = cfg.hydra.run.dir

if cfg.model.get("pretrained_ckpt_path"):
    cfg.model.pretrained_ckpt_path = None
if cfg.model.net.get("pretrained"):
    cfg.model.net.pretrained = False
if cfg.model.net.get("grad_checkpointing"):
    cfg.model.net.grad_checkpointing = False

ckpt_paths = []
for fold in range(n_fold):
    ckpt_name = get_ckpt_name(ckpt_type, fold)
    ckpt_path = os.path.join(cfg.paths.output_dir, "checkpoints", ckpt_name)
    ckpt_path = glob(ckpt_path)
    assert len(ckpt_path) == 1, f"ckpt_path: {ckpt_path}"
    ckpt_path = ckpt_path[0]
    ckpt_paths.append(ckpt_path)

# モデルの設定
model: LightningModule = hydra.utils.instantiate(cfg.model)
model.eval()
if fp16_mode:
    model.half()
    use_dtype = torch.float16
else:
    use_dtype = torch.float32

input_shape = [batch_size_pred, cfg.model.net.in_chans] + cfg.data.volume_size
inputs = [torch.randn(tuple(input_shape), dtype=use_dtype).cuda()]

for fold, ckpt_path in enumerate(ckpt_paths):
    model_fold = deepcopy(model)
    model_fold.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model_fold.to("cuda")

    # trt_ep is a torch.fx.GraphModule object
    print(f"compiling fold {fold}")
    trt_gm = torch_tensorrt.compile(
        model_fold,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=[use_dtype],
        # torch_executed_ops={"torch.ops.aten.convolution.default"}
    )
    save_path = os.path.join(
        output_dir, f"trt_{experiment}_fold{fold}_{ckpt_type}_bs{batch_size_pred}_fp16{fp16_mode}.ep"
    )
    torch_tensorrt.save(trt_gm, save_path, inputs=inputs)
    print(f"saved {save_path}")

# %%
