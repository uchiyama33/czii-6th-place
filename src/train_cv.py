# %%
import os
import numpy as np
import subprocess
import wandb
from omegaconf import OmegaConf
from hydra import initialize, compose
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.submit import get_ckpt_name

# %%

experiment_list = [
    "250101-particle_hard_masks_r0.5-focalTverskyPp-pretrained_241221_299-hengck23_tf_efficientnetv2_b2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
    "250102-hard_r0.5-focalTverskyPp-pretrained_241205_299-monai_unet_d32_512_res1_head1_bn-s64_128-lr1e-3-bs4_2_2-ep100-transV1-preV1",
    "250103-particle_hard_masks_r0.5-focalTverskyPp-hengck23_tf_efficientnetv2_b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV3-preV4",
    "250109-hard-focalTverskyPp-pretrained_241221_299-hengck23_v3_cn_nano_m2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
    "250111-focalTverskyPp-pretrained_241221_299-monai_segresnet_f16_bn_d1224-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
    "250113-focalTverskyPp-hengck23_enb2_d64_256-s64_256-lr1e-3-bs4_2_2-ep100-transV3-preV4",
    "250116-focalTverskyPp-pretrained_241221_299-hengck23_resnet34d_d64_256-s64_256-lr1e-3-bs4_2_2-ep50-transV3-preV4",
    "250117-focalTverskyPp-pretrained_241221_299-hengck23_resnet34d_d64_256-s64_256-lr1e-3-bs4_2_2-ep80-transV4-preV4",
    "250118-focalTverskyPp-hengck23_env2b2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-mix_sim-ep100-transV4-preV4",
    "250118-focalTverskyPp-hengck23_enb2_d64_256-disBA-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV4-preV4",
]

for experiment in experiment_list:
    if "classification" in experiment:
        n_fold = 7
    else:
        n_fold = 5

    resume_fold = []
    except_fold = []

    for fold in range(n_fold):
        if fold in except_fold:
            continue

        command = [
            "python",
            "/workspace/src/train.py",
            f"experiment={experiment}",
            f"data.fold={fold}",
            f"seed={fold}",
        ]
        if fold in resume_fold:
            ckpt_name = get_ckpt_name("last", fold)
            ckpt_path = os.path.join("/workspace/logs/train/runs", experiment, "checkpoints", ckpt_name)
            assert os.path.exists(ckpt_path), f"{ckpt_path} does not exist"
            command.append(f"ckpt_path={ckpt_path}")

        print(" ".join(command))
        subprocess.run(command)

    # metric_valueを取得して、resultsに格納
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

    save_dir = os.path.join(cfg.log_dir, "metric_values")

    result_dict = {}
    for fold in range(n_fold):
        with open(os.path.join(save_dir, f"metric_value_{fold}.txt"), "r") as f:
            result = float(f.read())
            result_dict[f"fold_{fold}"] = result

    result_dict["cv_score"] = np.array([result_dict[f"fold_{fold}"] for fold in range(n_fold)]).mean()

    if "classification" in experiment:
        project = "CZII_CV_classification"
    else:
        project = "CZII_CV"

    run = wandb.init(
        project=project,
        name=f"{cfg.experiment_name}",
        dir="/workspace/logs/wandb_cv",
        config=OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False),
    )
    run.log(result_dict)

    run.finish()

# %%
