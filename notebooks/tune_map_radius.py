# %%
import torch
import hydra
from hydra import compose, initialize
import os
import rootutils
from pathlib import Path
from monai.inferers import sliding_window_inference
import gc
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, ball

rootutils.setup_root(Path().resolve(), indicator=".project-root", pythonpath=True)

from src.data.components.czii_data_utils import (
    load_data,
    load_picks,
    extract_particle_coordinates,
    crop_center,
    split_train_val,
)
from src.czii_utils.inference_v2 import sliding_window_inference_edge_discard
from src.data.components.transforms import get_transforms
from scripts.submit import get_ckpt_name, ModelWrapper
from scripts.submit_all_data import tta_transform, inverse_tta_transform
from src.czii_utils.score import calc_score2
import pandas as pd

# %%
experiment_list = [
    "250128-ga03-focalTverskyPp_09-hengck23_env2b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV1-preV2",
    "250128-ga04-focalTverskyPp_09-hengck23_env2b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV1-preV2",
    "250128-ga05-focalTverskyPp_09-hengck23_env2b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV1-preV2",
    "250128-ga06-focalTverskyPp_09-hengck23_env2b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV1-preV2",
    # "250128-ga03-focalTverskyPp_09-hengck23_enb2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV4-preV4",
    # "250128-ga04-focalTverskyPp_09-hengck23_enb2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV4-preV4",
    # "250128-ga05-focalTverskyPp_09-hengck23_enb2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV4-preV4",
    # "250128-ga06-focalTverskyPp_09-hengck23_enb2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV4-preV4",
]

experiment_radius_list = [
    0.3,
    0.4,
    0.5,
    0.6,
    # 0.3,
    # 0.4,
    # 0.5,
    # 0.6,
]

n_fold = 7
overlap = 0.4
discard_ratio = 0.1

type = "classmap"  # "seg" or "}classmap"

inference_dtype = torch.bfloat16

# tta_list = [0]  # np.arange(0, 11)
n_tta = 3

ckpt_type = "last"  # "last", "best" or "epoch"
ckpt_epoch = None  # only used when ckpt_type == "epoch"

pred_size = None  # None
pred_bs = None  # None
ema = False
ensemble = "mean"  # "mean" or "max"
n_erosion = False

# %%
preprocessed_data = {}
for experiment in experiment_list:
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

    if cfg.data.preprocess_version not in preprocessed_data.keys():
        transforms_common, _, _, _ = get_transforms(
            cfg.data.transforms_version,
            cfg.data.num_classes + 1,
            cfg.data.volume_size,
            cfg.data.num_crop_samples,
        )
        data, picks_radius_dict = load_data(
            cfg.data.copick_config_path,
            cfg.data.num_classes + 1,
            transform=transforms_common,
            preprocess_version=cfg.data.preprocess_version,
            maps_name="gaussian_maps_r0.5",
        )
        preprocessed_data[cfg.data.preprocess_version] = data

picks_coord_dict = {}
for data_dict in list(preprocessed_data.values())[0]:
    picks, particle_names = load_picks(cfg.data.copick_config_path, data_dict["id"])
    particle_coordinates = extract_particle_coordinates(picks)
    picks_coord_dict[data_dict["id"]] = particle_coordinates

# %%
df_scores = pd.DataFrame()

# %%
T = 1

for experiment, radius in zip(experiment_list, experiment_radius_list):
    all_preds = []
    for fold in range(n_fold):
        print(f"Fold {fold}")
        cnt = 0
        image = data[fold]["image"]
        if type == "seg":
            preds_tmp = torch.zeros(tuple([7] + list(image.shape[-3:]))).to("cuda")
        else:
            preds_tmp = torch.zeros(tuple([6] + list(image.shape[-3:]))).to("cuda")

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

        image = preprocessed_data[cfg.data.preprocess_version][fold]["image"].to("cuda")

        model = hydra.utils.instantiate(cfg.model)
        model.eval().to("cuda")

        ckpt_name = get_ckpt_name(ckpt_type, fold, ckpt_epoch)
        ckpt_path = os.path.join(cfg.log_dir, "checkpoints", ckpt_name)
        ckpt_path = glob(ckpt_path)
        assert len(ckpt_path) == 1, f"ckpt_path: {ckpt_path}"
        ckpt_path = ckpt_path[0]
        state_dict = torch.load(ckpt_path)["state_dict"]
        if cfg.model.compile:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        if pred_size is not None:
            _pred_size = pred_size
        else:
            _pred_size = tuple(cfg.data.volume_size)

        if pred_bs is not None:
            _pred_bs = pred_bs
        else:
            _pred_bs = 4

        if ema:
            net = model.net_ema
        else:
            net = model.net

        with torch.autocast("cuda", dtype=inference_dtype), torch.no_grad(), torch.inference_mode():
            for _ in range(n_tta):
                input = tta_transform(image.as_tensor(), cnt).unsqueeze(0)
                if discard_ratio > 0:
                    outputs = sliding_window_inference_edge_discard(
                        inputs=input,
                        roi_size=_pred_size,
                        sw_batch_size=_pred_bs,  # one window is proecessed at a time
                        predictor=ModelWrapper(net, _pred_bs),
                        overlap=overlap,
                        discard_ratio=discard_ratio,
                    )["classmap"]
                else:
                    outputs = sliding_window_inference(
                        inputs=input,
                        roi_size=_pred_size,
                        sw_batch_size=_pred_bs,  # one window is proecessed at a time
                        predictor=ModelWrapper(net, _pred_bs),
                        overlap=overlap,
                    )["classmap"]
                if type == "seg":
                    preds = outputs.softmax(1)[0]
                else:
                    preds = (outputs / T).sigmoid()[0]

                preds = inverse_tta_transform(preds, cnt)

                if ensemble == "mean":
                    preds_tmp += preds
                elif ensemble == "max":
                    preds_tmp = torch.max(preds_tmp, preds)
                cnt += 1

            del model, net, state_dict, outputs, preds
            gc.collect()
            torch.cuda.empty_cache()

        if ensemble == "mean":
            preds_fold_mean = preds_tmp / cnt
        elif ensemble == "max":
            preds_fold_mean = preds_tmp

        all_preds.append(preds_fold_mean.cpu())

    all_df_results = []
    all_scores = []
    all_hit_pred_lists = []
    all_miss_truth_lists = []
    all_fp_pred_lists = []
    all_cc_lists = []
    all_hit_value_lists = []
    all_miss_value_lists = []
    all_fp_value_lists = []

    thresholds = np.arange(0.1, 1.0, 0.1)
    n_cols = 2
    n_rows = (n_fold + n_cols - 1) // n_cols

    plt.figure(figsize=(18, 6 * n_rows))

    for fold in range(n_fold):
        print(f"Fold {fold}")
        df_results_fold = {}
        scores_fold = {}
        hit_pred_fold = {}
        miss_truth_fold = {}
        fp_pred_fold = {}
        cc_fold = {}
        hit_value_fold = {}
        miss_value_fold = {}
        fp_value_fold = {}
        for threshold in thresholds:
            score, df_result, hit_pred_list, miss_truth_list, fp_pred_list, cc_list = calc_score2(
                [all_preds[fold]],
                [data[fold]["id"]],
                picks_coord_dict,
                picks_radius_dict,
                particle_names,
                thresh=threshold,
                binning=None,
                max_pooling=False,
                n_erosion=n_erosion,
            )
            print(f"threshold: {threshold:.1f}, score: {score:.4f}")

            threshold = str(round(threshold, 1))
            df_result["threshold"] = threshold

            # Save the lists for each threshold
            hit_pred_fold[threshold] = hit_pred_list[0]
            miss_truth_fold[threshold] = miss_truth_list[0]
            fp_pred_fold[threshold] = fp_pred_list[0]
            cc_fold[threshold] = cc_list[0]

            # Save df_result and score with threshold as key
            df_results_fold[threshold] = df_result
            scores_fold[threshold] = score

            # 各位置の予測値を取得
            hit_value_fold[threshold] = {}
            miss_value_fold[threshold] = {}
            fp_value_fold[threshold] = {}
            for i, particle_name in enumerate(particle_names):
                hit_pred = hit_pred_list[0][particle_name]
                if len(hit_pred) > 0:
                    hit_pred = (hit_pred / 10.012444 + 0.5).astype(int)
                    hit_value = all_preds[fold][i][hit_pred[:, 2], hit_pred[:, 1], hit_pred[:, 0]]
                else:
                    hit_value = []
                miss_truth = miss_truth_list[0][particle_name]
                if len(miss_truth) > 0:
                    miss_truth = (miss_truth / 10.012444 + 0.5).astype(int)
                    miss_value = all_preds[fold][i][miss_truth[:, 2], miss_truth[:, 1], miss_truth[:, 0]]
                else:
                    miss_value = []
                fp_pred = fp_pred_list[0][particle_name]
                if len(fp_pred) > 0:
                    fp_pred = (fp_pred / 10.012444 + 0.5).astype(int)
                    fp_value = all_preds[fold][i][fp_pred[:, 2], fp_pred[:, 1], fp_pred[:, 0]]
                else:
                    fp_value = []

                hit_value_fold[threshold][particle_name] = hit_value
                miss_value_fold[threshold][particle_name] = miss_value
                fp_value_fold[threshold][particle_name] = fp_value

        all_df_results.append(df_results_fold)
        all_scores.append(scores_fold)
        all_hit_pred_lists.append(hit_pred_fold)
        all_miss_truth_lists.append(miss_truth_fold)
        all_fp_pred_lists.append(fp_pred_fold)
        all_cc_lists.append(cc_fold)
        all_hit_value_lists.append(hit_value_fold)
        all_miss_value_lists.append(miss_value_fold)
        all_fp_value_lists.append(fp_value_fold)

        print("-" * 10)

    # 粒子ごとに最適なthresholdを探す
    particle_names = list(picks_coord_dict[list(picks_coord_dict.keys())[0]].keys())
    best_thresholds = []
    best_f_beta_list = []
    for particle_name in particle_names:
        print(f"Particle: {particle_name}")
        best_f_beta = 0
        for thresh in thresholds:
            thresh = str(round(thresh, 1))
            f_beta_list = []
            for fold in range(n_fold):
                score = (
                    all_df_results[fold][thresh]
                    .loc[all_df_results[fold][thresh]["particle_type"] == particle_name, "f-beta4"]
                    .values[0]
                )
                f_beta_list.append(score)

            f_beta = np.mean(f_beta_list)
            # print(f"thresh={thresh}: f-beta {f_beta:.4f}")

            if f_beta > best_f_beta:
                best_f_beta = f_beta
                best_thresh = thresh

        print(f"Best threshold: {best_thresh}, Best f-beta: {best_f_beta:.4f}")
        best_thresholds.append(best_thresh)
        best_f_beta_list.append(best_f_beta)

    score = (np.array(best_f_beta_list) * np.array([1, 0, 2, 1, 2, 1])).sum() / 7
    print(f"Tuning score: {score:.4f}")

    score_dict = {}
    score_dict["experiment"] = experiment
    score_dict["radius"] = radius
    score_dict["tuning_score"] = score
    for i, particle_name in enumerate(particle_names):
        score_dict[f"{particle_name}_fbeta"] = best_f_beta_list[i]
        score_dict[f"{particle_name}_threshold"] = best_thresholds[i]

    df_scores = pd.concat([df_scores, pd.DataFrame([score_dict])])


# %%
# 横軸radius, 縦軸best_f_betaのグラフをparticleごとに描画
plt.figure(figsize=(18, 6 * len(particle_names)))

for i, particle_name in enumerate(particle_names):
    plt.subplot(len(particle_names), 1, i + 1)
    plt.plot(df_scores["radius"], df_scores[f"{particle_name}_fbeta"], marker="o")
    plt.xlabel("Radius")
    plt.ylabel("Best f-beta")
    plt.title(particle_name)
