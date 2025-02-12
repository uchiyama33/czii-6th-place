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
from src.czii_utils.score import (
    calc_score2,
    do_one_eval,
    get_centroids_from_pred,
    calc_score_with_interpolation,
)
from src.czii_utils.utils_data import extract_region, visualize_particle
import pandas as pd
from src.czii_utils.czii_helper import dotdict

# %%
experiment_list = [
    "250101-particle_hard_masks_r0.5-focalTverskyPp-pretrained_241221_299-hengck23_tf_efficientnetv2_b2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
    "250102-hard_r0.5-focalTverskyPp-pretrained_241205_299-monai_unet_d32_512_res1_head1_bn-s64_128-lr1e-3-bs4_2_2-ep100-transV1-preV1",
    "250103-particle_hard_masks_r0.5-focalTverskyPp-hengck23_tf_efficientnetv2_b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV3-preV4",
    # "250104-hard_r05-focalTverskyPp-hengck23_convnext_nano_d64_256-s64_128-lr1e-3_decay02-bs4_2_2-ep100-transV1-preV2",
    # "250106-hard-focalTverskyPp-pretrained_241221_299-hengck23_v3_cn_nano_m2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4_1",
    # "250106-hard-focalTverskyPp-pretrained_241221_299-hengck23_v3_env2b2_m2_d64_256_scratch-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4_1",
    # "250109-hard-focalTverskyPp-pretrained_241221_299-hengck23_v3_cn_nano_m2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV3-preV4",
    # "250109-hard-focalTverskyPp-pretrained_241221_299-hengck23_v3_cn_nano_m2_d64_256-s64_128-lr1e-3_decay05e-bs4_2_2-ep100-transV3-preV4",
    # "250109-hard-focalTverskyPp-pretrained_241221_299-hengck23_v3_cn_nano_m2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
    # "250111-focalTverskyPp-pretrained_241221_299-monai_segresnet_f16_d1224-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
    # "250116-focalTverskyPp-hengck23_env2b2_d64_256-s64_128-2tomo-lr1e-3_decay05-bs4_2_2-ep50-transV3-preV4",
    # "250117-focalTverskyPp-pretrained_241221_299-hengck23_resnet34d_d64_256-s64_256-lr1e-3-bs4_2_2-ep50-transV4-preV4",
    # "250118-focalTverskyPp-hengck23_enb2_d64_256-disBA-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV4-preV4",
    # "250127-focalTverskyPp_09-hengck23_env2b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV1-preV2",
    # "250128-ga04-focalTverskyPp_09-hengck23_env2b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV1-preV2",
    # "250128-ga03-focalTverskyPp_09-hengck23_env2b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV1-preV2",
    # hard05
    # "250101-particle_hard_masks_r0.5-focalTverskyPp-pretrained_241221_299-hengck23_tf_efficientnetv2_b2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
    # "250103-particle_hard_masks_r0.5-focalTverskyPp-hengck23_tf_efficientnetv2_b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV3-preV4",
    "250109-hard-focalTverskyPp-pretrained_241221_299-hengck23_v3_cn_nano_m2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
    "250111-focalTverskyPp-pretrained_241221_299-monai_segresnet_f16_bn_d1224-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
    "250113-focalTverskyPp-hengck23_enb2_d64_256-s64_256-lr1e-3-bs4_2_2-ep100-transV3-preV4",
    "250116-focalTverskyPp-pretrained_241221_299-hengck23_resnet34d_d64_256-s64_256-lr1e-3-bs4_2_2-ep50-transV3-preV4",
    "250117-focalTverskyPp-pretrained_241221_299-hengck23_resnet34d_d64_256-s64_256-lr1e-3-bs4_2_2-ep80-transV4-preV4",
    "250118-focalTverskyPp-hengck23_env2b2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-mix_sim-ep100-transV4-preV4",
    "250118-focalTverskyPp-hengck23_enb2_d64_256-disBA-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV4-preV4",
    # "250120-focalTverskyPp-hengck23_env2s_d64_256-s64_128-mix_sim-lr1e-3_decay05-bs4_2_2-ep100-transV4-preV4",
    # ga04
    # "250129-ga04-focalTverskyPp_09-hengck23_enb2_d64_256-disBA-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV4-preV4",
    # "250129-ga04-focalTverskyPp_09-hengck23_enb2_d64_256-s64_256-lr1e-3-bs4_2_2-ep100-transV3-preV4",
    # "250129-ga04-focalTverskyPp_09-hengck23_env2b2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-mix_sim-ep100-transV4-preV4",
    # "250129-ga04-focalTverskyPp_09-hengck23_env2b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV3-preV4",
    # "250129-ga04-focalTverskyPp_09-pretrained_241221_299-hengck23_env2b2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
    # "250129-ga04-focalTverskyPp_09-pretrained_241221_299-hengck23_rn34d_d64_256-s64_256-lr1e-3-bs4_2_2-ep50-transV3-preV4",
    # "250129-ga04-focalTverskyPp_09-pretrained_241221_299-hengck23_rn34d_d64_256-s64_256-lr1e-3-bs4_2_2-ep80-transV4-preV4",
    # "250129-ga04-focalTverskyPp_09-pretrained_241221_299-hengck23_v3_cnnano_m2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
    # "250129-ga04-focalTverskyPp_09-pretrained_241221_299-monai_segresnet_f16_bn_d1224-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4",
]

n_fold = 5
overlap = 0.25  # [0.35, 0.45, 0.45]
discard_ratio = (0.08, 0.08, 0.08)  # None
thresh_low = 0.0
connectivity = None

type = "classmap"  # "seg" or "}classmap"

inference_dtype = torch.bfloat16

n_tta = 1

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
            maps_name=cfg.data.maps_name,
        )
        preprocessed_data[cfg.data.preprocess_version] = data

picks_coord_dict = {}
for data_dict in list(preprocessed_data.values())[0]:
    picks, particle_names = load_picks(cfg.data.copick_config_path, data_dict["id"])
    particle_coordinates = extract_particle_coordinates(picks)
    picks_coord_dict[data_dict["id"]] = particle_coordinates

# %%
T = 1

all_preds = []
for fold in range(n_fold):
    print(f"Fold {fold}")
    cnt = 0
    image = data[fold]["image"]
    if type == "seg":
        preds_tmp = torch.zeros(tuple([7] + list(image.shape[-3:]))).to("cuda")
    else:
        preds_tmp = torch.zeros(tuple([6] + list(image.shape[-3:]))).to("cuda")

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
                if discard_ratio is not None:
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


# %%
z_list = [40, 80, 120]
C = 6

fold = 4
preds = all_preds[fold].numpy()
if type == "seg":
    preds = preds[1:]
img = preprocessed_data[cfg.data.preprocess_version][fold]["image"].cpu().numpy()

plt.figure(figsize=(18, 8))
for i, z in enumerate(z_list):
    plt.subplot(len(z_list), C + 1, i * (C + 1) + 1)
    plt.imshow(img[0, z], cmap="gray")
    plt.axis("off")
    for c in range(C):
        plt.subplot(len(z_list), C + 1, i * (C + 1) + c + 2)
        plt.imshow(preds[c, z], cmap="viridis", vmin=0, vmax=1)
        plt.axis("off")

# %%
all_df_results = []
all_scores = []
all_hit_pred_lists = []
all_miss_truth_lists = []
all_fp_pred_lists = []
all_cc_lists = []
all_hit_value_lists = []
all_miss_value_lists = []
all_fp_value_lists = []

thresholds = np.arange(0.2, 1.0, 0.1)
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
        score, df_result, hit_pred_list, miss_truth_list, fp_pred_list, cc_list = (
            calc_score_with_interpolation(
                [all_preds[fold]],
                [data[fold]["id"]],
                picks_coord_dict,
                picks_radius_dict,
                particle_names,
                thresh=threshold,
                thresh_low=thresh_low,
                binning=None,
                max_pooling=False,
                connectivity=connectivity,
                n_erosion=n_erosion,
            )
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
        # # 各位置の予測値を取得
        # hit_value_fold[threshold] = {}
        # miss_value_fold[threshold] = {}
        # fp_value_fold[threshold] = {}
        # for i, particle_name in enumerate(particle_names):
        #     hit_pred = hit_pred_list[0][particle_name]
        #     if len(hit_pred) > 0:
        #         hit_pred = (hit_pred / 10.012444 + 0.5).astype(int)
        #         hit_value = all_preds[fold][i][hit_pred[:, 2], hit_pred[:, 1], hit_pred[:, 0]]
        #     else:
        #         hit_value = []
        #     miss_truth = miss_truth_list[0][particle_name]
        #     if len(miss_truth) > 0:
        #         miss_truth = (miss_truth / 10.012444 + 0.5).astype(int)
        #         miss_value = all_preds[fold][i][miss_truth[:, 2], miss_truth[:, 1], miss_truth[:, 0]]
        #     else:
        #         miss_value = []
        #     fp_pred = fp_pred_list[0][particle_name]
        #     if len(fp_pred) > 0:
        #         fp_pred = (fp_pred / 10.012444 + 0.5).astype(int)
        #         fp_value = all_preds[fold][i][fp_pred[:, 2], fp_pred[:, 1], fp_pred[:, 0]]
        #     else:
        #         fp_value = []

        #     hit_value_fold[threshold][particle_name] = hit_value
        #     miss_value_fold[threshold][particle_name] = miss_value
        #     fp_value_fold[threshold][particle_name] = fp_value

    ax = plt.subplot(n_rows, n_cols, fold + 1)
    combined_df = pd.concat(list(df_results_fold.values()))
    pivot_df = combined_df.pivot_table(index="threshold", columns="particle_type", values="f-beta4")
    pivot_df.plot(marker="o", ax=ax)
    ax.plot(thresholds, scores_fold.values(), marker="o", label="Score", color="black")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("F-beta4")
    ax.set_title(f"Fold {fold}")

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

plt.legend(title="Particle Type")

# %%
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


# %%
# recall, precision, f-betaをグラフにプロット
n_cols = 2
n_rows = (len(particle_names) + n_cols - 1) // n_cols
for i, particle_name in enumerate(particle_names):
    recall_list = []
    precision_list = []
    f_beta_list = []
    for threshold in thresholds:
        recall_list_threshold = []
        precision_list_threshold = []
        f_beta_list_threshold = []
        for fold in range(n_fold):
            df = all_df_results[fold][str(round(threshold, 1))]
            df = df[df["particle_type"] == particle_name]
            recall = df["recall"].values[0]
            precision = df["precision"].values[0]
            f_beta = df["f-beta4"].values[0]
            recall_list_threshold.append(recall)
            precision_list_threshold.append(precision)
            f_beta_list_threshold.append(f_beta)
        recall_list.append(np.mean(recall_list_threshold))
        precision_list.append(np.mean(precision_list_threshold))
        f_beta_list.append(np.mean(f_beta_list_threshold))

    plt.subplot(n_rows, n_cols, i + 1)
    plt.plot(thresholds, recall_list, marker="o", label="Recall")
    plt.plot(thresholds, precision_list, marker="o", label="Precision")
    plt.plot(thresholds, f_beta_list, marker="o", label="F-beta")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title(particle_name)
    if i == 0:
        plt.legend()


# %%
fold = 0
threshold = 0.3
particle_idx = 4
particle_name = particle_names[particle_idx]

preds = all_preds[fold][particle_idx].numpy()
threshold = str(round(threshold, 1))
cc = all_cc_lists[fold][threshold][particle_name]
maps = data[fold]["maps"][particle_idx].numpy()


# 3方向の断面を可視化
center = (100, 300, 300)
yz = preds[center[0]]
xz = preds[:, center[1]]
xy = preds[:, :, center[2]]

plt.figure(figsize=(10, 7))
plt.subplot(5, 2, (1, 5))
plt.imshow(yz, cmap="viridis", vmin=0, vmax=1)
plt.title("YZ plane")

plt.subplot(5, 2, 7)
plt.imshow(xz, cmap="viridis", vmin=0, vmax=1)
plt.title("XZ plane")

plt.subplot(5, 2, 9)
plt.imshow(xy, cmap="viridis", vmin=0, vmax=1)
plt.title("XY plane")

yz = maps[center[0]]
xz = maps[:, center[1]]
xy = maps[:, :, center[2]]

plt.subplot(5, 2, (2, 6))
plt.imshow(yz, cmap="viridis")  # , vmin=0, vmax=1)
plt.title("YZ plane")

plt.subplot(5, 2, 8)
plt.imshow(xz, cmap="viridis")  # , vmin=0, vmax=1)
plt.title("XZ plane")

plt.subplot(5, 2, 10)
plt.imshow(xy, cmap="viridis")  # , vmin=0, vmax=1)
plt.title("XY plane")


# %%
# miss_truth_list, fp_pred_listを可視化
fold = 0
particle_idx = 2
# threshold = "0.2"
threshold = best_thresholds[particle_idx]
name = particle_names[particle_idx]
data_id = data[fold]["id"]

print(f"Particle: {name}, Threshold: {threshold}")

image_size = (64, 64)
max_num = 10

hit_pred = all_hit_pred_lists[fold][threshold][name]
miss_truth = all_miss_truth_lists[fold][threshold][name]
fp_pred = all_fp_pred_lists[fold][threshold][name]
pred_map = all_preds[fold][particle_idx].numpy()
cc = all_cc_lists[fold][threshold][name]
image = data[fold]["image"].unsqueeze(0)

plt.figure(figsize=(18, 4))

# Visualize miss_truth
for i in range(min(max_num, len(miss_truth))):
    plt.subplot(2, max_num, i + 1)
    z = np.round(miss_truth[i][2] / 10).astype(int)
    y = np.round(miss_truth[i][1] / 10).astype(int)
    x = np.round(miss_truth[i][0] / 10).astype(int)

    img = (
        image[0, 0][
            z,
            max(0, y - image_size[0] // 2) : y + image_size[0] // 2,
            max(0, x - image_size[1] // 2) : x + image_size[1] // 2,
        ]
        .cpu()
        .numpy()
    )
    plt.imshow(img, cmap="gray")
    plt.axis("off")

# Visualize fp_pred
for i in range(min(max_num, len(fp_pred))):
    plt.subplot(2, max_num, i + max_num + 1)
    z = np.round(fp_pred[i][2] / 10).astype(int)
    y = np.round(fp_pred[i][1] / 10).astype(int)
    x = np.round(fp_pred[i][0] / 10).astype(int)

    img = (
        image[0, 0][
            z,
            max(0, y - image_size[0] // 2) : y + image_size[0] // 2,
            max(0, x - image_size[1] // 2) : x + image_size[1] // 2,
        ]
        .cpu()
        .numpy()
    )
    plt.imshow(img, cmap="gray")
    plt.axis("off")


# %% miss_trush
n = 3
center = (
    np.round(miss_truth[n][2] / 10).astype(int),
    np.round(miss_truth[n][1] / 10).astype(int),
    np.round(miss_truth[n][0] / 10).astype(int),
)
fig, axes = plt.subplots(4, 3, figsize=(5, 5))
cropped_volume = extract_region(image[0, 0], center)
visualize_particle(cropped_volume, axes=axes[0])

# maps (label)を可視化
cropped_volume = extract_region(data[fold]["maps"][particle_idx].numpy(), center)
visualize_particle(cropped_volume, vmin=0, vmax=1, cmap="viridis", axes=axes[1], title=False)

# predを可視化
cropped_volume = extract_region(pred_map, center)
visualize_particle(cropped_volume, vmin=0, vmax=1, cmap="viridis", axes=axes[2], title=False)

# ccを可視化
cropped_volume = extract_region(cc, center)
visualize_particle(cropped_volume, vmax=cropped_volume.max(), cmap="viridis", axes=axes[3], title=False)

# TODO ヒートマップが重なっているprediction mapを処理して分離させる

# %%
cc_high = all_cc_lists[fold]["0.6"][name]
cc_low = all_cc_lists[fold]["0.1"][name]

for i in range(cc_low.max()):
    cc_low_ = cc_low == i
    cc_high_in_low = cc_high[cc_low_]

    cc_high_in_low_not_zero = cc_high_in_low[cc_high_in_low != 0]
    cc_high_in_low_not_zero = np.unique(cc_high_in_low_not_zero)

    if len(cc_high_in_low_not_zero) > 1:
        print(i, cc_high_in_low_not_zero)

# %%
xyz_high, cc_high = get_centroids_from_pred(torch.from_numpy(pred_map), 0.6, 10.012444, return_cc=True)
xyz_low, cc_low = get_centroids_from_pred(torch.from_numpy(pred_map), 0.1, 10.012444, return_cc=True)

additional_xyz = []
for i in range(cc_low.max()):
    cc_low_ = cc_low == i
    cc_high_in_low = cc_high[cc_low_]

    cc_high_in_low_not_zero = cc_high_in_low[cc_high_in_low != 0]
    cc_high_in_low_not_zero = np.unique(cc_high_in_low_not_zero)

    if len(cc_high_in_low_not_zero) > 1:
        additional_xyz.append(xyz_high[cc_high_in_low_not_zero - 1].mean(0))

xyz2 = np.concatenate([xyz_high, additional_xyz])
xyz_truth = np.array(picks_coord_dict[data_id][name])

eval_df = []
hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_high, picks_radius_dict[name] * 0.5)
eval_df.append(
    dotdict(
        particle_type=name,
        P=metric[0],
        T=metric[1],
        hit=metric[2],
        miss=metric[3],
        fp=metric[4],
    )
)
hit, fp, miss, metric = do_one_eval(xyz_truth, xyz2, picks_radius_dict[name] * 0.5)
eval_df.append(
    dotdict(
        particle_type=name + "_additional",
        P=metric[0],
        T=metric[1],
        hit=metric[2],
        miss=metric[3],
        fp=metric[4],
    )
)

eval_df = pd.DataFrame(eval_df)
gb = eval_df.groupby("particle_type").agg("sum")
gb.loc[:, "precision"] = gb["hit"] / gb["P"]
gb.loc[:, "precision"] = gb["precision"].fillna(0)
gb.loc[:, "recall"] = gb["hit"] / gb["T"]
gb.loc[:, "recall"] = gb["recall"].fillna(0)
gb.loc[:, "f-beta4"] = 17 * gb["precision"] * gb["recall"] / (16 * gb["precision"] + gb["recall"])
gb.loc[:, "f-beta4"] = gb["f-beta4"].fillna(0)
gb
# %% fp_pred
n = 0
center = (
    np.round(fp_pred[n][2] / 10).astype(int),
    np.round(fp_pred[n][1] / 10).astype(int),
    np.round(fp_pred[n][0] / 10).astype(int),
)
fig, axes = plt.subplots(4, 3, figsize=(5, 5))
cropped_volume = extract_region(image[0, 0], center)
visualize_particle(cropped_volume, axes=axes[0])

# maps (label)を可視化
cropped_volume = extract_region(data[fold]["maps"][particle_idx].numpy(), center)
visualize_particle(cropped_volume, vmin=0, vmax=1, cmap="viridis", axes=axes[1], title=False)

# predを可視化
cropped_volume = extract_region(pred_map, center)
visualize_particle(cropped_volume, vmin=0, vmax=1, cmap="viridis", axes=axes[2], title=False)

# ccを可視化
cropped_volume = extract_region(cc, center)
visualize_particle(cropped_volume, vmin=0, vmax=1, cmap="viridis", axes=axes[3], title=False)


# %% hit_pred
n = 0
center = (
    np.round(hit_pred[n][2] / 10).astype(int),
    np.round(hit_pred[n][1] / 10).astype(int),
    np.round(hit_pred[n][0] / 10).astype(int),
)
fig, axes = plt.subplots(4, 3, figsize=(5, 5))
cropped_volume = extract_region(image[0, 0], center)
visualize_particle(cropped_volume, axes=axes[0])

# maps (label)を可視化
cropped_volume = extract_region(data[fold]["maps"][particle_idx].numpy(), center)
visualize_particle(cropped_volume, vmin=0, vmax=1, cmap="viridis", axes=axes[1], title=False)

# predを可視化
cropped_volume = extract_region(pred_map, center)
visualize_particle(cropped_volume, vmin=0, vmax=1, cmap="viridis", axes=axes[2], title=False)

# ccを可視化
cropped_volume = extract_region(cc, center)
visualize_particle(cropped_volume, vmin=0, vmax=1, cmap="viridis", axes=axes[3], title=False)


# %%
# スライスを１枚ずつ可視化 image
center = (
    np.round(miss_truth[n][2] / 10).astype(int),
    np.round(miss_truth[n][1] / 10).astype(int),
    np.round(miss_truth[n][0] / 10).astype(int),
)
cropped_volume = extract_region(image[0, 0], center)
size = cropped_volume.shape[0]
n_cols = 4

plt.figure(figsize=(6, 18))
for i in range(size):
    plt.subplot(size // n_cols, n_cols, i + 1)
    plt.imshow(cropped_volume[i], cmap="gray")
    plt.axis("off")
    plt.title(f"z={i-size//2}")

# %%
# スライスを１枚ずつ可視化  pred
center = (
    np.round(miss_truth[n][2] / 10).astype(int),
    np.round(miss_truth[n][1] / 10).astype(int),
    np.round(miss_truth[n][0] / 10).astype(int),
)
cropped_volume = extract_region(pred_map, center)
size = cropped_volume.shape[0]
n_cols = 4

plt.figure(figsize=(6, 18))
for i in range(size):
    plt.subplot(size // n_cols, n_cols, i + 1)
    plt.imshow(cropped_volume[i], vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"z={i-size//2}")


# %%
# スライスを１枚ずつ可視化  cc
center = (
    np.round(miss_truth[n][2] / 10).astype(int),
    np.round(miss_truth[n][1] / 10).astype(int),
    np.round(miss_truth[n][0] / 10).astype(int),
)
cropped_volume = extract_region(cc, center)
size = cropped_volume.shape[0]
n_cols = 4

plt.figure(figsize=(6, 18))
for i in range(size):
    plt.subplot(size // n_cols, n_cols, i + 1)
    plt.imshow(cropped_volume[i], vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"z={i-size//2}")

# %% fp_pred
n = 1
center = (
    np.round(fp_pred[n][2] / 10).astype(int),
    np.round(fp_pred[n][1] / 10).astype(int),
    np.round(fp_pred[n][0] / 10).astype(int),
)
cropped_volume = extract_region(image[0, 0], center)
visualize_particle(cropped_volume)

# %%
# スライスを１枚ずつ可視化
center = (
    np.round(fp_pred[n][2] / 10).astype(int),
    np.round(fp_pred[n][1] / 10).astype(int),
    np.round(fp_pred[n][0] / 10).astype(int),
)
cropped_volume = extract_region(image[0, 0], center)
size = cropped_volume.shape[0]
n_cols = 4

plt.figure(figsize=(6, 18))
for i in range(size):
    plt.subplot(size // n_cols, n_cols, i + 1)
    plt.imshow(cropped_volume[i], cmap="gray")
    plt.axis("off")
    plt.title(f"z={i-size//2}")

# %%
