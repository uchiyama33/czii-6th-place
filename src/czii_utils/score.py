from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cc3d
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.czii_utils.czii_helper import dotdict


@torch.no_grad()
def nms_3d_from_heatmap(heatmap, threshold=0.5, kernel_size=3, voxel_spacing=10):
    """
    3D ヒートマップに対して Non-Maximum Suppression を適用し、ピークを抽出する。

    Args:
        heatmap: Tensor, shape (D, H, W), 0~1 の値を持つ 3D ヒートマップ。
        threshold: float, ピークとして認識する最小値。
        kernel_size: int, NMS に使用する最大値プールのカーネルサイズ。

    Returns:
        peaks: List[Tuple[int, int, int, float]], ピークの座標と値のリスト。
    """
    # heatmapを21段階でビニング
    bins = 21
    heatmap = (heatmap * bins).to(torch.long)
    # ビンの値を0から10に制限
    heatmap = torch.clamp(heatmap, min=0, max=bins - 1) / (bins - 1)

    # 最大値プールで各ボクセルがローカル最大かを判定
    padding = kernel_size // 2
    pooled = (
        F.max_pool3d(
            heatmap.unsqueeze(0).unsqueeze(0),  # (1, 1, D, H, W)
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        .squeeze(0)
        .squeeze(0)
    )  # (D, H, W)

    # ローカル最大かつ閾値を超えた位置を抽出
    is_peak = (heatmap == pooled) & (heatmap >= threshold)
    # peak_coords = torch.nonzero(is_peak, as_tuple=False)  # (N, 3)

    # get_centroids
    cc, P = cc3d.connected_components(is_peak.int().cpu().numpy(), return_N=True)
    stats = cc3d.statistics(cc)
    peak_coords = torch.tensor(stats["centroids"][1:], dtype=torch.float32)

    # zyx -> xyz
    peak_coords = peak_coords.flip(1)
    peak_coords = peak_coords * voxel_spacing
    return peak_coords.cpu().numpy()


def get_centroids_from_segmentation(segmentation, threshold, voxel_spacing, connectivity=26):
    assert connectivity in [6, 18, 26]
    cc = cc3d.connected_components((segmentation > threshold).cpu().numpy(), connectivity=connectivity)
    stats = cc3d.statistics(cc)
    zyx = stats["centroids"][1:] * voxel_spacing
    xyz = np.ascontiguousarray(zyx[:, ::-1])
    return xyz


@torch.no_grad()
def get_centroids_from_pred(
    pred: torch.Tensor,
    threshold,
    voxel_spacing,
    binning=None,
    max_pooling=False,
    return_cc=False,
    connectivity=None,
    n_erosion=0,
):
    assert connectivity in [6, 18, 26, None]
    if binning is not None:
        pred = (pred * binning).to(torch.long)
        pred = torch.clamp(pred, min=0, max=binning - 1) / (binning - 1)
    if max_pooling:
        pooled = (
            F.max_pool3d(pred.unsqueeze(0).unsqueeze(0).cuda(), kernel_size=3, stride=1, padding=1)
            .squeeze(0)
            .squeeze(0)
        ).cpu()
        pred = (pred == pooled) & (pred >= threshold)
    else:
        pred = pred >= threshold
    if n_erosion > 0:
        pred = -(pred.float()).unsqueeze(0).unsqueeze(0).cuda()
        for _ in range(n_erosion):
            pred = F.max_pool3d(pred, kernel_size=3, stride=1, padding=1)
        pred = (-pred).bool().squeeze(0).squeeze(0).cpu()

    if connectivity is None:
        cc = cc3d.connected_components(pred.int().cpu().numpy())
    else:
        cc = cc3d.connected_components(pred.int().cpu().numpy(), connectivity=connectivity)
    stats = cc3d.statistics(cc)
    zyx = stats["centroids"][1:] * voxel_spacing
    xyz = np.ascontiguousarray(zyx[:, ::-1])
    if return_cc:
        return xyz, cc
    else:
        return xyz


def calc_score(
    preds,
    type,
    ids,
    picks_coord_dict,
    picks_radius_dict,
    particle_names,
    thresh=0.1,
    binning=None,
    max_pooling=False,
    return_df=False,
):
    assert type in ["classmap", "seg"]
    voxel_spacing = 10

    # threshが単一の場合はリストに変換
    if not isinstance(thresh, list):
        thresh = [thresh] * len(particle_names)
    assert len(thresh) == len(particle_names)

    eval_df = []
    for pred, id, th in zip(preds, ids, thresh):
        if len(pred) == len(particle_names) + 1:
            pred = pred[1:]

        particle_coordinates = picks_coord_dict[id]
        for i, name in enumerate(particle_names):
            xyz_truth = np.array(particle_coordinates[name])
            xyz_predict = get_centroids_from_pred(pred[i], th, voxel_spacing, binning, max_pooling)

            hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, picks_radius_dict[name] * 0.5)
            eval_df.append(
                dotdict(
                    id=id,
                    particle_type=name,
                    P=metric[0],
                    T=metric[1],
                    hit=metric[2],
                    miss=metric[3],
                    fp=metric[4],
                )
            )

    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby("particle_type").agg("sum").drop(columns=["id"])
    gb.loc[:, "precision"] = gb["hit"] / gb["P"]
    gb.loc[:, "precision"] = gb["precision"].fillna(0)
    gb.loc[:, "recall"] = gb["hit"] / gb["T"]
    gb.loc[:, "recall"] = gb["recall"].fillna(0)
    gb.loc[:, "f-beta4"] = 17 * gb["precision"] * gb["recall"] / (16 * gb["precision"] + gb["recall"])
    gb.loc[:, "f-beta4"] = gb["f-beta4"].fillna(0)

    gb = gb.sort_values("particle_type").reset_index(drop=False)
    # https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895
    gb.loc[:, "weight"] = [1, 0, 2, 1, 2, 1]
    lb_score = (gb["f-beta4"] * gb["weight"]).sum() / gb["weight"].sum()

    if return_df:
        return lb_score, gb
    else:
        return lb_score


def calc_score2(
    preds,
    ids,
    picks_coord_dict,
    picks_radius_dict,
    particle_names,
    thresh=0.1,
    binning=21,
    max_pooling=False,
    remove_corner_pred=False,
    connectivity=None,
    n_erosion=False,
):
    voxel_spacing = 10

    # threshが単一の場合はリストに変換
    if not isinstance(thresh, list):
        thresh = [thresh] * len(particle_names)
    assert len(thresh) == len(particle_names)

    eval_df = []
    hit_pred_list = []
    miss_truth_list = []
    fp_pred_list = []
    cc_list = []
    for pred, id in zip(preds, ids):
        if len(pred) == len(particle_names) + 1:
            pred = pred[1:]

        _hit_pred_dict = {}
        _miss_truth_dict = {}
        _fp_pred_dict = {}
        _cc_dict = {}

        particle_coordinates = picks_coord_dict[id]
        for i, (name, th) in enumerate(zip(particle_names, thresh)):
            xyz_truth = np.array(particle_coordinates[name])
            xyz_predict, cc = get_centroids_from_pred(
                pred[i],
                th,
                voxel_spacing,
                binning,
                max_pooling,
                return_cc=True,
                connectivity=connectivity,
                n_erosion=n_erosion,
            )
            if remove_corner_pred:
                xyz_predict = xyz_predict[
                    np.all(xyz_predict[:, [0, 1]] >= picks_radius_dict[name] * 0.2, axis=1)
                ]

            hit, fp, miss, metric, hit_pred, miss_truth, fp_pred = do_one_eval(
                xyz_truth, xyz_predict, picks_radius_dict[name] * 0.5, return_points=True
            )
            eval_df.append(
                dotdict(
                    id=id,
                    particle_type=name,
                    P=metric[0],
                    T=metric[1],
                    hit=metric[2],
                    miss=metric[3],
                    fp=metric[4],
                )
            )
            _hit_pred_dict[name] = hit_pred
            _miss_truth_dict[name] = miss_truth
            _fp_pred_dict[name] = fp_pred
            _cc_dict[name] = cc

        hit_pred_list.append(_hit_pred_dict)
        miss_truth_list.append(_miss_truth_dict)
        fp_pred_list.append(_fp_pred_dict)
        cc_list.append(_cc_dict)

    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby("particle_type").agg("sum").drop(columns=["id"])
    gb.loc[:, "precision"] = gb["hit"] / gb["P"]
    gb.loc[:, "precision"] = gb["precision"].fillna(0)
    gb.loc[:, "recall"] = gb["hit"] / gb["T"]
    gb.loc[:, "recall"] = gb["recall"].fillna(0)
    gb.loc[:, "f-beta4"] = 17 * gb["precision"] * gb["recall"] / (16 * gb["precision"] + gb["recall"])
    gb.loc[:, "f-beta4"] = gb["f-beta4"].fillna(0)

    gb = gb.sort_values("particle_type").reset_index(drop=False)
    # https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895
    gb.loc[:, "weight"] = [1, 0, 2, 1, 2, 1]
    lb_score = (gb["f-beta4"] * gb["weight"]).sum() / gb["weight"].sum()

    return lb_score, gb, hit_pred_list, miss_truth_list, fp_pred_list, cc_list


def calc_score_with_interpolation(
    preds,
    ids,
    picks_coord_dict,
    picks_radius_dict,
    particle_names,
    thresh=0.5,
    thresh_low=0.1,
    binning=21,
    max_pooling=False,
    connectivity=None,
    n_erosion=False,
):
    voxel_spacing = 10

    # threshが単一の場合はリストに変換
    if not isinstance(thresh, list):
        thresh = [thresh] * len(particle_names)
    assert len(thresh) == len(particle_names)

    eval_df = []
    hit_pred_list = []
    miss_truth_list = []
    fp_pred_list = []
    cc_list = []
    for pred, id in zip(preds, ids):
        if len(pred) == len(particle_names) + 1:
            pred = pred[1:]

        _hit_pred_dict = {}
        _miss_truth_dict = {}
        _fp_pred_dict = {}
        _cc_dict = {}

        particle_coordinates = picks_coord_dict[id]
        for i, (name, th) in enumerate(zip(particle_names, thresh)):
            xyz_truth = np.array(particle_coordinates[name])
            xyz_predict_high, cc_high = get_centroids_from_pred(
                pred[i],
                th,
                voxel_spacing,
                binning,
                max_pooling,
                return_cc=True,
                connectivity=connectivity,
                n_erosion=n_erosion,
            )
            if thresh_low == 0:
                xyz_predict = xyz_predict_high
            else:
                xyz_predict_low, cc_low = get_centroids_from_pred(
                    pred[i],
                    thresh_low,
                    voxel_spacing,
                    binning,
                    max_pooling,
                    return_cc=True,
                    connectivity=connectivity,
                    n_erosion=n_erosion,
                )

                additional_xyz = []
                for i in range(cc_low.max()):
                    cc_low_ = cc_low == i
                    cc_high_in_low = cc_high[cc_low_]

                    cc_high_in_low_not_zero = cc_high_in_low[cc_high_in_low != 0]
                    cc_high_in_low_not_zero = np.unique(cc_high_in_low_not_zero)

                    if len(cc_high_in_low_not_zero) > 1:
                        additional_xyz.append(xyz_predict_high[cc_high_in_low_not_zero - 1].mean(0))

                if len(additional_xyz) > 0:
                    additional_xyz = np.array(additional_xyz)
                    xyz_predict = np.concatenate([xyz_predict_high, additional_xyz])
                else:
                    xyz_predict = xyz_predict_high

            hit, fp, miss, metric, hit_pred, miss_truth, fp_pred = do_one_eval(
                xyz_truth, xyz_predict, picks_radius_dict[name] * 0.5, return_points=True
            )
            eval_df.append(
                dotdict(
                    id=id,
                    particle_type=name,
                    P=metric[0],
                    T=metric[1],
                    hit=metric[2],
                    miss=metric[3],
                    fp=metric[4],
                )
            )
            _hit_pred_dict[name] = hit_pred
            _miss_truth_dict[name] = miss_truth
            _fp_pred_dict[name] = fp_pred
            _cc_dict[name] = cc_high

        hit_pred_list.append(_hit_pred_dict)
        miss_truth_list.append(_miss_truth_dict)
        fp_pred_list.append(_fp_pred_dict)
        cc_list.append(_cc_dict)

    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby("particle_type").agg("sum").drop(columns=["id"])
    gb.loc[:, "precision"] = gb["hit"] / gb["P"]
    gb.loc[:, "precision"] = gb["precision"].fillna(0)
    gb.loc[:, "recall"] = gb["hit"] / gb["T"]
    gb.loc[:, "recall"] = gb["recall"].fillna(0)
    gb.loc[:, "f-beta4"] = 17 * gb["precision"] * gb["recall"] / (16 * gb["precision"] + gb["recall"])
    gb.loc[:, "f-beta4"] = gb["f-beta4"].fillna(0)

    gb = gb.sort_values("particle_type").reset_index(drop=False)
    # https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895
    gb.loc[:, "weight"] = [1, 0, 2, 1, 2, 1]
    lb_score = (gb["f-beta4"] * gb["weight"]).sum() / gb["weight"].sum()

    return lb_score, gb, hit_pred_list, miss_truth_list, fp_pred_list, cc_list


def calc_score_with_z_slide(
    preds,
    ids,
    picks_coord_dict,
    picks_radius_dict,
    particle_names,
    thresh=0.5,
    binning=21,
    max_pooling=False,
    connectivity=None,
    n_erosion=False,
):
    voxel_spacing = 10

    # threshが単一の場合はリストに変換
    if not isinstance(thresh, list):
        thresh = [thresh] * len(particle_names)
    assert len(thresh) == len(particle_names)

    eval_df = []
    hit_pred_list = []
    miss_truth_list = []
    fp_pred_list = []
    cc_list = []
    for pred, id in zip(preds, ids):
        if len(pred) == len(particle_names) + 1:
            pred = pred[1:]

        _hit_pred_dict = {}
        _miss_truth_dict = {}
        _fp_pred_dict = {}
        _cc_dict = {}

        particle_coordinates = picks_coord_dict[id]
        for i, (name, th) in enumerate(zip(particle_names, thresh)):
            xyz_truth = np.array(particle_coordinates[name])
            xyz_predict, cc = get_centroids_from_pred(
                pred[i],
                th,
                voxel_spacing,
                binning,
                max_pooling,
                return_cc=True,
                connectivity=connectivity,
                n_erosion=n_erosion,
            )

            xyz_predict_z_slide_up = xyz_predict.copy()
            xyz_predict_z_slide_up += np.array([0, 0, picks_radius_dict[name] * 0.5])
            xyz_predict_z_slide_down = xyz_predict.copy()
            xyz_predict_z_slide_down -= np.array([0, 0, picks_radius_dict[name] * 0.5])

            xyz_predict = np.concatenate([xyz_predict, xyz_predict_z_slide_up, xyz_predict_z_slide_down])

            hit, fp, miss, metric, hit_pred, miss_truth, fp_pred = do_one_eval(
                xyz_truth, xyz_predict, picks_radius_dict[name] * 0.5, return_points=True
            )
            eval_df.append(
                dotdict(
                    id=id,
                    particle_type=name,
                    P=metric[0],
                    T=metric[1],
                    hit=metric[2],
                    miss=metric[3],
                    fp=metric[4],
                )
            )
            _hit_pred_dict[name] = hit_pred
            _miss_truth_dict[name] = miss_truth
            _fp_pred_dict[name] = fp_pred
            _cc_dict[name] = cc

        hit_pred_list.append(_hit_pred_dict)
        miss_truth_list.append(_miss_truth_dict)
        fp_pred_list.append(_fp_pred_dict)
        cc_list.append(_cc_dict)

    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby("particle_type").agg("sum").drop(columns=["id"])
    gb.loc[:, "precision"] = gb["hit"] / gb["P"]
    gb.loc[:, "precision"] = gb["precision"].fillna(0)
    gb.loc[:, "recall"] = gb["hit"] / gb["T"]
    gb.loc[:, "recall"] = gb["recall"].fillna(0)
    gb.loc[:, "f-beta4"] = 17 * gb["precision"] * gb["recall"] / (16 * gb["precision"] + gb["recall"])
    gb.loc[:, "f-beta4"] = gb["f-beta4"].fillna(0)

    gb = gb.sort_values("particle_type").reset_index(drop=False)
    # https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895
    gb.loc[:, "weight"] = [1, 0, 2, 1, 2, 1]
    lb_score = (gb["f-beta4"] * gb["weight"]).sum() / gb["weight"].sum()

    return lb_score, gb, hit_pred_list, miss_truth_list, fp_pred_list, cc_list


# https://www.kaggle.com/code/hengck23/3d-unet-using-2d-image-encoder/comments
def do_one_eval(truth, predict, threshold, return_points=False):
    P = len(predict)
    T = len(truth)

    if P == 0:
        hit = [[], []]
        miss = np.arange(T).tolist()
        fp = []
        metric = [P, T, len(hit[0]), len(miss), len(fp)]
        if return_points:
            return hit, fp, miss, metric, [], truth, []
        else:
            return hit, fp, miss, metric

    if T == 0:
        hit = [[], []]
        fp = np.arange(P).tolist()
        miss = []
        metric = [P, T, len(hit[0]), len(miss), len(fp)]
        if return_points:
            return hit, fp, miss, metric, predict, [], []
        else:
            return hit, fp, miss, metric

    # ---
    distance = predict.reshape(P, 1, 3) - truth.reshape(1, T, 3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss, t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp, p_index)].tolist()

    metric = [P, T, len(hit[0]), len(miss), len(fp)]  # for lb metric F-beta copmutation

    if return_points:
        hit_pred = predict[hit[0]]
        miss_truth = truth[miss]
        fp_pred = predict[fp]

        return hit, fp, miss, metric, hit_pred, miss_truth, fp_pred
    else:
        return hit, fp, miss, metric


# copick_utils.segmentation.picks_from_segmentation.picks_from_segmentation
import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, binary_dilation, ball


def picks_from_segmentation(
    segmentation,
    thresh,
    segmentation_idx,
    maxima_filter_size,
    min_particle_size,
    max_particle_size,
    voxel_spacing=1,
):
    """
    Process a specific label in the segmentation, extract centroids, and save them as picks.

    Args:
        segmentation (np.ndarray): Multilabel segmentation array.
        segmentation_idx (int): The specific label from the segmentation to process.
        maxima_filter_size (int): Size of the maximum detection filter.
        min_particle_size (int): Minimum size threshold for particles.
        max_particle_size (int): Maximum size threshold for particles.
        session_id (str): Session ID for pick saving.
        user_id (str): User ID for pick saving.
        pickable_object (str): The name of the object to save picks for.
        run: A Copick run object that manages pick saving.
        voxel_spacing (int): The voxel spacing used to scale pick locations (default 1).
    """
    # Create a binary mask for the specific segmentation label
    binary_mask = (segmentation >= thresh).astype(int)

    # Skip if the segmentation label is not present
    if np.sum(binary_mask) == 0:
        print(f"No segmentation with label {segmentation_idx} found.")
        return np.array([])

    # Structuring element for erosion and dilation
    # dilated = binary_dilation(binary_mask, ball(1))
    eroded = binary_erosion(binary_mask, ball(1))
    dilated = binary_dilation(eroded, ball(1))

    # Distance transform and local maxima detection
    distance = ndi.distance_transform_edt(dilated)
    local_max = distance == ndi.maximum_filter(
        distance, footprint=np.ones((maxima_filter_size, maxima_filter_size, maxima_filter_size))
    )

    # Watershed segmentation
    markers, _ = ndi.label(local_max)
    watershed_labels = watershed(-distance, markers, mask=dilated)

    # Extract region properties and filter based on particle size
    all_centroids = []
    for region in regionprops(watershed_labels):
        if min_particle_size <= region.area <= max_particle_size:
            all_centroids.append(np.array(region.centroid) * voxel_spacing)

    # Save centroids as picks
    if all_centroids:
        print(f"Centroids for label {segmentation_idx} saved successfully.")
    else:
        print(f"No valid centroids found for label {segmentation_idx}.")

    return np.array(all_centroids)
