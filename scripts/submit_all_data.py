import argparse

# 必要なモジュールのインポート
import rootutils
import os
from hydra import compose, initialize
from pathlib import Path
from glob import glob
import numpy as np
import torch
import pickle
import pandas as pd
from tqdm import tqdm
import torch_tensorrt
import torch.distributed as dist
import torch.multiprocessing as mp
import gc
from monai.inferers import sliding_window_inference
import copick
from monai.data import MetaTensor
import time

rootutils.setup_root(Path().resolve(), indicator=".project-root", pythonpath=True)

from src.data.components.transforms import get_transforms
from src.czii_utils.score import get_centroids_from_pred
from src.czii_utils.czii_helper import dotdict
from src.data.components.czii_data_utils import preprocess_tomogram


# モデルのラッパークラスを定義
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.batch_size = batch_size

    def forward(self, x):
        if x.size(0) < self.batch_size:
            # バッチサイズに合わせてデータを複製
            original_size = x.size(0)
            repeats = (self.batch_size + original_size - 1) // original_size
            x = x.repeat(repeats, 1, 1, 1, 1)[: self.batch_size]
            outputs = self.model(x)
            if isinstance(outputs, dict):
                return {k: v[:original_size] for k, v in outputs.items()}
            return outputs[:original_size]
        outputs = self.model(x)
        return outputs


# 関連する関数を定義
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


def load_runs(copick_config_path):
    root = copick.from_file(copick_config_path)
    particle_names = [obj.name for obj in root.config.pickable_objects if obj.is_particle]
    picks_radius_dict = {obj.name: obj.radius for obj in root.config.pickable_objects if obj.is_particle}
    return root.runs, particle_names, picks_radius_dict


def load_tomogram(run, tomo_type, preprocess_version, voxel_size=10):
    tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(tomo_type).numpy()
    tomogram = preprocess_tomogram(tomogram, preprocess_version)
    if preprocess_version == 4:
        tomogram = np.clip(tomogram, -5, 2)
        tomogram = (tomogram - tomogram.min()) / (tomogram.max() - tomogram.min())
    if preprocess_version == 4.1:
        tomogram = np.clip(tomogram, -5, 2)
        tomogram = (tomogram - tomogram.min()) / (tomogram.max() - tomogram.min())
        tomogram = (tomogram - 0.5) * 2
    data_dict = {
        "image": MetaTensor(tomogram),
        "id": run.name,
    }
    return data_dict


# マルチプロセスのセットアップ関数を定義
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch_tensorrt.runtime.set_multi_device_safe_mode(True)


def load_configs(experiment_list):
    cfg_list = []
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
        if cfg.model.net.get("pretrained"):
            cfg.model.net.pretrained = False
        if cfg.model.net.get("grad_checkpointing"):
            cfg.model.net.grad_checkpointing = False

        cfg_list.append(cfg)

    return cfg_list


def tta_transform(image: torch.Tensor, tta_count: int) -> torch.Tensor:
    transformations = {
        0: lambda x: x,
        1: lambda x: torch.rot90(x, 1, [2, 3]),
        2: lambda x: torch.rot90(x, 2, [2, 3]),
        3: lambda x: torch.rot90(x, 3, [2, 3]),
        4: lambda x: torch.flip(x, [2]),
        5: lambda x: torch.flip(x, [3]),
        6: lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [2]),
        7: lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [3]),
        8: lambda x: torch.flip(x, [1]),  # 深さ方向フリップ
        9: lambda x: torch.flip(x, [1, 2]),  # 深さ＆高さフリップ
        10: lambda x: torch.flip(x, [1, 3]),  # 深さ＆幅フリップ
    }

    tta_count = tta_count % len(transformations)

    return transformations[tta_count](image)


def inverse_tta_transform(image: torch.Tensor, tta_count: int) -> torch.Tensor:
    """
    Apply the inverse of the specified TTA transformation to the input image.

    Args:
        image (torch.Tensor): The transformed image tensor. Shape: (C, D, H, W)
        tta_count (int): The TTA transformation index used during augmentation (0-7).

    Returns:
        torch.Tensor: The image tensor after applying the inverse transformation.
    """
    inverse_transformations = {
        0: lambda x: x,
        1: lambda x: torch.rot90(x, 3, [2, 3]),  # 逆回転90度（270度回転）
        2: lambda x: torch.rot90(x, 2, [2, 3]),  # 逆回転180度
        3: lambda x: torch.rot90(x, 1, [2, 3]),  # 逆回転270度（90度回転）
        4: lambda x: torch.flip(x, [2]),  # 反転の逆は同じ反転
        5: lambda x: torch.flip(x, [3]),  # 反転の逆は同じ反転
        6: lambda x: torch.rot90(torch.flip(x, [2]), 3, [2, 3]),  # 反転後に逆回転
        7: lambda x: torch.rot90(torch.flip(x, [3]), 3, [2, 3]),  # 反転後に逆回転
        8: lambda x: torch.flip(x, [1]),  # flip([1]) は自分自身が逆
        9: lambda x: torch.flip(x, [1, 2]),  # flip([1,2]) も自分自身が逆
        10: lambda x: torch.flip(x, [1, 3]),  # flip([1,3]) も自分自身が逆
    }

    tta_count = tta_count % len(inverse_transformations)

    return inverse_transformations[tta_count](image)


# 推論を行う関数を定義
def inference(rank, world_size, temp_dir, args, runs, particle_names):
    print(f"Running inference on rank {rank}.")
    setup(rank, world_size)

    cfg_list = load_configs(args.experiment_list)

    preprocess_versions = list(set(cfg.data.preprocess_version for cfg in cfg_list))

    for i_run, run in enumerate(tqdm(runs)):
        data_preprocess_dict = {}
        for preprocess_version in preprocess_versions:
            data_preprocess_dict[preprocess_version] = load_tomogram(run, args.tomo_type, preprocess_version)

        preds_tmp = torch.zeros(
            tuple([6] + list(data_preprocess_dict[preprocess_version]["image"].shape[-3:]))
        )
        cnt = 0

        for experiment, compiled_model_dir, cfg, fp16_mode, batch_size_pred in zip(
            args.experiment_list,
            args.compiled_model_dir_list,
            cfg_list,
            args.fp16_mode_list,
            args.batch_size_pred_list,
        ):

            transforms, _, _, _ = get_transforms(
                cfg.data.transforms_version,
                cfg.data.num_classes,
                cfg.data.volume_size,
                cfg.data.num_crop_samples,
            )

            data_dict = data_preprocess_dict[cfg.data.preprocess_version]
            data_dict = transforms(data_dict)
            inputs = []
            for tta_count in range(args.n_tta):
                image = data_dict["image"].as_tensor()
                image = tta_transform(image, tta_count)
                if fp16_mode:
                    image = image.half()
                inputs.append(image)

            start_time = time.time()
            save_path = os.path.join(
                compiled_model_dir,
                f"trt_{experiment}_all_data_{args.ckpt_type}_bs{batch_size_pred}_fp16{fp16_mode}.ep",
            )
            model = torch.export.load(save_path).module()
            model = ModelWrapper(model, batch_size_pred)
            end_time = time.time()
            print(f"Model loading time: {end_time - start_time:.2f} seconds")

            start_time = time.time()
            with torch.inference_mode(), torch.no_grad():
                outputs_tta_tmp = torch.zeros_like(preds_tmp).cuda()
                for tta_count in range(args.n_tta):
                    _inputs = inputs[tta_count].cuda().unsqueeze(0)
                    _outputs = sliding_window_inference(
                        inputs=_inputs,
                        roi_size=cfg.data.volume_size,
                        sw_batch_size=batch_size_pred,
                        predictor=model,
                        overlap=args.inference_overlap,
                    )
                    _outputs = _outputs["classmap"].sigmoid().squeeze(0)
                    _outputs = inverse_tta_transform(_outputs, tta_count)
                    outputs_tta_tmp += _outputs

                preds_tmp += (outputs_tta_tmp / args.n_tta).cpu()
                cnt += 1
            end_time = time.time()
            print(f"Inference time: {end_time - start_time:.2f} seconds")

            del model
            gc.collect()
            torch.cuda.empty_cache()

        preds_fold_mean = preds_tmp / cnt
        # １つ目だけpreds_fold_meanを保存
        if i_run == 0:
            preds_save_path = os.path.join(temp_dir, f"preds_fold_mean-{run.name}.pt")
            torch.save(preds_fold_mean, preds_save_path)

        pred_coord_dict = {}
        for p, name in enumerate(particle_names):
            if name == "beta-amylase":
                continue
            if args.only_specific_particle and name != args.only_specific_particle:
                continue
            xyz_pred = get_centroids_from_pred(
                preds_fold_mean[p], args.thresh[p], voxel_spacing=args.voxel_size
            )
            pred_coord_dict[name] = xyz_pred

        preds_save_path = os.path.join(temp_dir, f"pred_coord_dict-{run.name}.pickle")
        with open(preds_save_path, "wb") as f:
            pickle.dump(pred_coord_dict, f)

        del (
            preds_tmp,
            preds_fold_mean,
            pred_coord_dict,
            data_preprocess_dict,
            data_dict,
            inputs,
            outputs_tta_tmp,
            _inputs,
            _outputs,
        )
        gc.collect()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def none_or_str(value):
    if value == "None":
        return None
    return value


# 3つ目のセルの内容を引数として受け取る関数を定義
def main():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--copick_config_path", type=str, default="/kaggle/input/czii-copick-config/copick_sub.config"
    )
    parser.add_argument("--voxel_size", type=float, default=10)
    parser.add_argument("--tomo_type", type=str, default="denoised")
    parser.add_argument(
        "--experiment_list",
        type=str,
        nargs="+",
        default=[
            "241124-baseline-classmap_tversky07-hengck23_resnet34d_bn_relu_d64-ep300-transV2_2",
        ],
    )
    parser.add_argument(
        "--compiled_model_dir_list",
        type=str,
        nargs="+",
        default=[
            "/kaggle/input/czii-save-241124-classmap-tversky07-resnet34d",
        ],
    )
    parser.add_argument(
        "--fp16_mode_list",
        type=str2bool,
        nargs="+",
        default=[True],
        help="List of fp16_mode flags as True or False",
    )
    parser.add_argument("--inference_overlap", type=float, default=0.2)
    parser.add_argument("--thresh", type=float, nargs="+", default=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    parser.add_argument("--batch_size_pred_list", type=int, nargs="+", default=[8])
    parser.add_argument("--n_tta", type=int, default=1)
    parser.add_argument("--ckpt_type", type=str, default="last", choices=["last", "best"])
    parser.add_argument("--src_dir", type=str, default="/kaggle/working/src")
    parser.add_argument("--temp_dir", type=str, default="/kaggle/tmp/preds")
    parser.add_argument("--output_csv_path", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--only_specific_particle", type=none_or_str, nargs="?", default=None)

    args = parser.parse_args()

    # argsの中にあるリストの長さが全て同じであることを確認
    assert all(
        len(args.experiment_list) == len(lst)
        for lst in [
            args.compiled_model_dir_list,
            args.fp16_mode_list,
            args.batch_size_pred_list,
        ]
    )
    assert len(args.thresh) == 6

    mp.set_start_method("spawn", force=True)

    # 必要な設定
    os.chdir(args.src_dir)

    args = dotdict(vars(args))

    # データの準備
    runs, particle_names, picks_radius_dict = load_runs(args.copick_config_path)

    # 推論の実行
    runs_world = np.array_split(runs, args.world_size)

    processes = []

    os.makedirs(args.temp_dir, exist_ok=True)

    print(f"Temporary directory for storing predictions: {args.temp_dir}")

    for rank in range(args.world_size):
        p = mp.Process(
            target=inference,
            args=(rank, args.world_size, args.temp_dir, args, runs_world[rank], particle_names),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 推論結果の処理と保存
    results = []
    for run in runs:
        preds_save_path = os.path.join(args.temp_dir, f"pred_coord_dict-{run.name}.pickle")
        with open(preds_save_path, "rb") as f:
            pred_coord_dict = pickle.load(f)
        results.append({run.name: pred_coord_dict})

    # 結果をDataFrameに変換して保存
    data = []
    count = 0
    for result in results:
        for tomo_id, coord_dict in result.items():
            for name, coords in coord_dict.items():
                if name == "beta-amylase":
                    continue
                for coord in coords:
                    data.append(
                        {
                            "id": count,
                            "experiment": tomo_id,
                            "particle_type": name,
                            "x": coord[0],
                            "y": coord[1],
                            "z": coord[2],
                        }
                    )
                    count += 1

    df_submit = pd.DataFrame(data)
    df_submit.to_csv(args.output_csv_path, index=False)

    # 結果の確認（オプション）
    print(df_submit.head())


if __name__ == "__main__":
    main()

    # copick_config_path="/workspace/data/czii-cryo-et-object-identification/copick_sub_local.config",
    # compiled_model_dir_list=["/workspace/logs/compiled/czii-save-241124-classmap-tversky07-resnet34d"],
    # n_fold=2,
    # world_size=1,
    # src_dir="/workspace/src",
    # temp_dir="/workspace/tmp",
    # output_csv_path="/workspace/scripts/submission.csv",
