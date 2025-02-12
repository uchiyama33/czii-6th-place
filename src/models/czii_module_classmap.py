from typing import Any, Dict, Tuple

import os
import lightning
import wandb
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timm.utils import ModelEmaV3
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, Dice, FBetaScore, Recall, Precision
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.losses import FocalLoss
import rootutils
import random

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.czii_utils.score import calc_score


def mixup_3d(batch, alpha=0.2):
    """
    3Dデータに対するMixupの実装

    Args:
        x (torch.Tensor): 入力データ (batch_size, channels, depth, height, width)
        y (torch.Tensor): ラベル (batch_size, num_classes, depth, height, width)
        alpha (float): Beta分布のパラメータ (α)

    Returns:
        torch.Tensor: Mixup後のデータ
        torch.Tensor: Mixup後のラベル
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = batch["image"].size(0)
    # シャッフルインデックスを作成
    index = torch.randperm(batch_size)

    # Mixup
    batch["image"] = lam * batch["image"] + (1 - lam) * batch["image"][index, :]
    batch["maps"] = lam * batch["maps"] + (1 - lam) * batch["maps"][index, :]

    return batch


def cutmix_3d(batch, alpha=1.0):
    """
    3D CutMix: ランダムに切り出した立方体領域を別サンプルの同領域で置き換える。

    Args:
        batch (dict): {"image":(B,C,D,H,W), "maps":(B,C,D,H,W)などを想定}
        alpha (float): Beta分布用パラメータ
    """
    x = batch["image"]
    y = batch["maps"]
    B, C, D, H, W = x.shape

    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    # シャッフルインデックス
    index = torch.randperm(B)

    # 切り出しサイズを lam に応じて決定 (例: 体積比から一辺を決める)
    cut_d = int(D * lam ** (1 / 3))
    cut_h = int(H * lam ** (1 / 3))
    cut_w = int(W * lam ** (1 / 3))

    for i in range(B):
        dz = random.randint(0, max(D - cut_d, 0))
        dy = random.randint(0, max(H - cut_h, 0))
        dx = random.randint(0, max(W - cut_w, 0))

        x[i, :, dz : dz + cut_d, dy : dy + cut_h, dx : dx + cut_w] = x[
            index[i], :, dz : dz + cut_d, dy : dy + cut_h, dx : dx + cut_w
        ]
        y[i, :, dz : dz + cut_d, dy : dy + cut_h, dx : dx + cut_w] = y[
            index[i], :, dz : dz + cut_d, dy : dy + cut_h, dx : dx + cut_w
        ]

    batch["image"] = x
    batch["maps"] = y
    return batch


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, batch_size, preprocess_version):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.preprocess_version = preprocess_version

    def forward(self, x):
        if x.size(0) < self.batch_size:
            # Duplicate data to match batch size
            original_size = x.size(0)
            repeats = (self.batch_size + original_size - 1) // original_size
            x = x.repeat(repeats, 1, 1, 1, 1)[: self.batch_size]
            outputs = self.model(x)
            if isinstance(outputs, dict):
                return {k: v[:original_size] for k, v in outputs.items()}
            return outputs[:original_size]
        outputs = self.model(x)
        return outputs


class CZIILitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion_det: torch.nn.Module,
        criterion_classmap: torch.nn.Module,
        criterion_seg: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optimize_config,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        preprocess_version,
        eval_threshold: float = 0.3,
        num_classes=6,
        eval_normalize=False,
        is_kaggle_data=True,
        pretrained_ckpt_path: str = None,
        mixup_alpha: float = None,
        remove_pretrained_head: bool = True,
        scale_factor_focal_loss: float = 1,
        ema: bool = False,
        ema_decay: float = 0.99,
        disable_beta_amylase: bool = False,
        cutmix_alpha: float = None,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        if ema:
            self.net_ema = ModelEmaV3(
                self.net,
                decay=ema_decay,
                use_warmup=True,
                warmup_power=4 / 5,
            )

        if pretrained_ckpt_path is not None:
            pretrained_dict = torch.load(pretrained_ckpt_path)["state_dict"]
            model_dict = self.state_dict()
            # Filter out unnecessary keys
            if remove_pretrained_head:
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.size() == model_dict[k].size() and not k.startswith("net.head")
                }
            else:
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.size() == model_dict[k].size()
                }
            # Overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # Load the new state dict
            self.load_state_dict(model_dict, strict=False)

        # loss function
        self.criterion_det = criterion_det
        self.criterion_classmap = criterion_classmap
        self.criterion_seg = criterion_seg

        self.dummy_backgroud_idx = num_classes

        # for averaging loss across batches
        metrics = MetricCollection(
            {
                "loss": MeanMetric(),
                "recall": Recall(
                    task="multiclass",
                    average="macro",
                    num_classes=num_classes + 1,
                    ignore_index=self.dummy_backgroud_idx,
                ),
                "precision": Precision(
                    task="multiclass",
                    average="macro",
                    num_classes=num_classes + 1,
                    ignore_index=self.dummy_backgroud_idx,
                ),
                "fbeta": FBetaScore(
                    beta=4.0,
                    task="multiclass",
                    average="macro",
                    num_classes=num_classes + 1,
                    ignore_index=self.dummy_backgroud_idx,
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # self.val_score = MeanMetric()
        self.test_score = MeanMetric()

        # # for tracking best so far validation accuracy
        # self.val_score_best = MaxMetric()

        self.picks_coord_dict = None
        self.picks_radius_dict = None

    # def forward_with_batch_size(self, x, batch_size) -> torch.Tensor:
    #     """Perform a forward pass through the model `self.net` with a specified batch size.

    #     :param x: A tensor of images.
    #     :param batch_size: The batch size to use.
    #     :return: A tensor of logits.
    #     """
    #     if x.size(0) < batch_size:
    #         # Duplicate data to match batch size
    #         original_size = x.size(0)
    #         repeats = (batch_size + original_size - 1) // original_size
    #         x = x.repeat(repeats, 1, 1, 1, 1)[:batch_size]
    #         outputs = self.net(x)
    #         return outputs[:original_size]
    #     elif x.size(0) > batch_size:
    #         # Split data into chunks of batch size
    #         outputs = []
    #         for i in range(0, x.size(0), batch_size):
    #             chunk = x[i : i + batch_size]
    #             if chunk.size(0) < batch_size:
    #                 chunk = chunk.repeat((batch_size + chunk.size(0) - 1) // chunk.size(0), 1, 1, 1, 1)[
    #                     :batch_size
    #                 ]
    #             outputs.append(self.net(chunk))
    #         return torch.cat(outputs, dim=0)[: x.size(0)]
    #     else:
    #         return self.net(x)

    def forward(self, x) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        if self.hparams.ema and not self.training:
            net = self.net_ema
        else:
            net = self.net
        return net(x)

    def on_after_backward(self) -> None:
        if self.hparams.ema:
            self.net_ema.update(self.net, self.trainer.global_step)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_metrics.reset()
        # self.val_score_best.reset()

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        outputs = self.forward(batch["image"])

        preds_classmap = outputs["classmap"].sigmoid()

        if batch.get("maps") is None:
            # ラベルを [batch_size, D, H, W] の形状に変換
            labels = batch["label"].squeeze(1).long()  # shape: [B, D, H, W]
            # ワンホットエンコーディング（クラス数に応じて num_classes を設定）
            maps = F.one_hot(
                labels, num_classes=self.hparams.num_classes + 1
            )  # shape: [B, D, H, W, num_classes + 1]
            # 背景クラス（クラス0）を除外（必要に応じて）
            maps = maps[..., 1:]  # shape: [B, D, H, W, num_classes]
            # 次元を並べ替えて形状を統一
            maps = maps.permute(0, 4, 1, 2, 3).contiguous()  # shape: [B, num_classes, D, H, W]
            batch["maps"] = maps

            # maps = []
            # for i in range(self.hparams.num_classes):
            #     map = (batch["label"]==i+1).squeeze(1).long()
            #     maps.append(map)
            # batch["maps"] = torch.stack(maps, dim=1)

        if self.hparams.disable_beta_amylase:
            beta_amylase_idx = 1
            outputs["classmap"][:, beta_amylase_idx] = 0
            preds_classmap[:, beta_amylase_idx] = 0
            batch["maps"][:, beta_amylase_idx] = 0

        loss = self.criterion_classmap(outputs["classmap"], batch["maps"])
        if isinstance(self.criterion_classmap, FocalLoss):
            loss = self.hparams.scale_factor_focal_loss * loss

        if isinstance(outputs.get("det"), torch.Tensor):
            loss = loss + self.criterion_det(outputs["det"], batch["maps"].max(1, keepdims=True)[0])
        if isinstance(outputs.get("seg"), torch.Tensor):
            loss = loss + self.criterion_seg(outputs["seg"], batch["label"])
        return loss, preds_classmap

    def model_step_inference(
        self, batch: Tuple[torch.Tensor, torch.Tensor], overlap=0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        outputs = sliding_window_inference(
            inputs=batch["image"],
            roi_size=self.trainer.datamodule.hparams.volume_size,
            sw_batch_size=4,  # one window is proecessed at a time
            predictor=ModelWrapper(self, 4, self.hparams.preprocess_version),
            overlap=overlap,
        )
        preds_classmap = outputs["classmap"].sigmoid()

        return preds_classmap

    def on_train_epoch_start(self) -> None:
        pass

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        if self.hparams.mixup_alpha is not None:
            batch = mixup_3d(batch, alpha=self.hparams.mixup_alpha)

        if self.hparams.cutmix_alpha is not None:
            batch = cutmix_3d(batch, alpha=self.hparams.cutmix_alpha)

        loss, preds_classmap = self.model_step(batch)

        # 閾値以上で最大のクラスを予測ラベルとする（閾値以下の場合は背景ラベル）
        pred_class_idx = preds_classmap.argmax(1)
        pred_class_idx[preds_classmap.max(1).values < self.hparams.eval_threshold] = self.dummy_backgroud_idx
        target_binary = (batch["maps"] > self.hparams.eval_threshold).int()
        target_class_idx = target_binary.argmax(1)

        # 背景のラベルを変更
        target_class_idx[(target_binary.sum(1) == 0)] = self.dummy_backgroud_idx

        # update and log metrics
        try:
            self.train_metrics.update(
                value=loss.nan_to_num(),
                preds=pred_class_idx,
                target=target_class_idx,
            )
        except:
            self.train_metrics.update(
                value=loss.nan_to_num(),
                preds=torch.zeros(1).to(loss.device),
                target=torch.zeros(1).to(loss.device),
            )
        # 5エポックごとにモデルの予測結果を保存
        if (self.current_epoch + 1) % 5 == 0 and batch_idx == 0:
            self.log_image(batch, preds_classmap, "train")

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        output = self.train_metrics.compute()
        self.log_dict(output, on_epoch=True, prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_start(self) -> None:
        self.fold = self.trainer.datamodule.hparams.fold
        if self.hparams.is_kaggle_data:
            if self.picks_coord_dict is None or self.picks_radius_dict is None:
                self.picks_coord_dict = self.trainer.datamodule.picks_coord_dict
                self.picks_radius_dict = self.trainer.datamodule.picks_radius_dict
                self.particle_names = self.trainer.datamodule.particle_names

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds_classmap = self.model_step(batch)

        # 閾値以上で最大のクラスを予測ラベルとする（閾値以下の場合は背景ラベル）
        pred_class_idx = preds_classmap.argmax(1)
        pred_class_idx[preds_classmap.max(1).values < self.hparams.eval_threshold] = self.dummy_backgroud_idx
        target_binary = (batch["maps"] > self.hparams.eval_threshold).int()
        target_class_idx = target_binary.argmax(1)

        # 背景のラベルを変更
        target_class_idx[(target_binary.sum(1) == 0)] = self.dummy_backgroud_idx

        # update and log metrics
        try:
            self.val_metrics.update(
                value=loss.nan_to_num(),
                preds=pred_class_idx,
                target=target_class_idx,
            )
        except:
            self.val_metrics.update(
                value=loss.nan_to_num(),
                preds=torch.zeros(1).to(loss.device),
                target=torch.zeros(1).to(loss.device),
            )

        if (self.current_epoch + 1) % 5 == 0:
            if batch_idx == 0:
                self.log_image(batch, preds_classmap, "val")

            # score = calc_score(
            #     preds_map,
            #     batch["id"],
            #     self.picks_coord_dict,
            #     self.picks_radius_dict,
            #     self.particle_names,
            #     thresh=0.5,
            # )
            # self.val_score(score)
            # self.log("val/score", self.val_score, on_step=False, on_epoch=True, prog_bar=True)

    def log_image(self, batch, preds, split):
        os.makedirs(os.path.join(self.trainer.log_dir, "pred_images", f"fold{self.fold}"), exist_ok=True)

        z = batch["image"].shape[2] // 2
        N = min(3, batch["image"].shape[0])
        C = self.net.num_classes

        image = batch["image"].cpu().numpy()
        maps = batch["maps"].cpu().numpy()
        preds = preds.detach().float().cpu().numpy()

        plt.figure(figsize=(18, 10))
        for n in range(N):
            plt.subplot(N * 2, C + 1, 2 * (n * (C + 1)) + 1)
            plt.imshow(image[n, 0, z], cmap="gray", vmin=-1.5, vmax=1.5)
            plt.axis("off")
            for c in range(C):
                plt.subplot(N * 2, C + 1, 2 * (n * (C + 1)) + 2 + c)
                plt.imshow(maps[n, c, z], cmap="viridis", vmin=0, vmax=1)
                plt.axis("off")
                plt.subplot(N * 2, C + 1, 2 * (n * (C + 1)) + 2 + c + (C + 1))
                plt.imshow(preds[n, c, z], cmap="viridis", vmin=0, vmax=1)
                plt.axis("off")
        plt.savefig(
            os.path.join(
                self.trainer.log_dir,
                "pred_images",
                f"fold{self.fold}",
                f"epoch_{self.current_epoch}_{split}.png",
            )
        )
        plt.close()

    def on_validation_epoch_end(self) -> None:
        output = self.val_metrics.compute()
        self.log_dict(output, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_start(self) -> None:
        if self.hparams.is_kaggle_data:
            if self.picks_coord_dict is None or self.picks_radius_dict is None:
                self.picks_coord_dict = self.trainer.datamodule.picks_coord_dict
                self.picks_radius_dict = self.trainer.datamodule.picks_radius_dict
                self.particle_names = self.trainer.datamodule.particle_names
            self.thresholds = []
            self.scores = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds_classmap = self.model_step_inference(batch, overlap=0.7)

        # 閾値以上で最大のクラスを予測ラベルとする（閾値以下の場合は背景ラベル）
        pred_class_idx = preds_classmap.argmax(1)
        pred_class_idx[preds_classmap.max(1).values < self.hparams.eval_threshold] = self.dummy_backgroud_idx
        target_binary = (batch["maps"] > self.hparams.eval_threshold).int()
        target_class_idx = target_binary.argmax(1)

        # 背景のラベルを変更
        target_class_idx[(target_binary.sum(1) == 0)] = self.dummy_backgroud_idx

        # update and log metrics
        self.test_metrics.update(
            value=0,
            preds=pred_class_idx,
            target=target_class_idx,
        )

        if self.hparams.is_kaggle_data:
            for threshold in np.arange(0, 1.1, 0.1):
                score = calc_score(
                    preds_classmap,
                    "classmap",
                    batch["id"],
                    self.picks_coord_dict,
                    self.picks_radius_dict,
                    self.particle_names,
                    thresh=threshold,
                )
                self.thresholds.append(threshold)
                self.scores.append(score)

    def on_test_epoch_end(self) -> None:
        if self.hparams.is_kaggle_data:
            if isinstance(self.logger, lightning.pytorch.loggers.wandb.WandbLogger):
                if self.thresholds and self.scores:
                    table = wandb.Table(
                        data=list(zip(self.thresholds, self.scores)), columns=["Threshold", "Score"]
                    )
                    self.logger.experiment.log(
                        {
                            "Threshold vs Score": wandb.plot.line(
                                table, "Threshold", "Score", title="Threshold vs Score"
                            )
                        }
                    )

            # ベストスコアを保存
            best_score = max(self.scores)
            self.test_score(best_score)
            self.log("test/score", self.test_score, on_epoch=True, prog_bar=True)

            # リストをクリア
            self.thresholds.clear()
            self.scores.clear()

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        with torch.inference_mode():
            preds = self.model_step_inference(batch)
        return preds

    def on_predict_epoch_end(self) -> None:
        """Lightning hook that is called when a predict epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            if self.hparams.ema:
                self.net_ema = torch.compile(self.net_ema)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        if self.hparams.optimize_config.mode == "normal":
            optimizer = self.hparams.optimizer(params=self.parameters())

        elif self.hparams.optimize_config.mode == "target_decay":
            if self.hparams.optimize_config.get("target_names"):
                target_layer_list = []
                for target_name in self.hparams.optimize_config.target_names:
                    target_layer_list += self.get_target_param(target_name)
            else:
                target_layer_list = self.get_target_param(self.hparams.optimize_config.target_name)
            target_params = list(
                map(
                    lambda x: x[1],
                    list(filter(lambda kv: kv[0] in target_layer_list, self.named_parameters())),
                )
            )
            base_params = list(
                map(
                    lambda x: x[1],
                    list(filter(lambda kv: kv[0] not in target_layer_list, self.named_parameters())),
                )
            )

            optimizer = self.hparams.optimizer(
                params=[
                    {
                        "params": target_params,
                        "lr": self.hparams.optimize_config.lr_base
                        * self.hparams.optimize_config.lr_decay_coef,
                    },
                    {"params": base_params},
                ],
            )

        else:
            assert False, "optimize_mode"

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def get_target_param(self, target_name):
        layer_list = []
        for name, param in self.named_parameters():
            if target_name in name:
                # print(name, param.requires_grad)
                layer_list.append(name)

        assert len(layer_list) > 0

        return layer_list


# 検出点を可視化する関数
def visualize_detection(volume_shape, peaks, radius=3):
    """
    3D 画像と検出点のリストを受け取り、検出点を0/1で可視化する。

    Args:
        volume_shape: Tuple[int, int, int], 3D 画像のサイズ (D, H, W)。
        peaks: List[Tuple[int, int, int, float]], 検出点の座標と値のリスト。
        radius: int, 検出点の半径。

    Returns:
        vis: np.ndarray, shape (D, H, W), 検出点を可視化した画像。
    """
    vis = np.zeros(volume_shape, dtype=np.float32)
    for z, y, x, _ in peaks:
        z_start = max(0, z - radius)
        z_end = min(volume_shape[0], z + radius + 1)
        y_start = max(0, y - radius)
        y_end = min(volume_shape[1], y + radius + 1)
        x_start = max(0, x - radius)
        x_end = min(volume_shape[2], x + radius + 1)
        vis[z_start:z_end, y_start:y_end, x_start:x_end] = 1
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(vis.max(0), cmap="gray")
    ax[1].imshow(vis.max(1), cmap="gray")
    ax[2].imshow(vis.max(2), cmap="gray")
    plt.savefig("/workspace/g.png")


if __name__ == "__main__":
    _ = CZIILitModule()
