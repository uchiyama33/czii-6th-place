from typing import Any, Dict, Tuple

import os
import lightning
import wandb
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, Dice, FBetaScore, Recall, Precision
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.czii_utils.score import calc_score


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
        remove_pretrained_head: bool = True,
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

        # for averaging loss across batches
        metrics = MetricCollection(
            {
                "loss": MeanMetric(),
                "recall": Recall(
                    task="multiclass",
                    average="macro",
                    num_classes=num_classes + 1,
                    top_k=1,
                    ignore_index=0,
                ),
                "precision": Precision(
                    task="multiclass",
                    average="macro",
                    num_classes=num_classes + 1,
                    top_k=1,
                    ignore_index=0,
                ),
                "fbeta": FBetaScore(
                    beta=4.0,
                    task="multiclass",
                    average="macro",
                    num_classes=num_classes + 1,
                    top_k=1,
                    ignore_index=0,
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.test_score = MeanMetric()

        self.picks_coord_dict = None
        self.picks_radius_dict = None

    def forward(self, x) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_metrics.reset()

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

        preds_seg = outputs["seg"].softmax(1)

        loss = self.criterion_seg(outputs["seg"], batch["label"])
        if isinstance(outputs.get("det"), torch.Tensor):
            loss = loss + self.criterion_det(outputs["det"], batch["maps"].max(1, keepdims=True)[0])
        if isinstance(outputs.get("classmap"), torch.Tensor):
            loss = loss + self.criterion_classmap(outputs["classmap"], batch["maps"])
        return loss, preds_seg

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

        preds_seg = outputs["seg"].softmax(1)

        return preds_seg

    def on_train_epoch_start(self) -> None:
        pass

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds_seg = self.model_step(batch)

        # update and log metrics
        self.train_metrics.update(value=loss, preds=preds_seg, target=batch["label"].squeeze(1).long())

        # 5エポックごとにモデルの予測結果を保存
        if (self.current_epoch + 1) % 5 == 0 and batch_idx == 0:
            self.log_image(batch, preds_seg, "train")

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
        loss, preds_seg = self.model_step(batch)

        # update and log metrics
        self.val_metrics.update(
            value=loss.nan_to_num(), preds=preds_seg, target=batch["label"].squeeze(1).long()
        )

        if (self.current_epoch + 1) % 5 == 0 and batch_idx == 0:
            self.log_image(batch, preds_seg, "val")

    def log_image(self, batch, preds, split):
        os.makedirs(os.path.join(self.trainer.log_dir, "pred_images"), exist_ok=True)

        z = batch["image"].shape[2] // 2
        N = min(3, batch["image"].shape[0])

        image = batch["image"].cpu().numpy()
        label = batch["label"].cpu().numpy()
        preds = preds.detach().cpu().numpy()
        n_cols = self.net.num_classes + 2

        plt.figure(figsize=(18, 10))
        for n in range(N):
            plt.subplot(N, n_cols, n_cols * n + 1)
            plt.imshow(image[n, 0, z], cmap="gray", vmin=-1.5, vmax=1.5)
            plt.axis("off")
            plt.subplot(N, n_cols, n_cols * n + 2)
            plt.imshow(label[n, 0, z], cmap="viridis", vmin=0, vmax=self.net.num_classes)
            plt.axis("off")
            for c in range(self.net.num_classes):
                thresholded_preds = np.zeros_like(preds[n, 0, z])
                thresholded_preds[preds[n, c + 1, z] >= self.hparams.eval_threshold] = c + 1
                plt.subplot(N, n_cols, n_cols * n + 3 + c)
                plt.imshow(thresholded_preds, cmap="viridis", vmin=0, vmax=self.net.num_classes)
                plt.axis("off")
        plt.savefig(
            os.path.join(self.trainer.log_dir, "pred_images", f"epoch_{self.current_epoch}_{split}.png")
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
        preds_seg = self.model_step_inference(batch)

        # update and log metrics
        self.test_metrics.update(value=0, preds=preds_seg, target=batch["label"].squeeze(1).long())

        if self.hparams.is_kaggle_data:
            for threshold in np.arange(0, 1.1, 0.1):
                score = calc_score(
                    preds_seg,
                    "seg",
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
