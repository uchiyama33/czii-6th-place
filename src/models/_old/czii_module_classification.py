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
from monai.losses import FocalLoss
import rootutils

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
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        metrics = MetricCollection(
            {
                "loss": MeanMetric(),
                "recall": Recall(
                    task="multiclass",
                    average="macro",
                    num_classes=num_classes,
                    ignore_index=0,
                ),
                "precision": Precision(
                    task="multiclass",
                    average="macro",
                    num_classes=num_classes,
                    ignore_index=0,
                ),
                "fbeta": FBetaScore(
                    beta=4.0,
                    task="multiclass",
                    average="macro",
                    num_classes=num_classes,
                    ignore_index=0,
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
        preds = outputs.softmax(1)

        loss = self.criterion(outputs, batch["label"])

        return loss, preds

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

        loss, preds = self.model_step(batch)

        # update and log metrics
        self.train_metrics.update(
            value=loss.nan_to_num(),
            preds=preds,
            target=batch["label"],
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        output = self.train_metrics.compute()
        self.log_dict(output, on_epoch=True, prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_start(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds = self.model_step(batch)

        # update and log metrics
        self.val_metrics.update(
            value=loss.nan_to_num(),
            preds=preds,
            target=batch["label"],
        )

    def on_validation_epoch_end(self) -> None:
        output = self.val_metrics.compute()
        self.log_dict(output, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_start(self) -> None:
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds = self.model_step(batch)

        # update and log metrics
        self.test_metrics.update(
            value=loss.nan_to_num(),
            preds=preds,
            target=batch["label"],
        )

    def on_test_epoch_end(self) -> None:
        output = self.test_metrics.compute()
        self.log_dict(output, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

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


if __name__ == "__main__":
    _ = CZIILitModule()
