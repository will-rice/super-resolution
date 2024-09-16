"""Super resolution lightning module."""

from typing import Any

import torch
import torchvision
import wandb
from lightning_fabric.utilities import rank_zero_only
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from super_resolution.datasets import Batch
from super_resolution.losses import AuxiliaryLoss, HighFrequencyLoss


class SuperResolutionLightningModule(LightningModule):
    """Super resolution lightning module."""

    def __init__(self, model: nn.Module, sync_dist: bool = False):
        super().__init__()
        self.model = model
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
        )
        self.sync_dist = sync_dist
        self.loss_fn = nn.L1Loss()
        self.auxiliary_loss = AuxiliaryLoss()
        self.hf_loss = HighFrequencyLoss()

        self.metrics = MetricCollection(
            {
                "ssim": StructuralSimilarityIndexMeasure(data_range=(0, 1)),
                "psnr": PeakSignalNoiseRatio(data_range=(0, 1)),
                "ms_ssim": MultiScaleStructuralSimilarityIndexMeasure(
                    data_range=(0, 1)
                ),
            }
        )
        self.last_val_batch: dict[str, Any] = {}

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(image)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Training step."""
        inputs, targets = batch.source, batch.target
        outputs = self(inputs)
        l1_loss = self.loss_fn(outputs, targets)
        aux_loss = self.auxiliary_loss(outputs, targets)
        hf_loss = self.hf_loss(outputs, targets)
        loss = l1_loss + aux_loss + hf_loss
        self.log("train_loss", loss, sync_dist=self.sync_dist, prog_bar=True)
        self.log_dict(
            {
                "train_l1_loss": l1_loss,
                "train_aux_loss": aux_loss,
                "train_hf_loss": hf_loss,
            },
            sync_dist=self.sync_dist,
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Validate step."""
        inputs, targets = batch.source, batch.target
        outputs = self.ema_model(inputs)
        l1_loss = self.loss_fn(outputs, targets)
        aux_loss = self.auxiliary_loss(outputs, targets)
        hf_loss = self.hf_loss(outputs, targets)
        loss = l1_loss + aux_loss + hf_loss
        self.log("val_loss", loss, sync_dist=self.sync_dist, prog_bar=True)
        self.log_dict(
            {"val_l1_loss": l1_loss, "val_aux_loss": aux_loss, "val_hf_loss": hf_loss},
            sync_dist=self.sync_dist,
        )
        metrics = self.metrics(outputs.clip(0, 1), targets.clip(0, 1))
        self.log_dict(metrics, sync_dist=self.sync_dist)
        self.last_val_batch = {"source": inputs, "target": targets, "output": outputs}
        return loss

    @rank_zero_only
    def on_validation_end(self) -> None:
        """Log images."""
        source_images = self.last_val_batch["source"]
        sr_images = self.last_val_batch["output"]
        target_images = self.last_val_batch["target"]

        real_grid = torchvision.utils.make_grid(
            source_images.detach().cpu(), nrow=8, normalize=True
        ).permute(1, 2, 0)
        wandb.log(
            {
                "low_res_images": [
                    wandb.Image(real_grid.numpy(), caption="Low Resolution Images")
                ]
            }
        )

        sr_grid = torchvision.utils.make_grid(
            sr_images.detach().cpu(), nrow=8, normalize=True
        ).permute(1, 2, 0)
        wandb.log(
            {
                "sr_images": [
                    wandb.Image(sr_grid.numpy(), caption="Super Resolution Images")
                ]
            }
        )
        target_grid = torchvision.utils.make_grid(
            target_images.detach().cpu(), nrow=8, normalize=True
        ).permute(1, 2, 0)
        wandb.log(
            {
                "real_images": [
                    wandb.Image(target_grid.numpy(), caption="High Resolution Images")
                ]
            }
        )
        self.model.load_state_dict(self.ema_model.module.state_dict())
        self.model.model.push_to_hub("swin2sr-laion-hd")

        torch.save(
            self.ema_model.state_dict(), self.trainer.default_root_dir + "/model.pt"
        )
        garbage_collection_cuda()

    def on_before_zero_grad(self, *args, **kwargs):
        """Update EMA model."""
        self.ema_model.update_parameters(self.model)

    def configure_optimizers(self) -> Any:
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=0.5, milestones=[500000, 800000, 900000, 950000, 1000000]
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
