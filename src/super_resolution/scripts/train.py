"""Train script."""

import argparse
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from super_resolution.datamodule import SuperResolutionDataModule
from super_resolution.datasets.image import ImageDataset
from super_resolution.datasets.laion_hd import LaionHDDataset
from super_resolution.lightning_module import SuperResolutionLightningModule
from super_resolution.modeling.swin2sr.model import Swin2SR

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def main() -> None:
    """Train script."""
    parser = argparse.ArgumentParser(description="Super resolution training scripts.")
    parser.add_argument("name", type=str)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--log_path", default="logs", type=Path)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_devices", default=1, type=int)
    parser.add_argument("--ckpt_path", type=Path, default=None)
    parser.add_argument("--weights_path", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overfit", action="store_true")

    args = parser.parse_args()

    seed_everything(1234)

    log_path = args.log_path / args.name
    log_path.mkdir(exist_ok=True, parents=True)

    model = Swin2SR()
    lightning_module = SuperResolutionLightningModule(
        model=model, sync_dist=args.num_devices > 1
    )

    if args.weights_path:
        lightning_module.model.load_state_dict(
            torch.load(args.weights_path)["state_dict"]
        )

    if args.data_root:
        dataset: Any = ImageDataset(args.data_root)
    else:
        dataset = LaionHDDataset()

    datamodule = SuperResolutionDataModule(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    logger = WandbLogger(
        project="super-resolution",
        save_dir=log_path,
        name=args.name,
        offline=args.debug,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename="{epoch:02d}-{val_loss:.4f}",
        save_last=True,
        monitor="val_loss",
        save_top_k=5,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = log_path / "last.ckpt"

    trainer = Trainer(
        default_root_dir=log_path,
        max_epochs=10000,
        accelerator="auto",
        devices=args.num_devices,
        logger=logger,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback, lr_callback],
        limit_val_batches=100,
        val_check_interval=5000 if len(dataset) // args.batch_size > 5000 else None,
        overfit_batches=0.05 if args.overfit else 0.0,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        detect_anomaly=False,
        strategy="ddp_find_unused_parameters_true" if args.num_devices > 1 else "auto",
    )
    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )


if __name__ == "__main__":
    main()
