import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Instantiate logger
    logger = WandbLogger(
        entity="mike-i",
        project="test",
        # log_model=True,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    # Set lighting seed
    L.seed_everything(cfg.seed)

    # Enable tensor cores
    torch.set_float32_matmul_precision("high")

    # Instantiate model
    model = instantiate(cfg.model)

    # Instantiate data module (See: https://lightning.ai/docs/pytorch/stable/data/datamodule.html)
    data_module = instantiate(cfg.dataset)

    # Log model gradients and topology
    # logger.watch(model)

    # Instantiate callbacks
    callbacks = []
    # callbacks.append(
    #    EarlyStopping(
    #        monitor="val/accuracy",
    #        mode="max",
    #        patience=50,
    #    )
    # )
    callbacks.append(
        ModelCheckpoint(
            monitor="val/accuracy",
            mode="max",
            save_top_k=1,
        )
    )
    callbacks.append(
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=True,
        )
    )

    # Instantiate trainer
    trainer = L.Trainer(
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        max_epochs=300,
        benchmark=True,
        precision=32,
        callbacks=callbacks,
        # =-=-= Debugging =-=-=
        # fast_dev_run=True,
        # overfit_batches=150,
        enable_checkpointing=True,
    )

    # Train model
    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    # Test model
    trainer.test(
        model=model,
        ckpt_path="best",
        datamodule=data_module,
    )


if __name__ == "__main__":
    main()
