from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.optim.optimizer as optim
from lightning import LightningModule
from torch.nn import functional as F


class UANet(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        optimizer: Any,
        criterion: Any,
        accuracy: Any,
        scheduler: Any = None,
    ) -> None:
        super().__init__()

        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.accuracy = accuracy
        if scheduler is not None:
            self.scheduler = scheduler

        self.min_val_loss = float("inf")
        self.max_val_accuracy = float("-inf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _common_step(
        self, batch: Any, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        x, y = batch
        y_hat = self(x)
        loss = (
            self.criterion(y_hat[0], y)
            + self.criterion(y_hat[1], y)
            + self.criterion(y_hat[2], y)
            + self.criterion(y_hat[3], y)
        )
        y_hat = torch.argmax(y_hat[3], dim=1)
        y_hat = F.one_hot(y_hat, num_classes=2).movedim(-1, 1)

        return y.to(torch.long), y_hat, loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        y, y_hat, loss = self._common_step(batch, batch_idx)
        self.log("train/loss", loss)
        self.log(
            "train/accuracy",
            self.accuracy(y_hat, y)[1],
            on_epoch=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        y, y_hat, loss = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_hat, y)[1]
        self.log("val/loss", loss)
        self.log("val/accuracy", accuracy)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        y, y_hat, loss = self._common_step(batch, batch_idx)
        self.log("test/loss", loss)
        self.log("test/accuracy", self.accuracy(y_hat, y)[1])
        return loss

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        return self(x)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = self.optimizer(params=self.parameters())
        if hasattr(self, "scheduler"):
            return {
                "optimizer": optimizer,
                "lr_scheduler": self.scheduler(optimizer=optimizer),
            }
        else:
            return optimizer
