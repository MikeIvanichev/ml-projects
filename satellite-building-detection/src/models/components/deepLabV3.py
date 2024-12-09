from torch import nn
import torch
from torchvision import models


class DeepLabV3(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
    ):
        super().__init__()

        # Load the pre-trained DeepLabV3 model with MobileNetV3 backbone
        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(
            num_classes=num_classes
        )

    def forward(self, x):
        y = self.model(x)["out"]
        return y

