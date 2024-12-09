import os
import pickle
from typing import Callable, Optional, Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from lightning.pytorch import LightningDataModule
from torchvision import tv_tensors
import torchvision.transforms.v2 as T
from torch.nn import functional as F


class BuildingDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        transform: Optional[Callable] = None,
        common_train_transform: Optional[Callable] = None,
    ) -> None:
        self.data = data
        self.transform = transform
        self.common_train_transform = common_train_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        image = sample["image"] / 8000
        image[image > 1] = 1
        mask = sample["mask"]

        # Convert mask to tensor and apply one-hot encoding
        mask = torch.tensor(mask, dtype=torch.long)
        mask = F.one_hot(mask, num_classes=2).movedim(-1, 0)
        mask = tv_tensors.Mask(data=mask)

        # Apply image transformations if available
        if self.transform:
            image = self.transform(image)

        if self.common_train_transform:
            image, mask = self.common_train_transform(image, mask)
        else:
            image, mask = T.Resize(size=512)(image, mask)

        return (
            image.to(torch.float32),
            mask.to(torch.float32),
        )


class BuildingDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = ".",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.6,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=[0.18274285, 0.20637984, 0.22696759],
                    std=[0.04133136, 0.0484684, 0.0644507],
                ),
            ]
        )
        self.train_transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                # T.RandomSolarize(0.3),
                T.Normalize(
                    mean=[0.18274285, 0.20637984, 0.22696759],
                    std=[0.04133136, 0.0484684, 0.0644507],
                ),
            ]
        )
        self.common_train_transform = T.Compose(
            [
                # T.RandomHorizontalFlip(),
                # T.RandomVerticalFlip(),
                T.RandomResizedCrop(size=512, scale=(0.25, 1)),
            ]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_path = os.path.join(self.data_dir, "dataset.pickle")
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        self.train_data = BuildingDataset(
            data["train"],
            transform=self.train_transform,
            common_train_transform=self.common_train_transform,
        )
        val_dataset = BuildingDataset(data["val"], transform=self.transform)

        val_len = int(len(val_dataset) * self.val_split)
        test_len = len(val_dataset) - val_len
        self.val_data, self.test_data = random_split(val_dataset, [val_len, test_len])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
