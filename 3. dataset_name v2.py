# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 22:51:56 2025

@author: esmat
"""

"""NewDatasetDataModule datamodule."""

import os
from typing import Any

import kornia.augmentation as K
import torch

from .geo import NonGeoDataModule
from .dataset_name import KabulInformalSettlementDataset


class KabulInformalSettlementDatamodule(NonGeoDataModule):
    """LightningDataModule implementation for the NewDataset dataset.

    This DataModule should hove below such subfolders:

        root/
            train/images, train/masks
            val/images,   val/masks
            test/images,  test/masks
    """

    mean = torch.Tensor([0.5, 0.4, 0.3])
    std = torch.Tensor([1.5, 1.4, 1.3])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, size: int = 256, **kwargs: Any
    ) -> None:

        super().__init__(
            KabulInformalSettlementDataset,
            batch_size,
            num_workers,
            **kwargs
        )

        # augmentations for training
        self.train_aug = K.AugmentationSequential(
            K.Resize((size, size)),
            K.Normalize(self.mean, self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

        # augmentations for val/test
        self.aug = K.AugmentationSequential(
            K.Normalize(self.mean, self.std),
            K.Resize((size, size)),
            data_keys=None,
            keepdim=True,
        )

        self.size = size

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for train/val/test based on predefined folder splits."""

        # Load training + validation splits
        if stage in (None, "fit", "validate"):
            self.train_dataset = KabulInformalSettlementDataset(
                split="train",
                **self.kwargs
            )
            self.val_dataset = KabulInformalSettlementDataset(
                split="val",
                **self.kwargs
            )

        # Load test split
        if stage in (None, "test", "predict"):
            self.test_dataset = KabulInformalSettlementDataset(
                split="test",
                **self.kwargs
            )
