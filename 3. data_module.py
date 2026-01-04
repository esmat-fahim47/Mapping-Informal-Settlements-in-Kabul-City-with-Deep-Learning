# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 22:51:56 2025

@author: esmat
"""

import torch
import kornia.augmentation as K
from typing import Any

from .geo import NonGeoDataModule
from .dataset_name import KabulInformalSettlementDataset


class KabulInformalSettlementDatamodule(NonGeoDataModule):
    """LightningDataModule for Kabul Informal Settlements dataset."""

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, size: int = 256, **kwargs: Any
    ) -> None:
        super().__init__(
            KabulInformalSettlementDataset,
            batch_size,
            num_workers,
            **kwargs
        )

        self.size = size

        # --- Geometric augmentations for train (image + mask) ---
        self.train_geom_aug = K.AugmentationSequential(
            K.Resize((size, size)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],  # ensures image and mask are augmented identically
            keepdim=True,
        )

        # --- Normalization for image only ---
        self.img_normalize = K.Normalize(self.mean, self.std)

        # --- Validation/Test augmentation (no random flips, just resize + normalize) ---
        self.val_geom_aug = K.AugmentationSequential(
            K.Resize((size, size)),
            data_keys=["input", "mask"],
            keepdim=True,
        )

    def setup(self, stage: str | None = None) -> None:
        """Set up train, val, and test datasets."""

        if stage in (None, "fit", "validate"):
            self.train_dataset = KabulInformalSettlementDataset(
                split="train",
                transforms=self._train_transform,
                **self.kwargs
            )
            self.val_dataset = KabulInformalSettlementDataset(
                split="val",
                transforms=self._val_transform,
                **self.kwargs
            )

        if stage in (None, "test", "predict"):
            self.test_dataset = KabulInformalSettlementDataset(
                split="test",
                transforms=self._val_transform,
                **self.kwargs
            )

    # --- Transform functions to pass to dataset ---
    def _train_transform(self, sample: dict) -> dict:
        img, mask = sample["image"], sample["mask"]

        # Add batch dimension for Kornia
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # Apply geometric augmentation
        img_aug, mask_aug = self.train_geom_aug(img, mask)

        # Normalize image
        img_aug = self.img_normalize(img_aug)

        # Remove batch dimension
        sample["image"] = img_aug.squeeze(0)
        sample["mask"] = mask_aug.squeeze(0).long()  # ensure mask is long
        return sample

    def _val_transform(self, sample: dict) -> dict:
        img, mask = sample["image"], sample["mask"]

        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)

        img_aug, mask_aug = self.val_geom_aug(img, mask)
        img_aug = self.img_normalize(img_aug)

        sample["image"] = img_aug.squeeze(0)
        sample["mask"] = mask_aug.squeeze(0).long()
        return sample
