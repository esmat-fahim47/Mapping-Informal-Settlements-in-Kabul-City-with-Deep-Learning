import torch
import kornia.augmentation as K
from torchgeo.datamodules import NonGeoDataModule
from datasets.kabul_informal_settlement_dataset import KabulInformalSettlementDataset

class KabulInformalSettlementDatamodule(NonGeoDataModule):
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    
    def __init__(
        self,
        root: str,
        batch_size: int = 8,       # smaller batch for large images
        num_workers: int = 0,      # parallelize loading
    ) -> None:
        super().__init__(
            KabulInformalSettlementDataset,
            batch_size,
            num_workers,
            root=root,
        )
        self.root = root
        # Don't create augmentations here - they can't be pickled, create them in setup() instead
        
    def setup(self, stage: str | None = None) -> None:
        """Set up train, val, and test datasets."""

        self.train_geom_aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],  # loading the image and its mask together
            keepdim=True,
        )
        
        self.img_normalize = K.Normalize(self.mean, self.std)
        
        self.val_geom_aug = K.AugmentationSequential(    # No flipping or distortion for the val set
            data_keys=["input", "mask"],
            keepdim=True,
        )
        
        if stage in (None, "fit", "validate"):  #assigning transform functions and self.kwargs for further arguments
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
    
    # Application of the augmentatoins via the transform functions 
    def _train_transform(self, sample: dict) -> dict:
        img, mask = sample["image"], sample["mask"]
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        img_aug, mask_aug = self.train_geom_aug(img, mask)
        img_aug = self.img_normalize(img_aug)
        sample["image"] = img_aug.squeeze(0)
        sample["mask"] = mask_aug.squeeze(0).long()
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
