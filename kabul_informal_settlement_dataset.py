from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset


class KabulInformalSettlementDataset(NonGeoDataset):
    """
    Dataset for Kabul Informal Settlements.
    
    Class IDs:
    0 → Formal settlements
    1 → Background 
    2 → Informal settlements

    Expected folder structure:
    data/
        train/
            image/
            mask/
        val/
            image/
            mask/
        test/
            image/
            mask/
    """

    def __init__(self, root: str, split: str = "train", transforms=None):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        self.images_dir = self.root / split / "image"
        self.masks_dir = self.root / split / "mask"

        self.images = sorted(self.images_dir.glob("*.tif"))
        self.masks = sorted(self.masks_dir.glob("*.tif"))

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")
        if len(self.images) != len(self.masks):
            raise RuntimeError("Number of images and masks must match")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        img_path = self.images[index]
        mask_path = self.masks[index]

        # --- Load image (RGB) ---
        image = Image.open(img_path).convert("RGB")
        image = torch.from_numpy(
            np.array(image, dtype=np.float32)
        ).permute(2, 0, 1) / 255.0

        # --- Load mask (class IDs: 0,1,2) ---
        mask = Image.open(mask_path)
        mask = torch.from_numpy(
            np.array(mask, dtype=np.int64)
        )

        sample = {
            "image": image,
            "mask": mask,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
