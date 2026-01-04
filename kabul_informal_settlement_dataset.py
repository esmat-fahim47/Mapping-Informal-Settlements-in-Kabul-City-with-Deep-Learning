from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset


class KabulInformalSettlementDataset(NonGeoDataset):
    """Dataset for Kabul Informal Settlements.

    Expects folder structure:

    data/
        train/
            images/
            masks/
        val/
            images/
            masks/
        test/
            images/
            masks/

    Returns:
        sample: dict with:
            - "image": FloatTensor (C x H x W)
            - "mask": LongTensor (H x W)
    """

    # RGB → class id
    CLASS_COLORS = {
        (0, 0, 0): 0,     # Formal settlements
        (255, 0, 0): 1,   # Informal settlements
        (0, 0, 255): 2,   # Background
    }

    def __init__(self, root: str, split: str = "train", transforms=None):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        self.images_dir = self.root / split / "images"
        self.masks_dir = self.root / split / "masks"

        self.images = sorted(self.images_dir.glob("*.tif"))
        self.masks = sorted(self.masks_dir.glob("*.png"))

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

        if len(self.images) != len(self.masks):
            raise RuntimeError("Number of images and masks must match")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        img_path = self.images[index]
        mask_path = self.masks[index]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        image = torch.from_numpy(
            np.array(image, dtype=np.float32)
        ).permute(2, 0, 1) / 255.0

        mask_np = np.array(mask)
        mask_id = np.zeros(mask_np.shape[:2], dtype=np.int64)

        for rgb, class_id in self.CLASS_COLORS.items():
            mask_id[np.all(mask_np == rgb, axis=-1)] = class_id

        mask = torch.from_numpy(mask_id).long()

        sample = {
            "image": image,
            "mask": mask,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
