from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


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
        sample: dict with "image" (C x H x W float tensor) and "mask" (H x W long tensor)
    """

    # Define class colors (RGB) and their corresponding class IDs
    CLASS_COLORS = [
        (0, 0, 0),  # 0: Formal Settlements
        (255, 0, 0),  # 1: Informal Settlements
        (0, 0, 255),  # 2: Background
        
    ]

    def __init__(self, root: str = "data", split: str = "train", transforms=None):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        # Paths to images and masks
        self.images_dir = self.root / split / "images"
        self.masks_dir = self.root / split / "masks"

        # Get sorted list of files
        self.images = sorted(list(self.images_dir.glob("*.tif")))  # or *.png
        self.masks = sorted(list(self.masks_dir.glob("*.png")))

        # Ensure images and masks match
        assert len(self.images) == len(self.masks), "Number of images and masks must match"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        # Load image and mask
        img_path = self.images[index]
        mask_path = self.masks[index]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")  # keep RGB for color mapping

        # Convert image to tensor (C x H x W, float)
        img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Convert mask RGB to class IDs
        mask_np = np.array(mask)
        mask_id = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)

        for class_id, color in enumerate(self.CLASS_COLORS):
            equality = np.all(mask_np == color, axis=-1)
            mask_id[equality] = class_id

        mask_tensor = torch.tensor(mask_id, dtype=torch.long)

        sample = {"image": img, "mask": mask_tensor}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def plot(self, index: int = 0) -> Figure:
        """Plot an image and its mask side by side."""
        sample = self.__getitem__(index)
        img = sample["image"].permute(1, 2, 0).numpy()
        mask = sample["mask"].numpy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title("Image")
        ax[1].imshow(mask, cmap="tab20")
        ax[1].set_title("Mask")
        return fig

