from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

class KabulInformalSettlementDataset(NonGeoDataset):
    """Dataset for Kabul Informal Settlements.
    
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
    
    # RGB → class id
    CLASS_COLORS = {
        (0, 0, 0): 0,     # Formal settlements
        (255, 0, 0): 1,   # Informal settlements
        (0, 0, 255): 2,   # Background
    }
    
    def __init__(self, root: str, split: str = "train", transforms=None):   #constructor for the dataset
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        
        self.images_dir = self.root / split / "image"  #constructing paths to the subfolders
        self.masks_dir = self.root / split / "mask"
        
        self.images = sorted(self.images_dir.glob("*.tif"))   #finding and sorting images and masks
        self.masks = sorted(self.masks_dir.glob("*.png"))
        
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")  #check for sorting issues
        if len(self.images) != len(self.masks):
            raise RuntimeError("Number of images and masks must match")
    
    def __len__(self) -> int:     #return the number of samples in the dataset
        return len(self.images)
    
    def __getitem__(self, index: int) -> dict[str, Tensor]:  #return a single sample by index
        img_path = self.images[index]
        mask_path = self.masks[index]
        
        # Load images and masks
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        # Convert image to tensor
        image = torch.from_numpy(
            np.array(image, dtype=np.float32)
        ).permute(2, 0, 1) / 255.0
        
        # convert mask to a numpy array, 8-bit integers
        mask_np = np.array(mask, dtype=np.uint8)
        
        # Create unique values for each RGB combination
        mask_encoded = (mask_np[:, :, 0].astype(np.int32) * 65536 + 
                       mask_np[:, :, 1].astype(np.int32) * 256 + 
                       mask_np[:, :, 2].astype(np.int32))
        
        color_to_class = {
            0 * 65536 + 0 * 256 + 0: 0,      # Black (0,0,0) -> Formal
            255 * 65536 + 0 * 256 + 0: 1,    # Red (255,0,0) -> Informal
            0 * 65536 + 0 * 256 + 255: 2,    # Blue (0,0,255) -> Background
        }
        
        # Initialize mask with default class - background
        mask_id = np.full(mask_encoded.shape, 2, dtype=np.int64)
        
        # Vectorized mapping
        for encoded_color, class_id in color_to_class.items():
            mask_id[mask_encoded == encoded_color] = class_id
        
        mask = torch.from_numpy(mask_id).long()  # convert mask to LongTensor for use in CE
        
        
        sample = {   #create a dictionary for one sample
            "image": image,
            "mask": mask,
        }
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample
