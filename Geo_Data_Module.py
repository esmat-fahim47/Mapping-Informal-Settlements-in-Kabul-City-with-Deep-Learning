from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import GridGeoSampler
from torch.utils.data import DataLoader

class GeoDataModule(NonGeoDataModule):
    """GeoDataModule for tiled inference"""

    def __init__(self, big_image_path: str, batch_size: int = 1, tile_size: int = 512, stride: int = 256, num_workers: int = 0):
        super().__init__(RasterDataset, batch_size=batch_size, num_workers=num_workers, root=None)
        self.big_image_path = big_image_path
        self.tile_size = tile_size
        self.stride = stride
        self.num_workers = num_workers

    def predict_dataloader(self):
        dataset = RasterDataset(self.big_image_path)
        sampler = GridGeoSampler(dataset, size=self.tile_size, stride=self.stride)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
        )

