# 1 the dataset_name.py file

from collections.abc import Callable

from matplotlib.pyplot import Figure
from torch import Tensor

from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import Path


class KabulInformalSettlementDataset(NonGeoDataset):
    """MyNewDataset.

    Short summary of the dataset and link to its homepage.

    Dataset features:

    * number of classes
    * sensors
    * area covered
    * etc.

    Dataset format:

    * what file format and shape the input data comes in
    * what file format and shape the target data comes in
    * possible metadata files

    If you use this dataset in your research, please cite the following paper:

    * URL of publication or citation information

    .. versionadded:: next TorchGeo minor release version, e.g., 1.0
    """

    # In this part of the code you can define class attributes such as a list of
    # class names, color maps, url and checksums for data download, and other
    # attributes that one might require repeatedly in the subsequent class methods.

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        download: bool = False,
    ) -> None:
        """Initialize the dataset.

        The init parameters can include additional arguments, such as an option to
        select specific image bands, data modalities, or other arguments that give
        greater control over data loading. They should all have reasonable defaults.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
        """

    def __len__(self) -> int:
        """The length of the dataset.

        This is the total number of samples per epoch, and is used to define the
        maximum allow index that can be passed to `__getitem__`.
        """

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """A single sample from the dataset.

        Load a single input image and target label or mask, and return it in a
        dictionary.
        """

    def plot(self) -> Figure:
        """Plot a sample of the dataset for visualization purposes
        This might involve selecting the RGB bands, using a colormap to display a mask.
        adding a legend with class labels, etc."""