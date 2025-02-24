import os
from glob import glob
from typing import Dict, Union, Optional, Literal, Any, Callable, List
from itertools import cycle
from PIL.Image import Image
import torch
#from torchvision import datasets as tv_datasets
import torchvision.transforms.v2 as TT
import torchvision.io as IO
#from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset


ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


class BaseImageDataset(Dataset):
    """ Base class for both HuggingFace and local image datasets that includes the optional Matting Laplacian computation """
    def __init__(self, transform: Optional[Callable] = None): #, use_lap=True, win_rad=1):
        super().__init__()
        self.transform = transform if transform else TT.Compose([
            # TT.ToPureTensor(),
            TT.Lambda(lambda x: TT.functional.pil_to_tensor(x) if isinstance(x, Image) else x),
            # TODO: remove hard-coding and read from a config or arguments from the caller
            TT.Resize((256, 256)),
            TT.ToDtype(torch.float32, scale=True),
        ])

    # __len__ and __getitem__ are meant to be overridden by subclasses
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class LocalImageDataset(BaseImageDataset):
    def __init__(self, root, transform: Optional[Callable] = None, recursive: bool = True):
        super().__init__(transform)
        self.root = root
        self._scan_files(root, recursive)  # Scan the directory for image files


    def _scan_files(self, root, recursive):
        """ Scans the directory for image files and populates self.files """
        self.files: List[str] = []
        pattern = "**/*" if recursive else "*"
        # Gather all valid image files
        for f in glob(os.path.join(root, pattern), recursive=recursive):
            ext = os.path.splitext(f)[1].lower()
            if ext in ALLOWED_EXTS and os.path.isfile(f):
                self.files.append(f)
        if not self.files:
            raise FileNotFoundError(f"No image files found in {root} (recursive={recursive}).")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = self.transform(IO.read_image(img_path, mode=IO.ImageReadMode.RGB))
        # TODO: add content mask support
        return {"img": img}