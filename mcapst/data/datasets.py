import os
from glob import glob
from typing import Dict, Union, Optional, Literal, Any, Callable, List
from itertools import cycle
from PIL.Image import Image
import torch
import datasets
from torch.utils.data import DataLoader
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



class HFImageDataset(BaseImageDataset):
    """ HuggingFace dataset class for loading images from a specified dataset name and split """
    #!! TODO: remember to remove the hardcoding on split="train[:1%]" when testing with the full dataset
    def __init__(self, dataset_name, split="train[:1%]", transform: Optional[Callable] = None, streaming: bool = False): #, use_lap=True, win_rad=1):
        super().__init__(transform) #, use_lap, win_rad)
        self.streaming = streaming
        self.dataset = datasets.load_dataset(dataset_name, split=split, streaming=streaming)
        # streaming=True used together with .with_format("torch") doesn't work quite right
        if not streaming:
            self.dataset = self.dataset.with_format("torch")

    def __iter__(self):
        if self.streaming:
            for item in self.dataset:
                yield {"img": self.transform(item["image"])}

    def __len__(self):
        if self.streaming:
            # streaming datasets are "infinite" or unknown length so raise an error
            raise TypeError("HFImageDataset in streaming mode has no __len__.")
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        if self.streaming:
            raise RuntimeError("Cannot index a streaming dataset with __getitem__")
        image = self.dataset[idx]["image"]
        if self.transform:
            image = self.transform(image)
        # TODO: add content mask support
        return {"img": image}

    # def __del__(self):
    #     self.dataset.cleanup_cache_files()





def test_if_valid_hf_dataset(name: str) -> bool:
    """ checks if the provided dataset name is valid and accessible on HuggingFace """
    from urllib.request import urlopen
    from urllib.error import HTTPError
    try:
        # NOTE: should maybe be f"https://huggingface.co/datasets/{name}/blob/main/README.md"
        url = f"https://huggingface.co/datasets/{name}/resolve/main/README.md"
        response = urlopen(url)
        # return whether the HTTP 200 OK status code was returned by the server
        return response.status == 200
    except HTTPError as e:
        print(f"Error accessing dataset {name}: {e}")
        return False



class DataManager:
    """ wrapper around dataset classes for setting datasets, creating and iterating over the loaders, and loading image batches for training """
    def __init__(self, transfer_mode: Literal["artistic", "photorealistic", "art", "photo"], config: Any):
        # TODO: handle normalization better in the pipelines
        valid_modes = ["artistic", "photorealistic", "art", "photo"]
        if transfer_mode not in valid_modes:
            raise ValueError(f"Invalid transfer mode '{transfer_mode}'. Choose from {valid_modes}.")
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.streaming = getattr(config, "streaming", False)
        self.split = getattr(config, "split", "train")
        self.batch_size = getattr(config, "batch_size", 1)
        # TODO: generalized these arguments further in the near future
        self.content_loader = self._build_loader("train_content", transfer_mode, config.use_local_datasets, self.batch_size)
        self.style_loader = self._build_loader("train_style", transfer_mode, config.use_local_datasets, self.batch_size)
        # using itertools.cycle to replace the InfiniteSampler from the old code
        self.content_iter = cycle(self.content_loader)
        self.style_iter = cycle(self.style_loader)

    def _build_loader(self, root_or_name, transfer_mode, use_local, batch_size):
        if use_local:
            ds = LocalImageDataset(root_or_name)
        else:
            dataset_name = self._set_hf_dataset(root_or_name, transfer_mode)
            ds = HFImageDataset(dataset_name, split=self.split, transform=None, streaming=self.streaming)
        if not use_local and self.streaming:
            return ds  # return the dataset directly if streaming is enabled
            #return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True, timeout=30)  # no shuffling for streaming datasets
        # TODO: add more loader arguments later, like num_workers, pin_memory, etc.
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(not self.streaming))
        return loader

    def _set_hf_dataset(self, dataset_path, mode):
        """ setting the default datasets for training, following the original CAP-VSTNet paper """
        # TODO: extend this to use specific huggingface datasets specified by the config
        # TODO: set up a dictionary of these dataset names and splits for each transfer mode and make selection here more flexible
        if dataset_path == "train_content":
            return "bitmind/MS-COCO-unique-256"
        elif dataset_path == "train_style":
            return "huggan/wikiart" if mode in ["art", "artistic"] else "bitmind/MS-COCO-unique-256"
        else:
            if not test_if_valid_hf_dataset(dataset_path):
                raise ValueError(f"Invalid dataset name '{dataset_path}'. Please provide a valid HuggingFace dataset name.")
            return dataset_path

    # TODO: might just make this a __next__ method instead of a separate function so that I can use the iterator directly
    def get_next_batches(self):
        content_batch: Dict[str, torch.Tensor] = next(self.content_iter)
        style_batch: Dict[str, torch.Tensor] = next(self.style_iter)
        return content_batch, style_batch

    def __next__(self):
        return self.get_next_batches()
