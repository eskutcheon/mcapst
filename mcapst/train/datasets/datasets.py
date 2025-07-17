import os
from glob import glob
from typing import Dict, Union, Optional, Literal, Any, Callable, List
import random
from PIL.Image import Image
import torch
from torch.utils.data import Dataset, IterableDataset
import datasets # huggingface datasets
import torchvision.transforms.v2 as TT
import torchvision.io as IO



ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


class BaseImageDataset(Dataset):
    """ Base class for both HuggingFace and local image datasets that includes the optional Matting Laplacian computation """
    def __init__(self, transform: Optional[Callable] = None):
        super().__init__()
        DEFAULT_RESIZE_DIM = 256
        self.transform = transform if transform else TT.Compose([
            # TT.ToPureTensor(),
            TT.Lambda(lambda x: TT.functional.pil_to_tensor(x) if isinstance(x, Image) else x),
            TT.Resize((DEFAULT_RESIZE_DIM, DEFAULT_RESIZE_DIM)),
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
        #? NOTE: original CAP-VSTNet training scripts didn't use any masking during training, so don't bother supporting it here
        return {"img": img}



class HFImageDataset(BaseImageDataset):
    """ HuggingFace dataset class for loading images from a specified dataset name and split """
    def __init__(self, dataset_name, split="train", transform: Optional[Callable] = None):
        super().__init__(transform)
        # TODO: explore more of the options in datasets.load_dataset() to optimize loading
        self.dataset = datasets.load_dataset(dataset_name, split=split)
        # streaming=True used together with .with_format("torch") doesn't work quite right
        self.dataset = self.dataset.with_format("torch")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        if self.transform:
            image = self.transform(image)
        return {"img": image}


# REFERENCE LATER: https://medium.com/@amit25173/how-to-use-dataloader-with-iterabledataset-in-pytorch-an-advanced-practical-guide-898a49ace81c

class HFStreamingIterable(IterableDataset):
    """ IterableDataset that loads images from a streaming HuggingFace dataset.
        Optionally shuffles incoming data using a small buffer (inspired by TFRecord-like shuffling).
        :param dataset_name: e.g. "huggan/wikiart"
        :param split: e.g. "train" or "validation" or "train[:1%]"
        :param transform: optional torchvision-style transform to apply to each image
        :param buffer_size: if >0, do reservoir-like shuffle with that buffer size
    """
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        buffer_size: int = 0
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.buffer_size = buffer_size
        self._set_transforms(transform)
        # avoiding using .with_format("torch") like the other classes because streaming + with_format has issues
        self.raw_dataset = datasets.load_dataset(self.dataset_name, split=self.split, streaming=True)

    def _set_transforms(self, transform: TT.Compose = None):
        DEFAULT_SIZE = 256
        conversion_func = TT.Lambda(lambda x: TT.functional.pil_to_tensor(x) if isinstance(x, Image) else x)
        if transform:
            transform.transforms.insert(0, conversion_func)
            self.transform = transform
        else:
            self.transform = transform if transform else TT.Compose([
                conversion_func,
                TT.Resize((DEFAULT_SIZE, DEFAULT_SIZE)),
                TT.ToDtype(torch.float32, scale=True),
            ])

    def _shuffle_generator(self, iterator):
        """ Expects an iterator of (image) items, and uses a generator with pseudo-random ordering using a small in-memory buffer """
        buffer = []
        for x in iterator:
            if len(buffer) < self.buffer_size:
                buffer.append(x)
            else:
                # random index in [0, buffer_size)
                idx = random.randint(0, self.buffer_size-1)
                # yield one from the buffer
                yield buffer[idx]
                # replace it with the new item
                buffer[idx] = x
        # once we exhaust the iterator, yield all remaining in random order
        random.shuffle(buffer)
        for x in buffer:
            yield x

    def __iter__(self):
        # self.raw_dataset is an iterator itself
        base_iter = iter(self.raw_dataset)  # yields dict with "image", "label", etc.
        # if we want to shuffle, wrap base_iter with a shuffle generator
        sample_iter = self._shuffle_generator(base_iter) if self.buffer_size > 0 else base_iter
        # finally yield each item
        for sample in sample_iter:
            # apply transforms to image from the iterator item
            img = self.transform(sample["image"])
            yield {"img": img}

    def __len__(self):
        # ERROR: can't know the length of a streaming dataset
        raise TypeError("HFStreamingIterable has no length (infinite or unknown).")




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
        print(f"Error accessing dataset '{name}': {e}")
        print(f"Ensure this is a valid HuggingFace dataset and appropriate permissions are enabled.")
        return False




    # TODO: add a new dataset for other remote datasets, e.g. from Kaggle, cloud storage, or databases

    # TO ADD LATER::
        # TODO: include all necessary scripting to package trained models for upload to HuggingFace model hub
            # e.g. model card, recipe, readme, versioning, etc.
            # TODO: include a script to load my own trained models from the hub (for quick-start)
        # TODO: include all necessary scripting to package any NEW datasets for upload to HuggingFace Hub