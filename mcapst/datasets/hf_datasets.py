from typing import Dict, Union, Optional, Literal, Any, Callable, List
from itertools import cycle
import torch
from torch.utils.data import IterableDataset
import random
import datasets
from torch.utils.data import DataLoader
#from torchvision import datasets as tv_datasets
import torchvision.transforms.v2 as TT
# local imports
from .datasets import BaseImageDataset

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}



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
        self.transform = transform if transform else TT.Compose([
            TT.PILToTensor(),
            TT.Resize((256, 256)),
            TT.ToDtype(torch.float32, scale=True),
        ])
        self.buffer_size = buffer_size
        # avoiding using .with_format("torch") like the other classes because streaming + with_format has issues
        self.raw_dataset = datasets.load_dataset(self.dataset_name, split=self.split, streaming=True)

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
        # If we want to shuffle, wrap base_iter with a shuffle generator
        sample_iter = self._shuffle_generator(base_iter) if self.buffer_size > 0 else base_iter
        # Finally yield each item
        for sample in sample_iter:
            # apply transforms to image from the iterator item
            img = self.transform(sample["image"])
            yield {"img": img}

    def __len__(self):
        # We do not know the length of a streaming dataset
        raise TypeError("HFStreamingIterable has no length (infinite or unknown).")