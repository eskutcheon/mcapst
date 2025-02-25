import os
import math
from typing import Dict, Union, Optional, Literal, Any, Callable, List
from itertools import cycle
from multiprocessing import cpu_count
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as TT
# local imports
# TODO: add a factory method for HuggingFace datasets eventually
from .datasets import LocalImageDataset, HFImageDataset, HFStreamingIterable, test_if_valid_hf_dataset



def get_default_datasets(transfer_mode: Literal["artistic", "photorealistic", "art", "photo"]) -> Dict[str, str]:
    """ returns a dictionary of default datasets for the specified transfer mode """
    if transfer_mode in ["artistic", "art"]:
        return {"train_content": "bitmind/MS-COCO-unique-256", "train_style": "huggan/wikiart"}
    elif transfer_mode in ["photorealistic", "photo"]:
        return {"train_content": "bitmind/MS-COCO-unique-256", "train_style": "bitmind/MS-COCO-unique-256"}
    else:
        raise ValueError(f"Invalid transfer mode '{transfer_mode}'. Choose from ['artistic', 'photorealistic', 'art', 'photo'].")


class DataManager:
    """ wrapper around dataset classes for setting datasets, creating and iterating over the loaders, and loading image batches for training """
    def __init__(self, transfer_mode: Literal["artistic", "photorealistic", "art", "photo"], config: Any):
        self.transfer_mode = transfer_mode
        #self.modality = getattr(config, "modality", "image")
        self.streaming = getattr(config, "streaming", False)
        self.split = getattr(config, "split", "train")
        self.buffer_size = getattr(config, "shuffle_buffer", 0)
        self.batch_size = getattr(config, "batch_size", 1)
        self.use_local_datasets = getattr(config, "use_local_datasets", False)
            # use self.new_size for the resize transform
        resize_dim = getattr(config, "new_size", 256)
        preprocessor = TT.Compose([
            TT.Resize((resize_dim, resize_dim)),
            TT.ToDtype(torch.float32, scale=True),
        ])
        self.content_loader = self._build_loader(config.train_content, "content", preprocessor)
        self.style_loader = self._build_loader(config.train_style, "style", preprocessor)
        # using itertools.cycle to replace the InfiniteSampler from the old code
        # TODO: might just want to make the BaseImageDataset class into an infinite dataset setup
            # will be necessary after adding more parallelism arguments to the loaders
        self.content_iter = cycle(self.content_loader)
        self.style_iter = cycle(self.style_loader)


    def _validate_local_setting(self, root_or_name: str):
        """ checks if the provided local dataset path exists and is a valid directory """
        if not os.path.exists(root_or_name):
            raise FileNotFoundError(f"Local dataset path '{root_or_name}' does not exist, but use_local_datasets=True. " +
                                    "Please provide a valid local dataset path.")
        if not os.path.isdir(root_or_name):
            raise NotADirectoryError(f"Local dataset path '{root_or_name}' is not a directory as expected with use_local_dataset=True.")

    def _validate_hf_setting(self, root_or_name: str):
        if not test_if_valid_hf_dataset(root_or_name):
            if os.path.exists(root_or_name):
                raise ValueError(f"Invalid HuggingFace dataset path '{root_or_name}'. Provide a valid HuggingFace dataset name or set use_local_datasets=True.")
            raise ValueError(f"Invalid HuggingFace dataset name '{root_or_name}'. Provide a valid HuggingFace dataset name.")


    def _build_loader(self,
                      root_or_name: str,
                      loader_type: Optional[Literal["content", "style"]] = "content",
                      preprocessor: Optional[Callable] = None) -> DataLoader:
        # setting num_workers based on CPU count:
        # FIXME: won't work with pickling since the datasets have generators, but it'd be a lot slower without it - might just need to write a custom sampler
        #num_workers = 2**math.floor(math.log2(cpu_count())) if cpu_count() > 1 else 0
        if root_or_name is None:
            root_or_name = get_default_datasets(self.transfer_mode)[f"train_{loader_type}"]
        if self.use_local_datasets:
            self._validate_local_setting(root_or_name)
            ds = LocalImageDataset(root_or_name, transform=preprocessor)
            # for map-style dataset, use a normal DataLoader initialization
            return DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True) #, num_workers=num_workers)
        else:
            self._validate_hf_setting(root_or_name)
        # use HuggingFace streaming or map-style datasets
        if self.streaming: # and loader_type == "style":
            # NOTE: some HF sets only support "train" without options for "train[:1%]" or "train[1%:2%]" via the split
            ds = HFStreamingIterable(root_or_name, split=self.split, transform=preprocessor, buffer_size=self.buffer_size)
            # For an IterableDataset, we must set `batch_size` and `shuffle=False`in the data loaders
            # NOTE: can't random-shuffle an iterable dataset, so we use a buffer shuffle inside HFStreamingIterable.
            return DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=True) #, num_workers=num_workers)
        else:
            ds = HFImageDataset(dataset_name=root_or_name, split=self.split, transform=preprocessor)
            # NOTE: using drop_last = True since I have to have the same batch size for content and style images
                # might rewrite all datasets to group content and style later in a grouped dataset though
            return DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True) #, num_workers=num_workers)

    def get_next_batches(self):
        content_batch: Dict[str, torch.Tensor] = next(self.content_iter)
        style_batch: Dict[str, torch.Tensor] = next(self.style_iter)
        return content_batch, style_batch

    def __next__(self):
        return self.get_next_batches()
