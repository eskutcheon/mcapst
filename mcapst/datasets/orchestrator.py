import os
from typing import Dict, Union, Optional, Literal, Any, Callable, List
from itertools import cycle
import torch
from torch.utils.data import DataLoader
# local imports
from .datasets import LocalImageDataset
from .hf_datasets import HFImageDataset, HFStreamingIterable, test_if_valid_hf_dataset



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
        self._validate_transfer_mode(transfer_mode)
        self.transfer_mode = transfer_mode
        #self.modality = getattr(config, "modality", "image")
        self.streaming = getattr(config, "streaming", False)
        self.split = getattr(config, "split", "train")
        self.buffer_size = getattr(config, "shuffle_buffer", 0)
        self.batch_size = getattr(config, "batch_size", 1)
        self.use_local_datasets = getattr(config, "use_local_datasets", False)
        # TODO: create transform pipeline for the datasets here, pass it to _build_loader, and pass it to the dataset classes
            # use self.new_size for the resize transform
        self.new_size = getattr(config, "new_size", 512)
        self.content_loader = self._build_loader(config.train_content, "content")
        self.style_loader = self._build_loader(config.train_style, "style")
        # using itertools.cycle to replace the InfiniteSampler from the old code
        # TODO: might just want to make the BaseImageDataset class into an infinite dataset setup
        self.content_iter = cycle(self.content_loader)
        self.style_iter = cycle(self.style_loader)


    def _validate_local_setting(self, root_or_name: str):
        """ checks if the provided local dataset path exists and is a valid directory """
        if self.streaming:
            raise NotImplementedError(f"Disjoint config values 'use_local_datasets' and 'streaming' are both True. Set only one of these to True.")
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

    def _validate_transfer_mode(self, transfer_mode: str):
        # TODO: handle normalization better in the pipelines to avoid this
        valid_modes = ["artistic", "photorealistic", "art", "photo"]
        if transfer_mode not in valid_modes:
            raise ValueError(f"Invalid transfer mode '{transfer_mode}'. Choose from {valid_modes}.")


    def _build_loader(self, root_or_name, loader_type: Optional[Literal["content", "style"]] = "content"):
        if root_or_name is None:
            root_or_name = get_default_datasets(self.transfer_mode)[f"train_{loader_type}"]
        if self.use_local_datasets:
            self._validate_local_setting(root_or_name)
            ds = LocalImageDataset(root_or_name)
            # for map-style dataset, use a normal DataLoader initialization
            return DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        else:
            self._validate_hf_setting(root_or_name)
        # use HuggingFace streaming or map-style datasets
        if self.streaming:
            # NOTE: some HF sets only support "train" without options for "train[:1%]" or "train[1%:2%]" via the split
            ds = HFStreamingIterable(root_or_name, split=self.split, transform=None, buffer_size=self.buffer_size)
            # TODO: add self.transform later
            # For an IterableDataset, we must set `batch_size` and `shuffle=False`in the data loaders
            # NOTE: can't random-shuffle an iterable dataset, so we use a buffer shuffle inside HFStreamingIterable.
            return DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=True)
        else:
            ds = HFImageDataset(dataset_name=root_or_name, split=self.split)
            return DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    def get_next_batches(self):
        content_batch: Dict[str, torch.Tensor] = next(self.content_iter)
        style_batch: Dict[str, torch.Tensor] = next(self.style_iter)
        return content_batch, style_batch

    def __next__(self):
        return self.get_next_batches()
