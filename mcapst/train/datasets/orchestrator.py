
from typing import Dict, Optional, Literal, Any, Callable
from itertools import cycle
# import math
# from multiprocessing import cpu_count
from PIL.Image import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as TT
# local imports
# TODO: add a factory method for HuggingFace datasets
from .datasets import LocalImageDataset, HFImageDataset, HFStreamingIterable


#& update now doing validation and setting fallback default datasets earlier in the config

class DataManager:
    """ wrapper around dataset classes for setting datasets, creating and iterating over the loaders, and loading image batches for training """
    def __init__(self, transfer_mode: Literal["artistic", "photorealistic", "art", "photo"], config: Any):
        self.transfer_mode = transfer_mode
        self.streaming = getattr(config, "streaming", False)
        self.split = getattr(config, "split", "train")
        self.buffer_size = getattr(config, "shuffle_buffer", 0)
        self.batch_size = getattr(config, "batch_size", 1)
        self.use_local_data = getattr(config, "use_local_data", False)
        # use config.new_size for the resize transform or default to 256
        resize_dim = getattr(config, "new_size", 256)
        preprocessor = TT.Compose([
            TT.Lambda(lambda x: TT.functional.pil_to_tensor(x) if isinstance(x, Image) else x),
            TT.Resize((resize_dim, resize_dim)),
            TT.ToDtype(torch.float32, scale=True),
        ])
        self.content_loader = self._build_loader(config.train_content, "content", preprocessor)
        self.style_loader = self._build_loader(config.train_style, "style", preprocessor)
        # TODO: might just want to make the BaseImageDataset class into an infinite dataset setup (since it already doesn't use epochs)
            # will be necessary after adding more parallelism arguments to the loaders
        # using itertools.cycle to replace the InfiniteSampler from the old code
        self.content_iter = cycle(self.content_loader)
        self.style_iter = cycle(self.style_loader)


    # TODO: maybe make the dataset type a pydantic model since most of the conditional logic here is taken care of by the config
        # could access LocalImageDataset, HFImageDataset, or HFStreamingIterable from a registry or factory method and construct args to a common dataloader instantiation
    def _build_loader(
        self,
        root_or_name: str,
        loader_type: Optional[Literal["content", "style"]] = "content",
        preprocessor: Optional[Callable] = None
    ) -> DataLoader:
        #! setting num_workers based on CPU count:
        #! FIXME: won't work with pickling since the datasets have generators, but it'd be a lot slower without it - might just need to write a custom sampler
            # TODO: maybe add the streaming piped dataloader I wrote for the feature space analysis project
        #num_workers = 2**math.floor(math.log2(cpu_count())) if cpu_count() > 1 else 0
        ds = None
        # NOTE: using drop_last = True since I have to have the same batch size for content and style images
            # might rewrite all datasets to group content and style later in a grouped dataset though
        loader_kwargs: Dict[str, Any] = {"batch_size": self.batch_size, "shuffle": True, "drop_last": True} #, "num_workers": num_workers}
        if self.use_local_data:
            # for map-style dataset, use a normal DataLoader initialization with shuffle=True
            ds = LocalImageDataset(root_or_name, transform=preprocessor)
        # use HuggingFace streaming or map-style datasets
        elif self.streaming: # and loader_type == "style":
            # NOTE: can't random-shuffle an iterable dataset, so we use a buffer shuffle inside HFStreamingIterable.
            loader_kwargs["shuffle"] = False
            # NOTE: some HF sets only support "train" without options for "train[:1%]" or "train[1%:2%]" via the split
            ds = HFStreamingIterable(root_or_name, split=self.split, transform=preprocessor, buffer_size=self.buffer_size)
        else:
            ds = HFImageDataset(dataset_name=root_or_name, split=self.split, transform=preprocessor)
        return DataLoader(ds, **loader_kwargs) #, num_workers=num_workers)

    def get_next_batches(self):
        content_batch: Dict[str, torch.Tensor] = next(self.content_iter)
        style_batch: Dict[str, torch.Tensor] = next(self.style_iter)
        return content_batch, style_batch

    def __next__(self):
        return self.get_next_batches()
