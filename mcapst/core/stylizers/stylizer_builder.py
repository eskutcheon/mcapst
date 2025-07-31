
"""
Planning to implement an abstract factory / builder class to construct stylizers, preprocessors, postprocessors,
    and (possibly) a separate mask validator/generator.
- If I stick with the plan of using a new pydantic model `StylizerParams` for the stylizer parameters, it may be built here as well.
- This may be more than one function/class, but for now, I'll just start with the StylizerBuilder class.
- Some of the ideas that I had related to this refactor will probably be left to another version, but I wanted to write it down here first.
    - In that new version, I wanted to encapsulate the RevResNet model into the actual CAPVSTNet class and make it a torch.nn.Module,
        but it requires a TON of rewiring.
    - Currently, I'm considering creating a minimal CAPVSTNet base class and two subclasses: one without masking that's a TorchScript module, while the masked one isn't.
        - This may or may not be feasible, since I haven't explored the actual limitations of TorchScript with custom class inheritance yet.
    - CAPVSTNet would then have essentially 4 tasks: generating features with its RevResNet model, applying cWCT, interpolating aligned features,
        and passing around the RevResNet model (for training).
- Eventually, the basic stylizers will be much leaner and will basically be a lower level dispatcher for both submodules and the API endpoint(s),
    as well as an interface to the RevResNet model.
    - They'll no longer implement any preprocessing or postprocessing, but will instead rely on the StylizerBuilder to construct those components
        and act as an orchestrator/pipeline.
- I also planned to eliminate the `Masked*` subclasses entirely, since the masks are only ever really used in CAPVSTNet, while the stylizers currently just validate them.
    - This may end up being a strict necessity later if I want a lightweight, reusable method for creating masks on the fly, like the original.
"""


# in the short term, I'll probably Just be adding a lot of helper functions for parsing inputs here
# will also probably be adding a small factory function for the trainer and inference submodules to instantiate the stylizers for now

from typing import Union, Iterable
import torchvision.transforms.v2 as TT
from torchvision.io import read_image, ImageReadMode
import torch
from pathlib import Path
from ..utils.img_utils import (
    iterable_to_tensor, downscaling_resize, # get_scaled_dims, ensure_batch_tensor,
)



def prep_single_image(img: torch.Tensor, max_size: int, downscale: int = 4) -> torch.Tensor:
    """ Prepares a single image tensor for stylization by resizing it to the maximum size and downscaling it. """
    img = TT.functional.to_dtype(img, torch.float32, scale=True)
    img = downscaling_resize(img, max_size, downscale)
    return img


def load_styles_from_disk(
    style_paths: Union[str, Path, Iterable[str], Iterable[Path]],
    max_size: int,
    down_scale: int = 4,
    device: torch.device | None = None
) -> torch.Tensor:
    """ Read style images from disk and return a preprocessed batch tensor
        Args:
            style_paths: Path or iterable of image paths.
            max_size: Maximum size for the longest dimension after resizing.
            device: Optional device to move the resulting tensor to.
        Returns:
            torch.Tensor: Batch tensor of shape [B, C, H, W], dtype float32, and values in [0,1].
    """
    #& replaced the use of ensure_file_list_format with a simple check for str or Path
    # TODO: replace excessive type checking with Pydantic validation later
    # if isinstance(style_paths, (str, Path)):
    #     style_paths = [style_paths]
    if not (isinstance(style_paths, Iterable) and all(isinstance(p, (str, Path)) for p in style_paths)):
        raise ValueError("`style_paths` must be a path-like string or a list of path-like strings!")
    imgs = []
    for p in style_paths:
        img = read_image(p, ImageReadMode.RGB) #.pin_memory()
        img = prep_single_image(img, max_size, downscale=down_scale)
        imgs.append(img)
    batch = iterable_to_tensor(imgs, max_size)
    if device is not None:
        batch = batch.to(device)
    return batch.clamp(0, 1)


# TODO: replace with Pydantic validator
def process_style_sources(
    style_sources: Union[str, Path, torch.Tensor, Iterable[str], Iterable[Path], Iterable[torch.Tensor]],
    max_size: int = 1280,
    down_scale: int = 4
) -> torch.Tensor:
    # * NOTE: this is a method that should be called before the transform_preprocess decorator is applied to the stylize_from_images method
    if issubclass(type(style_sources), torch.Tensor):
        return prep_single_image(style_sources, max_size, down_scale) # return a resized and preprocessed tensor
    ### or if not a tensor, but an iterable (list) of tensors, collate them into a batch tensor
    elif isinstance(style_sources, Iterable) and all(isinstance(p, torch.Tensor) for p in style_sources):
        styles = [prep_single_image(p, max_size, down_scale) for p in style_sources]
        return iterable_to_tensor(styles, max_size) # return a resized and preprocessed batch tensor
    ### if style_paths is a string or iterable (typically list) of strings, load the style images from disk
    if isinstance(style_sources, (str, Path)):
        style_sources = [style_sources]  # ensure it's a list of paths
    if isinstance(style_sources, Iterable) and all(isinstance(p, (str, Path)) for p in style_sources):
        return load_styles_from_disk(style_sources, max_size, down_scale)
    raise ValueError("`style_paths` must be str, `torch.Tensor`, list of path-like strings, or a list of tensors!")