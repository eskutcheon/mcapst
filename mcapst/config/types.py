# types.py
from typing import NewType, List, Dict, Any, Union, Callable, Tuple, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:  # ensures imports are only for type-checking and not at runtime
    import torch
    import torchvision.tv_tensors as tv_tensors


# custom types for external classes and libraries
Tensor: TypeAlias = "torch.Tensor"
FloatTensor: TypeAlias = "torch.FloatTensor"
IntTensor: TypeAlias = "torch.IntTensor"
ByteTensor: TypeAlias = "torch.ByteTensor"
LongTensor: TypeAlias = "torch.LongTensor"

# custom types for project classes
if TYPE_CHECKING:
    from mcapst.models.containers import FeatureContainer

FeatureContainerType: TypeAlias = "FeatureContainer"

# Frequently-used complex types for annotations
Batch: TypeAlias = Dict[str, Union[Tensor, Any]]  # Example: A batch might be a dictionary with tensors.
ImgMaskPair: TypeAlias = Dict[str, Union[Tensor, tv_tensors.Image, tv_tensors.Mask, Any]]  # Example: A pair of image and mask tensors.





__all__ = [
    "Tensor",
    "FloatTensor",
    "IntTensor",
    "ByteTensor",
    "LongTensor",
    "FeatureContainerType",
    "Batch",
    "ImgMaskPair",
]
