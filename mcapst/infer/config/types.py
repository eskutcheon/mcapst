
from typing import NewType, List, Dict, Any, Union, Callable, Tuple, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:  # ensures imports are only for type-checking and not at runtime
    import torch
    import numpy as np


# custom types for external classes and libraries
Tensor: TypeAlias = "torch.Tensor"
FloatTensor: TypeAlias = "torch.FloatTensor"
IntTensor: TypeAlias = "torch.IntTensor"
ByteTensor: TypeAlias = "torch.ByteTensor"
LongTensor: TypeAlias = "torch.LongTensor"
NDArray: TypeAlias = "np.ndarray"



# custom types for project classes
if TYPE_CHECKING:
    # probably doing away with this wrapper class eventually anyway, but it's the only one I made a type for so far
    from mcapst.core.models.containers import FeatureContainer

FeatureContainerType: TypeAlias = "FeatureContainer"



__all__ = [
    "Tensor",
    "FloatTensor",
    "IntTensor",
    "ByteTensor",
    "LongTensor",
    "FeatureContainerType",
]