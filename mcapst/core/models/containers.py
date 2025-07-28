
import os
from typing import Dict, List, Literal, Union, Iterable, Optional, Callable
import functools
from dataclasses import dataclass, field
import numpy as np
import torch
import torchvision.transforms.v2 as TT
import torchvision.io as IO
# local imports
from ..utils.utils import ensure_file_list_format
from ..utils.img_utils import iterable_to_tensor, post_transfer_blending




@dataclass
class StyleWeights:
    """ Encapsulates weights for either content or style - objects of this type will be managed by StyleWeightContainer """
    weights: Optional[List[float]] = None
    weight_type: Literal["content", "style"] = "style"
    num_items: int = 1

    def __post_init__(self):
        if self.weights is None:
            self.weights = self._default_weights()
        self._validate_weights()
        self._normalize_weights()

    def _default_weights(self) -> List[float]:
        if self.weight_type == "content":
            return [0]
        return [round(1 / self.num_items, 4)] * self.num_items

    #& TEMP: I'll probably be moving some of this upstream to the parsing logic to throw argparse.ArgumentTypeError
        #& for non-numeric types, so this may be temporary
    def _validate_weights(self):
        # ensure the weights are a valid iterable type
        if not isinstance(self.weights, (list, tuple, np.ndarray, torch.Tensor)):
            self.weights = [self.weights]
        elif isinstance(self.weights, (np.ndarray, torch.Tensor)):
            self.weights = self.weights.tolist()
        # ensure self.num_items matches the number of style weights
        if len(self.weights) != self.num_items:
            #print(f"WARNING: got num_items = {self.num_items}, but len(weights) = {len(self.weights)}; using default weights instead...")
            self.weights = self._default_weights()
            self.num_items = len(self.weights)
        # ensure weights are valid numeric types in [0,1]
        if not all(isinstance(w, (int, float)) and 0 <= w <= 1 for w in self.weights):
            raise ValueError(f"All weights must be floats in the range [0, 1]; got {self.weights} with types {[type(w) for w in self.weights]}")

    def _normalize_weights(self):
        """ normalize the weights to sum to 1 """
        total: float = sum(self.weights) # assumes that self.weights is an iterable at this point
        TOL: float = 1e-6 # using a small tolerance to avoid floating point errors in comparing total to 0
        if total > TOL and self.weight_type == "style":
            # limiting precision for reproducibility
            self.weights: List[float] = [round(w/total, 4) for w in self.weights]

    #& I'm considering looking into the pytorch dunder method `__torch_function__` for this to override all calls by pytorch functions
    # https://pytorch.org/docs/stable/notes/extending.html#extending-torch-python-api
    # https://github.com/docarray/notes/blob/main/blog/02-this-weeks-in-docarray-01.md
    #& unused everywhere
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.weights, device=device, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.weights[idx]

    def __iter__(self):
        return iter(self.weights)

    def __next__(self):
        return next(self.weights)

    def __len__(self):
        return len(self.weights)

    def __repr__(self):
        return f"StyleWeight(weights={self.weights}, weight_type={self.weight_type}, num_items={self.num_items})"



def preprocess_and_postprocess(func):
    """ Decorator to preprocess and postprocess feature tensors contained in a dictionary w.r.t. their shape and dtype """
    @functools.wraps(func)
    def wrapper(self, feature_dict: Dict[str, FeatureContainer], *args, **kwargs):
        # preprocess each feature tensor
        for _, feature_container in feature_dict.items():
            feature_container.preprocess()
        # call the original function with preprocessed tensors
        result = func(self, feature_dict, *args, **kwargs)
        # postprocess the result tensor, defined by FeatureContainer class
        if isinstance(result, FeatureContainer):
            result.postprocess()
        else:
            raise NotImplementedError
        return result.feat
    return wrapper



# TODO: transition this to a hybrid builder/factory design pattern for the FeatureContainer class to use a different variant for masked style transfer
    #& I'm considering looking into the pytorch dunder method `__torch_function__` for this to override all calls by pytorch functions
    # https://pytorch.org/docs/stable/notes/extending.html#extending-torch-python-api
    # https://github.com/docarray/notes/blob/main/blog/02-this-weeks-in-docarray-01.md
class FeatureContainer(object):
    """encapsulation of each set of feature tensors with associated attributes, optional masks, optional scalar weights, etc"""
    def __init__(
        self,
        features: Union[torch.Tensor, Iterable[torch.Tensor]],
        # TODO: remove target tensor later to cut down on the unnecessary storage
        feature_type: Literal["content", "style", "target"],
        alpha: StyleWeights, #Union[float, List[float], None] = None,
        mask: Union[torch.Tensor, Iterable[torch.Tensor], None] = None,
        use_double=True,
        max_size=1280,
    ):
        self.feat: torch.Tensor = iterable_to_tensor(features, max_size, is_mask=False) if isinstance(features, list) else features
        # print(f"{feature_type} feature range after `iterable_to_tensor`: {self.feat.min(), self.feat.max()}")
        self.batch_size = self.feat.shape[0]
        #& totally unused everywhere
        self.feat_type: str = feature_type
        self.feat_shape_init: torch.Size = self.feat.shape
        self.feat_dtype_init: torch.dtype = self.feat.dtype
        # ? NOTE: should maybe move this to the FeatureFusionModule; I was thinking of doing the same for alpha_c, but I feel like I probably shouldn't
        self.alpha: StyleWeights = alpha
        # ? NOTE: may end up moving this as well to enfore mask consistency with the feature tensors
        if mask is not None:
            mask = iterable_to_tensor(mask, max_size, is_mask=True)
        self.mask: torch.Tensor | None = mask
        self.use_double: bool = use_double

    def preprocess(self):
        # flatten spatial dimensions for computation
        self.feat = self.feat.reshape(*self.feat_shape_init[:2], -1)  # [B, N, H*W]
        if self.use_double:
            self.feat = self.feat.double()
        if self.mask is not None:
            self.preprocess_mask()

    # should usually call this for the target features, which should be returned in the same shape and dtype of the original content features
    def postprocess(self):
        if self.use_double:
            self.feat = self.feat.to(dtype=self.feat_dtype_init)
        self.feat = self.feat.reshape(self.feat_shape_init)  # [B, N, H, W]
        # print(f"{self.feat_type} feature range before backward pass through RevResNet: {self.feat.min(), self.feat.max()}")

    def preprocess_mask(self):
        H, W = self.feat_shape_init[-2:]
        NUM_CLASSES = 4  # TODO: remove hardcoding later and make this a constructor argument
        # resize (by interpolation) masks to the proper [H,W] shape before flattening them
        if self.mask.shape[-2:] != self.feat_shape_init[-2:]:
            self.mask = TT.functional.resize(self.mask, size=(H, W), interpolation = TT.InterpolationMode.NEAREST)
        # flatten spatial dimensions to the same shape as the content features
        self.mask = self.mask.reshape(*self.mask.shape[:2], -1)  # [B, 1, H*W]
        # convert the mask to a one-hot encoded boolean tensor for later use in FeatureFusionModule._get_masked_target_features
        self.mask = self.mask.squeeze(1)
        self.mask = torch.nn.functional.one_hot(self.mask.to(dtype=torch.long), num_classes=NUM_CLASSES).to(dtype=torch.bool)
        # above function places channel dimension at the end using numpy convention for some reason - might be a holdover from before torchvision
        self.mask = self.mask.transpose(1, -1)
        assert (self.mask.shape[0] != self.batch_size), \
            f"ERROR: the number of style masks and number of style images should be the same; got {self.mask.shape[0]} and {self.batch_size}"
        if self.mask.device != self.feat.device:
            self.mask = self.mask.to(self.feat.device)

    #& unused everywhere, but this is the general approach I think I'll keep for the masked version later
    def get_mask_indices(self, label):
        # ? NOTE: pretty much wrote this while still assuming that we're iterating over labels, but passing the whole batch this time
        # ~ could always try the FeatureContainerIterable again later and use this with torch.vmap
        # ? NOTE: would need to take this check out, as well as other places if I resize masks to one-hot tensors and don't squeeze out the extra (second) dimension
        if self.mask is None or len(self.mask.shape) == 4:
            raise RuntimeError("mask-based method 'get_mask_indices' called before mask initialization and preprocessing")
        # returns a 2D LongTensor where each row is the index for a nonzero value
        indices: torch.LongTensor = torch.nonzero(self.mask == label)  # - essentially [matches, [B, C, matches]] where matches <= H*W
        # ~ IDEA: really wondering about the approach of saving masked features as a list in a new attribute
        """# select masked features in last dimension, based on indices to get N masked tensors with size equal to the idx tensor
        masked_view = torch.index_select(self.feat, -1, indices)"""
        return indices

    def __repr__(self):
        return (
            f"FeatureContainer("
            f"feat_type='{self.feat_type}', "
            f"feat_shape={self.feat_shape_init}, "
            f"feat_dtype={self.feat_dtype_init}, "
            f"alpha={self.alpha}, "
            f"mask_present={self.mask is not None})"
        )





@dataclass
class StylizerArgs:
    style_paths: Union[str, List[str], List[torch.Tensor], torch.Tensor]
    use_segmentation: bool = False
    use_blending: bool = False
    alpha_c: Union[float, None] = None
    alpha_s: Union[float, Iterable[float]] = None
    mask_paths: Union[str, List[str], None] = None
    cmask: Optional[torch.Tensor] = None  # Content mask from `sample`
    smask: Optional[List[torch.Tensor]] = field(default_factory=list)  # Style masks, loaded dynamically
    #! both are still used in the VideoStylizer, but not in the ImageStylizer - replace with a forward hook or just return videos to the caller to do it?
    # save_output: bool = True
    # output_path: Optional[str] = None

    def as_dict(self, supported_args: List[str]) -> Dict[str, any]:
        """ Filter arguments based on the supported ones for a specific class or method. """
        return {arg: getattr(self, arg) for arg in supported_args if hasattr(self, arg)}

    def construct_postprocessor(self) -> Union[Optional[Callable], None]:
        """ Dynamically create a postprocessor based on current argument values. """
        postprocessors = []
        if self.use_blending:
            postprocessors.append(post_transfer_blending)
        # FIXME: won't currently work when wrapped by torchvision.transforms container objects
        if not postprocessors:
            return None
        # combine all postprocessors into a single callable and return the function handle
        def combined_postprocessor(tensor: torch.Tensor) -> torch.Tensor:
            for postprocessor in postprocessors:
                tensor = postprocessor(tensor)
            return tensor
        return combined_postprocessor

    def load_style_masks(self, style_paths: List[str], default_mask_dir: Optional[str] = None, device: torch.device = None) -> None:
        """ Load style masks from provided paths or infer from style paths. """
        if self.mask_paths:
            self.smask = [IO.read_image(path, IO.ImageReadMode.UNCHANGED).to(device) for path in ensure_file_list_format(self.mask_paths)]
        elif default_mask_dir:
            # Infer mask paths based on style filenames
            inferred_paths = [os.path.join(default_mask_dir, os.path.basename(path)) for path in style_paths]
            self.smask = [IO.read_image(path, IO.ImageReadMode.UNCHANGED).to(device) for path in inferred_paths if os.path.exists(path)]

    def validate_segmentation(self, sample: Dict[str, torch.Tensor]) -> None:
        """ Ensure that segmentation masks are valid if required. """
        if self.use_segmentation:
            self.cmask = sample.get("mask")
            if self.cmask is None:
                raise ValueError("Segmentation enabled, but content mask (`cmask`) is missing in the sample.")