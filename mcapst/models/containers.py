import torch
import torchvision.transforms.v2 as TT
import functools
from typing import Dict, List, Literal, Union, Iterable, Callable, Tuple, Any
from ..utils.img_utils import iterable_to_tensor



# originally written and applied within src.dataset.transforms.AugmentationFunctionalWrapper
def cpu_wrapper(compute_on_cpu):
    def decorator(func):
        def wrapper(tensor, *args, **kwargs):
            if compute_on_cpu:
                try:
                    device_init = tensor.device
                    # ? NOTE: doing this because for some reason, a torvision.tv_tensors.Mask type gets converted to a torch.Tensor type when moved to the CPU
                    # apparently this works but not tensor.cpu(), so I'm guessing only the .to method is defined for tv_tensors
                    tensor = tensor.to(device="cpu")
                except AttributeError:
                    raise ValueError("The first argument to the function must be a tensor!")
                tensor = func(tensor, *args, **kwargs)
                # move back to original device
                return tensor.to(device=device_init)
            else:
                # if compute_on_cpu = False, just call the function directly
                return func(tensor, *args, **kwargs)
        return wrapper
    return decorator




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


class FeatureContainer(object):
    """encapsulation of each set of feature tensors with associated attributes, optional masks, optional scalar weights, etc"""
    def __init__(
        self,
        features: Union[torch.Tensor, Iterable[torch.Tensor]],
        # TODO: remove target tensor later to cut down on the unnecessary storage
        feature_type: Literal["content", "style", "target"],
        alpha: Union[float, List[float], None] = None,
        mask: Union[torch.Tensor, Iterable[torch.Tensor], None] = None,
        use_double=True,
        max_size=1280,
    ):
        self.feat: torch.Tensor = iterable_to_tensor(features, max_size, is_mask=False)
        # print(f"{feature_type} feature range after `iterable_to_tensor`: {self.feat.min(), self.feat.max()}")
        self.batch_size = self.feat.shape[0]
        self.feat_type: str = feature_type
        self.feat_shape_init: torch.Size = self.feat.shape
        self.feat_dtype_init: torch.dtype = self.feat.dtype
        # ? NOTE: should maybe move this to the FeatureFusionModule; I was thinking of doing the same for alpha_c, but I feel like I probably shouldn't
        self.set_alpha_value(alpha)
        # ? NOTE: may end up moving this as well to enfore mask consistency with the feature tensors
        if mask is not None:
            mask = iterable_to_tensor(mask, max_size, is_mask=True)
        # TODO: add a method to ensure feat and mask are on the same device, same shape (batch or spatial dims), same type (ndarray vs Tensor)
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

    def set_alpha_value(self, alpha):
        # ? NOTE: this will likely vary by the style transfer approach taken (regular, interpolated styles, segmentation guided, etc)
        if self.feat_type == "style":
            if (B := self.feat_shape_init[0]) > 1:
                if alpha is None:
                    alpha = [1 / B] * B
                assert (len(alpha) == B), "The number of alpha values (if provided) must match the number of style feature tensors!"
            if alpha is not None and not isinstance(alpha, (list, tuple, torch.Tensor)):
                alpha = [alpha]
        # ? NOTE: not necessary for anything except the interpolation methods
        elif self.feat_type == "content":
            # FIXME: kinda want to revisit this logic elsewhere since the only place it's used just checks (if alpha != 0.0)
            if alpha is None:
                alpha = 0.0
            assert (0.0 <= alpha < 1.0), "Content alpha must be in the half open range [0, 1), excluding 1 which returns the original image"
        # alpha remains None if neither conditional block executes
        self.alpha: float = alpha

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
            # f"feat_value_range={float(self.feat.min()), float(self.feat.max())}, "
            f"alpha={self.alpha}, "
            f"mask_present={self.mask is not None})"
        )


class MultiFeatureContainer(FeatureContainer):
    def __init__(self, feature_list: list, alpha_list: list):
        super().__init__(features=None)
        self.feature_list = feature_list
        self.alpha_list = alpha_list


