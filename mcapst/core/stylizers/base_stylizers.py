from typing import Literal, List, Dict, Callable, Iterable, Union, Tuple, Optional
import functools
import os
from dataclasses import dataclass, field
import torch
import torchvision.transforms.v2 as TT
import torchvision.io as IO
# local imports
from ..models.RevResNet import RevResNet
from ..models.CAPVSTNet import CAPVSTNet
from ..models.containers import FeatureContainer, StyleWeights
from ..utils.utils import ensure_file_list_format
from ..utils.img_utils import post_transfer_blending, get_scaled_dims, ensure_batch_tensor, iterable_to_tensor


# TODO: this whole file really needs to be cleaned up while eliminating redundant code



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
    save_output: bool = True
    output_path: Optional[str] = None

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


# TODO: Desperately need to refactor this and comply with the single responsibility principle a lot better
def transform_preprocess(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(
        # only showing arguments that should be common to all subclasses of BaseStylizer
        cls: BaseStylizer,
        sample: Union[torch.Tensor, Dict[str, torch.Tensor]],
        style_paths: Union[str, List[str], List[torch.Tensor], torch.Tensor],
        alpha_c: Union[float, None] = None,
        alpha_s: Union[float, Iterable[float]] = None,
        # should include postprocessor back in as an argument after I think over how the stylizers' structure may change
        #postprocessor: Optional[Callable] = None,
        #mask_paths: Union[str, List[str], None] = None,
        **kwargs
    ) -> torch.Tensor:
        # construct the StylizerArgs object from the received arguments
        args = StylizerArgs(
            style_paths=style_paths,
            alpha_c=alpha_c,
            alpha_s=alpha_s,
            #mask_paths=mask_paths,
            **kwargs
        )
        # Safeguards for ensuring proper formatting of `args.style_paths` + converting to batch tensor
        args.style_paths = cls.process_style_sources(args.style_paths)
        # handle the weights using initialization of StyleWeights objects
        # TODO: ensure proper types upstream and add error checking here
        content_batch_size = sample.shape[0] if isinstance(sample, torch.Tensor) else sample["img"].shape[0]
        args.alpha_c = StyleWeights(args.alpha_c, "content", num_items=content_batch_size)
        style_batch_size = args.style_paths.shape[0]
        args.alpha_s = StyleWeights(args.alpha_s, "style", num_items=style_batch_size)
        assert len(args.alpha_s) == style_batch_size, \
            f"ERROR: number of style weights ({len(args.alpha_s)}) must match the number of style images ({style_batch_size})!"
        # construct the postprocessor if applicable
        postprocessor = args.construct_postprocessor()
        if postprocessor:
            cls.postprocessor = postprocessor
        # fetch supported arguments dynamically from the class
        supported_args = getattr(cls, "supported_args", [])
        if not supported_args:
            raise AttributeError(f"Class {cls.__class__.__name__} must define `supported_args`.")
        filtered_args = args.as_dict(supported_args)
        return func(cls, sample, **filtered_args)
    return wrapper


"""
    ? NOTE on use_segmentation:
        - [x] going to have to deal with SegLabelMapper conversion for tensors later
        - [ ] also, I could potentially move the label filtering in cWCT.get_masked_target_features to an earlier preprocessing step
            in `stylize_from_images`,
        - [ ] still need to add error checking for the case where one mask is None and another isn't
"""

def get_default_revnet_args(mode: Literal["photo", "art"]):
    return {
        "nBlocks": [10, 10, 10],
        "nStrides": [1, 2, 2],
        "nChannels": [16, 64, 256],
        "in_channel": 3,
        "mult": 4,
        "hidden_dim": 16 if mode == "photo" else 64,
        "sp_steps": 2 if mode == "photo" else 1,
    }

def initialize_revnet_model(mode, device="cuda"):
    """ Initializes a Reversible Residual Network model based on the specified mode.
        Args:
            mode (str): Mode of the network, either 'photo' or 'art'.
        Returns:
            RevResNet: The initialized reversible network.
    """
    if "photo" in mode:
        mode = "photo"
    elif "art" in mode:
        mode = "art"
    else:
        raise ValueError(f"ERROR: only 'photo' and 'art' (or 'photorealistic' or 'artistic') accepted for 'mode' parameter; got {mode}")
    revnet_args = get_default_revnet_args(mode)
    return RevResNet(**revnet_args).to(device=device)



class BaseStylizer(object):
    def __init__(self,
                 mode: Literal["photo", "art"], # a class instance can use only (mutually exclusively) photorealistic or artistic style transfer modes
                 ckpt: str,                     # path to a Reversible Residual Network pre-trained model checkpoint
                 max_size: int,                 # maximum size to restrict both content and style images
                 postprocessor: Callable|None = None,
                 reg_method: str = "ridge",     # regularization method to apply to the output (if any)
                 train_mode: bool = False):
        if mode not in ["photo", "art"]:
            raise ValueError(f"ERROR: only 'photo' and 'art' accepted for 'mode' parameter; got {mode}")
        self.mode = mode
        self.max_size = max_size
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ? NOTE: should decide later whether to include the network parameters hardcoded in _set_revnet should be passed from an augment_cfg object (via policy managers)
        self.revnet = self._set_revnet(mode, ckpt)
        if train_mode:
            self.revnet.train()
        else:
            self.revnet.eval()
        self.feature_aligner = CAPVSTNet(max_size=self.max_size, train_mode=train_mode, reg_method=reg_method)
        #if postprocessor is not None:
        self.postprocessor = postprocessor

    def _set_revnet(self, mode: str, ckpt_path: str = None):
        """ Sets the reversible network based on the specified mode.
            Args:
                mode (str): Mode of the network, either 'photo' or 'art'.
            Returns:
                RevResNet: The initialized reversible network
        """
        revnet = initialize_revnet_model(mode, device=self.device)
        if isinstance(ckpt_path, str) and os.path.exists(ckpt_path):
            self._load_revnet_from_ckpt(revnet, ckpt_path)
        return revnet

    def _load_revnet_from_ckpt(self, revnet: RevResNet, ckpt_path: str):
        """ Initializes and loads the reversible network
            Args:
                ckpt_path (str): Path to the pre-trained model checkpoint
            Returns:
                RevResNet: The initialized reversible network
        """
        state_dict = torch.load(ckpt_path, weights_only=True, map_location=self.device)
        revnet.load_state_dict(state_dict['state_dict'])
        return revnet

    def preprocess(self, img: torch.Tensor):
        """ preprocess input tensor - move to device, convert to float32, ensure batching, resize, and ensure pixel range [0,1] """
        if not img.is_cuda:
            img = img.to(device=self.device)
        img = TT.functional.to_dtype(img, dtype=torch.float32, scale=True)
        img = self._resize(img).clamp(0,1)
        return img

    # ? NOTE: think this might be logically the same as the resize code I have written in a script on my laptop, with the exception of using down_scale
    def _resize(self, img: torch.Tensor, use_downscale=True):
        """ scale img so that its longest dimension <= max_size and again by a constant factor for input into the reversible network for multi-scale feature extraction """
        img = ensure_batch_tensor(img)
        H, W = img.shape[-2:]
        H_new, W_new = get_scaled_dims(img, self.max_size)
        down_scale = self.revnet.down_scale
        # Adjust to make dimensions multiples of down_scale - usually 4 for nStrides = [1, 2, 2]
        if use_downscale and down_scale is not None and (H_new % down_scale != 0 or W_new % down_scale != 0):
            H_new = (H_new//down_scale)*down_scale # same as H -= (H % down_scale) essentially
            W_new = (W_new//down_scale)*down_scale
        # if the dimensions have changed, resize the img tensor
        if H != H_new or W != W_new:
            return TT.functional.resize(img, [H_new, W_new], TT.InterpolationMode.BICUBIC, antialias=True)
        else:
            return img

    def process_style_sources(self, style_paths: Union[str, List[str], List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        # * NOTE: this is a method that should be called before the transform_preprocess decorator is applied to the stylize_from_images method
        if issubclass(type(style_paths), torch.Tensor):
            return self.preprocess(style_paths) # return a resized and preprocessed tensor
        ### or if not a tensor, but an iterable (list) of tensors, collate them into a batch tensor
        elif isinstance(style_paths, Iterable) and all(isinstance(p, torch.Tensor) for p in style_paths):
            return iterable_to_tensor([self.preprocess(p) for p in style_paths], self.max_size) # return a resized and preprocessed batch tensor
        ### if style_paths is a string or iterable (typically list) of strings, load the style images from disk
        elif isinstance(style_paths, str) or (isinstance(style_paths, Iterable) and all(isinstance(p, str) for p in style_paths)):
            return self.load_styles_from_disk(style_paths)
        else:
            raise ValueError("`style_paths` must be str, `torch.Tensor`, list of path-like strings, or a list of tensors!")

    def load_styles_from_disk(self, style_paths: Union[str, List[str], List[torch.Tensor], torch.Tensor]):
        """ Load style images from the specified paths. """
        if isinstance(style_paths, str):
            style_paths = ensure_file_list_format(style_paths)
        elif not (isinstance(style_paths, Iterable) and all(isinstance(p, str) for p in style_paths)):
            raise ValueError("`style_paths` must be a path-like string or a list of path-like strings!")
        # load style images from disk given that style paths should be an iterable of strings either way
        # !! try setting this back to the way it was with iterable_to_tensor called before self.preprocess
            # - I mistakenly thought a cache issue was an issue with saving the style images that way
        style_imgs = [self.preprocess(IO.read_image(p, IO.ImageReadMode.RGB).pin_memory()) for p in style_paths]
        return iterable_to_tensor(style_imgs, self.max_size)


    def stylize(self, content_features: FeatureContainer, style_features: FeatureContainer):
        z_cs = self.feature_aligner.transfer(content_features, style_features)
        with torch.no_grad(): # backward pass through reversible network acts as a decoder from feature space to image space
            stylized = self.revnet(z_cs, forward=False)
        if self.postprocessor is not None:
            stylized = self.postprocessor(stylized)
        del z_cs
        return stylized