from typing import Literal, List, Dict, Callable, Iterable, Union, Tuple, Optional
import functools
#from collections import Iterable
import os
from dataclasses import dataclass, field
from tqdm import tqdm
import torch
import torchvision.transforms.v2 as TT
import torchvision.io as IO
# local imports
from ..models.RevResNet import RevResNet
from ..models.CAPVSTNet import CAPVSTNet
from ..models.containers import FeatureContainer, StyleWeights
from ..utils.utils import ensure_file_list_format
from ..utils.img_utils import post_transfer_blending, get_scaled_dims, ensure_batch_tensor
from .video_processor import VideoProcessor


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



def transform_preprocess(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(
        self,
        sample: Union[torch.Tensor, Dict[str, torch.Tensor]],
        style_paths: Union[str, List[str], List[torch.Tensor], torch.Tensor],
        #use_segmentation: bool = False,
        #use_blending: bool = False,
        alpha_c: Union[float, None] = None,
        alpha_s: Union[float, Iterable[float]] = None,
        #mask_paths: Union[str, List[str], None] = None,
        **kwargs
    ) -> torch.Tensor:
        # construct the StylizerArgs object from the received arguments
        args = StylizerArgs(
            style_paths=style_paths,
            #use_segmentation=use_segmentation,
            #use_blending=use_blending,
            alpha_c=alpha_c,
            alpha_s=alpha_s,
            #mask_paths=mask_paths,
            **kwargs
        )
        # ensure style paths are properly formatted as a list of path-like strings
        if isinstance(args.style_paths, list):
            if not all(isinstance(p, (str, torch.Tensor)) for p in args.style_paths):
                raise ValueError("`style_paths` must be str, `torch.Tensor`, list of path-like strings, or a list of tensors!")
        elif isinstance(args.style_paths, str):
            args.style_paths = ensure_file_list_format(args.style_paths)
        elif not issubclass(type(args.style_paths), torch.Tensor):
            raise ValueError("`style_paths` must be str, `torch.Tensor`, list of path-like strings, or a list of tensors!")
        # handle the weights using initialization of StyleWeights objects
        # !! FIXME: this isn't actually set well depending on the number of content or style arguments
        num_c = 1 if args.alpha_c is None or not isinstance(args.alpha_c, (list, tuple, torch.Tensor)) else len(args.alpha_c)
        args.alpha_c = StyleWeights(args.alpha_c, "content", num_items=num_c)
        #print("content alpha: ", args.alpha_c)
        args.alpha_s = StyleWeights(args.alpha_s, "style", num_items=len(args.style_paths))
        #print("style alpha: ", args.alpha_s)
        assert len(args.alpha_s) == len(args.style_paths), \
            f"ERROR: number of style weights ({len(args.alpha_s)}) must match the number of style images ({len(args.style_paths)})!"
        # construct the postprocessor if applicable
        postprocessor = args.construct_postprocessor()
        if postprocessor:
            self.postprocessor = postprocessor
        # fetch supported arguments dynamically from the class
        supported_args = getattr(self, "supported_args", [])
        if not supported_args:
            raise AttributeError(f"Class {self.__class__.__name__} must define `supported_args`.")
        filtered_args = args.as_dict(supported_args)
        return func(self, sample, **filtered_args)
    return wrapper


"""
    ? NOTE on use_segmentation:
        - [x] going to have to deal with SegRemapping conversion for tensors later
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

    def stylize(self, content_features: FeatureContainer, style_features: FeatureContainer):
        z_cs = self.feature_aligner.transfer(content_features, style_features)
        with torch.no_grad(): # backward pass through reversible network acts as a decoder from feature space to image space
            stylized = self.revnet(z_cs, forward=False)
        if self.postprocessor is not None:
            stylized = self.postprocessor(stylized)
        del z_cs
        return stylized



class BaseImageStylizer(BaseStylizer):
    supported_args = ["style_paths", "alpha_c", "alpha_s"]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def stylize_from_images(
        self,
        content_img: torch.Tensor,
        style_paths: List[str],
        alpha_c: StyleWeights,
        alpha_s: StyleWeights,
        # leaving the segmentation mask arguments here so that base classes can pass them to this method
        cmask: Optional[torch.Tensor] = None,
        smask: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        style_images = []
        if all(isinstance(p, str) for p in style_paths):
            style_images = [self.preprocess(IO.read_image(p, IO.ImageReadMode.RGB).pin_memory()) for p in style_paths]
        elif isinstance(style_paths, (list, tuple)) and all(isinstance(p, torch.Tensor) for p in style_paths):
            style_images = [self.preprocess(p) for p in style_paths]
        elif isinstance(style_paths, torch.Tensor):
            style_images = [self.preprocess(style_paths)]
        else:
            raise ValueError("`style_paths` must be a list of path-like strings or a list of tensors!")
        with torch.no_grad(): # forward inference of self.revnet acts as the feature encoder
            content_img = self.preprocess(content_img)
            #self._resize(content_img).to(self.device)
            z_c = self.revnet(content_img, forward=True)
            z_s = [self.revnet(img, forward=True) for img in style_images]
        content_feat = FeatureContainer(z_c, "content", alpha_c, mask=cmask, max_size=self.max_size)
        style_feat = FeatureContainer(z_s, "style", alpha_s, mask=smask, max_size=self.max_size)
        pastiche = self.stylize(content_feat, style_feat)
        del content_feat, style_feat
        return pastiche

    def _postprocess_pastiche(self, pastiche: torch.Tensor):
        if self.postprocessor:
            pastiche = self.postprocessor(pastiche)
        return pastiche.clamp(0, 1)

    @transform_preprocess
    def transform(
        self,
        sample: Union[torch.Tensor, Dict[str, torch.Tensor]],
        style_paths: Union[str, List[str]],
        alpha_c: Union[float, None],
        alpha_s: Union[float, Iterable[float]],
    ):
        """ handle preprocessing where necessary then call stylize, then handle post-processing before returning the augmented sample """
        # TODO: need to use a dedicated function based on native pytorch decorators for this since they handle nested types natively
        content_img = sample if not isinstance(sample, dict) else sample["img"]
        # NOTE: the decorator turns alpha_c and alpha_s into StyleWeights objects
        pastiche: torch.Tensor = self.stylize_from_images(content_img, style_paths, alpha_c = alpha_c, alpha_s = alpha_s)
        pastiche = self._postprocess_pastiche(pastiche)
        if isinstance(sample, dict):
            sample["img"] = pastiche
        return pastiche



class BaseVideoStylizer(BaseStylizer):
    supported_args = ["style_paths", "alpha_c", "alpha_s", "save_output", "output_path"]
    def __init__(self, mode: Literal["art", "photo"], ckpt: str, max_size: int = 1280, postprocessor: Callable = None, reg_method: str = "ridge", fps: int = 10):
        """ video stylizer base class for photorealistic and artistic style transfer
            Args:
                mode (str): 'photo' or 'art' for photorealistic or artistic style transfer
                ckpt (str): Path to the pre-trained model checkpoint
                max_size (int): Maximum size for resizing frames
                reg_method: Regularization method to apply to the output (if any)
                fps (int): Frames per second for output video.
        """
        super().__init__(mode, ckpt, max_size, postprocessor, reg_method)
        self.fps = fps
        # TODO: need to add a input arguments for the batch size from the config
        self.max_batch_size = 4

    def stylize_video(self, frames: torch.Tensor, style_paths: List[str], alpha_c: StyleWeights, alpha_s: StyleWeights, cmask=None, smask=None):
        """ Applies style transfer to video frames and saves the result as a video file.
            Args:
                frames: Tensor of video frames with shape (T, C, H, W).
                style_paths: Style image template path(s)
        """
        style_images = []
        if all(isinstance(p, str) for p in style_paths):
            print("style paths: ", style_paths)
            style_images = [self.preprocess(IO.read_image(p, IO.ImageReadMode.RGB).pin_memory()) for p in style_paths]
        elif all(isinstance(p, torch.Tensor) for p in style_paths):
            style_images = [self.preprocess(p) for p in style_paths]
        else:
            raise ValueError("`style_paths` must be a list of path-like strings or a list of tensors!")
        with torch.no_grad(): # forward inference of self.revnet acts as the feature encoder
            z_c = self.revnet(frames, forward=True)
            z_s = [self.revnet(img, forward=True) for img in style_images]
        content_feat = FeatureContainer(z_c, "content", alpha_c, cmask, max_size=self.max_size)
        style_feat = FeatureContainer(z_s, "style", alpha_s, smask, max_size=self.max_size)
        # Initialize video writer
        with torch.no_grad():
            processed_frames = self.stylize(content_feat, style_feat).clamp(0, 1).cpu()
        # write stylized frames to the new video file
        return processed_frames

    # TODO: probably need to generalize the preprocessing decorator for videos or just write a new one
    @transform_preprocess
    def transform(
        self,
        sample: Union[str, torch.Tensor, Dict[str, torch.Tensor]],
        style_paths: Union[str, List[str]],
        # TODO: add the next two options to a higher level function call that will just add them to the postprocessor when calling this class
        #use_segmentation: bool,
        #use_blending: bool,
        alpha_c: Union[float, None],
        alpha_s: Union[float, Iterable[float]],
        #mask_paths: Union[str, List[str], None],
        # ! should definitely be False by default later, with caching enabled by default
        save_output: bool = True,
        output_path: Optional[str] = None
    ):
        content_vid = sample["src"] if isinstance(sample, dict) else sample
        assert content_vid is not None and (isinstance(content_vid, str) or issubclass(type(content_vid), torch.Tensor)), \
            "ERROR: 'sample' must be a path-like string or a tensor of video frames!"
        # NOTE: no need to check whether sample is a string or tensor since the VideoReader object can accept either as src
        vid_processor = VideoProcessor(content_vid, target_fps=self.fps, backend="torchvision")
        vid_generator = vid_processor.generate_frames(batch_size = self.max_batch_size)
        # NOTE: decorator turns alpha_c and alpha_s into containers.StyleWeights objects
        style_args = {"style_paths": style_paths, "alpha_c": alpha_c, "alpha_s": alpha_s}
        stylized_video = []
        # TODO: still need to find a more quantized way to do this
        for batch in vid_generator:
            batch = self.preprocess(batch).squeeze(0)
            pastiche: torch.Tensor = self.stylize_video(batch, **style_args).clamp(0, 1)
            stylized_video.append(pastiche.to(device="cpu"))
        pastiche = torch.cat(stylized_video, dim=0)
        if save_output:
            vid_processor.save_video_to_disk(pastiche, fps=self.fps, output_path=output_path)
        # TODO: rewrite to cache by default instead of writing to disk or returning a potentially gigantic tensor
        return pastiche




class MaskedImageStylizer(BaseImageStylizer):
    def __init__(
        self,
        mode: Literal["photo", "art"],
        ckpt: str,
        max_size: int,
        seg_model_ckpt: Optional[str] = None,
        postprocessor: Optional[callable] = None,
        reg_method: str = "ridge"
    ):
        """ A stylizer that supports segmentation mask-guided style transfer for images.
            Args:
                mode (Literal["photo", "art"]): Style transfer mode.
                ckpt (str): Path to the reversible residual network checkpoint.
                max_size (int): Maximum size for resizing images.
                seg_model_ckpt (Optional[str]): Path to the semantic segmentation model checkpoint.
                postprocessor (Optional[Callable]): Callable for post-processing stylized outputs.
                reg_method (str): Regularization method.
        """
        super().__init__(mode, ckpt, max_size, postprocessor, reg_method)
        self.segmentation_model = self._initialize_segmentation_model(seg_model_ckpt)

    def load_multi_style_masks(self, mask_paths: Union[str, Iterable[str]]):
        mask_paths = ensure_file_list_format(mask_paths)
        masks: List[torch.Tensor] = []
        for idx, path in enumerate(mask_paths):
            masks.append(IO.read_image(path, IO.ImageReadMode.UNCHANGED))
            while len(masks[idx].shape) < 4:
                masks[idx] = masks[idx].unsqueeze(0)
            masks[idx] = masks[idx].to(device=self.device)
            masks[idx] = self._resize(masks[idx])


    def _initialize_segmentation_model(self, ckpt: Optional[str]) -> Optional[torch.nn.Module]:
        """ Load a semantic segmentation model from a checkpoint.
            Args:
                ckpt (Optional[str]): Path to the segmentation model checkpoint.
            Returns:
                Optional[torch.nn.Module]: Loaded model or None if no checkpoint provided.
        """
        if ckpt is not None:
            raise NotImplementedError("Loading segmentation model from checkpoint not yet implemented.")
        # https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
        # NOTE: expects normalized (with ImageNet mean and std) tensors in shape [N, 3, H, W]
        model = torch.hub.load('pytorch/vision:v0.20.1', 'deeplabv3_resnet101', pretrained=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def generate_content_mask(self, content_img: torch.Tensor) -> torch.Tensor:
        """ Generate a content mask using the segmentation model.
            Args:
                content_img (torch.Tensor): Content image tensor of shape [C, H, W].
            Returns:
                torch.Tensor: One-hot encoded content mask of shape [C, H, W].
        """
        if self.segmentation_model is None:
            raise ValueError("Segmentation model not initialized. Provide a pre-trained checkpoint during instantiation.")
        with torch.no_grad():
            input_tensor = TT.functional.to_dtype(content_img, dtype=torch.float32).unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            output = self.segmentation_model(input_tensor)['out']
            segmentation = torch.argmax(output, dim=1, keepdim=False)
        # TODO: need to use SegRemapping to align the content and style segmentation classes
        one_hot_mask = torch.nn.functional.one_hot(segmentation.squeeze(0), num_classes=output.shape[1])
        return one_hot_mask.permute(2, 0, 1).bool()

    @transform_preprocess
    def transform(
        self,
        sample: Union[torch.Tensor, Dict[str, torch.Tensor]],
        style_paths: Union[str, List[str]],
        use_segmentation: bool = False,
        alpha_c: Union[float, None] = None,
        alpha_s: Union[float, Iterable[float]] = None,
        mask_paths: Union[str, List[str], None] = None,
    ) -> torch.Tensor:
        """ Handle preprocessing, call stylize, and post-process the augmented sample. """
        content_img = sample if not isinstance(sample, dict) else sample["img"]
        cmask = sample.get("mask", None) if use_segmentation and isinstance(sample, dict) else None
        if use_segmentation and cmask is None:
            cmask = self.generate_content_mask(content_img)
        smask = self.load_multi_style_masks(mask_paths) if use_segmentation and mask_paths else None
        pastiche = self.stylize_from_images(content_img, style_paths=style_paths, alpha_c=alpha_c, alpha_s=alpha_s, cmask=cmask, smask=smask)
        pastiche = self._postprocess_pastiche(pastiche)
        if isinstance(sample, dict):
            sample["img"] = pastiche
        return pastiche