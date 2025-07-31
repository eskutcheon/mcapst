from typing import Literal, List, Dict, Callable, Iterable, Union, Tuple, Optional
import functools
import torch
# local imports
from .base_stylizers import BaseStylizer, StylizerArgs
from ..models.containers import FeatureContainer #, StyleWeights
from ..utils.video_processor import VideoProcessor
from .stylizer_builder import process_style_sources, prep_single_image



# TEMPORARY SOLUTION: mirrors transform_preprocess decorator used with BaseStylizer and ImageStylizer
    # mostly because I don't want to keep most of the logic anyway, so writing a new one is easier than tweaking the old one
def transform_preprocess2(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(
        # only showing arguments that should be common to all subclasses of BaseStylizer
        cls: BaseStylizer,
        sample: Union[torch.Tensor, Dict[str, torch.Tensor]],
        style_paths: Union[str, List[str], List[torch.Tensor], torch.Tensor],
        alpha_c: Union[float, Iterable[float]],
        alpha_s: Union[float, Iterable[float]],
        # should include postprocessor back in as an argument after I think over how the stylizers' structure may change
        #postprocessor: Optional[Callable] = None,
        #mask_paths: Union[str, List[str], None] = None,
        **kwargs
    ) -> torch.Tensor:
        # construct the StylizerArgs object from the received arguments
        #! literally why tf did I do this - maybe this and the current decorators are where I need to start refactoring
            # - need to simplify inputs, add small dispatcher functions for argument parsing, and remove unnecessary container classes
        args = StylizerArgs(
            style_paths=style_paths,
            alpha_c=alpha_c,
            alpha_s=alpha_s,
            #mask_paths=mask_paths,
            **kwargs
        )
        # TODO: REMOVE path processing next
        # Safeguards for ensuring proper formatting of `args.style_paths` + converting to batch tensor
        args.style_paths = process_style_sources(args.style_paths, cls.max_size, down_scale=cls.revnet.down_scale)
        # handle the weights using initialization of StyleWeights objects
        # TODO: ensure proper types upstream and add error checking here
        #content_batch_size = sample.shape[0] if isinstance(sample, torch.Tensor) else sample["img"].shape[0]
        #args.alpha_c = StyleWeights(args.alpha_c, "content", num_items=cls.max_batch_size)
        style_batch_size = args.style_paths.shape[0]
        #args.alpha_s = StyleWeights(args.alpha_s, "style", num_items=style_batch_size)
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
        # take only supported arguments from the StylizerArgs object's attributes
        filtered_args = args.as_dict(supported_args)
        return func(cls, sample, **filtered_args)
    return wrapper




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
        # TODO: need to add input arguments for the batch size from the config
        self.max_batch_size = 4

    # FIXME: the `style_paths` argument should be brought in line with the ImageStylizer classes
    def stylize_video(
        self,
        frames: torch.Tensor,
        style_paths: List[str],
        alpha_c: Union[float, Iterable[float]],
        alpha_s: Union[float, Iterable[float]],
        cmask=None,
        smask=None
    ) -> torch.Tensor:
        """ Applies style transfer to video frames and saves the result as a video file.
            Args:
                frames: Tensor of video frames with shape (T, C, H, W).
                style_paths: Style image template path(s)
        """
        style_images = style_paths #[]
        with torch.no_grad(): # forward inference of self.revnet acts as the feature encoder
            z_c = self.revnet(frames, forward=True)
            z_s = self.revnet(style_images, forward=True)
            #z_s = [self.revnet(img, forward=True) for img in style_images]
        content_feat = FeatureContainer(z_c, "content", alpha_c, cmask, max_size=self.max_size)
        style_feat = FeatureContainer(z_s, "style", alpha_s, smask, max_size=self.max_size)
        # Initialize video writer
        with torch.no_grad():
            processed_frames = self.stylize(content_feat, style_feat).clamp(0, 1)
        # write stylized frames to the new video file
        return processed_frames

    # TODO: probably need to generalize the preprocessing decorator for videos or just write a new one
    @transform_preprocess2
    def transform(
        self,
        sample: Union[str, torch.Tensor, Dict[str, torch.Tensor]],
        style_paths: torch.Tensor, #Union[str, List[str]],
        # TODO: replace `use_blending` with a higher level function call that will just add them to the postprocessor when calling this method
        #use_blending: bool,
        alpha_c: Union[float, None],
        alpha_s: Union[float, Iterable[float]],
        #! REMOVE last two args LATER
        save_output: bool = True,
        output_path: Optional[str] = None
    ):
        content_vid = sample["src"] if isinstance(sample, dict) else sample
        assert content_vid is not None and (isinstance(content_vid, str) or issubclass(type(content_vid), torch.Tensor)), \
            "ERROR: 'sample' must be a path-like string or a tensor of video frames!"
        # NOTE: no need to check whether sample is a string or tensor since the VideoReader object can accept either as src
        vid_processor = VideoProcessor(content_vid, target_fps=self.fps, backend="torchvision")
        vid_generator = vid_processor.frame_generator(batch_size = self.max_batch_size)
        stylized_video = []
        # TODO: still need to find a more efficient way to do this (i.e., moving chunks to a disk cache and finally joining them in an iterated fashion)
        for batch in vid_generator:
            batch = prep_single_image(batch, self.max_size, self.revnet.down_scale).squeeze(0).clamp(0, 1)
            if batch.device != style_paths.device:
                style_paths = style_paths.to(device=batch.device)
            pastiche = self.stylize_video(batch, style_paths, alpha_c, alpha_s).clamp(0, 1)
            stylized_video.append(pastiche.to(device="cpu"))
        pastiche = torch.cat(stylized_video, dim=0)
        if save_output:
            vid_processor.save_video_to_disk(pastiche, fps=self.fps, output_path=output_path)
        # TODO: rewrite to cache by default instead of writing to disk or returning a potentially gigantic tensor
        return pastiche



# TODO: add a new class for masked video stylization
# class MaskedVideoStylizer(BaseVideoStylizer):
#     def __init__(self, *args, **kwargs):
#         raise NotImplementedError("Masked video stylization is not yet implemented.")
#         #super().__init__(*args, **kwargs)

#     def stylize_video(
#         self,
#         frames: torch.Tensor,
#         style_paths: List[str],
#         alpha_c: Union[float, Iterable[float]],
#         alpha_s: Union[float, Iterable[float]],
#         cmask=None, smask=None
#     ):
#         """ Applies style transfer to video frames and saves the result as a video file.
#             Args:
#                 frames: Tensor of video frames with shape (T, C, H, W).
#                 style_paths: Style image template path(s)
#         """
#         raise NotImplementedError

#     #~ This one will be a later addition after tweaking, but I'm incoporating the label remapping while knocking out the MaskedImageStylizer class:
#         #~ main difference between the two (ref: old video_transfer.py script) is that style masks are remapped once and content masks are remapped for each frame
#     def transform(
#         self,
#         sample: Union[str, torch.Tensor, Dict[str, torch.Tensor]],
#         style_paths: Union[str, List[str]],
#         alpha_c: Union[float, None],
#         alpha_s: Union[float, Iterable[float]],
#         #mask_paths: Union[str, List[str], None],
#     ):
#         raise NotImplementedError