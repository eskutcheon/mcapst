from typing import Literal, List, Dict, Callable, Iterable, Union, Tuple, Optional
import torch
import torchvision.io as IO
# local imports
from .base_stylizers import BaseStylizer, transform_preprocess
from ..models.containers import FeatureContainer, StyleWeights
from ..datasets.video_processor import VideoProcessor



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
        style_images = style_paths #[]
        # if all(isinstance(p, str) for p in style_paths):
        #     print("style paths: ", style_paths)
        #     style_images = [self.preprocess(IO.read_image(p, IO.ImageReadMode.RGB).pin_memory()) for p in style_paths]
        # elif all(isinstance(p, torch.Tensor) for p in style_paths):
        #     style_images = [self.preprocess(p) for p in style_paths]
        # else:
        #     raise ValueError("`style_paths` must be a list of path-like strings or a list of tensors!")
        with torch.no_grad(): # forward inference of self.revnet acts as the feature encoder
            z_c = self.revnet(frames, forward=True)
            z_s = self.revnet(style_images, forward=True)
            #z_s = [self.revnet(img, forward=True) for img in style_images]
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



# TODO: add a new class for masked video stylization
class MaskedVideoStylizer(BaseVideoStylizer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Masked video stylization is not yet implemented.")
        #super().__init__(*args, **kwargs)

    def stylize_video(self, frames: torch.Tensor, style_paths: List[str], alpha_c: StyleWeights, alpha_s: StyleWeights, cmask=None, smask=None):
        """ Applies style transfer to video frames and saves the result as a video file.
            Args:
                frames: Tensor of video frames with shape (T, C, H, W).
                style_paths: Style image template path(s)
        """
        raise NotImplementedError