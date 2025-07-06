
import os
from abc import ABC, abstractmethod
import torch
from typing import Dict, Optional, Union, List, Any
from torchvision.io import read_image, write_jpeg, ImageReadMode
from torchvision.transforms.v2 import Compose, ToDtype, Lambda, Resize
# local imports
#from .dispatcher import StyleTransferDispatcher
from mcapst.config.configure import ConfigManager, InferenceConfig
from mcapst.utils.utils import ensure_file_list_format




TRANSFER_MODE_ALIASES = {
    "photorealistic": "photo",
    "artistic": "art",
}

class BaseInferenceOrchestrator(ABC):
    """ Base class for performing inference using trained style transfer models.
        Subclasses should implement domain-specific logic (e.g., image vs. video).
    """
    def __init__(self, config: Union[InferenceConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            # if config is a dictionary, convert it to an InferenceConfig object
            config = InferenceConfig(**config)
        self.config = config
        self._validate_config()
        self._normalize_mode(mode = self.config.transfer_mode)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = Compose([
            #Resize((self.config.max_size, self.config.max_size)),
            ToDtype(torch.float32, scale=True),
            Lambda(lambda x: x.unsqueeze(0)),  # Add batch dimension
            Lambda(lambda x: x.to(self.device) if x.device != self.device else x)  # Move to GPU if available
        ])
        # Create stylizer instance with a factory method to differentiate between base image, masked image, or video stylizers.
        self.stylizer = self._load_stylizer()

    @abstractmethod
    def _validate_config(self):
        """ should be overridden by subclasses to validate config parameters for specific inference tasks """
        pass

    @abstractmethod
    def _load_stylizer(self):
        """ factory method to create the appropriate stylizer object. Subclasses may override to handle more specific use cases """
        raise NotImplementedError("Subclasses must implement `_load_stylizer` to return a stylizer instance for the specific data modality.")

    @abstractmethod
    def run_inference(self, *args, **kwargs):
        """ High-level pipeline for performing inference, to be overridden by subclasses.
            The typical sequence should be:
                1. gather input data
                2. transform (style transfer)
                3. save or return outputs
        """
        raise NotImplementedError("Subclasses must implement `run_inference` to define the inference pipeline.")

    @abstractmethod
    def _parse_run_inputs(self, *args, **kwargs) -> Dict[str, Any]:
        """ base method that child classes may override for unique argument logic.
            By default, it does nothing or raises exception if not overridden.
        """
        raise NotImplementedError("Subclasses must implement `_parse_run_inputs` to handle specific input-parsing logic.")

    def _normalize_mode(self, mode: str):
        """ ensure self.config.transfer_mode is one of {'art', 'photo'} """
        all_modes = list(TRANSFER_MODE_ALIASES.keys()) + list(TRANSFER_MODE_ALIASES.values())
        if mode not in TRANSFER_MODE_ALIASES.values():
            # Attempt to map from 'photorealistic' -> 'photo', etc.
            try:
                self.config.transfer_mode = TRANSFER_MODE_ALIASES[mode]
            except KeyError:
                raise ValueError(f"Invalid transfer mode: '{mode}'!\nExpected one of {all_modes}.")

    def save_pastiche_to_disk(self, pastiche: torch.Tensor, output_path: str):
        write_jpeg(pastiche.clamp(0,1).mul(255).byte().cpu(), output_path)

    def _log_inference_details(self):
        """ Logs inference-related details. """
        print(f"Inference Mode: {self.config.transfer_mode}")
        print(f"Input Path: {self.config.input_path}")
        print(f"Output Path: {self.config.output_path}")
        print(f"Using Segmentation: {self.config.use_segmentation}")




class ImageInferenceOrchestrator(BaseInferenceOrchestrator):
    """ High-level inference class for image-based style transfer.
        essentially a wrapper around one of
        - BaseImageStylizer
        - MaskedImageStylizer (if config.use_segmentation is True)
    """
    def __init__(self, config: InferenceConfig):
        super().__init__(config)

    def _validate_config(self):
        """ Validates necessary parameters for image inference. """
        if self.config.modality != "image":
            raise ValueError("ImageInferenceOrchestrator should only be used for image inference tasks.")

    def _load_stylizer(self):
        """ selects either BaseImageStylizer or MaskedImageStylizer depending on config.use_segmentation """
        from mcapst.stylizers.image_stylizers import BaseImageStylizer, MaskedImageStylizer
        mode = self.config.transfer_mode  # "art" or "photo"
        ckpt_path = self.config.ckpt_path
        max_size = self.config.max_size
        if self.config.use_segmentation:
            # Example: optionally pass in segmentation checkpoint if you have one
            # NOTE: # seg_model_ckpt may be a path to a segmentation checkpoint
                # TODO: need to add this to the config
            return MaskedImageStylizer(mode=mode, ckpt=ckpt_path, max_size=max_size, seg_model_ckpt=None, reg_method="ridge")
        else:
            # TODO: need to refactor how these stylizers are loaded to allow users to pass in custom postprocessors and other arguments
            return BaseImageStylizer(mode=mode, ckpt=ckpt_path, max_size=max_size, postprocessor=None, reg_method="ridge", train_mode=False)

    def _parse_run_inputs(self, *args, **kwargs) -> Dict[str, Any]:
        """ merge user-provided arguments with config-based defaults, returning final arguments for run_inference """
        # if user passed `input_files` explicitly, use it; otherwise fallback to config.input_path
        input_files = kwargs.get("input_files", self.config.input_path)
        input_files = ensure_file_list_format(input_files)
        # handle remaining arguments in the same way
        style_files = kwargs.get("style_files", self.config.input_path)  # Possibly config.style_path if you have it
        style_files = ensure_file_list_format(style_files)
        alpha_c = kwargs.get("alpha_c", self.config.alpha_c)
        alpha_s = kwargs.get("alpha_s", self.config.alpha_s)
        # for user to specify whether to save output
        save_output = kwargs.get("save_output", True)  # or default to False if desired
        output_path = kwargs.get("output_path", self.config.output_path)
        if save_output:
            os.makedirs(output_path, exist_ok=True)
        # return the dictionary of final arguments for run_inference
        return {
            "input_files": input_files,
            "style_files": style_files,
            "alpha_c": alpha_c,
            "alpha_s": alpha_s,
            "save_output": save_output,
            "output_path": output_path
        }

    def run_inference(self, *args, **kwargs) -> List[torch.Tensor]:
        """ reads images from self.config.input_path, applies style transfer, and saves to self.config.output_path """
        parsed = self._parse_run_inputs(*args, **kwargs)
        # NOTE: while I could return this as a single batch tensor, it would be way less flexible for variable-sized images
        stylized_images = []
        for idx, img_path in enumerate(parsed["input_files"]):
            content_tensor = self.preprocessor(read_image(img_path, ImageReadMode.RGB))
            # `stylizer.transform` method uses the @transform_preprocess decorator and is typically:
                # transform(sample=..., style_paths=..., alpha_s=..., ...)
            # in a real scenario, you’d pass in your actual style images or style mask paths:
            pastiche = self.stylizer.transform(
                sample = content_tensor,
                style_paths = parsed["style_files"],
                alpha_c = parsed["alpha_c"],
                alpha_s = parsed["alpha_s"],
                # mask_paths=..., or use_segmentation=..., etc. if relevant
            )
            if parsed["save_output"]:
                out_name = os.path.join(parsed["output_path"], os.path.basename(img_path))
                self.save_pastiche_to_disk(pastiche.squeeze(0), out_name)
            # TODO: need to make this more consistent with the video inference output later
            stylized_images.append(pastiche)
        return stylized_images


class VideoInferenceOrchestrator(BaseInferenceOrchestrator):
    """ High-level inference class for video-based style transfer that wraps BaseVideoStylizer or a future masked video stylizer """
    def __init__(self, config: Union[InferenceConfig, dict]):
        super().__init__(config)

    def _validate_config(self):
        """ Validates necessary parameters for video inference. """
        if self.config.modality != "video":
            raise ValueError(f"ERROR: encountered modality={self.config.modality}. Only 'video' is valid for video inference tasks.")

    def _load_stylizer(self):
        """ returns BaseVideoStylizer instance for 'art' or 'photo' style transfer on video """
        from mcapst.stylizers.video_stylizers import BaseVideoStylizer
        if self.config.use_segmentation:
            # TODO: implement a "MaskedVideoStylizer"
            raise NotImplementedError("Segmented video style transfer not yet implemented.")
        return BaseVideoStylizer(
            mode=self.config.transfer_mode,
            ckpt=self.config.ckpt_path,
            max_size=self.config.max_size,
            postprocessor=None,        # Or a custom postprocessor
            reg_method="ridge",
            fps=10,                    # Or read from config
        )

    def _parse_run_inputs(self, *args, **kwargs) -> Dict[str, Any]:
        # if user passed `input_files` explicitly, use it; otherwise fallback to config.input_path
        video_list = kwargs.get("video_list", self.config.input_path)
        video_list = ensure_file_list_format(video_list)
        alpha_c = kwargs.get("alpha_c", self.config.alpha_c)
        alpha_s = kwargs.get("alpha_s", self.config.alpha_s)
        save_output = kwargs.get("save_output", True)
        output_path = kwargs.get("output_path", self.config.output_path)
        if save_output:
            os.makedirs(output_path, exist_ok=True)
        return {
            "video_list": video_list,
            "alpha_c": alpha_c,
            "alpha_s": alpha_s,
            "save_output": save_output,
            "output_path": output_path
        }

    def run_inference(self, *args, **kwargs) -> List[torch.Tensor]:
        """ Returns a list of stylized tensors, or an empty list if 'save_output=True', since the stylizer internally saves by default """
        parsed = self._parse_run_inputs(*args, **kwargs)
        stylized_results: List[torch.Tensor] = []
        for video_path in parsed["video_list"]:
            # pass a video path directly into stylizer.transform, where it internally does reading and batch processing
            out_name = os.path.join(parsed["output_path"], os.path.basename(video_path)) if parsed["save_output"] else None
            # if BaseVideoStylizer returns the frames, we store them in stylized_results
            # TODO: need to refactor how these stylizers are loaded to allow users to pass in custom postprocessors and other arguments
            stylized_frames = self.stylizer.transform(
                sample=video_path,
                # TODO: add a sampler to select style images from a default directory defined in the config
                # !! Remove later - temporaryly hard-coded to use the same style image for all videos
                style_paths = [os.path.realpath(r"data/style/01.jpg")],   # or pass something else
                alpha_c = parsed["alpha_c"],
                alpha_s = parsed["alpha_s"],
                save_output = parsed["save_output"],    # let stylizer write to disk if True
                output_path=out_name,
            )
            # stylized_frames might be large, so the user may prefer not to hold them in memory
            if not parsed["save_output"]:
                stylized_results.append(stylized_frames)
        return stylized_results # may be list of tensors or empty list if save_output=True


def stage_inference_pipeline(config_path: Optional[str] = None):
    """ Top-level convenience function used when calling from CLI or programmatically as
        ```python -m mcapst.pipelines.infer --mode inference --config_path path/to/infer_config.yaml```
    """
    # uses a ConfigManager to parse user config and pass an object directly
    config_manager = ConfigManager(mode="inference", config_path=config_path)
    config = config_manager.get_config()
    # routes to the correct inference class based on modality (image or video)
    if config.modality == "image":
        runner = ImageInferenceOrchestrator(config)
    elif config.modality == "video":
        runner = VideoInferenceOrchestrator(config)
    else:
        raise ValueError(f"Unsupported modality: {config.modality}")
    # By default, run with no direct overrides => read from config
    results = runner.run_inference()
    # return a list of stylized images/frames:
    return results


if __name__ == "__main__":
    stage_inference_pipeline()