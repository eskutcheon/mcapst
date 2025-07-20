
import os
from typing import List, Union, Optional, Literal, Sequence
from dataclasses import dataclass #, field, fields
# for subclassing BaseConfigManager:
from mcapst.core.utils.config_manager import BaseConfigManager
from mcapst.core.utils.utils import validate_path_arg


# TODO: replace a lot of this with argparse-dataclass or fancy-dataclass instances to replace some of the argparse boilerplate
    # TODO: also should create some custom types (e.g. natural_number for int >= 1, normed_float for float in [0,1], etc.)


# TODO: need to actually test inputs with these extensions -
SUPPORTED_IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
SUPPORTED_VID_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


HELP_MESSAGES = {
    # TODO: still not used anywhere, but it should be used to make subdirectories under `output_path`
    "run_name": "Base name for the inference run, used to create output directories.",
    "transfer_mode": "Transfer mode for the inference run, either 'photorealistic' or 'artistic'.",
    "modality": "Modality of the input data: either 'image' or 'video'.",
    ### MIGHT REMOVE LATER, or at least default to None
    "input_path": "Path to input data for inference.",
    "style_paths": "Path to style template data to apply. If not provided, defaults to 'data/style'.",
    # TODO: need to add a flag for saving the output or just feeding it to some resource queue
        # OR for full inference pipelines, just default to an `mcapst` subdirectory under the current working directory (unless the cwd == "mcapst" already)
        # should probably simplify the orchestrators and make a separate entry point to use with the Python API version
            # e.g. a simple functional API endpoint to input the minimal args needed and return the stylized outputs
    "output_path": "Path to save generated images/videos. Defaults to 'results' subdirectory in the current working directory.",
    "output_path": "Path to save generated images/videos.",
    ###
    "alpha_c": "Content weight blending factor for inference.",
    "alpha_s": "Style weight blending factor for inference.",
    "use_segmentation": "Use segmentation-based style transfer.",
    "mask_path": "Path to the segmentation mask file.",
    "max_size": "Maximum size for input images during inference.",
    "ckpt_path": "Path to the checkpoint file for inference. If not provided, defaults to the appropriate checkpoint based on transfer_mode and modality.",
}


DEFAULT_INFERENCE_CHECKPOINTS = {
    "artistic": {
        "image": ("checkpoints/art_image.pt", "1i7EalkfclbyB7xvgAxI_b96em3yuGZGQ"),
        "video": ("checkpoints/art_video.pt", "1-gIg-IvCbu02hmpcgY75_VdejwoNwIX2"),
    },
    "photorealistic": {
        "image": ("checkpoints/photo_image.pt", "1hjoltyItPQmaRTNzWZqOG0aj8H50OYpm"),
        "video": ("checkpoints/photo_video.pt", "1mKKijuDKDS-UGJCI9D0_iNd029vStMlw")
    },
}


@dataclass
class InferenceConfig:
    run_name: Optional[str] = None
    transfer_mode: str = "photorealistic"
    modality: Literal['image', 'video'] = "image"
    # FIXME: the paths below may be removed altogether in final refactors
    input_path: Union[str, List[str]] = r"mcapst/data/content"
    style_paths: Union[str, List[str]] = r"mcapst/data/style"
    output_path: str = r"mcapst/results"
    # TODO: need to revisit this since I've changed the way alpha_c and alpha_s are used from the original implementation and they're no longer independent
        # they're implicitly normalized in the StyleWeights dataclass now, but that kind of "hidden" behavior might not be ideal for other users
    alpha_c: float = 0.0
    alpha_s: float = 0.5
    use_segmentation: bool = False
    mask_path: Optional[str] = None
    max_size: int = 1280
    ckpt_path: str = None # will be set to the default checkpoint based on transfer_mode and modality if not provided
    # TODO: maybe add regularization options for cWCT during inference (low priority since ridge regression works fine for now)

    def validate_checkpoint(self):
        mode = "artistic" if self.transfer_mode in ["artistic", "art"] else "photorealistic"
        default_ckpt, default_id = DEFAULT_INFERENCE_CHECKPOINTS[mode][self.modality]
        if not self.ckpt_path:
            self.ckpt_path = default_ckpt
        if not os.path.exists(self.ckpt_path):
            # if the path doesn't exist and it's the default checkpoint, prompt user to download it
            if self.ckpt_path == default_ckpt:
                from mcapst.core.setup.downloads import prompt_to_download_checkpoint
                # prompt user to download the default checkpoint if it doesn't exist
                prompt_to_download_checkpoint(self.ckpt_path, default_id)
            # if it's a non-default checkpoint and the user just passed an invalid path, raise an error
            else:
                raise FileNotFoundError(f"Non-default CAP-VSTNet checkpoint at '{self.ckpt_path}' does not exist. Please provide a valid path or download the default checkpoint.")

    def __post_init__(self):
        # previous checks from the old `BaseConfig` class
        ###############################################################################################################
        valid_modes = ["photorealistic", "artistic", "photo", "art"]
        if self.transfer_mode not in valid_modes:
            raise ValueError(f"transfer_mode '{self.transfer_mode}' is invalid. Must be one of {valid_modes}.")
        if self.modality not in ["image", "video"]:
            raise ValueError(f"modality '{self.modality}' is invalid. Must be 'image' or 'video'.")
        ###############################################################################################################
        if self.modality == "video" and self.use_segmentation:
            #? NOTE: think this one only supported auto-segmentation, not manual masks, in the original implementation
            raise NotImplementedError("Segmentation-based style transfer is not yet implemented for video data.")
        # check if the input path exists or use the default if not provided - differs from train config 
        self.validate_checkpoint()
        allowed_ext = SUPPORTED_IMG_EXTENSIONS if self.modality == "image" else SUPPORTED_VID_EXTENSIONS
        self.input_path = validate_path_arg(self.input_path, "input_path", allowed_ext)
        self.style_paths = validate_path_arg(self.style_paths, "style_paths", SUPPORTED_IMG_EXTENSIONS)
        if isinstance(self.style_paths, Sequence):
            # might want to add a user confirmation prompt here?
            if 3 < len(self.style_paths) <= 8:
                print(f"WARNING: Using {len(self.style_paths)} style images for inference. This will interpolate between them, which you may not have intended. Be aware of increased runtime and memory usage.")
            elif len(self.style_paths) > 8:
                raise RuntimeError(f"ERROR: Using {len(self.style_paths)} style images for inference. Now terminating to avoid accidental use of an entire directory.")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)


class InferenceConfigManager(BaseConfigManager):
    """ Config manager subclass specialized for inference mode """
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(InferenceConfig, config_path, help_messages=HELP_MESSAGES, description="Inference configuration")