
from typing import Dict, Union, Optional, Type, Literal
from dataclasses import dataclass #, field, fields
# for subclassing BaseConfigManager:
from mcapst.core.utils.config_manager import BaseConfigManager


# TODO: replace a lot of this with argparse-dataclass or fancy-dataclass instances to replace some of the argparse boilerplate
# TODO: also should create some custom types (e.g. natural_number for int >= 1, normed_float for float in [0,1], etc.)

# TODO: need to write some helper functions to call in setup.py to download the default checkpoints to the top level data/ directory if they don't exist
    # may go a different direction depending on how I handle this in the next few weeks
DEFAULT_INFERENCE_CHECKPOINTS = {
    "artistic": {
        "image": "checkpoints/art_image.pt",
        "video": "checkpoints/art_video.pt"
    },
    "photorealistic": {
        "image": "checkpoints/photo_image.pt",
        "video": "checkpoints/photo_video.pt"
    },
}


HELP_MESSAGES = {
    "base_name": "Base name for the inference run, used to create output directories.",
    "transfer_mode": "Transfer mode for the inference run, either 'photorealistic' or 'artistic'.",
    "modality": "Modality of the input data: either 'image' or 'video'.",
    ### MIGHT REMOVE LATER, or at least default to None
    "input_path": "Path to input data for inference.",
    "output_path": "Path to save generated images/videos.",
    ###
    "alpha_c": "Content weight blending factor for inference.",
    "alpha_s": "Style weight blending factor for inference.",
    "use_segmentation": "Use segmentation-based style transfer.",
    "mask_path": "Path to the segmentation mask file.",
    "max_size": "Maximum size for input images during inference.",
    "ckpt_path": "Path to the checkpoint file for inference. If not provided, defaults to the appropriate checkpoint based on transfer_mode and modality.",
}




@dataclass
class InferenceConfig:
    base_name: Optional[str] = None
    transfer_mode: str = "photorealistic"
    modality: Literal['image', 'video'] = "image"
    # FIXME: the paths below may be removed altogether in final refactors
    input_path: str = "data/test_input"
    output_path: str = "data/test_output"
    # TODO: need to revisit this since I've changed the way alpha_c and alpha_s are used from the original implementation and they're no longer independent
        # they're implicitly normalized in the StyleWeights dataclass now, but that kind of "hidden" behavior might not be ideal for other users
    alpha_c: float = 0.0
    alpha_s: float = 0.5
    use_segmentation: bool = False
    mask_path: Optional[str] = None
    max_size: int = 1280
    ckpt_path: str = ""
    # TODO: maybe add regularization options for cWCT during inference (low priority since ridge regression works fine for now)

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
        if not self.ckpt_path:
            mode = "artistic" if self.transfer_mode in ["artistic", "art"] else "photorealistic"
            self.ckpt_path = DEFAULT_INFERENCE_CHECKPOINTS[mode][self.modality]



# TODO: figure out how we want to incorporate `ConfigManager`
class InferenceConfigManager(BaseConfigManager):
    """ Config manager subclass specialized for inference mode """
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(InferenceConfig, config_path, help_messages=HELP_MESSAGES, description="Inference configuration")