# mcapst/infer/config/config.py

# import json
from pathlib import Path
from typing import Union, Optional, Sequence
from pydantic import Field, model_validator, ValidationInfo
# base class that InferenceConfig inherits from
from mcapst.core.utils.config_utils import BaseConfigModel, PathList, AlphaWeights



# TODO: need to actually test inputs with ALL of these extensions - haven't tested ".bmp", ".tiff", ".mov", ".mkv", or ".webm" yet
SUPPORTED_IMG_EXTENSIONS = set({".jpg", ".jpeg", ".png", ".bmp", ".tiff"})
SUPPORTED_VID_EXTENSIONS = set({".mp4", ".avi", ".mov", ".mkv", ".webm"})


# TODO: add a flag for saving the output or just feeding it to some resource queue (esp. for the Python API version where stylization may be an augmentation)
    # should probably simplify the orchestrators and make a separate entry point to use with the Python API version
        # e.g. a simple functional API endpoint to input the minimal args needed and return the stylized outputs

# TODO: might move these to a separate file with other project-wide "constants" later
DEFAULT_INFERENCE_CHECKPOINTS = {
    "art": {
        "image": ("checkpoints/art_image.pt", "1i7EalkfclbyB7xvgAxI_b96em3yuGZGQ"),
        "video": ("checkpoints/art_video.pt", "1-gIg-IvCbu02hmpcgY75_VdejwoNwIX2"),
    },
    "photo": {
        "image": ("checkpoints/photo_image.pt", "1hjoltyItPQmaRTNzWZqOG0aj8H50OYpm"),
        "video": ("checkpoints/photo_video.pt", "1mKKijuDKDS-UGJCI9D0_iNd029vStMlw")
    },
}

class InferenceConfig(BaseConfigModel):
    """ Configuration model for inference mode, with fields for input data, style templates, and output settings. """
    input_paths: PathList = Field(..., description="Path to input (content) data for inference.")
    style_paths: PathList = Field(..., description="Path to style image templates to apply (up to 8)")
    # TODO: revise this so that it always creates a subdirectory under the cwd, but checks if the cwd is already `mcapst` and creates "mcapst/results" if not (like gallery-dl)
    output_path: Union[str, Path] = Field('results',
        description="Path to save generated images/videos. Defaults to 'results' subdirectory in the current working directory."
    )
    # TODO: need to revisit this since I've changed the way alpha_c and alpha_s are used from the original implementation and they're no longer independent
        # they're implicitly normalized in the StyleWeights dataclass now, but that kind of "hidden" behavior might not be ideal for other users
    alpha_c: AlphaWeights = Field(0.0, description="Content weight blending factor for stylization. Must be between 0.0 and 1.0.")
    alpha_s: AlphaWeights = Field(1.0,
        description="Style weights blending factors for stylization. Must be a list of floats with length equal to the number of style images provided (up to 8)."
    )
    #? NOTE: leaving the max huge in case someone wants to try super-resolution inference, but should probably be limited to 1280-2160 in most cases
    max_size: int = Field(1280, ge=128, le=4096, description="Maximum size (of longest side) for input images during inference.")
    ckpt_path: Optional[Path] = Field(None,
        description="Path to the checkpoint file for inference. If not provided, defaults to the appropriate checkpoint based on transfer mode and modality."
    )
    # TODO: determine whether to remove this argument altogether in favor of either inferring it from cmask_paths and smask_paths or another for auto-segmentation
    use_segmentation: bool = Field(False, description="Whether to use segmentation-based style transfer.")
    cmask_paths: Optional[PathList] = Field(None,
        description="Path to the content segmentation mask file(s). If provided, must match the names and number of arguments to 'input_paths'."
    )
    smask_paths: Optional[PathList] = Field(None,
        description="Path to the style segmentation mask file(s). If provided, must match the names and number of arguments to 'style_paths'."
    )

    @model_validator(mode='after')
    def filter_path_lists(self, info: ValidationInfo) -> "InferenceConfig":
        print("input_paths: ", self.input_paths)
        print("style_paths: ", self.style_paths)
        allowed_ext = SUPPORTED_IMG_EXTENSIONS if self.modality == "image" else SUPPORTED_VID_EXTENSIONS
        # 1. extension filtering
        exts = SUPPORTED_IMG_EXTENSIONS if self.modality=="image" else SUPPORTED_VID_EXTENSIONS
        self.input_paths = [p for p in self.input_paths if p.suffix.lower() in exts]
        assert self.input_paths, f"No valid input files found in {self.input_paths}. Please provide valid paths with extensions: {', '.join(allowed_ext)}."
        self.style_paths = [p for p in self.style_paths if p.suffix.lower() in SUPPORTED_IMG_EXTENSIONS]
        assert self.style_paths, f"No valid style files found in {self.style_paths}. Please provide valid paths with extensions: {', '.join(SUPPORTED_IMG_EXTENSIONS)}."
        # 2. max number of styles
        N = len(self.style_paths)
        print("alpha_s: ", self.alpha_s)
        if N > 8:
            raise ValueError(f"Too many style images provided: {len(self.style_paths)}. Please provide at most 8 style images.")
        # warn the user when using 4-8 style images, just in case they didn't intend to
        elif 3 < N <= 8:
            msg = f"WARNING: Using {len(self.style_paths)} style images for inference, which you may not have intended. Be aware of increased runtime and memory usage."
            print(f"\x1b[33m{msg}\x1b[0m")  # yellow color for warning
        # ensure the number of alpha_s values matches the number of style images
        if len(self.alpha_s) != N:
            msg = f"WARNING: number of style alpha weights != number style inputs; Defaulting to {N} equal weights for {N} style inputs..."
            print(f"\x1b[33m{msg}\x1b[0m")  # yellow color for warning
            self.alpha_s = [1/N] * N  # normalize to equal weights if not provided
        return self

    @model_validator(mode='after')
    def validate_mask_paths(self, m: ValidationInfo) -> "InferenceConfig":
        """ Validates the segmentation mask paths if provided. """
        # TODO: remove these (or add additional condition) eventually once auto-segmentation is implemented
        if self.use_segmentation:
            if not self.cmask_paths or not self.smask_paths:
                raise ValueError("Segmentation masks are enabled, but no mask paths were provided. Provide valid paths to both `cmask_paths` and `smask_paths`.")
            field_iter_dict = {"content": (self.cmask_paths, self.input_paths), "style": (self.smask_paths, self.style_paths)}
            for img_type, (mask_paths, input_paths) in field_iter_dict.items():
                masks = []
                input_fnames = set({p.stem for p in input_paths})  # get the file names without extensions for matching
                # TODO: might want to test whether set difference would be faster here - this just allows it to fail early for missing files
                for p in mask_paths:
                    # ensure mask file names match some input image file name
                    if p.stem not in input_fnames:
                        raise ValueError(f"Mask file '{p}' does not match any input {img_type} image file names. Ensure mask files have the same names as input images.")
                    # filter the mask paths to only include valid image files (eventually support other mask formats like COCO-style JSON)
                    if p.suffix.lower() in SUPPORTED_IMG_EXTENSIONS:
                        masks.append(p)
                # ensure the number of masks matches the number of input images (or has extras we can ignore)
                if len(masks) >= len(input_paths):
                    raise ValueError(f"Number of {img_type} masks ({len(masks)}) does not match number of {img_type} images ({len(input_paths)}).")
                # update the mask paths in the model with the filtered list
                if img_type == "content":
                    self.cmask_paths = masks
                else:
                    self.smask_paths = masks
        return self


    #? NOTE: look into whether I can have more than one "after" validator
    # needs to be an after-validator to ensure the checkpoint exists after all other fields are validated since it depends on transfer_mode and modality
    @model_validator(mode='after')
    def assign_and_check_ckpt(self) -> "InferenceConfig":
        # segmentation-based style transfer is not yet implemented for video data, so raise an error if requested
        if self.modality == "video" and self.use_segmentation:
            raise NotImplementedError("Segmentation-based style transfer is not yet implemented for video data.")
        default_ckpt, default_id = DEFAULT_INFERENCE_CHECKPOINTS[self.transfer_mode][self.modality]
        # set default if missing and convert to Path
        self.ckpt_path = self.ckpt_path or default_ckpt
        if not Path(self.ckpt_path).exists():
            # prompt download if default path is missing
            if self.ckpt_path == default_ckpt:
                # if the ckpt_path is the default and isn't found, prompt the user to download the default checkpoint
                from mcapst.core.setup.downloads import prompt_to_download_checkpoint
                prompt_to_download_checkpoint(default_ckpt, default_id)
            else: # if it's a non-default checkpoint and the user just passed an invalid path, raise an error
                raise FileNotFoundError(f"Checkpoint at '{self.ckpt_path}' does not exist. Please provide a valid path or download the default checkpoint.")
        self.ckpt_path = Path(self.ckpt_path).resolve()  # ensure the path is absolute
        return self
