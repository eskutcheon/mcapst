# mcapst/infer/config/config.py

# import json
from pathlib import Path
from typing import List, Union, Optional, Sequence
from pydantic import Field, field_validator, model_validator, ValidationInfo
# for subclassing BaseConfigManager:
from mcapst.core.utils.config_utils import BaseConfigModel #, ConfigManager #, PathList



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
    input_paths: Union[str, Path, List[str], List[Path]] = Field(..., description="Path to input (content) data for inference.")
    style_paths: Union[str, Path, List[str], List[Path]] = Field(..., description="Path to style image templates to apply (up to 8)")
    # TODO: revise this so that it always creates a subdirectory under the cwd, but checks if the cwd is already `mcapst` and creates "mcapst/results" if not (like gallery-dl)
    output_path: Union[str,Path] = Field('results',
        description="Path to save generated images/videos. Defaults to 'results' subdirectory in the current working directory."
    )
    # TODO: need to revisit this since I've changed the way alpha_c and alpha_s are used from the original implementation and they're no longer independent
        # they're implicitly normalized in the StyleWeights dataclass now, but that kind of "hidden" behavior might not be ideal for other users
    alpha_c: float = Field(0.0, ge=0.0, le=1.0, description="Content weight blending factor for stylization. Must be between 0.0 and 1.0.")
    # TODO: annotate these better with their allowed ranges and `nargs`
    alpha_s: List[float] = Field(...,
        description="Style weights blending factors for stylization. Must be a list of floats with length equal to the number of style images provided (up to 8)."
    )
    #? NOTE: leaving the max huge in case someone wants to try super-resolution inference, but should probably be limited to 1280-2160 in most cases
    max_size: int = Field(1280, ge=128, le=4096, description="Maximum size (of longest side) for input images during inference.")
    ckpt_path: Optional[Path] = Field(None,
        description="Path to the checkpoint file for inference. If not provided, defaults to the appropriate checkpoint based on transfer_mode and modality."
    )
    # TODO: determine whether to remove this argument altogether in favor of either inferring it from cmask_paths and smask_paths or another for auto-segmentation
    use_segmentation: bool = Field(False, description="Whether to use segmentation-based style transfer.")
    cmask_paths: Optional[List[Path]] = Field(None,
        description="Path to the content segmentation mask file(s). If provided, must match the names and number of arguments to --input-paths."
    )
    smask_paths: Optional[List[Path]] = Field(None,
        description="Path to the style segmentation mask file(s). If provided, must match the names and number of arguments to --style-paths."
    )

    # TODO: (MAYBE) create a new helper function and add it to a new typing.Annotated type for PathList that handles the coercion and validation of input paths
    @field_validator("input_paths", "style_paths", "cmask_paths", "smask_paths", mode="before")
    def _coerce_input_paths(cls, v, info: ValidationInfo) -> Optional[List[Path]]:
        """ Coerce input paths to a list of Path objects, validating that they exist and are either files or a directory with target files
            Accepts:
                - a single path (str or Path) to a file or directory
                - a list of paths (str or Path) to files
            Returns:
                - a flat list of Path objects for valid files
        """
        # handling possible JSON-encoded lists, which shouldn't happen to begin with so I still need to figure out where that comes from
        if isinstance(v, (str, Path)):
            v = str(v)
            if v.strip().startswith('["') and v.strip().endswith('"]'):
                print("removing chars from start and end of input paths: ", v)
                v = v.lstrip('["').rstrip('"]')
        # for a single path, convert to Path; add to a list if it's a file or unpack a directory of files into the list
        if isinstance(v, (str, Path)):
            p = Path(v)
            if p.is_file():
                return [p]
            elif p.is_dir():
                return [f for f in p.iterdir() if f.is_file()]
        # if it's an iterable (list or tuple), convert each item to Path and filter out non-files; if it's a single directory in the list, unpack it
        if isinstance(v, (list, tuple)):
            if len(v) == 1 and Path(v[0]).is_dir():
                p = Path(v[0])
                return [f for f in p.iterdir() if f.is_file()]
            return [Path(x) for x in v if Path(x).is_file()]
        if v is None and info.field_name in ("cmask_paths", "smask_paths"):
            # allow None for mask paths, but not for input or style paths
            return None
        raise ValueError(f"Expected a path, dir, or list of files; got {v!r}")

    @model_validator(mode='after')
    def filter_path_lists(self, info: ValidationInfo) -> "InferenceConfig":
        allowed_ext = SUPPORTED_IMG_EXTENSIONS if self.modality == "image" else SUPPORTED_VID_EXTENSIONS
        # 1. extension filtering
        exts = SUPPORTED_IMG_EXTENSIONS if self.modality=="image" else SUPPORTED_VID_EXTENSIONS
        self.input_paths = [p for p in self.input_paths if p.suffix.lower() in exts]
        assert self.input_paths, f"No valid input files found in {self.input_paths}. Please provide valid paths with extensions: {', '.join(allowed_ext)}."
        self.style_paths = [p for p in self.style_paths if p.suffix.lower() in SUPPORTED_IMG_EXTENSIONS]
        assert self.style_paths, f"No valid style files found in {self.style_paths}. Please provide valid paths with extensions: {', '.join(SUPPORTED_IMG_EXTENSIONS)}."
        # 2. max number of styles
        N = len(self.style_paths)
        if N > 8:
            raise ValueError(f"Too many style images provided: {len(self.style_paths)}. Please provide at most 8 style images.")
        # warn the user when using 4-8 style images, just in case they didn't intend to
        elif 3 < N <= 8:
            print(f"WARNING: Using {len(self.style_paths)} style images for inference, which you may not have intended. Be aware of increased runtime and memory usage.")
        # ensure the number of alpha_s values matches the number of style images
        # TODO: actually may just want to expand alpha_s with N copies of the same (normalized) value instead - think I already do this in StyleWeights
        assert len(self.alpha_s) == N, "The number of values in alpha_s must match the number of style images provided."
        return self

    @model_validator(mode='after')
    def validate_mask_paths(self, m: ValidationInfo) -> "InferenceConfig":
        """ Validates the segmentation mask paths if provided. """
        # TODO: remove these (or add additional condition) eventually once auto-segmentation is implemented
        if self.use_segmentation:
            if not self.cmask_paths or not self.smask_paths:
                raise ValueError("Segmentation masks are enabled, but no mask paths were provided. Provide valid paths to both `cmask_paths` and `smask_paths`.")
            for img_type, (mask_paths, input_paths) in {"content": (self.cmask_paths, self.input_paths), "style": (self.smask_paths, self.style_paths)}.items():
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

    # TODO: move some of the style weight normalization logic from the StyleWeights dataclass to here
    @field_validator('alpha_s', mode='before')
    def normalize_alpha_s(cls, v: Union[float, Sequence[float]]) -> List[float]:
        # assume user gave float or list; coerce to list & normalize
        lst = v if isinstance(v, (list, tuple)) else [v]
        if len(lst) == 1:
            # TODO: remove this after adding ranges to `alpha_s` Field - need to figure out how to do it with Unions of lists and floats
            return lst
        # only normalize with multiple styles since multi-style interpolation blends them by their individual weights
        total = sum(lst)
        if total == 0:
            raise ValueError("sum(alpha_s) must be >0")
        return [x/total for x in lst]

    #? NOTE: look into whether I can have more than one "after" validator
    # needs to be an after-validator to ensure the checkpoint exists after all other fields are validated since it depends on transfer_mode and modality
    @model_validator(mode='after')
    def assign_and_check_ckpt(self) -> "InferenceConfig":
        # segmentation-based style transfer is not yet implemented for video data, so raise an error if requested
        if self.modality == "video" and self.use_segmentation:
            raise NotImplementedError("Segmentation-based style transfer is not yet implemented for video data.")
        default_ckpt, default_id = DEFAULT_INFERENCE_CHECKPOINTS[self.transfer_mode][self.modality]
        # set default if missing and convert to Path
        self.ckpt_path = Path(self.ckpt_path or default_ckpt)
        if not self.ckpt_path.exists():
            # prompt download if default path is missing
            if self.ckpt_path == default_ckpt:
                # if the ckpt_path is the default and isn't found, prompt the user to download the default checkpoint
                from mcapst.core.setup.downloads import prompt_to_download_checkpoint
                prompt_to_download_checkpoint(default_ckpt, default_id)
            else: # if it's a non-default checkpoint and the user just passed an invalid path, raise an error
                raise FileNotFoundError(f"Checkpoint at '{self.ckpt_path}' does not exist. Please provide a valid path or download the default checkpoint.")
        return self
