# mcapst/train/config/config.py
from datetime import datetime
from pathlib import Path
from pydantic import (
    BaseModel, Field, ConfigDict, field_validator, model_validator, ValidationInfo,
    PositiveInt, PositiveFloat, NonNegativeFloat, NonNegativeInt, DirectoryPath
)
from typing import Optional, Literal, Union, Dict, Any, Annotated
# local imports
from mcapst.core.utils.config_utils import BaseConfigModel, ConfigManager
from mcapst.core.utils.utils import test_if_valid_hf_dataset



SUPPORTED_IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
# default VGG19 checkpoint path (where to look for or save it) and the Google Drive ID to download from if the latter
DEFAULT_VGG_CKPT = ("checkpoints/vgg_normalised.pth", "1HChq_ab5DmdUMDqO_5KTQhwoOw4f5Rhc")

# encapsulated dataclasses dedicated to specific groups of configuration attributes
class LossConfig(BaseModel):
    style_weight: float = Field(1.0, ge=0.0, le=1.0, description="Weight for style loss component.")   # original default: 1.0
    content_weight: float = Field(0.0, ge=0.0, le=1.0, description="Weight for content loss component.")  # original default: 0.0
    lap_weight: NonNegativeFloat = Field(200.0, description="Weight for Matting Laplacian loss component. Set to 0 to omit.") # original default: 1500
    rec_weight: PositiveFloat = Field(10.0, description="Weight for reconstruction (L1) loss component.") # original default: 10.0
    temporal_weight: NonNegativeFloat = Field(0.0, description="Weight for temporal loss in video style transfer. Default=0 when modality != 'video'") # original default: 60.0
    vgg_ckpt: Path = Field(DEFAULT_VGG_CKPT[0], description="Path to the VGG19 checkpoint for style encoding")

    @field_validator("vgg_ckpt", mode="before")
    def _validate_or_download_vgg(cls, vgg_path: str) -> Path:
        p = Path(vgg_path)
        if not p.exists():
            # if the path doesn't exist and it's the default VGG checkpoint, prompt user to download it
            if p.name == DEFAULT_VGG_CKPT[0]:
                from mcapst.core.setup.downloads import prompt_to_download_checkpoint
                prompt_to_download_checkpoint(p, DEFAULT_VGG_CKPT[1])
            # otherwise raise an error if the user just provided a checkpoint path that doesn't exist
            else:
                raise FileNotFoundError(f"VGG checkpoint file '{p}' does not exist. Provide a valid path or download the default checkpoint.")
        return p


#     # TODO: add these later and update train.py to use them - they're currently hardcoded in the orchestrator
#     # split: str = "train"  # split of the dataset to use (e.g. "train", "validation", "test")
#     # buffer_size: int = 0  # buffer size for shuffling in streaming datasets

class LocalDatasetConfig(BaseModel):
    model_config = ConfigDict(discriminator="use_local_data")
    use_local_data: Literal[True] = Field(
        True, description="Whether to use local directories for train_content & train_style."
    )
    streaming: bool = Field(False, description="Unused when use_local_data=True.")
    train_content: DirectoryPath = Field(..., description="Local directory of content images for training.")
    train_style: DirectoryPath = Field(..., description="Local directory of style images for training.")
    batch_size: PositiveInt = Field(4, description="Number of samples per batch.")
    new_size: int = Field(512, ge=128, description="Resize images to this size during training.")

    @model_validator(mode="after")
    def forbid_streaming(self):
        if self.streaming:
            raise ValueError("Cannot stream when use_local_data=True")
        return self

    @field_validator("train_content", "train_style", mode="after")
    def check_directories(cls, v: DirectoryPath, info: ValidationInfo) -> DirectoryPath:
        """ ensure that the directory contains valid image files and enough to train with """
        dir_size = len(list(filter(lambda f: f.suffix.lower() in SUPPORTED_IMG_EXTENSIONS, Path(v).iterdir())))
        if info.field_name == "train_content" and dir_size < 100:
            raise ValueError(f"Directory '{v}' must contain at least 100 valid image files for training.")
        elif info.field_name == "train_style" and dir_size < 20:
            raise ValueError(f"Directory '{v}' must contain at least 20 valid image files for training.")
        return v

class HFDatasetConfig(BaseModel):
    model_config = ConfigDict(discriminator="use_local_data")
    use_local_data: Literal[False] = Field(
        False, description="Whether to use local directories for train_content & train_style."
    )
    streaming: bool = Field(False, description="Whether to stream from HF in map or streaming mode.")
    train_content: Optional[str] = Field(None, description="Hugging Face dataset ID or full URL for content.")
    train_style: Optional[str] = Field(None, description="Hugging Face dataset ID or full URL for style.")
    batch_size: PositiveInt = Field(4, description="Number of samples per batch.")
    new_size: int = Field(512, ge=128, description="Resize images to this size during training.")

    # TODO: add default datasets from `orchestrator.py` to be set here instead (and notify user) - also might want this to be a field_validator instead
    @model_validator(mode="after")
    def check_hf_names(self):
        for name in (self.train_content, self.train_style):
            if name is not None and not test_if_valid_hf_dataset(name):
                raise ValueError(f"Invalid HF dataset: {name!r}")
        return self


DatasetConfig = Annotated[
    Union[LocalDatasetConfig, HFDatasetConfig],
    Field(discriminator="use_local_data",)
]


class TrainingConfig(BaseConfigModel):
    """ Configuration model for training mode, with fields for datasets, training parameters, and loss settings. """
    data_cfg: DatasetConfig
    loss_cfg: LossConfig
    logs_directory: Path = Field("logs", description="Directory to save training logs. Defaults to a new top-level directory named 'logs/'")
    resume: bool = Field(False, description="Whether to resume training from the provided 'ckpt_path'.")
    log_interval: NonNegativeInt = Field(10, description="Interval for logging training progress; Log every `log_interval` batches.")
    lr: PositiveFloat = Field(1e-4, description="Learning rate for training optimization.")
    #& now defaulting to 0, so that the learning rate is constant during training
    # TODO: still need to actually implement LR scheduler support
    lr_decay: NonNegativeFloat = Field(0, description="Decay rate for learning rate during training. Default is 0 (constant LR).")
    # the number of batches the trainer goes through - might want to refactor to use epochs eventually, but the original authors used this
    train_iter: PositiveInt = Field(160_000, description="Total number of training iterations (number of batches to process).")
    ckpt_interval: NonNegativeInt = Field(5000, description="Interval for saving model checkpoints; Log every `ckpt_interval` batches.")
    grad_max_norm: PositiveFloat = Field(5.0, description="Maximum norm for gradient clipping during training.")  # clamp max norm of the gradient to this
    # destination path with default name based on current date and time or if provided while resume is True, the path to the checkpoint to resume from
    ckpt_path: Path = Field(
        default_factory=lambda: Path("checkpoints")/Path(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")).with_suffix(".pt"),
        description="Path to the checkpoint file to resume training from; Otherwise defaults to a new checkpoint based on datetime."
    )

    # @field_validator('data_cfg', 'loss_cfg', mode='before')
    # def coerce_nested_cfg(cls, v, info: ValidationInfo) -> Union[DatasetConfig, LossConfig]:
    #     # ensure that data_cfg and loss_cfg are instances of the respective dataclasses if they're passed as dictionaries
    #     if isinstance(v, dict):
    #         if info.field_name == 'data_cfg':
    #             try:
    #                 print("data_cfg dictionary: ", v)
    #                 return DatasetConfig.model_validate(v)
    #             except ValidationError as e:
    #                 raise ValueError(f"Invalid data_cfg: {e}") from e
    #         elif info.field_name == 'loss_cfg':
    #             return LossConfig(**v)
    #     elif not isinstance(v, (LocalDatasetConfig, HFDatasetConfig, LossConfig)):
    #         raise TypeError(f"{info.field_name} must be a DatasetConfig or LossConfig instance or a dictionary")
    #     return v

    @field_validator('ckpt_interval', mode='before')
    def check_ckpt_interval(cls, v):
        """ just print a warning if the interval is 0, but don't raise an error, in case the user doesn't actually want to run in debug mode """
        if v == 0:
            print("WARNING: ckpt_interval is set to 0, which means no checkpoints will be saved during training.")
        elif v > cls.train_iter:
            raise ValueError(f"ckpt_interval ({v}) must be less than or equal to train_iter ({cls.train_iter}).")
        return v

    @model_validator(mode='after')
    def cross_validate(self):
        # video must have temporal weight >0
        if self.modality=='video' and self.loss_cfg.temporal_weight == 0.0:
            print("WARNING: `temporal_weight` must be greater than 0 for video training; Defaulting to temporal_weight=20.0")
            self.loss_cfg.temporal_weight = 20.0
        return self


def get_training_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """ Returns a ConfigManager for the TrainingConfig model """
    return ConfigManager(TrainingConfig, config_path, description="Training configuration")
