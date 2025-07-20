
import os
import datetime
from typing import Optional, Literal
from dataclasses import dataclass, field
# for subclassing BaseConfigManager:
from mcapst.core.utils.config_manager import BaseConfigManager


# TODO: replace a lot of this with argparse-dataclass or fancy-dataclass instances to replace some of the argparse boilerplate
    # TODO: also create some custom types (e.g. natural_number for int >= 1, normed_float for float in [0,1], etc.)


HELP_MESSAGES = {
    # TODO: still unused but could be used to name new checkpoints or logging subdirectories
    "run_name": "Base name for the training run, used to create output directories.",
    "transfer_mode": "Transfer mode for the training run, either 'photorealistic' or 'artistic'.",
    # dataset options
    "modality": "Modality of the input datasets: either 'image' or 'video'.",
    # TODO: test all combinations of the next 4 options for compatibility tests
    "use_local_data": "Whether to use local datasets instead of HuggingFace datasets.",
    "train_content": "Path to the content dataset for training.",
    "train_style": "Path to the style dataset for training.",
    "streaming": "Whether to use streaming datasets (e.g. from Hugging Face) for training.",
    # loss options (mostly weights)
    # TODO: add non-negativity checks for all weights
    "style_weight": "Weight for style loss component.",
    "content_weight": "Weight for content loss component.",
    "lap_weight": "Weight for Laplacian loss component.",
    "rec_weight": "Weight for reconstruction loss component.",
    "temporal_weight": "Weight for temporal loss in video style transfer.",
    "vgg_ckpt": "Path to the VGG19 checkpoint for style encoding.",
    # training parameters
    "resume": "Whether to resume training from a checkpoint.",
    "ckpt_path": "Path to the checkpoint file to resume training from; Defaults to a new checkpoint based on datetime.",
    # TODO: add type restrictions: positive integers > 0
    "batch_size": "Number of samples per batch.",
    # TODO: add type restrictions: positive integers > 128
    "new_size": "Resize images to this size before processing.",
    #"crop_size": "Crop images to this size after resizing.",
    # TODO: add type restrictions for both below: floats in [0, 1]
    "lr": "Learning rate for optimization.",
    "lr_decay": "Decay rate for learning rate.",
    # TODO: add type restrictions: positive integers > 0
    "train_iter": "Total number of iterations for training.",
    # TODO: add type restrictions for both below: positive integers > 0
    "model_save_interval": "Interval for saving model checkpoints.",
    "log_interval": "Interval for logging training progress.",
    # TODO: add type restrictions: positive floats > 0.0
    "grad_max_norm": "Maximum norm for gradient clipping during training.",
    # other options
    "logs_directory": "Directory to save training logs. Defaults to a new top-level directory named 'logs/'",
}

# default VGG19 checkpoint path (where to look for or save it) and the Google Drive ID to download from if the latter
DEFAULT_VGG_CKPT = ("checkpoints/vgg_normalised.pth", "1HChq_ab5DmdUMDqO_5KTQhwoOw4f5Rhc")

# experimenting with greater encapsulation to have dataclasses dedicated to specific groups of configuration attributes
@dataclass
class LossConfig:
    style_weight: Optional[float] = 1.0 #None    # original project default: 1.0
    content_weight: Optional[float] = 0.0 #None  # original project default: 0.0
    lap_weight: float = 200.0 # original implementation used 1500 - seems way too high since most losses are on the order of 1e-3
    rec_weight: float = 10.0
    temporal_weight: float = 0.0 # original implementation used 60.0 for video stylization - seems slightly high, so trying ~ 20.0
    vgg_ckpt: str = DEFAULT_VGG_CKPT[0]  # path to VGG19 checkpoint file

    def validate_checkpoint(self):
        """ Validates the VGG checkpoint path and downloads it if necessary """
        # if the path doesn't exist and it's the default VGG checkpoint, prompt user to download it
        if self.vgg_ckpt == DEFAULT_VGG_CKPT[0]:
            from mcapst.core.setup.downloads import prompt_to_download_checkpoint
            prompt_to_download_checkpoint(self.vgg_ckpt, DEFAULT_VGG_CKPT[1])
        # if it's a non-default VGG checkpoint and the user just passed an invalid path, raise an error
        else:
            raise FileNotFoundError(f"Non-default VGG checkpoint file '{self.vgg_ckpt}' does not exist. Please provide a valid path or download the default VGG checkpoint.")

    def __post_init__(self):
        if self.style_weight is not None and not 0.0 <= self.style_weight <= 1.0:
            raise ValueError(f"style_weight={self.style_weight} out of valid range [0, 1].")
        if self.content_weight is not None and not 0.0 <= self.content_weight <= 1.0:
            raise ValueError(f"content_weight={self.content_weight} out of valid range [0, 1].")
        if not self.vgg_ckpt:
            self.vgg_ckpt = DEFAULT_VGG_CKPT[0]
        if not os.path.exists(self.vgg_ckpt):
            self.validate_checkpoint()

@dataclass
class DatasetConfig:
    # NOTE: paths here don't exist but are placeholders for the user to provide their own datasets
    train_content: str = None #"data/content"
    train_style: str = None #"data/style"
    batch_size: int = 2
    new_size: int = 512
    # crop_size: int = 256
    use_local_data: bool = False
    streaming: bool = False

    def __post_init__(self):
        # Example: if using streaming but also local => conflict => raise error
        if self.streaming and self.use_local_data:
            raise NotImplementedError(f"Disjoint config values 'use_local_data' and 'streaming' are both True. Set at most one of these to True.")
        if self.batch_size <= 0 or not isinstance(self.batch_size, int):
            raise ValueError("batch_size must be an integer >= 1.")
        # TODO: add a lot more of the validation checks (from datasets.py) for the given paths to fail earlier



@dataclass
class TrainingConfig:
    run_name: Optional[str] = None
    transfer_mode: str = "photorealistic"
    modality: Literal['image', 'video'] = "image"
    logs_directory: str = "logs"
    resume: bool = False
    log_interval: int = 1
    lr: float = 1e-4
    lr_decay: float = 5e-5
    data_cfg: DatasetConfig = field(default_factory=DatasetConfig)
    loss_cfg: LossConfig = field(default_factory=LossConfig)
    # the number of batches the trainer goes through - might want to refactor to use epochs eventually, but the original authors used this
    #? NOTE: still using the majority of the original config attributes' values, but most have been removed or renamed
    train_iter: int = 160000
    model_save_interval: int = 10000
    grad_max_norm: float = 5.0  # gradient clipping max norm
    # destination path with default name based on current date and time or if provided while resume is True, the path to the checkpoint to resume from
    ckpt_path: str = field(default_factory=lambda: os.path.join("checkpoints", f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"))

    def __post_init__(self):
        # previous checks from the old `BaseConfig` class
        ###############################################################################################################
        valid_modes = ["photorealistic", "artistic", "photo", "art"]
        if self.transfer_mode not in valid_modes:
            raise ValueError(f"transfer_mode '{self.transfer_mode}' is invalid. Must be one of {valid_modes}.")
        if self.modality not in ["image", "video"]:
            raise ValueError(f"modality '{self.modality}' is invalid. Must be 'image' or 'video'.")
        ###############################################################################################################
        # for dictionary overrides, we need to ensure that the nested dataclasses are instantiated properly
        if not isinstance(self.data_cfg, DatasetConfig):
            self.data_cfg = DatasetConfig(**self.data_cfg)
        if not isinstance(self.loss_cfg, LossConfig):
            self.loss_cfg = LossConfig(**self.loss_cfg)
        if self.loss_cfg.temporal_weight > 0.0 and self.modality != "video":
            print("WARNING: temporal_weight > 0.0 but modality is not 'video'. This will have no effect.")
            self.loss_cfg.temporal_weight = 0.0  # reset to 0.0 if not video
        elif self.loss_cfg.temporal_weight == 0.0 and self.modality == "video":
            # TODO: might want to just fall back on the default temporal weight (think it's 10.0)
            raise ValueError("temporal_weight must be > 0.0 for video stylization. Please specify a positive value or omit the argument.")
        # Additional checks
        if self.train_iter < 1 or not isinstance(self.train_iter, int):
            raise ValueError("train_iter must be an integer >= 1.")
        if self.model_save_interval <= 0 or not isinstance(self.model_save_interval, int):
            raise ValueError("model_save_interval must be an integer > 0.")




class TrainingConfigManager(BaseConfigManager):
    """ Config manager subclass specialized for training mode """
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(TrainingConfig, config_path, help_messages=HELP_MESSAGES, description="Training configuration")