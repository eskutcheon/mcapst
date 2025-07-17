import os
# import argparse
import datetime
# import yaml
from typing import Dict, Union, Optional, Type, Literal
from dataclasses import dataclass, field
# for subclassing BaseConfigManager:
from mcapst.core.utils.config_manager import BaseConfigManager


# TODO: replace a lot of this with argparse-dataclass or fancy-dataclass instances to replace some of the argparse boilerplate
# TODO: also should create some custom types (e.g. natural_number for int >= 1, normed_float for float in [0,1], etc.)


HELP_MESSAGES = {
    "base_name": "Base name for the training run, used to create output directories.",
    "transfer_mode": "Transfer mode for the training run, either 'photorealistic' or 'artistic'.",
    # dataset options
    "modality": "Modality of the input datasets: either 'image' or 'video'.",
    "use_local_datasets": "Whether to use local datasets instead of HuggingFace datasets.",
    "train_content": "Path to the content dataset for training.",
    "train_style": "Path to the style dataset for training.",
    "streaming": "Whether to use streaming datasets (e.g. from Hugging Face) for training.",
    # loss options (mostly weights)
    "style_weight": "Weight for style loss component.",
    "content_weight": "Weight for content loss component.",
    "lap_weight": "Weight for Laplacian loss component.",
    "rec_weight": "Weight for reconstruction loss component.",
    "temporal_weight": "Weight for temporal loss in video style transfer.",
    "vgg_ckpt": "Path to the VGG19 checkpoint for style encoding.",
    # training parameters
    "resume": "Whether to resume training from a checkpoint.",
    "ckpt_path": "Path to the checkpoint file to resume training from; Defaults to a new checkpoint based on datetime.",
    "batch_size": "Number of samples per batch.",
    "new_size": "Resize images to this size before processing.",
    "crop_size": "Crop images to this size after resizing.",
    "lr": "Learning rate for optimization.",
    "lr_decay": "Decay rate for learning rate.",
    "training_iterations": "Total number of iterations for training.",
    "model_save_interval": "Interval for saving model checkpoints.",
    "log_interval": "Interval for logging training progress.",
    "grad_max_norm": "Maximum norm for gradient clipping during training.",
    # other options
    "logs_directory": "Directory to save training logs. Defaults to a new top-level directory named 'logs/'",
}



# experimenting with greater encapsulation to have dataclasses dedicated to specific groups of configuration attributes
@dataclass
class LossConfig:
    style_weight: Optional[float] = 1.0 #None    # original project default: 1.0
    content_weight: Optional[float] = 0.0 #None  # original project default: 0.0
    lap_weight: float = 200.0 # original implementation used 1500 - seems way too high since most losses are on the order of 1e-3
    rec_weight: float = 10.0
    temporal_weight: float = 0.0 # original implementation used 60.0 for video stylization - seems slightly high, so trying ~ 20.0
    vgg_ckpt: str = "checkpoints/vgg_normalised.pth"

    def __post_init__(self):
        # if not self.style_weight:
        #     self.style_weight = 1.0 if self.transfer_mode == "artistic" else 0.5
        if self.style_weight is not None and not 0.0 <= self.style_weight <= 1.0:
            raise ValueError(f"style_weight={self.style_weight} out of valid range [0, 1].")
        if self.content_weight is not None and not 0.0 <= self.content_weight <= 1.0:
            raise ValueError(f"content_weight={self.content_weight} out of valid range [0, 1].")
        # might also want temporal_weight to be in [0,1].

@dataclass
class DatasetConfig:
    # NOTE: paths here don't exist but are placeholders for the user to provide their own datasets
    train_content: str = None #"data/content"
    train_style: str = None #"data/style"
    batch_size: int = 2
    new_size: int = 512
    crop_size: int = 256
    use_local_datasets: bool = False
    streaming: bool = False

    def __post_init__(self):
        # Example: if using streaming but also local => conflict => raise error
        if self.streaming and self.use_local_datasets:
            raise NotImplementedError(f"Disjoint config values 'use_local_datasets' and 'streaming' are both True. Set at most one of these to True.")
        if self.batch_size <= 0 or not isinstance(self.batch_size, int):
            raise ValueError("batch_size must be an integer >= 1.")



@dataclass
class TrainingConfig:
    base_name: Optional[str] = None
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
    training_iterations: int = 160000
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
        if self.training_iterations < 1 or not isinstance(self.training_iterations, int):
            raise ValueError("training_iterations must be an integer >= 1.")
        if self.model_save_interval <= 0 or not isinstance(self.model_save_interval, int):
            raise ValueError("model_save_interval must be an integer > 0.")




class TrainingConfigManager(BaseConfigManager):
    """ Config manager subclass specialized for training mode """
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(TrainingConfig, config_path, help_messages=HELP_MESSAGES, description="Training configuration")