import os
import argparse
import datetime
import yaml
from typing import Dict, Union, Optional, Type
from dataclasses import dataclass, field, fields



@dataclass
class BaseConfig:
    base_name: Optional[str] = None
    transfer_mode: str = "photorealistic"
    modality: str = "image"


# experimenting with greater encapsulation to have dataclasses dedicated to specific groups of configuration attributes
@dataclass
class LossConfig:
    style_weight: Optional[float] = None
    content_weight: Optional[float] = None
    lap_weight: float = 1.0
    rec_weight: float = 10.0
    temporal_weight: float = 0.0
    use_lap: bool = True
    vgg_ckpt: str = "checkpoints/vgg_normalised.pth"


@dataclass
class DatasetConfig:
    # NOTE: paths here don't exist but are placeholders for the user to provide their own datasets
    train_content: str = "data/content"
    train_style: str = "data/style"
    batch_size: int = 2
    new_size: int = 512
    crop_size: int = 256
    use_local_datasets: bool = False



@dataclass
class TrainingConfig(BaseConfig):
    logs_directory: str = "logs"
    resume: bool = False
    log_interval: int = 1
    lr: float = 1e-4
    lr_decay: float = 5e-5
    data_cfg: DatasetConfig = field(default_factory=DatasetConfig)
    loss_cfg: LossConfig = field(default_factory=LossConfig)
    # the number of batches the trainer goes through - might want to refactor to use epochs eventually, but the original authors used this
    training_iterations: int = 160000
    #image_display_iter: int = 1000
    #image_save_iter: int = 10000
    model_save_interval: int = 10000
    # destination path with default name based on current date and time or if provided while resume is True, the path to the checkpoint to resume from
    ckpt_path: str = field(default_factory=lambda: os.path.join("checkpoints", f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"))

    def __post_init__(self):
        if not isinstance(self.data_cfg, DatasetConfig):
            self.data_cfg = DatasetConfig(**self.data_cfg)
        if not isinstance(self.loss_cfg, LossConfig):
            self.loss_cfg = LossConfig(**self.loss_cfg)

@dataclass
class InferenceConfig(BaseConfig):
    input_path: str = "data/test_input"
    output_path: str = "data/test_output"
    alpha_c: float = None
    alpha_s: float = 0.5
    use_segmentation: bool = False
    mask_path: Optional[str] = None
    max_size: int = 1280
    # TODO: maybe add regularization options for cWCT during inference

    def __post_init__(self):
        if self.modality == "image":
            self.ckpt_path = "checkpoints/photo_image.pt"
        elif self.modality == "video":
            self.ckpt_path = "checkpoints/photo_video.pt"
        else:
            raise ValueError("Unsupported modality. Choose 'image' or 'video'.")



# ** Configuration Registry **
CONFIG_REGISTRY: Dict[str, Type[BaseConfig]] = {
    "training": TrainingConfig,
    "inference": InferenceConfig,
}


class ConfigManager:
    # TODO: might have to refactor with nested dictionaries to reflect the new dataclasses like LossConfig
    HELP_MESSAGES = {
        "train_content": "Path to the content dataset for training.",
        "train_style": "Path to the style dataset for training.",
        "vgg_ckpt": "Path to the VGG19 checkpoint for style encoding.",
        "batch_size": "Number of samples per batch.",
        "new_size": "Resize images to this size before processing.",
        "crop_size": "Crop images to this size after resizing.",
        "use_lap": "Enable Matting Laplacian loss for style transfer.",
        #"win_rad": "Radius of the window used in Matting Laplacian computation.",
        "lr": "Learning rate for optimization.",
        "lr_decay": "Decay rate for learning rate.",
        "style_weight": "Weight for style loss component.",
        "content_weight": "Weight for content loss component.",
        "lap_weight": "Weight for Laplacian loss component.",
        "rec_weight": "Weight for reconstruction loss component.",
        "temporal_weight": "Weight for temporal loss in video style transfer.",
        "training_iterations": "Total number of iterations for training.",
        #"fine_tuning_iterations": "Number of iterations for fine-tuning.",
        #"image_display_iter": "How often to display intermediate results.",
        #"image_save_iter": "How often to save training images.",
        "model_save_interval": "Interval for saving model checkpoints.",
        ### REMOVE LATER
        "input_path": "Path to input data for inference.",
        "output_path": "Path to save generated images/videos.",
        ###
        "alpha_c": "Content weight blending factor for inference.",
        "alpha_s": "Style weight blending factor for inference.",
        "use_segmentation": "Use segmentation-based style transfer.",
        "mask_path": "Path to the segmentation mask file.",
        "log_interval": "Interval for logging training progress.",
        "use_local_datasets": "Use local datasets instead of HuggingFace datasets.",
        "max_size": "Maximum size for input images during inference.",
    }

    """ Manages configuration settings for training and other run modes supporting both CLI arguments and YAML configuration files. """
    def __init__(self, mode: str = "training", config_path: str = None):
        print("mode:", mode)
        self.mode = mode.lower()
        if self.mode not in CONFIG_REGISTRY:
            raise ValueError(f"Unknown mode: {self.mode}. Available modes: {list(CONFIG_REGISTRY.keys())}")
        self.config: BaseConfig = CONFIG_REGISTRY[self.mode]()
        if config_path:
            self._load_yaml_config(config_path)
        self._parse_cli_args()
        print("self.config:", self.config)

    def _load_yaml_config(self, config_path: str):
        """Loads settings from a YAML configuration file and updates defaults."""
        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)
        for key, value in yaml_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def _parse_cli_args(self):
        """ Parses command-line arguments and updates configuration settings dynamically. """
        parser = argparse.ArgumentParser(description=f"Configuration for {self.mode} mode")
        # Extract all fields from the current config class
        config_class = CONFIG_REGISTRY[self.mode]
        for field in fields(config_class):
            cli_flag = f"--{field.name.replace('_', '-')}"
            arg_type = field.type if field.type in [int, float, bool] else str
            parser.add_argument(cli_flag, type=arg_type, help=f"{field.name.replace('_', ' ').capitalize()} (default: {getattr(self.config, field.name)})")
            # field_name = field.name
            # field_type = field.type
            # # Define argument type
            # arg_type = str  # Default type
            # if field_type in [int, float, bool]:
            #     arg_type = field_type
            # elif field_type == Optional[float]:
            #     arg_type = float
            # elif field_type == Optional[str]:
            #     arg_type = str
            # # Define CLI flag name
            # cli_flag = f"--{field_name.replace('_', '-')}"
            # # Check if it's a boolean flag
            # if field_type == bool:
            #     parser.add_argument(cli_flag, action='store_true', help=self.HELP_MESSAGES.get(field_name, ""))
            # else:
            #     parser.add_argument(cli_flag, type=arg_type, help=self.HELP_MESSAGES.get(field_name, ""))
        args, unknown = parser.parse_known_args()
        # Apply CLI overrides
        for key, value in vars(args).items():
            if value is not None and hasattr(self.config, key):
                setattr(self.config, key, value)

    def get_config(self) -> BaseConfig:
        """ Returns the final configuration object """
        return self.config

# Example usage:
# config_manager = ConfigManager("config.yaml")
# config = config_manager.get_config()
# print(config)