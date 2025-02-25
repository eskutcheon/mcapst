import os
import argparse
import datetime
import yaml
from typing import Dict, Union, Optional, Type
from dataclasses import dataclass, field, fields



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

@dataclass
class BaseConfig:
    base_name: Optional[str] = None
    transfer_mode: str = "photorealistic"
    modality: str = "image"

    def __post_init__(self):
        valid_modes = ["photorealistic", "artistic", "photo", "art"]
        if self.transfer_mode not in valid_modes:
            raise ValueError(f"transfer_mode '{self.transfer_mode}' is invalid. Must be one of {valid_modes}.")
        if self.modality not in ["image", "video"]:
            raise ValueError(f"modality '{self.modality}' is invalid. Must be 'image' or 'video'.")


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
    train_content: str = "data/content"
    train_style: str = "data/style"
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
    model_save_interval: int = 10000
    # destination path with default name based on current date and time or if provided while resume is True, the path to the checkpoint to resume from
    ckpt_path: str = field(default_factory=lambda: os.path.join("checkpoints", f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"))

    def __post_init__(self):
        super().__post_init__()  # from BaseConfig
        # for dictionary overrides, we need to ensure that the nested dataclasses are instantiated properly
        if not isinstance(self.data_cfg, DatasetConfig):
            self.data_cfg = DatasetConfig(**self.data_cfg)
        if not isinstance(self.loss_cfg, LossConfig):
            self.loss_cfg = LossConfig(**self.loss_cfg)
        # Additional checks
        if self.training_iterations < 1 or not isinstance(self.training_iterations, int):
            raise ValueError("training_iterations must be an integer >= 1.")
        if self.model_save_interval <= 0 or not isinstance(self.model_save_interval, int):
            raise ValueError("model_save_interval must be an integer > 0.")

@dataclass
class InferenceConfig(BaseConfig):
    input_path: str = "data/test_input"
    output_path: str = "data/test_output"
    alpha_c: float = 0.0
    alpha_s: float = 0.5
    use_segmentation: bool = False
    mask_path: Optional[str] = None
    max_size: int = 1280
    ckpt_path: str = ""
    # TODO: maybe add regularization options for cWCT during inference

    def __post_init__(self):
        super().__post_init__()  # from BaseConfig
        if not self.ckpt_path:
            mode = "art" if self.transfer_mode in ["artistic", "art"] else "photo"
            self.ckpt_path = DEFAULT_INFERENCE_CHECKPOINTS[mode][self.modality]
        # ~~possibly check alpha_c, alpha_s~~ should be unnecessary since the StyleWeights class has a normalization step
        # if self.alpha_c < 0 or self.alpha_c > 1:
        #     raise ValueError(f"alpha_c={self.alpha_c} must be in [0,1].")
        # if self.alpha_s < 0 or self.alpha_s > 1:
        #     raise ValueError(f"alpha_s={self.alpha_s} must be in [0,1].")



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
        "lr": "Learning rate for optimization.",
        "lr_decay": "Decay rate for learning rate.",
        "style_weight": "Weight for style loss component.",
        "content_weight": "Weight for content loss component.",
        "lap_weight": "Weight for Laplacian loss component.",
        "rec_weight": "Weight for reconstruction loss component.",
        "temporal_weight": "Weight for temporal loss in video style transfer.",
        "training_iterations": "Total number of iterations for training.",
        "model_save_interval": "Interval for saving model checkpoints.",
        ### MIGHT REMOVE LATER, or at least default to None
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
        # TODO: add remaining config options here
    }

    """ Manages configuration settings for training and other run modes supporting both CLI arguments and YAML configuration files. """
    def __init__(self, mode: str = "training", config_path: Optional[str] = None):
        self.mode = mode.lower()
        if self.mode not in CONFIG_REGISTRY:
            raise ValueError(f"Unknown mode: {self.mode}. Available modes: {list(CONFIG_REGISTRY.keys())}")
        self.config: BaseConfig = CONFIG_REGISTRY[self.mode]()
        # load YAML config if provided
        if config_path:
            self._load_yaml_config(config_path)
        self._parse_cli_args()
        # After we have merged YAML + CLI, the dataclass’s __post_init__() runs
        # automatically in the constructor. If we need to re-trigger it, do so:
        # (In Python 3.10+ you can do dataclasses.replace(...) to re-run post_init.)
        # But typically it runs automatically if we pass them in the constructor from scratch.
        # So let's do it manually:
        self._rerun_post_init()

    def _load_yaml_config(self, config_path: str):
        """Loads settings from a YAML configuration file and updates defaults."""
        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)
        for key, value in yaml_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"WARNING: Unrecognized config key: {key}")

    def _parse_cli_args(self):
        """ Parses command-line arguments and updates configuration settings dynamically """
        parser = argparse.ArgumentParser(description=f"Configuration for {self.mode} mode")
        # Extract all fields from the current config class
        config_class = CONFIG_REGISTRY[self.mode]
        # Example function to add arguments for a single dataclass instance
        def add_dataclass_args(dc_instance, prefix=""):
            for fobj in fields(dc_instance):
                field_name = fobj.name
                full_name = f"{prefix}{field_name}"
                # attempt to find a help message
                help_msg = self.HELP_MESSAGES.get(field_name, f"{field_name} (no help available)")
                # or if prefix is "data_cfg.", we might do self.HELP_MESSAGES.get("data_cfg."+field_name, help_msg)
                argtype = fobj.type
                is_bool = (argtype == bool)
                # Build CLI flag
                cli_flag = f"--{full_name.replace('_', '-')}"
                if is_bool:
                    parser.add_argument(cli_flag, action="store_true", help=help_msg)
                else:
                    parser.add_argument(cli_flag, type=str, help=help_msg)
        # add arguments for top-level config
        add_dataclass_args(self.config, prefix="")
        # if training or inference has nested dataclasses, parse them
        if isinstance(self.config, TrainingConfig):
            add_dataclass_args(self.config.data_cfg, prefix="data_cfg.")
            add_dataclass_args(self.config.loss_cfg, prefix="loss_cfg.")
        elif isinstance(self.config, InferenceConfig):
            # Inference has no nested dataclasses by default, but that may change soon
            pass
        # actually parse input collected so far
        args, unknown = parser.parse_known_args()
        # apply CLI overrides for top-level arguments with helper function:
        def apply_overrides(dc_instance, args_namespace, prefix=""):
            for fobj in fields(dc_instance):
                field_name = fobj.name
                full_name = f"{prefix}{field_name}"
                cli_key = full_name.replace(".", "_")
                if hasattr(args_namespace, cli_key):
                    raw_val = getattr(args_namespace, cli_key)
                    # If it's not None, set it
                    if raw_val is not None:
                        cast_val = self._cast_value(fobj.type, raw_val)
                        setattr(dc_instance, field_name, cast_val)

        apply_overrides(self.config, args, prefix="")
        if isinstance(self.config, TrainingConfig):
            apply_overrides(self.config.data_cfg, args, prefix="data_cfg.")
            apply_overrides(self.config.loss_cfg, args, prefix="loss_cfg.")

        # args, unknown = parser.parse_known_args()
        # # Apply CLI overrides
        # for key, value in vars(args).items():
        #     if value is not None and hasattr(self.config, key):
        #         setattr(self.config, key, value)

    def _cast_value(self, field_type, raw_val):
        """ helper function to cast raw strings from CLI to the correct type if needed
            e.g. if field_type = int, we do int(raw_val), if field_type is bool, we do (raw_val.lower() in ['true','1','yes']), etc.
            for nested or complex types, I might need to just raise a NotImplementedError for now
        """
        origin_type = getattr(field_type, "__origin__", None)
        if field_type == int:
            return int(raw_val)
        elif field_type == float:
            return float(raw_val)
        elif field_type == bool:
            # we used action="store_true" so if present => True, else => False or just parse manually:
            return raw_val.lower() in ["true", "1", "yes"]
        elif field_type == str:
            return str(raw_val)
        elif origin_type == Union:
            # We have Union type => pick the first that doesn't fail
            for t in field_type.__args__:
                if t == type(None):
                    continue
                try:
                    return self._cast_value(t, raw_val)
                except:
                    pass
            # fallback
            return raw_val
        else:
            # fallback
            return raw_val

    def _rerun_post_init(self):
        """ Re-run the dataclass's __post_init__ to ensure validation. """
        # We do it by re-constructing the config class with the updated fields:
        cls = CONFIG_REGISTRY[self.mode]
        self.config = cls(**vars(self.config))


    def get_config(self) -> BaseConfig:
        """ Returns the final configuration object """
        return self.config

# Example usage:
# config_manager = ConfigManager("config.yaml")
# config = config_manager.get_config()
# print(config)