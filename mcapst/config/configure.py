import os
import argparse
import datetime
import yaml
from typing import Dict, Union, Optional, Type, Literal
from dataclasses import dataclass, field, fields


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

@dataclass
class BaseConfig:
    base_name: Optional[str] = None
    transfer_mode: str = "photorealistic"
    modality: Literal['image', 'video'] = "image"

    def __post_init__(self):
        valid_modes = ["photorealistic", "artistic", "photo", "art"]
        if self.transfer_mode not in valid_modes:
            raise ValueError(f"transfer_mode '{self.transfer_mode}' is invalid. Must be one of {valid_modes}.")
        if self.modality not in ["image", "video"]:
            raise ValueError(f"modality '{self.modality}' is invalid. Must be 'image' or 'video'.")


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
    grad_max_norm: float = 5.0  # gradient clipping max norm
    # destination path with default name based on current date and time or if provided while resume is True, the path to the checkpoint to resume from
    ckpt_path: str = field(default_factory=lambda: os.path.join("checkpoints", f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"))

    def __post_init__(self):
        super().__post_init__()  # from BaseConfig
        # for dictionary overrides, we need to ensure that the nested dataclasses are instantiated properly
        if not isinstance(self.data_cfg, DatasetConfig):
            self.data_cfg = DatasetConfig(**self.data_cfg)
        if not isinstance(self.loss_cfg, LossConfig):
            self.loss_cfg = LossConfig(**self.loss_cfg)
        if self.loss_cfg.temporal_weight > 0.0 and self.modality != "video":
            print("WARNING: temporal_weight > 0.0 but modality is not 'video'. This will have no effect.")
            self.loss_cfg.temporal_weight = 0.0  # reset to 0.0 if not video
        elif self.loss_cfg.temporal_weight == 0.0 and self.modality == "video":
            raise ValueError("temporal_weight must be > 0.0 for video stylization. Please specify a positive value or omit the argument.")
        # Additional checks
        if self.training_iterations < 1 or not isinstance(self.training_iterations, int):
            raise ValueError("training_iterations must be an integer >= 1.")
        if self.model_save_interval <= 0 or not isinstance(self.model_save_interval, int):
            raise ValueError("model_save_interval must be an integer > 0.")


@dataclass
class InferenceConfig(BaseConfig):
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
        super().__post_init__()  # from BaseConfig
        if self.modality == "video" and self.use_segmentation:
            #? NOTE: think this one only supported auto-segmentation, not manual masks, in the original implementation
            raise NotImplementedError("Segmentation-based style transfer is not yet implemented for video data.")
        if not self.ckpt_path:
            mode = "artistic" if self.transfer_mode in ["artistic", "art"] else "photorealistic"
            self.ckpt_path = DEFAULT_INFERENCE_CHECKPOINTS[mode][self.modality]



# ** Configuration Registry **
CONFIG_REGISTRY: Dict[str, Type[BaseConfig]] = {
    "training": TrainingConfig,
    "inference": InferenceConfig,
}


class ConfigManager:
    # TODO: might move this to a separate file if I further separate config definitions from the argument parsing logic
    HELP_MESSAGES = {
        "train_content": "Path to the content dataset for training.",
        "train_style": "Path to the style dataset for training.",
        "vgg_ckpt": "Path to the VGG19 checkpoint for style encoding.",
        "batch_size": "Number of samples per batch.",
        "new_size": "Resize images to this size before processing.",
        "crop_size": "Crop images to this size after resizing.",
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
        "use_local_datasets": "Whether to use local datasets instead of HuggingFace datasets.",
        "max_size": "Maximum size for input images during inference.",
        "grad_max_norm": "Maximum norm for gradient clipping during training.",
        # TODO: add remaining config options here
    }

    """ Manages configuration settings for training and other run modes supporting both CLI arguments and YAML configuration files. """
    def __init__(self, mode: str = "training", config_path: Optional[str] = None):
        self.mode = mode.lower()
        if self.mode not in CONFIG_REGISTRY:
            raise ValueError(f"Unknown mode: {self.mode}. Available modes: {list(CONFIG_REGISTRY.keys())}")
        # load YAML config if provided
        if config_path:
            yaml_args = self._load_yaml_config(config_path)
        self.config: BaseConfig = CONFIG_REGISTRY[self.mode](**yaml_args) if config_path else CONFIG_REGISTRY[self.mode]()
        self._parse_cli_args()
        # After we have merged YAML + CLI, the dataclass’s __post_init__() runs
            # automatically in the constructor. If we need to re-trigger it, do so:
        # (In Python 3.10+ you can do dataclasses.replace(...) to re-run post_init.)
            # But typically it runs automatically if we pass them in the constructor from scratch.
            # So let's do it manually:
        #self._rerun_post_init()

    def _load_yaml_config(self, config_path: str):
        """ Loads settings from a YAML configuration file and updates defaults """
        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)
        # merge YAML config into the current config object
        return yaml_config

    #& added to rearrange logic to allow for instantiating the TrainingConfig just once
    def _reset_config_from_yaml(self, yaml_config: Dict[str, Union[str, int, float, bool]]):
        """ Reset the current config object from a YAML dictionary """
        # merge YAML config into the current config object
        for key, value in yaml_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"WARNING: Unrecognized config key: {key} in YAML reset.")
        # re-run post_init to validate the new config
        self._rerun_post_init()

    def _parse_cli_args(self):
        """ Build an ArgumentParser dynamically from fields in the dataclass, then optionally override from CLI. """
        parser = argparse.ArgumentParser(description=f"Configuration for {self.mode} mode")
        # config_cls = CONFIG_REGISTRY[self.mode]
        # might handle nested configs by flattening.
        # e.g., if self.config is a TrainingConfig, it has data_cfg (DatasetConfig) and loss_cfg (LossConfig).
        # could do a recursion or manual approach like
            #  - parse top-level fields in the config
            #  - parse fields in data_cfg if present
            #  - parse fields in loss_cfg if present
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
        # first, add arguments for top-level config
        add_dataclass_args(self.config, prefix="")
        # if training or inference has nested dataclasses, parse them
        if isinstance(self.config, TrainingConfig):
            # print("Data Config type: ", type(self.config.data_cfg))
            # print("Loss Config type: ", type(self.config.loss_cfg))
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
        # apply CLI overrides to the top-level config
        apply_overrides(self.config, args, prefix="")
        if isinstance(self.config, TrainingConfig):
            apply_overrides(self.config.data_cfg, args, prefix="data_cfg.")
            apply_overrides(self.config.loss_cfg, args, prefix="loss_cfg.")


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
            #! FIXME: I assumed it would read all of this as a string by default - fix later
            if isinstance(raw_val, bool):
                return raw_val
            # we used action="store_true" so if present => True, else => False or just parse manually:
            return raw_val.lower() in ["true", "1", "yes"]
        elif field_type == str:
            return str(raw_val)
        elif origin_type == Union:
            # Union type => pick the first that doesn't fail
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