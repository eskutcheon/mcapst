import os
import argparse
import datetime
import yaml
from typing import Dict

class ConfigManager:
    """ Manages configuration settings for training, supporting both CLI arguments and YAML configuration files. """
    def __init__(self, config_path: str = None):
        self.config = self._load_default_config()
        if config_path:
            self._load_yaml_config(config_path)
        self._parse_cli_args()

    def _load_default_config(self) -> Dict:
        """ Returns default configuration settings. """
        return {
            "base_name": None,
            "mode": "photorealistic",
            "modality": "image",
            "ckpt_path": os.path.join("checkpoints", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
            "vgg_ckpoint": "checkpoints/vgg_normalised.pth",
            "train_content": "data/train_content",
            "train_style": "data/train_style",
            "batch_size": 2,
            "new_size": 512,
            "crop_size": 256,
            "use_lap": True,
            "win_rad": 1,
            "lr": 1e-4,
            "lr_decay": 5e-5,
            "style_weight": None,
            "content_weight": None,
            "lap_weight": 1,
            "rec_weight": 10,
            "temporal_weight": 0,
            "training_iterations": 160000,
            "fine_tuning_iterations": 10000,
            "resume": False,
            "resume_iter": -1,
            "logs_directory": "logs",
            "display_size": 16,
            "image_display_iter": 1000,
            "image_save_iter": 10000,
            "model_save_interval": 10000
        }

    def _load_yaml_config(self, config_path: str):
        """ Loads settings from a YAML configuration file and updates defaults. """
        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)
        self.config.update(yaml_config)

    def _parse_cli_args(self):
        """ Parses command-line arguments and updates configuration settings. """
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, help='Path to configuration file')
        parser.add_argument('--base_name', type=str, help='Directory name to save')
        parser.add_argument('--mode', type=str, choices=['photorealistic', 'artistic'], help='Mode of training')
        parser.add_argument('--modality', type=str, choices=['image', 'video'], help='Modality of training data')
        parser.add_argument('--ckpt_path', type=str, help='Path to save checkpoints')
        parser.add_argument('--vgg_ckpoint', type=str, help='Path to VGG checkpoint')
        parser.add_argument('--train_content', type=str, help='Path to content training dataset')
        parser.add_argument('--train_style', type=str, help='Path to style training dataset')
        parser.add_argument('--batch_size', type=int, help='Batch size for training')
        parser.add_argument('--new_size', type=int, help='New size for resizing images')
        parser.add_argument('--crop_size', type=int, help='Crop size for dataset')
        # TODO: FIXME: not explicitly using --use_lap defaults to False no matter what and not using it isn't supported yet
        parser.add_argument('--use_lap', action='store_true', help='Use Laplacian loss')
        parser.add_argument('--win_rad', type=int, help='Window radius for Laplacian loss')
        parser.add_argument('--lr', type=float, help='Learning rate')
        parser.add_argument('--lr_decay', type=float, help='Learning rate decay')
        parser.add_argument('--style_weight', type=float, help='Weight for style loss')
        parser.add_argument('--content_weight', type=float, help='Weight for content loss')
        parser.add_argument('--lap_weight', type=float, help='Weight for Laplacian loss')
        parser.add_argument('--rec_weight', type=float, help='Weight for reconstruction loss')
        parser.add_argument('--temporal_weight', type=float, help='Weight for temporal loss')
        parser.add_argument('--training_iterations', type=int, help='Number of training iterations')
        parser.add_argument('--fine_tuning_iterations', type=int, help='Number of fine-tuning iterations')
        parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
        parser.add_argument('--resume_iter', type=int, help='Iteration to resume training from')
        parser.add_argument('--logs_directory', type=str, help='Directory for logs')
        parser.add_argument('--display_size', type=int, help='Number of images to display in logs')
        parser.add_argument('--image_display_iter', type=int, help='Frequency of displaying images')
        parser.add_argument('--image_save_iter', type=int, help='Frequency of saving images')
        parser.add_argument('--model_save_interval', type=int, help='Interval for saving model checkpoints')
        args = parser.parse_args()
        for key, value in vars(args).items():
            if value is not None:
                self.config[key] = value

    def get_config(self) -> Dict:
        """ Returns the final configuration dictionary. """
        return self.config

# Example usage:
# config_manager = ConfigManager("config.yaml")
# config = config_manager.get_config()
# print(config)