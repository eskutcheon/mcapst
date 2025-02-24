import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from mcapst.datasets.datasets import HFImageDataset
from mcapst.pipelines.train import ImageTrainer
from mcapst.config.configure import ConfigManager

def main():
    # Create a temporary config file
    config_path = "temp_config.yaml"
    config = {
        "base_name": "test_run",
        "mode": "photorealistic",
        #"win_rad": 1,
        "lr": 1e-4,
        "lr_decay": 5e-5,
        "data_cfg": {
            "train_content": "data/train_content",
            "train_style": "data/train_style",
            "batch_size": 2,
            "new_size": 512,
            "crop_size": 256,
            "use_local_datasets": False,
            #"use_segmentation": False, # not implemented with the Laplacian yet
        },
        "loss_cfg": {
            "style_weight": 1.0,
            "content_weight": 1.0,
            "lap_weight": 1,
            "rec_weight": 10,
            "temporal_weight": 0,
            "vgg_ckpt": "checkpoints/vgg_normalised.pth",
            "use_lap": True,
        },
        "training_iterations": 10,
        #"fine_tuning_iterations": 1,
        "resume": False,
        #"resume_iter": -1,
        "logs_directory": "logs",
        #"display_size": 16,
        #"image_display_iter": 1,
        #"image_save_iter": 100,
        "model_save_interval": 100,
        #"checkpoint_directory": "checkpoints",
        "log_interval": 1,
    }
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    # Load ConfigManager and initialize training
    config_manager = ConfigManager(mode="training", config_path=config_path)
    trainer = ImageTrainer(config_manager.get_config())
    trainer.train()

    # Clean up
    os.remove(config_path)

if __name__ == "__main__":
    main()