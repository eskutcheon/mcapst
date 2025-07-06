import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from mcapst.datasets.datasets import HFImageDataset
from mcapst.pipelines.train import ImageTrainer, VideoTrainer
from mcapst.config.configure import ConfigManager

def main():
    # Create a temporary config file
    config_path = "temp_config.yaml"
    config = {
        "base_name": "test_run_video", # "test_run",
        "transfer_mode": "photorealistic",
        "modality": "video",  # "image",
        "lr": 1e-4,
        "lr_decay": 5e-5,
        "data_cfg": {
            #"train_content": "data/train_content",
            #"train_style": "data/train_style",
            "batch_size": 2,
            "new_size": 512,
            "crop_size": 256,
            "use_local_datasets": False,
            # "use_segmentation": False, # not implemented with the Laplacian yet
        },
        "loss_cfg": {
            "style_weight": 1.0,
            "content_weight": 0.0,
            "lap_weight": 200.0,      # Laplacian weight
            "rec_weight": 10,       # Reconstruction (cycle consistency) loss weight
            "temporal_weight": 20.0,   # Temporal loss weight (only relevant for video stylization)
            "vgg_ckpt": "checkpoints/vgg_normalised.pth",
        },
        "training_iterations": 20,
        "resume": False,
        "logs_directory": "logs",
        "model_save_interval": 100,
        "log_interval": 5,  # log every 5 iterations
        "grad_max_norm": 5.0,
    }
    with open(config_path, "w") as file:
        yaml.dump(config, file)
    # load ConfigManager and initialize training
    config_manager = ConfigManager(mode="training", config_path=config_path)
    #trainer = ImageTrainer(config_manager.get_config())
    trainer = VideoTrainer(config_manager.get_config())
    trainer.train()
    # clean up
    os.remove(config_path)

if __name__ == "__main__":
    main()