import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from mcapst.datasets.datasets import HFImageDataset
from mcapst.train import ImageTrainer, VideoTrainer, TrainingConfig #, get_training_config_manager

def main():
    # Create a temporary config file
    config_path = "temp_config.yaml"
    config = {
        "run_name": "test_run_images", # "test_run",
        "transfer_mode": "photorealistic",
        "modality": "image",  # "image",
        "lr": 1e-4,
        "lr_decay": 5e-5,
        "data_cfg": {
            #"train_content": "data/train_content",
            #"train_style": "data/train_style",
            "batch_size": 4,
            "new_size": 512,
            "use_local_data": False,
        },
        "loss_cfg": {
            "style_weight": 1.0,
            "content_weight": 0.0,
            "lap_weight": 200.0,      # Laplacian weight
            "rec_weight": 10,       # Reconstruction (cycle consistency) loss weight
            "temporal_weight": 0.0,   # Temporal loss weight (only relevant for video stylization)
            "vgg_ckpt": "checkpoints/vgg_normalised.pth",
        },
        "train_iter": 20,
        "resume": False,
        "logs_directory": r"logs",
        "ckpt_interval": 0,
        "log_interval": 5,  # log every 5 iterations
        "grad_max_norm": 5.0,
    }
    with open(config_path, "w") as file:
        yaml.dump(config, file)
    # load ConfigManager and initialize training
    # config_manager = get_training_config_manager(config_path=config_path)
    cfg = TrainingConfig(config_path=config_path)
    trainer = ImageTrainer(cfg)
    #trainer = VideoTrainer(config_manager.config_model)
    trainer.train()
    # clean up
    os.remove(config_path)

if __name__ == "__main__":
    main()