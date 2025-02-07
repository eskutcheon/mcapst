import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from mcapst.data.datasets import HFImageDataset
from mcapst.pipelines.train import ImageTrainer

def main():
    # Create a temporary config file
    config_path = "temp_config.yaml"
    config = {
        "base_name": "test_run",
        "mode": "photorealistic",
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
        "style_weight": 1.0,
        "content_weight": 0.0,
        "lap_weight": 1,
        "rec_weight": 10,
        "temporal_weight": 0,
        "training_iterations": 10,
        "fine_tuning_iterations": 1,
        "resume": False,
        "resume_iter": -1,
        "logs_directory": "logs",
        "display_size": 16,
        "image_display_iter": 1,
        "image_save_iter": 1,
        "model_save_interval": 1,
        "checkpoint_directory": "checkpoints",
        "log_interval": 1,
        "checkpoint_interval": 1
    }
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    # Load the HuggingFace datasets
    #content_dataset = HFImageDataset("bitmind/MS-COCO-unique-256", split="train[:1%]", use_lap=False)
    #style_dataset = HFImageDataset("huggan/wikiart", split="train[:1%]", use_lap=False)

    # Create data loaders
    #content_loader = DataLoader(content_dataset, batch_size=config["batch_size"], shuffle=True)
    #style_loader = DataLoader(style_dataset, batch_size=config["batch_size"], shuffle=True)

    # Fetch a batch of data
    #content_batch = next(iter(content_loader))
    #style_batch = next(iter(style_loader))
    trainer = ImageTrainer(config_path)
    trainer.train()

    # Clean up
    os.remove(config_path)

if __name__ == "__main__":
    main()