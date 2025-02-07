import os
from typing import Dict, Union
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# local imports
#from mcapst.models.RevResNet import RevResNet
from mcapst.models.CAPVSTNet import CAPVSTNet
from mcapst.data.datasets import DataManager
from mcapst.config.configure import ConfigManager
#from mcapst.training.losses import LossManager
from mcapst.loss.manager import LossManager


TRANSFER_MODE_ALIASES = {
    "photorealistic": "photo",
    "artistic": "art",
}

class Trainer:
    def __init__(self, config_path: str):
        self.config = ConfigManager(config_path).get_config()
        if self.config["mode"] not in (mode_abbr := ["photo", "art"]):
            try:
                self.config["mode"] = TRANSFER_MODE_ALIASES[self.config["mode"]]
            except KeyError:
                raise ValueError(f"Invalid mode: {self.config['mode']}. Expected one of {list(TRANSFER_MODE_ALIASES.keys()).extend(mode_abbr)}.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: replace with the use of a stylizer class in data.managers later
        self.transfer_module = CAPVSTNet(max_size=self.config["new_size"], train_mode=True)
        self.data_manager = DataManager(self.config)
        self.loss_manager = LossManager(self.config)
        self.current_iter = 0
        self.total_iterations = self.config["training_iterations"] + self.config["fine_tuning_iterations"]
        self.writer = SummaryWriter(log_dir=self.config["logs_directory"])

    def set_model_and_optimizer(self, model: nn.Module):
        self.model = model
        # Initialize model and optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        # Resume if needed
        if self.config["resume"]:
            self._resume_checkpoint()

    def _resume_checkpoint(self):
        checkpoint = torch.load(self.config["ckpt_path"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_iter = self.config["resume_iter"]
        print(f"Resumed from checkpoint at iteration {self.current_iter}")

    def train(self):
        raise NotImplementedError("Train method should be implemented in subclasses.")

    def _log_progress(self, losses):
        if self.current_iter % self.config["log_interval"] == 0:
            print(f"Iteration {self.current_iter}/{self.total_iterations} | " + " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()]))
            self.writer.add_scalar("Total Loss", losses["total"], self.current_iter)

    def _save_checkpoint(self):
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": torch.tensor([self.current_iter], dtype=torch.int32)
        }, self.config["ckpt_path"])



class ImageTrainer(Trainer):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        from mcapst.data.managers import BaseImageStylizer
        if self.config["modality"] != "image":
            raise ValueError("The ImageTrainer class is only for image training data!")
        self.transfer_module = BaseImageStylizer(mode=self.config["mode"], ckpt=self.config["ckpt_path"], max_size=self.config["new_size"], train_mode=True)
        self.set_model_and_optimizer(self.transfer_module.revnet)


    # NOTE: for now, the way to call both the image and video stylizer classes' transform methods are the same, so I may be able to just stick with the base class
    def train(self):
        for _ in tqdm(range(self.current_iter, self.total_iterations), desc = "Training Progress"):
            content_batch, style_batch = self.data_manager.get_next_batches()
            # Prepare Laplacian matrices if needed
            # TODO: probably going to remove the computation of the Laplacian from the data manager entirely and let the LossManager handle it
            #laplacian_list = content_batch["laplacian_m"].to(device=self.device) if self.config["lap_weight"] > 0 else None
            content_batch = content_batch["img"].to(self.device)
            style_batch = style_batch["img"].to(self.device)
            # Forward pass
            stylized_batch = self.transfer_module.transform(content_batch, style_batch, alpha_c = self.config["content_weight"], alpha_s = self.config["style_weight"])
            losses = self.loss_manager.compute_losses(content_batch, style_batch, stylized_batch) #, laplacian_list)
            # Backward and optimize
            total_loss = losses["total"]
            print(f"Total loss: {total_loss}")
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            # Logging and checkpointing
            self._log_progress(losses)
            if self.current_iter % self.config["checkpoint_interval"] == 0:
                self._save_checkpoint()
            self.current_iter += 1