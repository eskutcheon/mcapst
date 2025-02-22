import os
from typing import Dict, Union, Optional, Any, Callable
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# local imports
from mcapst.models.CAPVSTNet import CAPVSTNet
from mcapst.models.VGG import VGG19
from mcapst.data.datasets import DataManager
from mcapst.config.configure import ConfigManager, TrainingConfig
from mcapst.loss.manager import LossManager


TRANSFER_MODE_ALIASES = {
    "photorealistic": "photo",
    "artistic": "art",
}

class TrainerBase:
    def __init__(self, config: Union[TrainingConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            config = TrainingConfig(**config)
        self.config = config
        self._validate_config()
        self._normalize_mode(mode = self.config.transfer_mode)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_iter = 0
        self.total_iterations = self.config.training_iterations # + self.config.fine_tuning_iterations
        self.writer = SummaryWriter(log_dir=self.config.logs_directory)
        # TODO: replace with the use of a stylizer class in data.managers later
        #self.transfer_module = CAPVSTNet(max_size=self.config.new_size, train_mode=True)
        self.data_manager = DataManager(self.config.transfer_mode, self.config.data_cfg)
        #!! FIXME: problem with VGG19 hard-coding I think
        style_encoder: Callable = VGG19(self.config.loss_cfg.vgg_ckpt).to(device=self.device)
        self.loss_manager = LossManager(self.config.loss_cfg, style_encoder=style_encoder)
        # Let child classes define self.model, switch to using the stylizer classes like with BaseImageStylizer, etc.
            # would probably need to immediately save a model to a checkpoint after initialization to pass as a checkpoint to the stylizer class
        self.model = None
        self.optimizer = None


    def _validate_config(self):
        """ should be overridden by subclasses to validate config parameters for specific training tasks """
        pass

    def _normalize_mode(self, mode: str):
        """ ensure self.config.transfer_mode is one of {'art', 'photo'} """
        all_modes = list(TRANSFER_MODE_ALIASES.keys()) + list(TRANSFER_MODE_ALIASES.values())
        if mode not in TRANSFER_MODE_ALIASES.values():
            # Attempt to map from 'photorealistic' -> 'photo', etc.
            try:
                self.config.transfer_mode = TRANSFER_MODE_ALIASES[mode]
            except KeyError:
                raise ValueError(f"Invalid transfer mode: '{mode}'!\nExpected one of {all_modes}.")

    def set_model_and_optimizer(self, model: nn.Module):
        self.model = model
        # Initialize model and optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        # Resume if needed
        if self.config.resume:
            self._resume_checkpoint()

    def _resume_checkpoint(self):
        if not os.path.isfile(self.config.ckpt_path):
            raise FileNotFoundError(f"Cannot resume: checkpoint path '{self.config.ckpt_path}' not found.")
        checkpoint = torch.load(self.config.ckpt_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_iter = int(checkpoint["iteration"].item())
        if self.current_iter >= self.total_iterations:
            raise ValueError(f"Resume iteration {self.current_iter} exceeds total iterations {self.total_iterations}.")
        print(f"Resumed from checkpoint at iteration {self.current_iter}")

    def train(self):
        raise NotImplementedError("Train method should be implemented in subclasses.")

    def _save_checkpoint(self):
        os.makedirs(os.path.dirname(self.config.ckpt_path), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": torch.tensor([self.current_iter], dtype=torch.int32)
        }, self.config.ckpt_path)

    @staticmethod
    def get_loss_log_string(losses: Dict[str, float]) -> str:
        return " | ".join([f"{k}: {v:.3e}" for k, v in losses.items() if v > 0])

    def _log_progress(self, losses):
        if self.current_iter % self.config.log_interval == 0:
            self.writer.add_scalar("Total Loss", losses["total"], self.current_iter)



class ImageTrainer(TrainerBase):
    def __init__(self, config: Union[TrainingConfig, Dict[str, Any]]):
        super().__init__(config)
        from mcapst.data.managers import BaseImageStylizer
        # TODO: if I keep using stylizer classes, I'll have to add some staging for choosing this or the MaskedImageStylizer class
        self.transfer_module = BaseImageStylizer(
            mode=self.config.transfer_mode,
            ckpt=self.config.ckpt_path,
            max_size=self.config.data_cfg.new_size,
            train_mode=True
        )
        self.set_model_and_optimizer(self.transfer_module.revnet)

    def _validate_config(self):
        if self.config.modality != "image":
            raise ValueError(f"ImageTrainer only supports 'image' modality; got '{self.config.modality}'.")


    # NOTE: for now, the way to call both the image and video stylizer classes' transform methods are the same, so I may be able to just stick with the base class
    def train(self):
        #!! FIXME: align with the old implementation since now alpha_c and alpha_s are treated differently after my major refactor for inference
            #! might require new config options
        alpha_c = self.config.loss_cfg.content_weight
        alpha_s = self.config.loss_cfg.style_weight
        pbar = tqdm(range(self.current_iter, self.total_iterations), desc="Training Progress")
        for _ in pbar:
            content_batch, style_batch = self.data_manager.get_next_batches()
            content_batch = content_batch["img"].to(self.device)
            style_batch = style_batch["img"].to(self.device)
            # Forward pass
            stylized_batch = self.transfer_module.transform(content_batch, style_batch, alpha_c = alpha_c, alpha_s = alpha_s)
            losses = self.loss_manager.compute_losses(content_batch, style_batch, stylized_batch) #, laplacian_list)
            # Backward and optimize
            total_loss = losses["total"]
            self.optimizer.zero_grad()
            total_loss.backward()
            # TODO: might want to add the gradient clipping magnitude to the config options
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            # logging and checkpointing steps
            pbar.set_description(self.get_loss_log_string(losses))
            self._log_progress(losses)
            # save model checkpoints - would be refactored more like my semantic segmentation project if I switch to epochs instead of iterations
            if self.current_iter % self.config.model_save_interval == 0:
                self._save_checkpoint()
            self.current_iter += 1


class VideoTrainer(TrainerBase):
    """
        Specialized trainer for video-based style transfer.
        Incomplete - still need to adapt data loading for the video frames - might need a new data manager class that wraps the video processor
            it presents some challenges since dataloaders can't pickle methods that use generators
    """
    def __init__(self, config: Union[TrainingConfig, Dict[str, Any]]):
        if config.loss_cfg.temporal_weight == 0:
            raise ValueError("Temporal weight must be greater than 0 for video training!")
        super().__init__(config)
        # using the BaseImageStylizer since the original implementation only generated fake optical flow data between unrelated images in a batch
            # eventually, I'll add a new data manager for batching video frames and generating real optical flow data in the same manner as it does now.
        # TODO: if I keep using stylizer classes, I'll have to add some staging for choosing this or the MaskedImageStylizer class
        from mcapst.data.managers import BaseImageStylizer
        self.transfer_module = BaseImageStylizer(
            mode=self.config.transfer_mode,
            ckpt=self.config.ckpt_path,
            max_size=self.config.data_cfg.new_size,
            train_mode=True
        )
        # from mcapst.data.managers import BaseVideoStylizer
        # self.transfer_module = BaseVideoStylizer(
        #     mode=self.config.transfer_mode,
        #     ckpt=self.config.ckpt_path,
        #     max_size=self.config.data_cfg.new_size,
        #     reg_method="ridge",
        #     train_mode=True)
        self.set_model_and_optimizer(self.transfer_module.revnet)

    def _validate_config(self):
        if self.config.modality != "video":
            raise ValueError(f"VideoTrainer only supports 'video' modality; got '{self.config.modality}'.")

    def train(self):
        #!! FIXME: align with the old implementation since now alpha_c and alpha_s are treated differently after my major refactor for inference
            #! -- might require new config options
        alpha_c = self.config.loss_cfg.content_weight
        alpha_s = self.config.loss_cfg.style_weight
        pbar = tqdm(range(self.current_iter, self.total_iterations), desc="Video Training Progress")
        for _ in pbar:
            #! PLACEHOLDER: need to implement a new data manager for video frames and find a good video dataset
            content_batch, style_batch = self.data_manager.get_next_batches()
            content_batch = content_batch["img"].to(self.device)
            style_batch = style_batch["img"].to(self.device)
            # Forward pass
            stylized_batch = self.transfer_module.transform(content_batch, style_batch, alpha_c = alpha_c, alpha_s = alpha_s)
            temp_stylizer_callback = lambda x: self.transfer_module.transform(x, style_batch, alpha_c = alpha_c, alpha_s = alpha_s)
            losses = self.loss_manager.compute_losses(content_batch, style_batch, stylized_batch, stylizer_callback=temp_stylizer_callback) #, laplacian_list)
            # Backward and optimize
            total_loss = losses["total"]
            self.optimizer.zero_grad()
            total_loss.backward()
            # TODO: might want to add the gradient clipping magnitude to the config options
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            # logging and checkpointing steps
            pbar.set_description(self.get_loss_log_string(losses))
            self._log_progress(losses)
            # save model checkpoints - would be refactored more like my semantic segmentation project if I switch to epochs instead of iterations
            if self.current_iter % self.config.model_save_interval == 0:
                self._save_checkpoint()
            self.current_iter += 1



def stage_training_pipeline(config_path: Optional[str] = None):
    """ Top-level convenience function for launching training from CLI or programmatic usage:
        ```python -m mcapst.pipelines.train --mode training --config_path path/to/train_config.yaml```
    """
    config_manager = ConfigManager(mode="training", config_path=config_path)
    config = config_manager.get_config()
    if config.modality == "image":
        trainer = ImageTrainer(config)
    elif config.modality == "video":
        trainer = VideoTrainer(config)
    else:
        raise ValueError(f"Unsupported modality: {config.modality}")
    trainer.train()


if __name__ == "__main__":
    stage_training_pipeline()