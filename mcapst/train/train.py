import os
from typing import Dict, Union, Optional, Any, Callable
from tqdm import tqdm
import torch
# TODO: extract to a new logging file later
from torch.utils.tensorboard import SummaryWriter
# local imports
from mcapst.train.config.config import TrainingConfig #, get_training_config_manager


#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # to avoid fragmentation issues with CUDA memory allocation


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
        self.total_iterations = self.config.train_iter # + self.config.fine_tuning_iterations
        self.writer = SummaryWriter(log_dir=self.config.logs_directory)
        # lazy import to avoid circular imports and improve startup time
        from mcapst.train.datasets.orchestrator import DataManager
        from mcapst.train.loss.manager import LossManager
        self.data_manager = DataManager(self.config.transfer_mode, self.config.data_cfg)
        self.loss_manager = LossManager(self.config.loss_cfg, self.device)
        self.optimizer = None
        # small container class for tracking running means of losses (primarily for logging purposes)
        from mcapst.train.loss.loss_utils import RunningMeanLoss
        self.mean_losses = RunningMeanLoss()


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

    def set_model_and_optimizer(self, model: torch.nn.Module):
        # Initialize model and optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        # Resume if needed
        if self.config.resume:
            self._resume_checkpoint()

    def _resume_checkpoint(self, model: torch.nn.Module):
        # TODO: remove after being handled by Pydantic validation (unless this is used as an API checkpoint separate from the config)
        if not os.path.isfile(self.config.ckpt_dest):
            raise FileNotFoundError(f"Cannot resume: checkpoint path '{self.config.ckpt_dest}' not found.")
        checkpoint = torch.load(self.config.ckpt_dest, weights_only=True, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_iter = int(checkpoint["iteration"].item())
        if self.current_iter >= self.total_iterations:
            raise ValueError(f"Resume iteration {self.current_iter} exceeds config's train_iter={self.total_iterations}.")
        print(f"Resumed from checkpoint at iteration {self.current_iter}")

    def train(self):
        raise NotImplementedError("Train method should be implemented in subclasses.")

    def _save_checkpoint(self, model: torch.nn.Module):
        if self.config.ckpt_interval > 0 and self.current_iter % self.config.ckpt_interval == 0:
            os.makedirs(os.path.dirname(self.config.ckpt_dest), exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": torch.tensor([self.current_iter], dtype=torch.int32)
            }, self.config.ckpt_dest)

    @staticmethod
    def get_loss_log_string(losses: Dict[str, float]) -> str:
        # format losses and pad to align 'Progress'
        nonzero_losses = {k: v for k, v in losses.items() if v > 0}
        loss_str = ' | '.join([f'{k}: {v:.4}' for k, v in nonzero_losses.items()])
        prefix = f"[ Mean Losses ({loss_str}) ]"
        # pad to rough character length, then append Progress
        # TODO: make this more robust to different loss names and lengths
        padding = max(int(22 * len(nonzero_losses)), len(prefix) + 10)  # adjust padding based on number of losses
        padded = prefix.ljust(padding)
        return f"{padded} Progress"

    def _log_progress(self, losses, pbar: tqdm = None):
        if self.config.log_interval > 0 and (self.current_iter + 1) % self.config.log_interval == 0:
            # TODO: separate out the progress bar updates from the logging to TensorBoard
            loss_str = self.get_loss_log_string(self.mean_losses.get_means())
            if self.current_iter % 100 == 0:
                self.mean_losses.reset()  # reset running means every 100 iterations
            pbar.set_description(loss_str, refresh=False)
            self.writer.add_scalar("Total Loss", losses["total"], self.current_iter)



class ImageTrainer(TrainerBase):
    def __init__(self, config: Union[TrainingConfig, Dict[str, Any]]):
        super().__init__(config)
        from mcapst.core.stylizers.image_stylizers import BaseImageStylizer
        # TODO: if I keep using stylizer classes in this way, I'll have to add some staging for choosing this or the MaskedImageStylizer class
        self.transfer_module = BaseImageStylizer(
            mode=self.config.transfer_mode,
            ckpt=self.config.ckpt_dest,
            max_size=self.config.data_cfg.new_size,
            # TODO: add support for post-processors passed to stylizer classes here
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
        # TODO: go back to using the default alpha_c and alpha_s from the original CAP-VSTNet repo to separate their logic from the content-style loss weights
        alpha_c = self.config.loss_cfg.content_weight
        alpha_s = self.config.loss_cfg.style_weight
        grad_clip_magnitude = getattr(self.config, "grad_max_norm", 5.0)  # default value if not specified
        pbar = tqdm(range(self.current_iter, self.total_iterations), miniters=self.config.log_interval, desc="Training Progress")
        for _ in pbar:
            content_batch, style_batch = self.data_manager.get_next_batches()
            content_batch = content_batch["img"].to(self.device)
            style_batch = style_batch["img"].to(self.device)
            # forward pass
            self.optimizer.zero_grad()
            stylized_batch = self.transfer_module.transform(content_batch, [style_batch], alpha_c = alpha_c, alpha_s = alpha_s)
            losses = self.loss_manager.compute_losses(content_batch, style_batch, stylized_batch)
            self.mean_losses.update(losses)
            # back-propagation gradient computation and optimization
            total_loss: torch.Tensor = losses["total"]
            total_loss.backward()
            del stylized_batch, content_batch, style_batch  # free up memory
            torch.nn.utils.clip_grad_norm_(self.transfer_module.revnet.parameters(), grad_clip_magnitude)
            self.optimizer.step()
            # logging and checkpointing steps (only log every log_interval iterations)
            self._log_progress(losses, pbar=pbar)
            # save model checkpoints - would be refactored more like my semantic segmentation project if I switch to epochs instead of iterations
            self._save_checkpoint(self.transfer_module.revnet)
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
        # TODO: need to add some staging for choosing the stylizers (e.g. the MaskedImageStylizer class if use_segmentation is True)
        from mcapst.core.stylizers.image_stylizers import BaseImageStylizer
        self.transfer_module = BaseImageStylizer(
            mode=self.config.transfer_mode,
            ckpt=self.config.ckpt_dest,
            max_size=self.config.data_cfg.new_size,
            reg_method=self.config.reg_method,  # e.g. 'ridge' for cWCT
            train_mode=True
        )
        # TODO: revisit how this is used later - the stylizer below was primarily made for inference and isn't currently suited to training
        # from mcapst.data.managers import BaseVideoStylizer
        # self.transfer_module = BaseVideoStylizer(
        self.set_model_and_optimizer(self.transfer_module.revnet)

    def _validate_config(self):
        if self.config.modality != "video":
            raise ValueError(f"VideoTrainer only supports 'video' modality; got '{self.config.modality}'.")

    def train(self):
        #!! FIXME: align with the old implementation since now alpha_c and alpha_s are treated differently after my major refactor for inference
            #! -- might require new config options
        alpha_c = self.config.loss_cfg.content_weight
        alpha_s = self.config.loss_cfg.style_weight
        grad_clip_magnitude = getattr(self.config, "grad_max_norm", 5.0)  # default value if not specified
        pbar = tqdm(range(self.current_iter, self.total_iterations), miniters=self.config.log_interval, desc="Training Progress")
        for _ in pbar:
            #! PLACEHOLDER: need to implement a new data manager for video frames and find a good video dataset
            content_batch, style_batch = self.data_manager.get_next_batches()
            content_batch = content_batch["img"].to(self.device)
            style_batch = style_batch["img"].to(self.device)
            # Forward pass
            stylized_batch = self.transfer_module.transform(content_batch, [style_batch], alpha_c = alpha_c, alpha_s = alpha_s)
            temp_stylizer_callback = lambda x: self.transfer_module.transform(x, [style_batch], alpha_c = alpha_c, alpha_s = alpha_s)
            self.optimizer.zero_grad()
            losses = self.loss_manager.compute_losses(content_batch, style_batch, stylized_batch, stylizer_callback=temp_stylizer_callback)
            self.mean_losses.update(losses)
            # Back-propagation and optimization
            total_loss: torch.Tensor = losses["total"]
            total_loss.backward()
            del stylized_batch, content_batch, style_batch  # free up memory
            torch.nn.utils.clip_grad_norm_(self.transfer_module.revnet.parameters(), grad_clip_magnitude)
            self.optimizer.step()
            # logging and checkpointing steps (only log every log_interval iterations)
            self._log_progress(losses, pbar=pbar)
            # save model checkpoints - would be refactored more like my semantic segmentation project if I switch to epochs instead of iterations
            self._save_checkpoint(self.transfer_module.revnet)
            self.current_iter += 1



def stage_training_pipeline(config_path: Optional[str] = None, config: Optional[TrainingConfig] = None):
    """ Top-level convenience function for launching training from CLI or programmatic usage:
        ```python -m mcapst.pipelines.train --mode training --config_path path/to/train_config.yaml```
    """
    if config is None:
        # config_manager = get_training_config_manager(config_path=config_path)
        # config: TrainingConfig = config_manager.config_model
        config = TrainingConfig(config_path=config_path)
    if config.modality == "image":
        trainer = ImageTrainer(config)
    elif config.modality == "video":
        trainer = VideoTrainer(config)
    else:
        raise ValueError(f"Unsupported modality: {config.modality}")
    trainer.train()


if __name__ == "__main__":
    stage_training_pipeline()