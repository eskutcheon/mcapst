import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# local imports
from mcapst.utils.dataset import DataManager
from mcapst.config.configure import ConfigManager




class LossManager:
    def __init__(self, config):
        self.config = config
        self.l1_loss = nn.L1Loss()
        self.lap_weight = config.get("lap_weight", 0)
        self.temporal_weight = config.get("temporal_weight", 0)
        if self.temporal_weight > 0:
            from mcapst.utils.TemporalLoss import TemporalLoss
            self.temporal_loss = TemporalLoss()
        if self.lap_weight > 0:
            from mcapst.utils.MattingLaplacian import laplacian_loss_grad
            self.laplacian_loss_grad = laplacian_loss_grad

    def transfer(self, content_features, style_features):
        from mcapst.models.cWCT import cWCT
        transfer_module = cWCT(train_mode=True)
        return transfer_module.transfer(content_features, style_features)

    def compute_losses(self, content_img, style_img, stylized_img, laplacian_list=None):
        """ Computes all losses, including Matting Laplacian loss if enabled. """
        losses = {}
        # Content and Style Loss
        losses["content"] = self._compute_content_loss(content_img, stylized_img)
        losses["style"] = self._compute_style_loss(style_img, stylized_img)
        # Reconstruction Loss
        losses["reconstruction"] = self._compute_reconstruction_loss(content_img, stylized_img)
        # Matting Laplacian Loss
        if self.lap_weight > 0 and laplacian_list is not None:
            losses["laplacian"] = self._compute_laplacian_loss(stylized_img, laplacian_list)
        else:
            losses["laplacian"] = 0
        # Temporal Loss (if applicable)
        if self.temporal_weight > 0:
            losses["temporal"] = self._compute_temporal_loss(content_img, stylized_img)
        else:
            losses["temporal"] = 0
        # Total Loss
        losses["total"] = (self.config["content_weight"] * losses["content"] +
                           self.config["style_weight"] * losses["style"] +
                           self.config["reconstruction_weight"] * losses["reconstruction"] +
                           self.config["lap_weight"] * losses["laplacian"] +
                           self.config["temporal_weight"] * losses["temporal"])
        return losses

    def _compute_laplacian_loss(self, stylized_img, laplacian_list):
        """ Computes the Matting Laplacian loss using the precomputed sparse Laplacian matrices. """
        batch_size = stylized_img.size(0)
        lap_losses, gradients = [], []
        for i in range(batch_size):
            lap_loss, gradient = self.laplacian_loss_grad(stylized_img[i], laplacian_list[i])
            lap_losses.append(lap_loss)
            gradients.append(gradient)
        lap_loss_mean = torch.mean(torch.stack(lap_losses))
        self._apply_laplacian_gradients(stylized_img, gradients)
        return lap_loss_mean

    def _apply_laplacian_gradients(self, stylized_img, gradients):
        """ Directly applies Laplacian gradients to the stylized image for backpropagation. """
        grad_tensor = torch.stack(gradients)
        grad_tensor = grad_tensor * self.lap_weight
        grad_tensor = grad_tensor.clamp(-0.05, 0.05)  # Gradient clipping
        stylized_img.backward(grad_tensor, retain_graph=True)

    def _compute_content_loss(self, content_img, stylized_img):
        # Placeholder: Implement VGG-based content loss or similar
        return self.l1_loss(content_img, stylized_img)

    def _compute_style_loss(self, style_img, stylized_img):
        # Placeholder: Implement Gram matrix-based style loss or similar
        return self.l1_loss(style_img, stylized_img)

    def _compute_reconstruction_loss(self, content_img, stylized_img):
        # Placeholder: L1 loss for reconstruction
        return self.l1_loss(content_img, stylized_img)

    def _compute_temporal_loss(self, content_img, stylized_img):
        # Placeholder: Temporal loss computation
        return self.temporal_loss(content_img, stylized_img)



class Trainer:
    def __init__(self, config_path: str):
        self.config = ConfigManager(config_path).get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_manager = DataManager(self.config)
        self.loss_manager = LossManager(self.config)
        # Initialize model and optimizer
        self.model = self._initialize_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.current_iter = 0
        self.total_iterations = self.config["training_iterations"] + self.config["fine_tuning_iterations"]
        self.writer = SummaryWriter(log_dir=self.config["logs_directory"])
        # Resume if needed
        self._resume_checkpoint()

    def _initialize_model(self):
        from ..models.RevResNet import RevResNet
        if self.config["mode"] == "photorealistic":
            model = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=16, sp_steps=2)
        elif self.config["mode"] == "artistic":
            model = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=64, sp_steps=1)
        else:
            raise ValueError("Unsupported mode: {}".format(self.config["mode"]))
        return model.to(self.device).train()

    def _resume_checkpoint(self):
        if self.config["resume"]:
            checkpoint_path = os.path.join(self.config["checkpoint_directory"], "last.pt")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.current_iter = self.config["resume_iter"]
            print(f"Resumed from checkpoint at iteration {self.current_iter}")

    def train(self):
        while self.current_iter < self.total_iterations:
            content_batch, style_batch = self.data_manager.get_next_batches()
            # Prepare Laplacian matrices if needed
            laplacian_list = None
            if self.config["lap_weight"] > 0:
                laplacian_list = self.data_manager.compute_laplacian_matrices(content_batch)
            # Forward pass
            content_features = self.model(content_batch, forward=True)
            style_features = self.model(style_batch, forward=True)
            # Compute stylized image
            stylized_features = self.loss_manager.transfer(content_features, style_features)
            stylized_image = self.model(stylized_features, forward=False)
            # Compute losses
            losses = self.loss_manager.compute_losses(content_batch, style_batch, stylized_image, laplacian_list)
            # Backward and optimize
            total_loss = losses["total"]
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            # Logging and checkpointing
            self._log_progress(losses)
            self._save_checkpoint()
            self.current_iter += 1

    def _log_progress(self, losses):
        if self.current_iter % self.config["log_interval"] == 0:
            message = f"Iteration {self.current_iter}/{self.total_iterations} | " + \
                    " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
            print(message)
            self.writer.add_scalar("Total Loss", losses["total"], self.current_iter)

    def _save_checkpoint(self):
        if self.current_iter % self.config["checkpoint_interval"] == 0:
            checkpoint_path = os.path.join(self.config["checkpoint_directory"], "last.pt")
            torch.save({
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": self.current_iter
            }, checkpoint_path)
