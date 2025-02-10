
""" will be moving the temporal loss and Matting Laplacian loss to this file while refactoring to use pure pytorch """

from typing import Dict, Union, Literal
import torch
import torch.nn as nn
# local imports
from ..models.VGG import VGG19
from .matting_laplacian import MattingLaplacianLoss
from .temporal_loss import TemporalLoss




class LossManager:
    def __init__(self, config: Dict[str, Union[str, int, float, bool]], style_encoder=None):
        """ Loss manager for computing various losses during training.
            Args:
                config (dict): Configuration dictionary.
                style_encoder (BaseStyleEncoder, optional): A style encoding model for computing content/style loss.
        """
        self.config = config
        self.l1_loss = nn.L1Loss()
        self.lap_weight = config.get("lap_weight", 0)
        self.temporal_weight = config.get("temporal_weight", 0)
        # !!! FIXME: fix arguments for TemporalLoss and fully test the functionality of the class - currently not working bc of a shape mismatch
        """ #TODO: address this error:
            vgrid = grid - flo
            RuntimeError: The size of tensor a (256) must match the size of tensor b (257) at non-singleton dimension 3
        """
        self.temporal_loss = TemporalLoss() if self.temporal_weight > 0 else None
        self.style_encoder = style_encoder if style_encoder is not None else VGG19(config["vgg_ckpt"])  # Store style encoder or use default VGG19
        #self.laplacian_loss_grad = laplacian_loss_grad if self.lap_weight > 0 else None
        self.laplacian_loss_module = MattingLaplacianLoss(win_rad=config.get("win_rad", 1), objective="mse") if self.lap_weight > 0 else None


    def compute_losses(self, content_img, style_img, stylized_img, mask = None): #, laplacian_list=None):
        """ Computes all losses, including Matting Laplacian loss if enabled, while ensuring differentiability for backpropagation. """
        content_loss, style_loss = self._compute_content_style_loss(content_img, style_img, stylized_img, self.config["content_weight"], self.config["style_weight"])
        #content_loss = self.config["content_weight"] * self._compute_content_loss(content_img, stylized_img)
        #style_loss = self.config["style_weight"] * self._compute_style_loss(style_img, stylized_img)
        reconstruction_loss = self._compute_reconstruction_loss(content_img, stylized_img, self.config["rec_weight"])
        laplacian_loss = torch.tensor(0.0, device=content_img.device, requires_grad=True)
        if self.lap_weight > 0:
            laplacian_loss = self._compute_laplacian_loss(content_img, stylized_img, mask, self.config["lap_weight"]) #, laplacian_list)
        temporal_loss = torch.tensor(0.0, device=content_img.device, requires_grad=True)
        if self.temporal_weight > 0:
            temporal_loss = self._compute_temporal_loss(content_img, stylized_img, self.config["temporal_weight"])
        # dynamically construct the losses dictionary from __local__ variables, removing `_loss` suffix from keys
        losses = {key.replace("_loss", ""): value for key, value in locals().items() if key.endswith("_loss")}
        # Total Loss - Ensure summation does not break autograd tracking
        losses["total"] = sum(value for value in losses.values())
        return losses

    def _compute_laplacian_loss(self, content_img, stylized_img, mask=None, weight=1.0): #, laplacian_list):
        """ Computes the Matting Laplacian loss using the precomputed sparse Laplacian matrices.
            laplacian_list [torch.Tensor]: (B, H, W) computed from the content images via our compute_laplacian function.
            NOTE: gradient differentiation is handled by pytorch autograd with MattingLaplacianLoss's superclass nn.Module
        """
        return weight * self.laplacian_loss_module(content_img, stylized_img)
        # # Use mean squared error between the stylized response and the precomputed response.
        # return F.mse_loss(lap_stylized, laplacian_list)

    def _compute_content_style_loss(self, content_img, style_img, stylized_img, cweight=1.0, sweight=1.0):
        """ Computes both content and style losses using the style encoder. """
        closs, sloss =  self.style_encoder(content_img, style_img, stylized_img, n_layer=4, content_weight=self.config["content_weight"])
        return cweight * closs, sweight * sloss

    def _compute_reconstruction_loss(self, content_img, stylized_img, weight=1.0):
        # Placeholder: L1 loss for reconstruction
        return weight * self.l1_loss(content_img, stylized_img)

    def _compute_temporal_loss(self, content_img, stylized_img, weight=1.0):
        # Placeholder: Temporal loss computation
        # TODO: double check that this is properly implemented from the old train.py script (around line 177)
        return weight * self.temporal_loss(content_img, stylized_img)