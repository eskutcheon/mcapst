
""" will be moving the temporal loss and Matting Laplacian loss to this file while refactoring to use pure pytorch """

from typing import Dict, Union, Literal, Callable
import torch
import torch.nn as nn
# local imports
from ..models.VGG import VGG19
from .matting_laplacian import MattingLaplacianLoss
from .temporal_loss import TemporalLoss




class LossManager:
    # TODO: replace with custom type hinting later
    def __init__(self, config: Dict[str, Union[str, int, float, bool]], style_encoder=None):
        """ Loss manager for computing various losses during training.
            Args:
                config (dict): Configuration dictionary.
                style_encoder (BaseStyleEncoder, optional): A style encoding model for computing content/style loss.
        """
        self.l1_loss = nn.L1Loss()
        self.content_weight = config.content_weight
        self.style_weight = config.style_weight
        self.rec_weight = config.rec_weight
        self.lap_weight = config.lap_weight
        self.temporal_weight = config.temporal_weight
        self.temporal_loss = TemporalLoss() if self.temporal_weight > 0 else None
        self.style_encoder = style_encoder if style_encoder is not None else VGG19(config.vgg_ckpt)  # Store style encoder or use default VGG19
        # NOTE: using win_rad = 1 because only 3x3 kernels are supported for now - the original never supported larger kernels either
        self.laplacian_loss_module = MattingLaplacianLoss(win_rad=1) if self.lap_weight > 0 else None

    @staticmethod
    def _toggle_grad(*args):
        """ Helper function to toggle gradient computation for a list of tensors. """
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg.requires_grad = not arg.requires_grad

    def compute_losses(self, content_img, style_img, stylized_img, mask = None, stylizer_callback: Callable = None): #, laplacian_list=None):
        """ Computes all losses, including Matting Laplacian loss if enabled, while ensuring differentiability for backpropagation. """
        content_loss, style_loss = self._compute_content_style_loss(content_img, style_img, stylized_img, self.content_weight, self.style_weight)
        # Toggle gradient computation for content and stylized images and mask if provided (to avoid tracking content-style loss above)
        self._toggle_grad(content_img, stylized_img, mask)
        reconstruction_loss = self._compute_reconstruction_loss(content_img, stylized_img, self.rec_weight)
        temporal_loss = self._compute_temporal_loss(content_img, stylized_img, self.temporal_weight, stylizer_callback)
        del stylizer_callback, style_img  # free up memory
        laplacian_loss = self._compute_laplacian_loss(content_img, stylized_img, mask, self.lap_weight)
        # dynamically construct the losses dictionary from __local__ variables, removing `_loss` suffix from keys
        losses = {key.replace("_loss", ""): value for key, value in locals().items() if key.endswith("_loss")}
        # Total Loss - Ensure summation does not break autograd tracking
        losses["total"] = sum(losses.values())
        return losses

    def _compute_laplacian_loss(self, content_img, stylized_img, mask=None, weight=1.0): #, laplacian_list):
        """ Computes the Matting Laplacian loss using the precomputed sparse Laplacian matrices.
            laplacian_list [torch.Tensor]: (B, H, W) computed from the content images via our compute_laplacian function.
            NOTE: gradient differentiation is handled by pytorch autograd with MattingLaplacianLoss's superclass nn.Module
        """
        if mask is not None:
            raise NotImplementedError("Masking is not implemented for Laplacian loss yet.")
        if weight == 0:
            return torch.tensor(0.0, device=content_img.device, requires_grad=True)
        return weight * self.laplacian_loss_module(content_img, stylized_img, mask)

    def _compute_content_style_loss(self, content_img, style_img, stylized_img, cweight=0.0, sweight=1.0):
        """ Computes both content and style losses using the style encoder. """
        # NOTE: content_weight is only passed here so that the encoder avoids computing the content loss if content_weight == 0 (from original authors)
        # TODO: inputs might absolutely have to have shape [256,256] since the VGG19 model is trained on 256x256 images
        closs, sloss =  self.style_encoder(content_img, style_img, stylized_img, n_layer=4, content_weight=self.content_weight)
        return cweight * closs, sweight * sloss

    def _compute_reconstruction_loss(self, content_img, stylized_img, weight=1.0):
        """ Computes the reconstruction loss (L1 loss) between content and stylized images - referred to as the cycle consistency loss in the original paper """
        return weight * self.l1_loss(content_img, stylized_img)

    def _compute_temporal_loss(self, content_img: torch.Tensor, stylized_img: torch.Tensor, weight=1.0, callback: Callable = None):
        """ computes the temporal loss between (spoofed) sequential stylized images to train video stylization models """
        if weight == 0:
            return torch.tensor(0.0, device=content_img.device, requires_grad=True)
        # NOTE: the original authors used the first frame of the content image as the previous frame for computing optical flow
        #return weight * self.temporal_loss(stylized_img[:-1], stylized_img[1:], prev_flow=None, use_fake_flow=True)
        second_frame, flow = self.temporal_loss.generate_fake_data(content_img)
        # stylize second frames with the callback (might want a more elegant way to do this in the future)
        stylized_second = callback(second_frame)
        # compute temporal loss between stylized frames
        loss_tmp, _ = self.temporal_loss.compute_temporal_loss(stylized_img, stylized_second, flow)
        # optional “ground truth” checks
        #loss_tmp_GT, _ = self.temporal_loss.compute_temporal_loss(content_img, second_frame, flow)
        return weight * loss_tmp