from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
# helper class for mesh grid caching
from mcapst.core.utils.loss_utils import MeshGridCache



#^ Temporal loss code formerly in mcapst/utils/TemporalLoss.py
class TemporalLoss(nn.Module):
    """ Handles both the original "fake flow" approach and (optionally) real flow for future implementations
        By default, it replicates the old approach:
            - GenerateFakeData: produce second_frame + flow from first_frame
            - forward_temporal: warp the first frame with the flow, measure L1 diff
    """
    #Regularization from paper: Consistent Video Style Transfer via Compound Regularization: https://daooshee.github.io/CompoundVST/
    def __init__(self, use_fake_flow=True, warp_flag=True, noise_level=1e-3, motion_level=8, shift_level=10, flow_model=None):
        """
            Args:
            use_fake_flow:              whether to generate "fake" flow for training
            warp_flag:                  if True, warp the first frame before computing the difference
            noise_level:                how much random noise to apply to the second frame
            motion_level, shift_level:  parameters controlling the magnitude of fake flow
            flow_model:                 if providing a real optical-flow model (e.g. RAFT).
        """
        super(TemporalLoss, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.mesh_cache = MeshGridCache(max_size=8)
        self.use_fake_flow = use_fake_flow
        self.warp_flag = warp_flag
        self.noise_level = noise_level
        self.motion_level = motion_level
        self.shift_level = shift_level
        if not self.use_fake_flow:
            self._initialize_real_flow_model(flow_model)  # Initialize the real optical flow model if provided
    """ Flow should have most values in the range of [-1, 1].
        For example, values x = -1, y = -1 is the left-top pixel of input,
        and values  x = 1, y = 1 is the right-bottom pixel of input.
        Flow should be from pre_frame to cur_frame
    """

    def _initialize_real_flow_model(self, flow_model):
        """ Initialize the real optical flow model if provided """
        if not self.use_fake_flow and flow_model is None:
            from torchvision.models.optical_flow import raft_small
            self.flow_model = raft_small(pretrained=True).eval()
        elif flow_model is not None:
            self.flow_model = flow_model
        else:
            raise ValueError("No optical flow model provided.")

    def warp(self, x: torch.Tensor, flo: torch.Tensor, padding_mode: str = 'border') -> torch.Tensor:
        """ warp tensor x according to flow, which is shape (B,2,H,W) in [-1,1] """
        B, C, H, W = x.size()
        # get base grid from mesh cache
        grid2d = self.mesh_cache.get(H, W, x.device)
        grid = grid2d.unsqueeze(0).expand(B, -1, -1, -1)
        # flow is from first_frame -> second_frame, so we compute "vgrid = base - flow" to align "first_frame" with "second_frame".
        vgrid = grid - flo
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        #** NOTE: previously used mode='nearest', but 'bilinear' is more common
        return F.grid_sample(x, vgrid, padding_mode=padding_mode, mode='bilinear', align_corners=False)

    #@torch.no_grad()
    def compute_optical_flow(self, frameA: torch.Tensor, frameB: torch.Tensor) -> torch.Tensor:
        """ If we're using an optical-flow model (e.g. RAFT), compute real flow from A->B.
            Args:
                frameA, frameB: tensors of shape [B,C,H,W].
        """
        # RAFT typically expects frames in shape [N,C,H,W]. If B>1, RAFT should support batched input
        if frameA.shape[0] != 1 or frameB.shape[0] != 1:
            raise NotImplementedError("Batch optical flow not currently implemented.")
        flow = self.flow_model(frameA, frameB)[-1]  # RAFT returns a list of flows
        return flow

    #** NOTE: original implementation had more conditionals based on self.warp_flag, where fake flow was created if self.warp_flag was True
    def generate_fake_data(self, first_frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Creates a 'fake' second frame + flow from first_frame """
        B, C, H, W = first_frame.shape
        # generate random flow
        flow_2d = self._generate_fake_flow(H, W).to(first_frame.device) # shape (2,H,W)
        # repeat for the batch dimension
        flow_4d = flow_2d.unsqueeze(0).expand(B, -1, -1, -1)  # shape (B,2,H,W)
        # warp first_frame => second_frame
        second_frame = self.warp(first_frame, flow_4d)
        # optionally add some random noise
        if self.noise_level > 0:
            second_frame = self._add_gaussian_noise(second_frame, stddev=self.noise_level)
        return second_frame, flow_4d

    def _generate_fake_flow(self, height: int, width: int) -> torch.Tensor:
        """ the original implementation's logic: random normal, random shift, large blur => "fake" motion """
        if self.motion_level > 0:
            flow = torch.normal(0, self.motion_level, size=(2, height, width)) # shape (2,H,W)
            flow += torch.randint(-self.shift_level, self.shift_level + 1, size=(2, height, width))
            # approximate smoothing with very large kernel
            #flow = F.avg_pool2d(flow.unsqueeze(0), kernel_size=100, stride=1, padding=50, ceil_mode=True).squeeze(0)
            #! REVISIT LATER - may not work as intended since a large kernel may be needed to avoid artifacts
            flow = F.adaptive_avg_pool2d(flow.unsqueeze(0), (height, width)).squeeze(0) # shape (2,H,W)
        else:
            # fallback using constant flow
            flow = torch.ones(2, height, width) * torch.randint(-self.shift_level, self.shift_level + 1, (1,))
            # print("generating constant flow with shape: ", flow.shape)
        return flow # shape (2,H,W)

    def _add_gaussian_noise(self, tensor: torch.Tensor, mean: float = 0, stddev: float = 1e-3) -> torch.Tensor:
        stddev = stddev + torch.rand(1).item() * stddev
        noise = torch.normal(mean, stddev, size=tensor.shape, device=tensor.device)
        return tensor + noise

    def compute_temporal_loss(self, first_frame: torch.Tensor, second_frame: torch.Tensor, flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ (almost) equivalent to the original authors' "forward" method """
        # optionally warp the first frame with the given flow
        if self.warp_flag:
            warped = self.warp(first_frame, flow)
        else:
            warped = first_frame
        # L1 difference between warped first_frame and second_frame
        temporalloss = torch.mean(torch.abs(warped - second_frame))
        return temporalloss, warped


    def forward(self, seqA: torch.Tensor, seqB: torch.Tensor, flow: torch.Tensor = None) -> torch.Tensor:
        """ optional single `forward` method conforming to nn.Module interface:
            - If flow is None and use_optical_flow=True, compute real flow between A->B
            - If flow is None and use_fake_flow=True, generate random flow. (Though typically you'd call generate_fake_data upstream)
            - Then warp A, compute L1 difference from B, return scalar loss
        """
        B, C, H, W = seqA.shape
        if flow is None:
            if self.use_fake_flow:
                # generate random flow of shape (B,2,H,W)
                flow_2d = self._generate_fake_flow(H, W).to(seqA.device)
                flow = flow_2d.unsqueeze(0).expand(B, -1, -1, -1)
            else:
                flow = self.compute_optical_flow(seqA, seqB)
        # compute L1 difference
        temporal_loss, _ = self.compute_temporal_loss(seqA, seqB, flow)
        return temporal_loss
