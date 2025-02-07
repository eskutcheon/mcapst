from typing import Dict, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
# local imports



#^ Temporal loss code formerly in mcapst/utils/TemporalLoss.py
class TemporalLoss(nn.Module):
    """ Regularization from paper: Consistent Video Style Transfer via Compound Regularization """
    def __init__(self, use_fake_flow=True, use_optical_flow=False, data_w=True, noise_level=1e-3,
                 motion_level=8, shift_level=10, flow_model=None):
        super(TemporalLoss, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.use_fake_flow = use_fake_flow
        self.use_optical_flow = use_optical_flow
        self.data_w = data_w
        self.noise_level = noise_level
        self.motion_level = motion_level
        self.shift_level = shift_level
        self.flow_model = flow_model
        if use_optical_flow and flow_model is None:
            from torchvision.models.optical_flow import raft_small
            self.flow_model = raft_small(pretrained=True).eval()
    """ Flow should have most values in the range of [-1, 1]. 
        For example, values x = -1, y = -1 is the left-top pixel of input, 
        and values  x = 1, y = 1 is the right-bottom pixel of input.
        Flow should be from pre_frame to cur_frame """

    @staticmethod
    def warp(x, flo, padding_mode='border'):
        """ Optical flow warping function """
        B, C, H, W = x.size()
        # Mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(device=x.device)
        print("grid shape: ", grid.shape)
        print("flo shape: ", flo.shape)
        vgrid = grid - flo
        # Scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        #** NOTE: previously used mode='nearest', but 'bilinear' is more common
        return F.grid_sample(x, vgrid, padding_mode=padding_mode, mode='bilinear')

    def GaussianNoise(self, ins, mean=0, stddev=0.001):
        stddev = stddev + torch.rand(1).item() * stddev
        noise = torch.normal(mean, stddev, size=ins.shape, device=ins.device)
        return ins + noise

    def GenerateFakeFlow(self, height, width):
        if self.motion_level > 0:
            flow = torch.normal(0, self.motion_level, size=(2, height, width), device="cpu")
            flow += torch.randint(-self.shift_level, self.shift_level + 1, size=(2, height, width), device="cpu")
            flow = F.avg_pool2d(flow.unsqueeze(0), kernel_size=100, stride=1, padding=50).squeeze(0)
        else:
            flow = torch.ones(2, height, width, device="cpu") * torch.randint(-self.shift_level, self.shift_level + 1, (1,), device="cpu")
        return flow

    #** NOTE: in the original implementation, this function had more conditionals based on self.data_w, where GenerateFakeFlow was called only when self.data_w was True
    def GenerateFakeData(self, first_frame):
        forward_flow = self.GenerateFakeFlow(first_frame.shape[2], first_frame.shape[3]).to(first_frame.device)
        forward_flow = forward_flow.unsqueeze(0).expand(first_frame.shape[0], -1, -1, -1)
        second_frame = self.warp(first_frame, forward_flow)
        if self.noise_level > 0:
            second_frame = self.GaussianNoise(second_frame, stddev=self.noise_level)
        return second_frame, forward_flow

    #** NEW: added since the original only implemented fake data generation
    @torch.no_grad()
    def compute_optical_flow(self, prev_frame, current_frame):
        """ Uses the pre-trained RAFT model to estimate optical flow between frames """
        prev_frame = prev_frame.unsqueeze(0) if prev_frame.dim() == 3 else prev_frame
        current_frame = current_frame.unsqueeze(0) if current_frame.dim() == 3 else current_frame
        flow = self.flow_model(prev_frame, current_frame)[-1]
        return flow

    def forward(self, first_frame, second_frame, prev_flow=None):
        #** NOTE: previously only called `warp` if self.data_w was True
        """ Compute temporal consistency loss. """
        B, C, H, W = first_frame.shape
        # Use optical flow if enabled
        if self.use_optical_flow:
            forward_flow = self.compute_optical_flow(first_frame, second_frame) if prev_flow is None else prev_flow
        elif self.use_fake_flow:
            forward_flow = self.GenerateFakeFlow(H, W).to(first_frame.device).unsqueeze(0).expand(B, -1, -1, -1)
        else:
            raise ValueError("At least one flow method must be enabled.")
        # Warp first frame using estimated flow
        warped_frame = self.warp(first_frame, forward_flow)
        # Compute temporal consistency loss
        temporal_loss = torch.mean(torch.abs(warped_frame - second_frame))
        return temporal_loss, forward_flow