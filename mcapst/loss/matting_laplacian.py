
from typing import List, Optional, Sequence # Dict, Union, Literal,
import torch
import torch.nn.functional as F
from mcapst.utils.loss_utils import IndexCache


def _construct_final_L(indices_b: torch.Tensor, vals_b: torch.Tensor, N: int, win_units: int, mask: torch.Tensor=None):
    # indices_b: (2, nnz), vals_b: (nnz,)
    # optionally drop all windows that lie entirely outside the mask
    if mask is not None:
        # window_mask[b] is shape (N,), so repeat each element win_size**2 times
        keep = mask.repeat_interleave(win_units)
        vals_b = vals_b[keep]
        indices_b = indices_b[:, keep]
    # compute sparse matrix L for the current batch element
    L = torch.sparse_coo_tensor(indices_b, vals_b, size=(N, N), device=vals_b.device)
    # compute the Kronecker delta correction D = Diag(sum_L)
    sum_L = torch.sparse.sum(L, dim=1).to_dense() # shape (N,)
    D = torch.sparse_coo_tensor(
        torch.arange(N, device=L.device).unsqueeze(0).repeat(2, 1),
        sum_L, size=(N, N), device=vals_b.device)
    # return final Laplacian from (Eq. 12)
    return D - L


def _compute_window_mask(mask: torch.Tensor, win_radius: int = 1) -> torch.BoolTensor:
    """ compute a mask for which patches overlap the input mask (after a dilation)
        Args:
            mask: (B, 1, H, W) or (B, H, W) boolean mask
        Returns
            (B, N) boolean tensor
    """
    win_diam = win_radius * 2 + 1
    win_size = win_diam ** 2
    # ensure mask has shape (B, 1, H, W)
    if mask.dim()==3:
        mask = mask.unsqueeze(1)
    # binary dilation over each window so that any patch touching a True gets marked
    dilated = F.max_pool2d(mask.float(), kernel_size=win_diam, stride=1, padding=win_radius) # (B,1,H,W)
    dilated = dilated > 0.5
    # extract patch‐blocks of the dilated mask
    m_patches = dilated.unfold(2, win_diam, 1).unfold(3, win_diam, 1) # shape (B, 1, win_d, win_d, H', W')
    # flatten each (win_d, win_d) patch and ask if any True
    m_patches = m_patches.reshape(mask.shape[0], 1, win_size, -1) # shape (B, 1, win_size, N)
    return m_patches.any(dim=2).squeeze(1)  # shape (B, N)



#^ MattingLaplacian code formerly in mcapst/utils/MattingLaplacian.py
class MattingLaplacianLoss(torch.nn.Module):
    def __init__(self, eps=1e-7, win_rad=1):
        super(MattingLaplacianLoss, self).__init__()
        self.eps = eps
        if win_rad != 1:
            raise ValueError(f"Only win_rad=1 is supported (from the original authors), but got {win_rad}.")
        self.win_radius = win_rad
        #! original implementation requiresd that window diameter == C, possibly unintentionally - thus we assume eye(C) == eye(win_diam)
        self.win_diam = win_rad * 2 + 1
        self.win_size = self.win_diam ** 2
        #self.ident = torch.eye(self.win_diam) #.view(1, 1, self.win_diam, self.win_diam)
        # identity matrix saved to a module buffer for normalizing the covariance matrix and solving for its inverse
        self.register_buffer("ident", torch.eye(self.win_diam), persistent=False)
        # add index cache for the COO indices of the sparse matrix into the IndexCache submodule (MLL is still stateless except for this)
        self._cache = IndexCache(self.win_diam) #? NOTE: submodule should be automatically registered in the ModuleDict of the parent class


    def _extract_patches(self, img: torch.Tensor) -> torch.Tensor:
        """ Extracts local patches using explicit indexing instead of F.unfold with padding.
            Args:
                img (Tensor): Input image tensor of shape (B, C, H, W).
            Returns:
                patches (Tensor): Extracted patches of shape (B, C, win_size, H' = H-2*win_rad, W' = W-2*win_rad).
        """
        # Apply unfold on height and width separately (removing extra padding)
        patches = img.unfold(2, self.win_diam, 1).unfold(3, self.win_diam, 1)
        # Reshape correctly to match NumPy's `_rolling_block()` output
        patches = patches.contiguous().permute(0, 1, 4, 5, 2, 3)  # (B, C, win_diam, win_diam, H', W')
        #patches = patches.reshape(B, C, H'*W', self.win_size).permute(0, 1, 3, 2)
        return patches  # Shape: (B, C, win_diam, H', W')


    def compute_local_statistics(self, patches: torch.Tensor):
        """ Computes local mean and covariance for each spatial location.
            Args:
                img (Tensor): Input image tensor of shape (B, C, H, W).
            # NOTE: using the notation H' and W' in subsequent comments, meaning H' = H - 2*win_rad, W' = W - 2*win_rad
            Returns:
                local_mean (Tensor): Local mean tensor of shape (B, C, 1, H', W').
                cov (Tensor): Local covariance tensor of shape (B, C, C, H', W').
        """
        # compute local mean per channel
        local_mean = patches.mean(dim=(2,3), keepdim=True).squeeze(2)  # (B, C, 1, H', W')
        # TODO: rewrite Einstein summation to go ahead and reshape patches earlier
        # compute per-pixel E[X X^T]
        patch_sq_sum = torch.einsum('... i m n h w, ... j m n h w -> ... i j h w', patches, patches) / self.win_size # shape: (B, C, win_diam, H' W')
        # compute outer product of local mean: E[X]E[X]^T
        mean_sq = torch.einsum('... i k h w, ... j k h w -> ... i j h w', local_mean, local_mean) # shape: (B, C, win_diam, H', W')
        # compute covariance: E[X X^T] - E[X]E[X]^T
        cov = patch_sq_sum - mean_sq  # shape: (B, C, win_diam, H', W')
        del patch_sq_sum, mean_sq  # free up memory
        return local_mean, cov


    def compute_quadratic_term(self, patches: torch.Tensor, local_mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """ Computes the quadratic term for each pixel based on local statistics within each window in patches
            Args:
                patches:    Tensor (B, C, win_size, N)
                local_mean: Tensor (B, C, 1, N)
                cov:        Tensor (B, C, C, N)
            Returns:
                Tensor (B, N, win_size, win_size)
        """
        # # first, ensure identity matrix is on the correct device
        # if self.ident.device != patches.device: # might add this to the preprocessor function later (or insist the whole module be on the same device)
        #     self.ident = self.ident.to(patches.device)
        B, C = patches.shape[:2]
        patches = patches.reshape(B, C, self.win_size, -1)  # Shape: (B, C, win_size, H' * W')
        # regularize the covariance matrix and invert it
        # !!! requires that win_diam == C but there's no way around it without compromising the whole algorithm
        cov = cov.flatten(start_dim=-2).permute(0, 3, 2, 1)  # Shape: (B, C, win_diam, H' * W') -> (B, H' * W', win_diam, C)
        cov += (self.eps / self.win_size) * self.ident  # Shape: # (B, H' * W', win_diam, C)
        # more numerically stable matrix inversion:
        inv_cov = torch.linalg.solve(cov, self.ident).float() # shape (B, H' * W', C (or win_diam), C)
        # compute the difference between image pixels and the local mean.
        local_mean = local_mean.flatten(start_dim=-2)  # Shape: (B, C, 1, H' * W')
        diff = patches - local_mean # shape: (B, C, win_size, H' * W')
        # compute the quadratic form for each pixel: diff.T * inv_cov * diff to yield a scalar per spatial location.
        diff = diff.permute(0, 3, 2, 1) # shape: (B, H' * W', win_size, C)
        # quadratic form from einsum should also be equivalent to (I - mu).T @ inv_cov @ (I - mu) from the Kaiming He Paper
        #quadratic = diff @ inv_cov @ diff.transpose(2,3) # shape (B, N = H' * W', win_size, win_size)
        # compute per‐patch energies in one shot (should avoid intermediate large N^2 tensors)
        quadratic = torch.einsum('... s c, ... c d, ... t d -> ... s t', diff, inv_cov, diff) # shape (B, N = H' * W', win_size, win_size)
        del diff, inv_cov  # free up memory
        quadratic += 1.0
        # division by window
        return quadratic.div(self.win_size) # shape (B, H' * W', win_size, win_size)


    def construct_laplacian_parallel(self, laplacian: torch.Tensor, indices: torch.Tensor,
                                     img_shape: Sequence[int], mask: Optional[torch.Tensor] = None):
        """ Constructs the Laplacian matrix asynchronously for each batch index using torchscript """
        B, C, H, W = img_shape
        lap_vals = laplacian.flatten(start_dim=1)  # (B, nnz = H' * W' * win_size**2)
        N = H * W
        del laplacian  # free memory since lap_vals has the needed data now
        # use `torch.jit.fork` for asynchronous Laplacian construction
        futures = [
            torch.jit.fork(
                _construct_final_L, indices[:, b, :], lap_vals[b], N, self.win_size**2, mask[b] if mask is not None else None)
            for b in range(B)
        ]
        # torch.cuda.synchronize()  # ensure all GPU operations are complete before proceeding
        # wait and return fully computed Laplacians
        return [torch.jit.wait(f) for f in futures]


    def construct_laplacian_sequential(
        self,
        laplacian: torch.Tensor,
        indices: torch.Tensor,
        img_shape,
        mask: torch.Tensor = None
    ) -> List[torch.sparse.Tensor]:
        """ Constructs the sparse Laplacian matrix sequentially for each batch index
            Args:
                laplacian (Tensor): Laplacian response of shape (B, H' * W', win_size, win_size)
                indices (Tensor): Row and column indices for the sparse matrix of shape (2, B, H' * W' * win_size**2)
                img_shape (tuple): Shape of the input image (B, C, H, W)
        """
        B, C, H, W = img_shape
        # Construct sparse matrices batch-wise
        # _construct_final_L, indices[:, b, :], lap_vals[b], N, self.win_size**2, mask[b] if mask is not None else None)
        sparse_laplacians = []
        for b in range(B):
            lap_vals = laplacian[b].flatten()  # Shape: (H' * W' * win_size**2)
            indices_b = indices[:, b, :]  # Shape: (2, H' * W' * win_size**2)
            L_b = _construct_final_L(indices_b, lap_vals, H*W, self.win_size**2, mask[b] if mask is not None else None)
            # #? NOTE: I tried multiplying the Laplacian response by -1 first then do in-place addition here, but regression tests failed hard
            # L_b = diag_L_b - L_b # shape: (HW, HW)
            sparse_laplacians.append(L_b)
        return sparse_laplacians  # List of (H*W, H*W) sparse tensors



    def compute_laplacian_response(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """ Computes the Matting Laplacian response based on local covariance statistics.
            Args:
                img (Tensor): Input image tensor of shape (B, C, H, W).
                mask (Tensor, optional): Optional mask tensor to weight the local statistics.
            Returns:
                laplacian (Tensor): Laplacian response of shape (B, H, W).
        """
        # NOTE: using the notation H' and W' in subsequent comments, meaning H' = H - 2*win_rad, W' = W - 2*win_rad
        B, _, H, W = img.shape  # B: batch size, C: channels, H: height, W: width
        # TODO: there must be a more memory efficient way to do this (with fewer extra dimensions)
        patches = self._extract_patches(img)  # (B, C, win_diam, win_diam, H', W')
        # patches = F.unfold(img, kernel_size=self.win_diam, padding=self.win_radius) # shape: (B, C * win_size, H' * W')
        local_mean, cov = self.compute_local_statistics(patches)
        # Regularize the covariance matrix and invert it.
        laplacian = self.compute_quadratic_term(patches, local_mean, cov) # shape: (B, H' * W', win_size, win_size)
        # TODO: (maybe) add some thresholding to the the Laplacian to enforce meaningful sparsity - maybe anything below 1e-8 to zero?
        del patches, local_mean, cov  # free up memory
        #indices = self.get_coo_indices(img.shape, img.device) # shape: (2, B, H' * W' * win_size**2)
        indices = self._cache.get(H, W, img.device).unsqueeze(1).expand(2, B, -1)  # shape: (2, B, H' * W' * win_size**2)
        #return laplacian, indices # REMOVE: using for debugging in comparing device speedup (accumulating error meant I had to use the same `laplacian`)
        #? NOTE: to support multiclass masks, we'll need to compute window_mask over each channel and construct the final Laplacian iteratively
        window_mask = _compute_window_mask(mask, self.win_radius) if mask is not None else None # shape: (B, N)
        # TODO: may decouple settings for device and parallelism later - main problem is that CPU multithreading could interfere with DataLoader workers
        # construct sparse matrices batch-wise
        if img.is_cuda:
            return self.construct_laplacian_parallel(laplacian, indices, img.shape, window_mask)
        return self.construct_laplacian_sequential(laplacian, indices, img.shape, window_mask)


    def _preprocess(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # leaving this here in case I want to add additional preprocessing steps to `forward()`
        if img.ndim not in [3,4]:
            raise ValueError(f"Expected 3D or 4D tensor, but got {img.dim()}D tensor of shape {tuple(img.shape)}")
        if img.ndim == 3:  # if input is (C, H, W), add batch dimension
            img = img.unsqueeze(0)
        if img.max() > 1.0: # TODO: replace with more robust normalization functions from utils of other projects later
            img = img.float() / 255.0
        if mask is not None:
            if (mask.shape[-2:] != img.shape[-2:]) and (mask.shape[0] != img.shape[0]):
                raise ValueError(f"Mask shape {mask.shape} does not match image shape {tuple(img.shape)}")
            # TODO: may add support for multiclass masks later, but it requires refactoring some channel-wise ops in `compute_laplacian_response`
                # might have to do this in `construct_laplacian_sequential` and `construct_laplacian_parallel`
                # honestly, mask support is unnecessary for the current use case since masks aren't used in training, but the original CAP-VSTNet repo supported it
            #! change the following after implementing multiclass mask support
            assert mask.dtype == torch.bool, f"Only boolean masks are supported for now; got {mask.dtype}"
        # this is taken care of implicitly if we move the whole module to the same device first, but this is a simple fallback
        if self.ident.device != img.device:
            self.ident = self.ident.to(img.device)
        return img


    def _postprocess(self, laplacian: torch.Tensor):
        if isinstance(laplacian, list):
            laplacian = torch.stack(laplacian)
        if isinstance(laplacian, torch.sparse.Tensor) and not laplacian.is_coalesced():
            # Convert to COO format for safe element-wise operations
            laplacian = laplacian.coalesce()
        return laplacian


    def forward(self, content_img, stylized_img, mask=None):
        """ Computes the Matting Laplacian loss between stylized and content images, optionally considering only masked regions
            Args:
                content_img (Tensor): The content image tensor of shape (B, C, H, W).
                stylized_img (Tensor): The stylized image tensor of shape (B, C, H, W).
                mask (Tensor, optional): Optional mask tensor of shape (B, 1, H, W) or (B, H, W).
            Returns:
                loss (Tensor): A scalar tensor representing the mean squared error between the laplacian responses.
        """
        content_img = self._preprocess(content_img, mask=mask)
        stylized_img = self._preprocess(stylized_img)
        lap_content = self.compute_laplacian_response(content_img, mask=mask)
        lap_content = self._postprocess(lap_content)  # enable gradients for content Laplacian
        # dispatch to custom Function
        return MLLossFn.apply(lap_content, stylized_img)



########################### Helper class for custom autograd function ###########################

class MLLossFn(torch.autograd.Function):
    """ Custom autograd function for computing the Matting Laplacian loss and its gradient
        - follows the same sparse quadratic form as the original project, but relies on PyTorch's autograd for differentiation
        - uses a custom forward and backward pass to compute the loss and gradient efficiently (while scaling and clipping the gradient)
    """
    @staticmethod
    def forward(ctx, laplacian, stylized_img):
        """ Computes both scalar loss and raw gradient w.r.t. stylized_img in one pass """
        B, C, H, W = stylized_img.shape
        loss_accum = 0.0
        grad = torch.zeros_like(stylized_img)
        for b in range(B):
            x = stylized_img[b].reshape(C, -1)
            # raw gradient: Lx / (H*W)
            grad_b = torch.sparse.mm(laplacian[b], x.T).T / (H * W)
            # scalar loss: x^T (L x) / (H*W)
            # essentially a single dot product per channel with summation over channels
            loss_b = (x * grad_b).sum()
            loss_accum = loss_accum + loss_b
            grad[b] = grad_b.view_as(stylized_img[b])
        loss = loss_accum / B # mean over already-summed batch
        # save raw gradient for backward pass
        ctx.save_for_backward(2.0 * grad)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is scalar (dTotal/dLoss)
        (raw_grad,) = ctx.saved_tensors
        # combine and clamp
        g = raw_grad * grad_output
        g = g.clamp(-0.05, 0.05)
        # propagate only into stylized_img; other inputs get None
        return None, g #, None, None
