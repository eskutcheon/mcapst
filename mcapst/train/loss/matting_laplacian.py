from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
from mcapst.train.loss.loss_utils import IndexCache
import concurrent.futures


def _construct_final_L(indices_b: torch.Tensor, vals_b: torch.Tensor, N: int, win_units: int, mask: torch.Tensor=None):
    # indices_b: (2, nnz), vals_b: (nnz,)
    L = None
    if mask is not None:
        # window_mask[b] is shape (N,), so repeat each element win_size**2 times
        keep = mask.repeat_interleave(win_units)
        # compute sparse matrix L with masked indices and values
        L = torch.sparse_coo_tensor(indices_b[:, keep], vals_b[keep], size=(N, N), device=vals_b.device)
    else:
        # compute sparse matrix L for the current batch element without masking
        L = torch.sparse_coo_tensor(indices_b, vals_b, size=(N, N), device=vals_b.device)
    # compute the Kronecker delta correction D = Diag(sum_L)
    sum_L = torch.sparse.sum(L, dim=1).to_dense() # shape (N,)
    #? NOTE: this is really only necessary because operating on sparse matrices with slicing and diagonal methods isn't supported
    D = torch.sparse_coo_tensor(
        torch.arange(N, device=L.device).unsqueeze(0).repeat(2, 1), # diagonal matrix indices (i,i) for i in [0, N-1]
        sum_L, size=(N, N), device=vals_b.device) # sum_L values fill diagonal in new sparse diagonal matrix D
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


def _extract_patches(img: torch.Tensor, win_diam: int) -> torch.Tensor:
    """ Extracts local patches using explicit indexing instead of F.unfold with padding.
        Args:
            img (Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            patches (Tensor): Extracted patches of shape (B, C, win_size, H' = H-2*win_rad, W' = W-2*win_rad).
    """
    # Apply unfold on height and width separately (removing extra padding)
    patches = img.unfold(2, win_diam, 1).unfold(3, win_diam, 1).contiguous()
    B, C, H, W = patches.shape[:4]
    # Reshape correctly to match NumPy's `_rolling_block()` output
    patches = patches.reshape(B, C, H*W, win_diam**2).transpose(2, 3)  # (B, C, win_diam, win_diam, H', W')
    return patches  # Shape: (B, C, win_diam, H', W')


#^ MattingLaplacian code formerly in mcapst/utils/MattingLaplacian.py
class MattingLaplacianLoss(torch.nn.Module):
    def __init__(self, eps=1e-7, win_rad=1):
        super(MattingLaplacianLoss, self).__init__()
        self.eps = eps
        if win_rad != 1:
            raise ValueError(f"Only win_rad=1 is supported (from the original authors), but got {win_rad}.")
        self.win_radius = win_rad
        #! original implementation required that window diameter == C, possibly unintentionally - thus we assume eye(C) == eye(win_diam)
        self.win_diam = win_rad * 2 + 1
        self.win_size = self.win_diam ** 2
        # identity matrix saved to a module buffer for regularizing the covariance matrix and solving for its inverse
        self.register_buffer("ident", torch.eye(self.win_diam, dtype=torch.float64), persistent=False)
        # add index cache for the COO indices of the sparse matrix into the IndexCache submodule (MLL is still stateless except for this)
        self._cache = IndexCache(self.win_diam) #? NOTE: submodule should be automatically registered in the ModuleDict of the parent class


    def compute_local_statistics(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes per-window mean and covariance from content patches.
            Args:
                patches (Tensor): windowed input tensor of shape (B, C, win_size, N)
            # NOTE: using the notation H' and W' in subsequent comments, meaning H' = H - 2*win_rad, W' = W - 2*win_rad
            Returns:
                local_mean (Tensor): Local mean tensor of shape (B, C, 1, N).
                cov (Tensor): Local covariance tensor of shape (B, C, C, N).
        """
        # compute local mean per channel
        local_mean = patches.mean(dim=2, keepdim=True) # shape (B, C, 1, N)
        # compute per-pixel E[X X^T]
        # element-wise mult with mean over flat window dims: (B, 1, C, win_size, N) * (B, C, 1, win_size, N) = (B, C, C, win_size, N)
        patch_sq_sum = (patches.unsqueeze(1) * patches.unsqueeze(2)).mean(dim=3)  # (B, C, C, N)
        # compute outer product of local mean: E[X]E[X]^T
        mean_sq = local_mean * local_mean.transpose(1, 2) # shape: (B, C, C, N)
        # compute covariance: E[X X^T] - E[X]E[X]^T
        cov = patch_sq_sum - mean_sq  # shape: (B, C, C, H', W')
        del patch_sq_sum, mean_sq  # free up memory
        return local_mean, cov


    def compute_quadratic_term(self, patches: torch.Tensor, local_mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """ Computes the quadratic term for each pixel based on local statistics within each window in patches
            For notation's sake, let N = H' * W' be the number of spatial locations in the input image
            Args:
                patches:    Tensor (B, C, win_size, N)
                local_mean: Tensor (B, C, 1, N)
                cov:        Tensor (B, C, C, N)
            Returns:
                Tensor (B, N, win_size, win_size)
        """
        # regularize the covariance matrix and invert it
        # !!! requires that win_diam == C but there's no way around it without compromising the whole algorithm
        cov = cov.permute(0, 3, 2, 1).double()  # Shape: (B, C, win_diam, N) -> (B, N, win_diam, C)
        cov += (self.eps / self.win_size) * self.ident  # Shape: # (B, N, win_diam, C)
        # more numerically stable matrix inversion:
        inv_cov = torch.linalg.solve(cov, self.ident).float() # shape (B, N, win_diam, C)
        # compute the difference between image pixels and the local mean.
        diff = (patches - local_mean).permute(0, 3, 2, 1) # shape: (B, C, win_size, N) -> (B, N, win_size, C)
        # compute the quadratic form for each pixel: diff.T * inv_cov * diff to yield a scalar per spatial location.
        # quadratic form from einsum should also be equivalent to (I - mu).T @ inv_cov @ (I - mu) from the Kaiming He Paper
        # TODO: look into how opt_einsum might be able to replace this with compiled and reusable expressions (with better intermediate memory usage)
        quadratic = torch.einsum('... s c, ... c d, ... t d -> ... s t', diff, inv_cov, diff) # shape (B, N = H' * W', win_size, win_size)
        #quadratic = diff @ inv_cov @ diff.transpose(2,3) # shape (B, N = H' * W', win_size, win_size)
        del diff, inv_cov  # free up memory
        quadratic += 1.0
        # division by window size to normalize the quadratic term
        return quadratic.div(self.win_size) # shape (B, H' * W', win_size, win_size)


    def construct_laplacian_parallel(
        self,
        laplacian: torch.Tensor,
        indices: torch.Tensor,
        img_shape: Sequence[int],
        mask: Optional[torch.Tensor] = None
    ) -> List[torch.sparse.Tensor]:
        """ Constructs the Laplacian matrix in parallel like the sequential version, but over batch using ThreadPoolExecutor """
        B, _, H, W = img_shape
        N = H * W
        # parallel construction using threads (supports CPU and CUDA)
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(
                    _construct_final_L, indices, laplacian[b].flatten(), N, self.win_size**2,
                    mask[b] if mask is not None else None)
                    for b in range(B)
                ]
            # wait for all futures to complete and return a list of fully computed Laplacians
            return [f.result() for f in futures]
        except Exception as e:
            raise RuntimeError(f"Error constructing Laplacian matrices in parallel: {e}")

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
                indices (Tensor): Row and column indices for the sparse matrix of shape (2, H' * W' * win_size**2)
                img_shape (tuple): Shape of the input image (B, C, H, W)
        """
        B, _, H, W = img_shape
        sparse_laplacians = []
        for b in range(B):
            mask_b = mask[b] if mask is not None else None
            lap_vals = laplacian[b].flatten()  # Shape: (H' * W' * win_size**2)
            L_b = _construct_final_L(indices, lap_vals, H*W, self.win_size**2, mask_b)
            sparse_laplacians.append(L_b)
        return sparse_laplacians  # List of (H*W, H*W) sparse tensors



    def compute_laplacian_response(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """ Computes the Matting Laplacian response based on local covariance statistics
            Args:
                img (Tensor): Input image tensor of shape (B, C, H, W)
                mask (Tensor, optional): Optional mask tensor to weight the local statistics
            Returns:
                response (Tensor): Laplacian response of shape (B, H, W)
        """
        # NOTE: using the notation H' and W' in subsequent comments, meaning H' = H - 2*win_rad, W' = W - 2*win_rad
        H, W = img.shape[-2:]  # B: batch size, C: channels, H: height, W: width
        patches = _extract_patches(img, self.win_diam)  # (B, C, win_diam, win_diam, H', W')
        local_mean, cov = self.compute_local_statistics(patches)
        # regularize the covariance matrix and invert it
        response = self.compute_quadratic_term(patches, local_mean, cov) # shape: (B, H' * W', win_size, win_size)
        # TODO: (maybe) add some thresholding to the the Laplacian to enforce meaningful sparsity - maybe anything below 1e-8 to zero?
        del patches, local_mean, cov  # free up memory
        # retrieve or build indices from the cache (now saves the indices after calling `indices.expand(2, -1)`)
        indices = self._cache.get(H, W, img.device) # shape: (2, B, H' * W' * win_size**2)
        #? NOTE: to support multiclass masks, we'll need to compute window_mask over each channel and construct the final Laplacian iteratively
        window_mask = _compute_window_mask(mask, self.win_radius) if mask is not None else None # shape: (B, N)
        # construct sparse matrices batch-wise
        if torch.cpu.device_count() > 1: # if enough CPU cores are available, use parallel construction
            return self.construct_laplacian_parallel(response, indices, img.shape, window_mask)
        return self.construct_laplacian_sequential(response, indices, img.shape, window_mask)


    def _preprocess(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # leaving this here in case I want to add additional preprocessing steps to `forward()`
        if img.ndim not in [3,4]:
            raise ValueError(f"Expected 3D or 4D tensor, but got {img.dim()}D tensor of shape {tuple(img.shape)}")
        if img.ndim == 3:  # if input is (C, H, W), add batch dimension
            img = img.unsqueeze(0)
        if img.max() > 1.0:
            img = img.float() / 255.0
        if mask is not None:
            if (mask.shape[-2:] != img.shape[-2:]) and (mask.shape[0] != img.shape[0]):
                raise ValueError(f"Mask shape {mask.shape} does not match image shape {tuple(img.shape)}")
            # TODO: may add support for multiclass masks later, but it requires refactoring some channel-wise ops in `construct_laplacian_*`
                # honestly, mask support is unnecessary for the current use case since masks aren't used in training, but the original CAP-VSTNet repo supported it
            # TODO: change the following after implementing multiclass mask support
            assert mask.dtype == torch.bool, f"Only boolean masks are supported for now; got {mask.dtype}"
            if mask.device != img.device:
                # move mask to the same device as the image tensor
                mask = mask.to(img.device)
        # this is taken care of implicitly if we move the whole module to the same device first, but this is a simple fallback
        if self.ident.device != img.device:
            self.ident = self.ident.to(img.device)
        return img


    def forward(self, content_img, stylized_img, mask=None):
        """ Computes the Matting Laplacian loss between stylized and content images, optionally considering only masked regions
            Args:
                content_img (Tensor): The content image tensor of shape (B, C, H, W).
                stylized_img (Tensor): The stylized image tensor of shape (B, C, H, W).
                mask (Tensor, optional): Optional mask tensor of shape (B, 1, H, W) or (B, H, W).
            Returns:
                loss (Tensor): A scalar tensor representing the mean squared error between the laplacian responses.
        """
        with torch.no_grad():
            # gradient tracking shouldn't be needed for the content image Laplacian, since the loss w.r.t. weights only depends on the stylized image
            content_img = self._preprocess(content_img, mask=mask)
            lap_content = self.compute_laplacian_response(content_img, mask=mask)
        stylized_img = self._preprocess(stylized_img)
        if stylized_img.device != content_img.device:
            # move stylized image to the same device as the content image Laplacian
            stylized_img = stylized_img.to(content_img.device)
        # dispatch to custom Function
        return MLLossFn.apply(lap_content, stylized_img)



########################### Helper class for custom autograd function ###########################

class MLLossFn(torch.autograd.Function):
    """ Custom autograd function for computing the Matting Laplacian loss and its gradient
        - follows the same sparse quadratic form as the original project, but relies on PyTorch's autograd for differentiation
        - uses a custom forward and backward pass to compute the loss and gradient efficiently (while scaling and clipping the gradient)
    """
    @staticmethod
    def forward(ctx, laplacian: List[torch.Tensor], stylized_img: torch.Tensor) -> torch.Tensor:
        """ Computes both scalar loss and raw gradient w.r.t. stylized_img in one pass """
        B, C, H, W = stylized_img.shape
        loss_accum = 0.0
        grad = torch.zeros_like(stylized_img)
        # have to iterate over each batch index since torch.sparse.mm only supports 2D tensors
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

    # TODO: need to validate the gradient output from the backward pass with torch.autograd.gradcheck
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # grad_output is scalar (dTotal/dLoss)
        (raw_grad,) = ctx.saved_tensors
        # combine and clamp
        g = raw_grad * grad_output
        g = g.clamp(-0.05, 0.05)
        # propagate only into stylized_img; other inputs get None
        return None, g #, None, None
