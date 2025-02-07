
""" will be moving the temporal loss and Matting Laplacian loss to this file while refactoring to use pure pytorch """

from typing import Dict, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


def compare_tensor_with_benchmark(target, filename, exact = True, rtol=1e-3, atol=1e-5, names = ["src", "tgt"]):
    """ Loads saved tensor from NumPy implementation and compares with PyTorch target tensor.
        Args:
            target (torch.Tensor): Extracted indices from PyTorch.
            filename (str): Path to saved NumPy-based `source.pt`.
    """
    # Load the saved indices
    source = torch.load(filename, weights_only=True)
    # Ensure shapes match
    if source.shape != target.shape:
        print(f"❌ Shape Mismatch: {names[0]} {source.shape}, {names[1]} {target.shape}")
        return False
    # Check if values match
    if exact:
        if torch.equal(source, target):
            print(f"✅ {names[1]} matches {names[0]}!")
        else:
            print(f"❌ {names[1]} does NOT match {names[0]}!")
    else:
        if torch.allclose(source, target, rtol=rtol, atol=atol):
            print(f"✅ {names[1]} is close to {names[0]}!")
        else:
            print(f"❌ {names[1]} is NOT close to {names[0]}!")


#^ MattingLaplacian code formerly in mcapst/utils/MattingLaplacian.py
class MattingLaplacianLoss(nn.Module):
    def __init__(self, eps=1e-7, win_rad=1, objective: Literal["mse"] = "mse"):
        super(MattingLaplacianLoss, self).__init__()
        self.eps = eps
        self.win_radius = win_rad
        self.win_size = (win_rad * 2 + 1) ** 2
        if objective not in ["mse"]:
            raise ValueError(f"Unsupported objective: {objective}")
        self.objective = objective

    def extract_patches(self, img):
        """ Extracts local patches using explicit indexing instead of F.unfold with padding.
            Args:
                img (Tensor): Input image tensor of shape (B, C, H, W).
            Returns:
                patches (Tensor): Extracted patches of shape (B, C, win_size, H' = H-2*win_rad, W' = W-2*win_rad).
        """
        win_diam = self.win_radius * 2 + 1  # Window diameter
        # Apply unfold on height and width separately (removing extra padding)
        patches = img.unfold(2, win_diam, 1).unfold(3, win_diam, 1)
        # Reshape correctly to match NumPy's `_rolling_block()` output
        patches = patches.contiguous().permute(0, 1, 4, 5, 2, 3)  # (B, C, win_diam, win_diam, H', W')
        #patches = patches.reshape(B, C, H'*W', self.win_size).permute(0, 1, 3, 2)
        return patches  # Shape: (B, C, win_diam, H', W')


    def compute_local_statistics(self, patches):
        """ Computes local mean and covariance for each spatial location.
            Args:
                img (Tensor): Input image tensor of shape (B, C, H, W).
            # NOTE: using the notation H' and W' in subsequent comments, meaning H' = H - 2*win_rad, W' = W - 2*win_rad
            Returns:
                local_mean (Tensor): Local mean tensor of shape (B, C, 1, H', W').
                cov (Tensor): Local covariance tensor of shape (B, C, C, H', W').
        """
        # Compute local mean
        local_mean = patches.mean(dim=(2,3), keepdim=True).squeeze(2)  # (B, C, 1, H', W')
        # Compute covariance: E[X X^T] - E[X]E[X]^T
        # TODO: rewrite Einstein summation to go ahead and reshape patches earlier (avoids permutation of `diff` in compute_quadratic term later)
        patch_sq_sum = torch.einsum('... i m n h w, ... j m n h w -> ... i j h w', patches, patches) / self.win_size # shape: (B, C, win_diam, H' W')
        mean_sq = torch.einsum('... i k h w, ... j k h w -> ... i j h w', local_mean, local_mean) # shape: (B, C, win_diam, H', W')
        cov = patch_sq_sum - mean_sq  # shape: (B, C, win_diam, H', W')
        return local_mean, cov


    def compute_quadratic_term(self, patches, local_mean, cov):
        B, C = patches.shape[:2]
        patches = patches.reshape(B, C, self.win_size, -1)  # Shape: (B, C, win_size, H' * W')
        # Regularize the covariance matrix and invert it.
        # !!! requires that win_diam == C but there's no way around it without compromising the whole algorithm
        cov = cov.flatten(start_dim=-2).permute(0, 3, 2, 1)  # Shape: (B, C, win_diam, H' * W') -> (B, H' * W', win_diam, C)
        cov += (self.eps / self.win_size) * torch.eye(C, device=cov.device)  # Shape: # (B, H' * W', win_diam, C)
        #inv_cov = torch.linalg.inv(cov.double()).float() # (B, H' * W', win_diam, C)
        # more numerically stable matrix inversion:
        inv_cov = torch.linalg.solve(cov.double(), torch.eye(cov.shape[-1], device=cov.device).double()).float()
        # Compute the difference between image pixels and the local mean.
        local_mean = local_mean.flatten(start_dim=-2)  # Shape: (B, C, 1, H' * W')
        diff = patches - local_mean # shape: (B, C, win_size, H' * W')
        # Compute the quadratic form for each pixel: diff.T * inv_cov * diff to yield a scalar per spatial location.
        diff = diff.permute(0, 3, 2, 1) # shape: (B, H' * W', win_size, C)
        # quadratic form from einsum should be equivalent to (I - mu).T @ inv_cov @ (I - mu) from the Kaiming He Paper
        quadratic = diff @ inv_cov @ diff.transpose(2,3) # shape (B, H' * W', win_size, win_size)
        quadratic += torch.ones_like(quadratic)
        #quadratic = torch.einsum('... c s n, ... d r n, ... e t n -> ... s t n', diff, inv_cov, diff) # shape (B, win_size, win_size, H' * W')
        return quadratic.div(self.win_size) # shape (B, H' * W', win_size, win_size)

    def get_coo_indices(self, img_shape, device="cuda"):
        """ Computes the row and column indices for the sparse matrix.
            Args:
                patches (Tensor): Extracted patches of shape (B, C, win_diam, H', W').
            Returns:
                indices (Tensor): Row and column indices for the sparse matrix.
        """
        B, C, H, W = img_shape
        win_diam = 2 * self.win_radius + 1
        patch_indices = torch.arange(H * W, device=device).reshape(1, H, W)
        patch_indices = patch_indices.unfold(1, win_diam, 1).unfold(2, win_diam, 1) # shape: (1, H', W', win_diam, win_diam)
        patch_indices = patch_indices.reshape(1, -1, self.win_size).expand(B, -1, -1)  # Shape: (B, H' * W', win_size)
        # generate row and column indices for the sparse matrix
        indices = torch.stack([
            patch_indices.view(B, -1, 1).repeat(1, 1, self.win_size).flatten(start_dim=1),
            patch_indices.repeat(1, 1, self.win_size).flatten(start_dim=1),
        ], dim=0) # shape: (2, B, H' * W' * win_size**2)
        return indices


    def construct_laplacian_parallel(self, laplacian, indices, img_shape):
        """ Constructs the Laplacian matrix in parallel for each batch. """
        B, C, H, W = img_shape
        lap_vals = laplacian.flatten(start_dim=1)  # (B, H' * W' * win_size**2)
        # Use `torch.jit.fork` for parallel Laplacian construction
        futures = []
        for i in range(B):
            futures.append(torch.jit.fork(
                torch.sparse_coo_tensor, indices[:, i, :], lap_vals[i], size=torch.Size([H * W, H * W]), device=lap_vals.device
            ))
        # wait for all parallel tasks to complete
        sparse_laplacians = [torch.jit.wait(f) for f in futures]
        # compute the Kronecker delta correction in parallel
        futures = []
        for i in range(B):
            sum_L_b = torch.sparse.sum(sparse_laplacians[i], dim=1).to_dense()  # (H * W,)
            diag_L_b = torch.sparse_coo_tensor(
                torch.arange(H * W, device=sum_L_b.device).unsqueeze(0).repeat(2, 1),
                sum_L_b, (H * W, H * W)
            )
            futures.append(torch.jit.fork(lambda L, D: D - L, sparse_laplacians[i], diag_L_b))
        torch.cuda.synchronize()  # Ensure all GPU operations are complete before proceeding
        # return fully computed Laplacians
        return [torch.jit.wait(f) for f in futures]


    def construct_laplacian_sequential(self, laplacian, indices, img_shape):
        B, C, H, W = img_shape
        # Construct sparse matrices batch-wise
        sparse_laplacians = []
        for i in range(B):
            lap_vals = laplacian[i].flatten()  # Shape: (H' * W' * win_size**2)
            indices_b = indices[:, i, :]  # Shape: (2, H' * W' * win_size**2)
            # Create batchwise sparse tensor
            L_b = torch.sparse_coo_tensor(indices_b, lap_vals, torch.Size([H * W, H * W])) #.coalesce() # shape: (HW, HW)
            # Compute the sum of each row in L (Kronecker delta term from Eq. (5))
            sum_L_b = torch.sparse.sum(L_b, dim=1).to_dense()  # Shape: (H * W,)
            # Create diagonal matrix from sum_L_b
            diag_L_b = torch.sparse_coo_tensor(
                torch.arange(H * W, device=L_b.device).unsqueeze(0).repeat(2, 1),
                sum_L_b, (H * W, H * W), device=L_b.device
            ) # shape: (HW, HW)
            # Compute final Matting Laplacian: L = Diagonal(sum_L) - L (Equation (5))
            L_b = diag_L_b - L_b # shape: (HW, HW)
            sparse_laplacians.append(L_b)
        return sparse_laplacians  # List of (H*W, H*W) sparse tensors


    def compute_laplacian(self, img: torch.Tensor, mask=None):
        """ Computes the Matting Laplacian response based on local covariance statistics.
            Args:
                img (Tensor): Input image tensor of shape (B, C, H, W).
                mask (Tensor, optional): Optional mask tensor to weight the local statistics.
            Returns:
                laplacian (Tensor): Laplacian response of shape (B, H, W).
        """
        if img.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {img.dim()}D tensor of shape {tuple(img.shape)}")
        # FIXME: shouldn't be necessary to be exactly the same shape as long as it has the same spatial and batch dimensions
        if mask is not None and mask.shape != img.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {tuple(img.shape)}")
        # NOTE: using the notation H' and W' in subsequent comments, meaning H' = H - 2*win_rad, W' = W - 2*win_rad
        patches = self.extract_patches(img)  # (B, C, win_diam, win_diam, H', W')
        local_mean, cov = self.compute_local_statistics(patches)
        # Regularize the covariance matrix and invert it.
        laplacian = self.compute_quadratic_term(patches, local_mean, cov) # shape: (B, H' * W', win_size, win_size)
        indices = self.get_coo_indices(img.shape, img.device) # shape: (2, B, H' * W' * win_size**2)
        #return laplacian, indices # REMOVE: using for debugging in comparing device speedup (accumulating error meant I had to use the same `laplacian`)
        # Construct sparse matrices batch-wise
        if img.is_cuda:
            return self.construct_laplacian_parallel(laplacian, indices, img.shape)
        return self.construct_laplacian_sequential(laplacian, indices, img.shape)


    def _apply_laplacian(self, pastiche, laplacian):
        return torch.sparse.mm(laplacian, pastiche.flatten(start_dim=2).T).T.reshape_as(pastiche)
        #return torch.matmul(laplacian, pastiche.flatten(start_dim=2).T).T.reshape_as(pastiche)


    # NOTE: following two functions follow the cost function defined in "Fast Matting Using Large Kernel Matting Laplacian Matrices"
    ################################################################################################################################
    # TODO: if I end up keeping both implementations, I should just make this a single function with a decorator to switch between sparse and dense
    def get_sparse_laplacian_loss(self, pastiche: torch.Tensor, laplacian: torch.sparse.Tensor, mask: torch.Tensor = None):
        # Use sparse matrix multiplication (only works on CPU for now)
        B, C = pastiche.shape[:2]
        loss = 0
        for c in range(C):
            x = pastiche[:, c, :].reshape(B, -1)  # Shape (B, C, HW) -> (B, HW)
            lap_x = torch.sparse.mm(laplacian, x.T).T  # Shape (B, HW)
            loss += torch.sum(x * lap_x)  # Sum over all pixels
        return loss / (B * C)

    def get_dense_laplacian_loss(self, pastiche: torch.Tensor, laplacian: Union[torch.Tensor, torch.sparse.Tensor], mask: torch.Tensor = None):
        # Dense matrix multiplication (better for back-propagation on CUDA)
        B, C = pastiche.shape[:2]
        laplacian_dense = laplacian.to_dense() if laplacian.is_sparse else laplacian
        loss = 0
        for c in range(C):
            x = pastiche[:, c, :].reshape(B, -1)  # Shape (B, C, HW) -> (B, HW)
            lap_x = torch.matmul(laplacian_dense, x.T).T  # Shape (B, HW)
            loss += torch.sum(x * lap_x)  # Quadratic form: x^T L x
        return loss / (B * C)
    ################################################################################################################################



    def get_laplacian_mse_loss(self, lap_pastiche: torch.Tensor, lap_content: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            # Ensure mask is broadcastable to (B, H, W)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # Remove channel dimension if present.
            mask = mask.squeeze(1)
            # Compute the mean squared error only over valid (masked) pixels.
            diff_sq = (lap_pastiche - lap_content) ** 2
            return (diff_sq * mask).sum() / (mask.sum() + self.eps)
        return F.mse_loss(lap_pastiche, lap_content)


    def forward(self, content_img, stylized_img, mask=None):
        """ Computes the Matting Laplacian loss between stylized and content images, optionally considering only masked regions
            Args:
                content_img (Tensor): The content image tensor of shape (B, C, H, W).
                stylized_img (Tensor): The stylized image tensor of shape (B, C, H, W).
                mask (Tensor, optional): Optional mask tensor of shape (B, 1, H, W) or (B, H, W).
            Returns:
                loss (Tensor): A scalar tensor representing the mean squared error between the laplacian responses.
        """
        lap_content = self.compute_laplacian(content_img, mask=mask)
        if self.objective == "mse":
            lap_stylized = self.compute_laplacian(stylized_img, mask=mask)
            return self.get_laplacian_mse_loss(lap_stylized, lap_content, mask=mask)
        elif self.objective == "sparse":
            return self.get_sparse_laplacian_loss(stylized_img, lap_content, mask=mask)
        elif self.objective == "dense":
            return self.get_dense_laplacian_loss(stylized_img, lap_content, mask=mask)
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")