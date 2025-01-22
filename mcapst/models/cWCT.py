import torch
import torch.jit
from typing import Union, Iterable, Callable, Tuple




class cWCT(torch.jit.ScriptModule):
    def __init__(
        self,
        train_mode:bool = False,
        eps:float = 2e-5,
        verbose:bool = False,
        reg_method: str = "ridge",
        temperature: float = 1.0,
        alpha: float = 0.01
    ):
        super().__init__()
        self.train_mode = train_mode
        self.eps = eps
        self.verbose = verbose
        self.reg_method = reg_method    # Choose the regularization method
        self.temperature = temperature  # For softmax-based smoothing
        self.alpha = alpha              # For diagonal loading regularization
        if reg_method not in ["diagonal_loading", "ridge", "softmax_temperature", "log_space"]:
            raise ValueError(f"Unknown regularization method: {self.reg_method}")

    #@staticmethod
    # TODO: should still try to replace this check with a more lightweight method if possible
    @torch.jit.script_method
    def is_positive_definite(self, A: torch.Tensor):
        """ Check if all eigenvalues are positive (A is positive definite) - test must be scriptable """
        #? NOTE: saving previous checks by attempting Cholesky Decomposition for reference - no longer possible in a torch.jit.script_method
            # though was almost definitely computationally more efficient since now the eigenvalues always have to be computed first whether it would fail or not
        # try:
        #     _ = torch.linalg.cholesky(A)
        #     return True
        # except RuntimeError:
        #     return False
        # Compute eigenvalues to check positive definiteness
        eigenvalues = torch.linalg.eigvalsh(A)
        return torch.all(eigenvalues > 0)

    #@staticmethod
    @torch.jit.script_method
    def diagonal_loading(self, A: torch.Tensor) -> torch.Tensor:
        """ regularize the matrix A by adding a fraction of its trace to the diagonal"""
        # Compute the trace for each matrix in the batch
        traces = torch.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        identity = torch.eye(A.shape[-1], device=A.device).expand_as(A)
        return A + self.alpha*traces.unsqueeze(-1)*identity

    #@staticmethod
    @torch.jit.script_method
    def ridge_regularization(self, A: torch.Tensor) -> torch.Tensor:
        """ regularize the matrix A using ridge regression (Tikhonov regularization)"""
        identity = torch.eye(A.shape[-1], device=A.device)
        return A + self.eps*identity


    @torch.jit.script_method
    def softmax_temperature_regularization(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Apply temperature-scaled softmax regularization."""
        exp_vals = torch.exp(eigenvalues/self.temperature)
        return exp_vals/exp_vals.sum()


    @torch.jit.script_method
    def log_space_regularization(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Apply log-space regularization to the eigenvalues."""
        return torch.exp(torch.log(eigenvalues + self.eps))


    @torch.jit.script_method
    def regularize(self, A: torch.Tensor) -> torch.Tensor:
        """ Apply the selected regularization method. """
        if self.reg_method == "diagonal_loading":
            return self.diagonal_loading(A)
        elif self.reg_method == "ridge":
            return self.ridge_regularization(A)
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        if self.reg_method == "softmax_temperature":
            eigenvalues = self.softmax_temperature_regularization(eigenvalues)
        elif self.reg_method == "log_space":
            eigenvalues = self.log_space_regularization(eigenvalues)
        # Reconstruct the matrix using the regularized eigenvalues
        return torch.matmul(eigenvectors, torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2))


    @torch.jit.script_method
    def cholesky_dec(self, conv: torch.Tensor, invert:bool=False) -> torch.Tensor:
        """ Perform Cholesky decomposition with optional inversion
            :param conv: Covariance matrix [..., N, N]
            :param invert: Whether to invert the Cholesky factor.
            :return: Cholesky factor or its inverse [..., N, N]
            * NOTE: GPT-generated docstring
        """
        #conv_copy = torch.empty_like(conv)
        while not self.is_positive_definite(conv):
            # if decomposition fails and in training mode, raise an error
            if self.train_mode:
                raise RuntimeError('Cholesky Decomposition fails. Gradient infinity.')
            conv = self.regularize(conv)
        # initial Cholesky Decomposition, which would raise RuntimeError if conv is not positive definite, but regularization should prevent that
        L = torch.linalg.cholesky(conv)
        # invert the Cholesky factor if performed on the backwards pass
        if invert:
            L = torch.inverse(L)
        # ? should note that the order in which things are inverted and when dtype is converted is backwards from what it once was in the interpolation function
        return L

    @torch.jit.script_method
    def get_feature_covariance_and_decomp(
        self,
        X: torch.Tensor,
        invert:bool=False,
        update_mean:bool=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ get covariance matrix as part of the operations for both whitening and coloring transforms
            :param X: Input features [B, N, H*W]
            :param invert: Whether to invert the Cholesky factor before returning
            :param update_mean: Whether to reshape the mean tensor to match the input features before returning
            :return: feature covariance matrix [B, N, N]
        """
        # compute mean of the input features along the last dimension
        mean = torch.mean(X, -1)                    # [B, N]
        if update_mean:
            # updates the mean before it gets passed back
            mean = mean.unsqueeze(-1).expand_as(X)    # [B, N, H*W]
            X -= mean
        # standardize input features
        else:
            # only uses the mean without changing underlying storage
            X -= mean.unsqueeze(-1).expand_as(X)    # [B, N, H*W]
        # compute covariance matrix
        conv = torch.bmm(X, X.transpose(-1, -2)).div(X.shape[-1] - 1)   # [B, N, N]
        # perform Cholesky Decomposition; get L in the case of coloring transform or L^-1 for whitening transform
            # (note from CAP-VSTNet authors: "interpolate Conv works well; interpolate L seems to be slightly better")
        # ? NOTE: saving L differently (being a lower triangular matrix) could reduce memory requirements by about half
        L = self.cholesky_dec(conv, invert=invert)              # [B, N, N]
        return mean, conv, L

    @torch.jit.script_method
    def whitening(self, x: torch.Tensor) -> torch.Tensor:
        """ Perform the whitening transform in WCT
            :param x: Input features [B, N, H*W]
            :return: Whitened features [B, N, H*W]
        """
        # x should be changed since it's pass-by-reference (kinda forget without pointer syntax)
        _, _, inv_L = self.get_feature_covariance_and_decomp(x, invert=True, update_mean=True)
        # Apply the whitening transform to the input features
        whiten_x = torch.bmm(inv_L, x)                                   # [B, N, H*W]
        return whiten_x

    @torch.jit.script_method
    def coloring(self, whiten_xc: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """ Perform the coloring transform.
            :param whiten_xc: Whitened content features [B_c, N, H*W]
            :param xs: Style features [B_s, N, H*W]
            :return: Colored features [B_c, N, H*W]
        """
        xs_mean, _, Ls = self.get_feature_covariance_and_decomp(xs, invert=False, update_mean=False)
        # apply the coloring transform to the whitened content features
        coloring_cs = torch.bmm(Ls.expand(whiten_xc.size(0), -1, -1), whiten_xc)                      # [B_c, N, H*W]
        # Add the mean of the style features back to the colored features
        coloring_cs += xs_mean.unsqueeze(-1).expand_as(coloring_cs) # [B_c, N, H*W]
        return coloring_cs