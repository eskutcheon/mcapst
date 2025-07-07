
import torch
from collections import OrderedDict
from typing import Any, Dict



class BaseBufferCache:
    """ Generic LRU cache of small-to-medium tensors derived from a shape + device key.
        Subclasses must implement `_build(key: Tuple[Any, ...]) -> torch.Tensor`.
    """
    def __init__(self, max_size: int = 8):
        super().__init__()
        self.max_size = max_size
        # maps key -> buffer name
        self._cache = OrderedDict() # maybe consider using deque to easily set a max size

    def get(self, *key_parts: Any) -> torch.Tensor:
        """ Key is the tuple `key_parts` (e.g. (H, W, device)). Returns a tensor, caching the result of `_build` """
        key = tuple(key_parts)
        if key in self._cache:  # if there was a hit, move it to the front of the queue and return the tensor
            val = self._cache.pop(key)
            self._cache[key] = val
            return val
        # else if it's a miss, then build, register as buffer, insert in LRU
        tensor = self._build(*key_parts)
        self._cache[key] = tensor #bufname
        # evict (by LRU policy) if needed
        if len(self._cache) > self.max_size:
            _ = self._cache.popitem(last=False) # retrieve with FIFO order
        return tensor

    def _build(self, *key_parts: Any) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement `_build` to create the tensor based on the key parts.")



class IndexCache(BaseBufferCache):
    """ LRU cache for indices of a sparse matrix constructed from a sliding window over an image.
            - computed based on the spatial dimensions (H, W) and the window size (win_d).
            - used to construct a sparse matrix for the Matting Laplacian loss
    """
    def __init__(self, win_d: int, max_size: int = 8):
        super().__init__(max_size)
        self.win_d = win_d
        self.win_size = win_d**2

    def _build(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """ Computes the row and column COO indices for the sparse matrix while maintaining LRU cache
            Args:
                H, W: spatial dimensions of the input image
                device (str or torch.device): Device on which to construct the indices (default: "cpu")
            Returns:
                indices (Tensor): Row and column indices for the sparse matrix of shape (2, H' * W' * win_size**2)
        """
        # build meshgrid of indices for the patches to be unfolded
        patch_idx = torch.arange(H * W, device=device).view(1, H, W)
        # extract valid windows of size win_d
        patches = (
            patch_idx
            .unfold(1, self.win_d, 1)
            .unfold(2, self.win_d, 1)
            .reshape(1, -1, self.win_size)
        ) # shape: (1, num_windows, win_size)
        rows = patches.reshape(-1, 1).repeat(1, self.win_size).flatten()
        cols = patches.repeat(1, 1, self.win_size).flatten()
        indices  = torch.stack([rows, cols], dim=0)   # (2, num_windows * win_size)
        return indices



class MeshGridCache(BaseBufferCache):
    """ LRU cache for 2D mesh grids to avoid recomputing them frequently while generating deformed video frames using optical flow.
        The grid is used to warp the first frame in a sequence to simulate the second frame during video style transfer training.
        The grid is built based on the spatial dimensions (H, W) and the device
    """
    def _build(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        return torch.stack((xx, yy), dim=0).float()  # shape (2, H, W)
        # returned grid is repeated along a batch dimension to deform input tensors according to the optical flow input
        # grid = grid2d.unsqueeze(0).expand(B, -1, -1, -1)




class RunningMeanLoss:
    """ Maintains a running mean of losses using Welford's online variance algorithm, modified for means only.
        This is useful for tracking the average loss (in a numerically stable way) during training without storing all individual losses.
    """
    def __init__(self):
        self._loss_means: Dict[str, float] = {}
        self._loss_counts: Dict[str, int] = {}

    def update(self, losses: Dict[str, float]):
        """ update running means of losses using Welford's online algorithm """
        for key, value in losses.items():
            if key not in self._loss_means:
                self._loss_counts[key] = 1
                self._loss_means[key] = value
            else: # else perform Welford's algorithm update on existing key
                self._loss_counts[key] += 1
                delta = value - self._loss_means[key]
                self._loss_means[key] += delta / self._loss_counts[key]

    def reset(self):
        """ resets the running means and counts to start a new running mean calculation """
        self._loss_means.clear()
        self._loss_counts.clear()


    def get_means(self) -> Dict[str, float]:
        """ returns the current running means of losses """
        return self._loss_means