# tests/test_matting_laplacian_regression.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import torchvision.transforms.v2 as TT
import torchvision.io as IO

from mcapst.train.loss.matting_laplacian import MattingLaplacianLoss as MLL_new
from scripts.MattingLaplacian import compute_laplacian as compute_laplacian_np

# Paths to test images
IMAGE_PATHS = [
    "data/content/01.jpg",
    "data/style/02.png"
]

# Hyperparameters
EPSILON = 1e-6
WIN_RAD = 1
RTOL = 1e-2
ATOL = 1e-4
IMG_SIZE = [256, 256]

torch_preprocessor = TT.Compose([
    TT.Resize(IMG_SIZE, interpolation=TT.InterpolationMode.BILINEAR),
    TT.ToDtype(torch.float32, scale=True),
    TT.Lambda(lambda x: x.unsqueeze(0))
])

# --- Fixtures ---

@pytest.fixture(params=IMAGE_PATHS)
def image_pair(request):
    """Load and return (path, numpy_image, torch_image)."""
    path = request.param
    img_t = IO.read_image(path, mode=IO.ImageReadMode.RGB)
    img_t = torch_preprocessor(img_t)        # (1, C, H, W), float32 in [0,1]
    img_np = img_t.permute(0, 2, 3, 1).numpy()   # (1, H, W, C)
    return path, img_np, img_t

@pytest.fixture(params=[IMAGE_PATHS[0]])
def single_image(request):
    """Load and return (path, torch_image batch)."""
    path = request.param
    img = IO.read_image(path, mode=IO.ImageReadMode.RGB)
    img = torch_preprocessor(img) # shape: (1, C, H, W)
    return path, img

# --- Helpers ---

def numpy_to_sparse_tensor(M_np):
    """ Convert SciPy COO to PyTorch sparse tensor. """
    indices = torch.tensor([M_np.row, M_np.col], dtype=torch.long)
    values  = torch.tensor(M_np.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=M_np.shape)

def compare_sparse(a: torch.Tensor, b: torch.Tensor, rtol=RTOL, atol=ATOL):
    """ assert two sparse COO tensors have the same sparsity pattern and close values """
    a, b = a.coalesce(), b.coalesce()
    assert torch.equal(a.indices(), b.indices()), "Sparse indices mismatch"
    assert torch.allclose(a.values(), b.values(), rtol=rtol, atol=atol), "Sparse values mismatch"

# --- Tests ---

def test_numpy_vs_old_torch_laplacian(image_pair):
    """Original NumPy vs. legacy PyTorch sparse Laplacian."""
    _, img_np, img_t = image_pair
    # NumPy Laplacian
    M_np = compute_laplacian_np(img_np, eps=EPSILON, win_rad=WIN_RAD)
    lap_np = numpy_to_sparse_tensor(M_np)
    # Legacy PyTorch
    loss_new = MLL_new(eps=EPSILON, win_rad=WIN_RAD, objective="mse")
    lap_new = loss_new.compute_laplacian_response(img_t).cpu()
    #lap_new = lap_new.to_sparse()
    compare_sparse(lap_np, lap_new)



def test_new_laplacian_response_shape_and_nonnegativity(single_image):
    """ New impl's per-pixel diag entries have correct shape & ≥0. """
    _, img = single_image
    resp = MLL_new(eps=EPSILON, win_rad=WIN_RAD).compute_laplacian_response(img)
    assert resp.ndim == 3 and list(resp.shape[1:]) == IMG_SIZE
    assert torch.all(resp >= 0)




# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns



# def compare_density_overlay(
#     a: np.ndarray,
#     b: np.ndarray,
#     bins: int = 1000,
#     x_percentile: float = 99.0,
#     log_y: bool = True,
#     figsize=(10, 6),
# ):
#     """ Overlay density of a and b on one plot and show their residual below """
#     x1 = a.ravel()
#     x2 = b.ravel()
#     # symmetric x-limits at percentile
#     lim1 = np.percentile(np.abs(x1), x_percentile)
#     lim2 = np.percentile(np.abs(x2), x_percentile)
#     xlim = (-max(lim1, lim2), max(lim1, lim2))
#     # compute histograms
#     h1, edges = np.histogram(x1, bins=bins, range=xlim, density=True)
#     h2, _     = np.histogram(x2, bins=bins, range=xlim, density=True)
#     centers   = (edges[:-1] + edges[1:]) * 0.5
#     # make figure with two rows: overlay and residual
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
#     # top: overlay
#     sns.lineplot(x=centers, y=h1, ax=ax1, label="old", drawstyle="steps-mid", linewidth=3)
#     sns.lineplot(x=centers, y=h2, ax=ax1, label="new", drawstyle="steps-mid", linewidth=1, alpha=0.7)
#     ax1.set_xlim(xlim)
#     if log_y:
#         ax1.set_yscale("log")
#     ax1.set_ylabel("Density")
#     ax1.legend()
#     ax1.set_title("Density Overlay")
#     # bottom: residual
#     resid = h2 - h1
#     sns.lineplot(x=centers, y=resid, ax=ax2, color="gray", drawstyle="steps-mid")
#     ax2.axhline(0, linestyle="--", color="black", linewidth=1)
#     ax2.set_xlim(xlim)
#     ax2.set_ylabel(r"$\Delta$ Density")
#     ax2.set_xlabel("Value")
#     ax2.set_title("Residual (new - old)")
#     plt.tight_layout()
#     return fig, (ax1, ax2)



# # testing this here because the constant Pytest refactoring is annoying me at the moment
# if __name__ == "__main__":
#     import os, sys
#     import torchvision.io as IO
#     import torchvision.transforms.v2 as TT
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
#     from scripts.MattingLaplacian import compute_laplacian as compute_laplacian_np, laplacian_loss_grad

#     # Hyperparameters
#     EPSILON = 1e-6
#     WIN_RAD = 1
#     RTOL = 1e-2
#     ATOL = 1e-4
#     IMG_SIZE = [512, 512]

#     torch_preprocessor = TT.Compose([
#         TT.Resize(IMG_SIZE, interpolation=TT.InterpolationMode.BILINEAR),
#         TT.ToDtype(torch.float32, scale=True),
#         TT.Lambda(lambda x: x.unsqueeze(0))
#     ])

#     def ensure_coalesced(sparse_tensor: torch.sparse.Tensor):
#         """ Ensure the sparse tensor is in coalesced (COO) format. """
#         if sparse_tensor.is_sparse and not sparse_tensor.is_coalesced():
#             return sparse_tensor.coalesce()
#         return sparse_tensor

#     def compare_sparse(a: torch.Tensor, b: torch.Tensor, rtol=RTOL, atol=ATOL):
#         """ assert two sparse COO tensors have the same sparsity pattern and close values """
#         a = ensure_coalesced(a)
#         b = ensure_coalesced(b)
#         print(f"differences in nnz values: a: {a._nnz()}, b: {b._nnz()}")
#         assert torch.equal(a.indices(), b.indices()), "Sparse indices mismatch"
#         assert torch.allclose(a.values(), b.values(), rtol=rtol, atol=atol), "Sparse values mismatch"

#     def numpy_to_sparse_tensor(M_np):
#         """ Convert SciPy COO to PyTorch sparse tensor. """
#         indices = torch.tensor([M_np.row, M_np.col], dtype=torch.long)
#         values  = torch.tensor(M_np.data, dtype=torch.float32)
#         return torch.sparse_coo_tensor(indices, values, size=M_np.shape)

#     #root = os.path.abspath(os.path.dirname(__file__), "../..")
#     root = r"E:/matting_laplacian_tests"
#     IMAGE_PATHS = [
#         os.path.join(root, "imperial_boy_dusk.jpeg"),
#         os.path.join(root, "imperial_boy_night.jpg"),
#         os.path.join(root, "imperial_boy_dusk_mask.png")
#     ]
#     content = torch_preprocessor(IO.read_image(IMAGE_PATHS[0], mode=IO.ImageReadMode.RGB))
#     stylized = torch_preprocessor(IO.read_image(IMAGE_PATHS[1], mode=IO.ImageReadMode.RGB))
#     content_np = content.squeeze(0).permute(1, 2, 0).numpy()
#     # content = content.to(device="cuda")
#     # stylized = stylized.to(device="cuda")


#     MLL = MLL_new(eps=EPSILON, win_rad=WIN_RAD, objective="sparse")

#     def loss_regression_test():
#         loss_new = MLL(content, stylized)
#         lap_np = compute_laplacian_np(content_np, eps=EPSILON, win_rad=WIN_RAD)
#         lap_np = numpy_to_sparse_tensor(lap_np)
#         lap_np = ensure_coalesced(lap_np)
#         print("stylized shape: ", stylized.shape)
#         print("lap_np shape: ", lap_np.shape)
#         loss_np, _ = laplacian_loss_grad(stylized.squeeze(0).cpu(), lap_np)
#         print(f"Numpy loss: {loss_np}, PyTorch loss: {loss_new.item()}")
#         assert np.isclose(loss_np, round(loss_new.item(), 4), rtol=RTOL, atol=ATOL), "Loss values do not match"


#     def laplacian_regression_test(mask_ndarray = None, mask_tensor = None):
#         # old implementation
#         lap_np = compute_laplacian_np(content_np, mask_ndarray, eps=EPSILON, win_rad=WIN_RAD)
#         lap_np = numpy_to_sparse_tensor(lap_np)
#         lap_np = ensure_coalesced(lap_np)
#         # new implementation
#         lap_new = MLL.compute_laplacian_response(content, mask_tensor) #.to(device="cuda")).cpu()
#         lap_new = MLL.postprocess(lap_new)
#         lap_new = ensure_coalesced(lap_new)
#         # free memory
#         #del content, stylized, content_np, stylized_np, M_np
#         # inspecting shapes and stuff before any assertions
#         lap_range = lambda x: (x.values().min().item(), x.values().max().item())
#         print(f"Sparse Laplacian range comparison: old = {lap_range(lap_np)}, new = {lap_range(lap_new)}")
#         print(f"Sparse Laplacian shape comparison: old = {lap_np.shape}, new = {lap_new.shape}")
#         print(f"Sparse Laplacian nnz comparison: old = {lap_np._nnz()}, new = {lap_new._nnz()}")
#         print(f"Sparse laplacian mean comparison: old = {lap_np.values().mean().item()}, new = {lap_new.values().mean().item()}")
#         # plot density overlay of the old and new implementations' Laplacian matrix values and their residuals
#         compare_density_overlay(
#             lap_np.values().cpu().numpy(),
#             lap_new.values().cpu().numpy(),
#             bins=2000,
#             x_percentile=99.9,
#             log_y=True,
#         )
#         plt.show()
#         compare_sparse(lap_new, lap_np, rtol=RTOL, atol=ATOL)

#     def masked_regression_test():
#         mask = IO.read_image(IMAGE_PATHS[2], mode=IO.ImageReadMode.GRAY)
#         mask = TT.Resize(IMG_SIZE, interpolation=TT.InterpolationMode.NEAREST)(mask)
#         mask = (mask > 127).unsqueeze(0)  # boolean mask
#         mask_np = mask.squeeze(0).squeeze(0).numpy()
#         laplacian_regression_test(mask_np, mask)


#     loss_regression_test()
#     #laplacian_regression_test()
#     #masked_regression_test()