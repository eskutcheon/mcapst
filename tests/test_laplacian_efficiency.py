import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pytest
import torch
import torchvision.transforms.v2 as TT
import torchvision.io as IO

#from mcapst.training.losses import MattingLaplacianLoss  # Ensure this is the latest PyTorch implementation
from mcapst.train.loss.matting_laplacian import MattingLaplacianLoss as MLL_old
from scripts.MattingLaplacian import (
    compute_laplacian as compute_laplacian_np,
    #get_local_statistics
)

# Paths to test images (Update these to match your actual test images)
IMAGE_PATHS = [
    "data/content/01.jpg",
    "data/style/02.png"
]

# Hyperparameters
EPSILON = 1e-6
WIN_RAD = 1
# the allowable deviations in precision for the sparse Laplacian comparisons
RTOL = 1e-2
ATOL = 1e-4
IMG_SIZE = [512, 512]


torch_preprocessor = TT.Compose([
    TT.Resize(IMG_SIZE, interpolation=TT.InterpolationMode.BILINEAR),
    TT.ToDtype(torch.float32, scale=True),
])


# FIXME - I had this comparing another implementation of the Matting Laplacian, but I scrapped it and updated the original
    # which should be compared against the original repo's numpy implementation



# # --- Fixtures ---
# @pytest.fixture(params=IMAGE_PATHS)
# def image_pair(request): #& replaces the old `load_and_resize_numpy` and `load_and_resize_torch` fixtures
#     """ Load and return (path, numpy_image, torch_image) """
#     path = request.param
#     print("[DEBUGGING] path variable in image_pair fixture: ", path)
#     img_t = IO.read_image(path, mode=IO.ImageReadMode.RGB)
#     img_t = torch_preprocessor(img_t)  # (C, H, W), float32 in [0,1]
#     img_np = img_t.permute(1, 2, 0).numpy()  # (H, W, C)
#     return path, img_np, img_t

# @pytest.fixture(params=[IMAGE_PATHS[0]])
# def single_image(request):
#     path = request.param
#     img = IO.read_image(path, mode=IO.ImageReadMode.RGB)
#     img = torch_preprocessor(img).unsqueeze(0)
#     return path, img


# # --- Helper Functions ---
# def numpy_to_sparse_tensor(M_np): #& replaces `numpy_laplacian_to_sparse_tensor`
#     """ convert SciPy COO matrix to PyTorch sparse tensor """
#     indices = torch.tensor([M_np.row, M_np.col], dtype=torch.long)
#     values = torch.tensor(M_np.data, dtype=torch.float32)
#     print("[DEBUGGING] indices shape: ", indices.shape)
#     return torch.sparse_coo_tensor(indices, values, size=M_np.shape)

# #& replaces the old `compare_sparse_laplacians` but removes most console output
# def compare_sparse(a: torch.Tensor, b: torch.Tensor, rtol=RTOL, atol=ATOL):
#     """ assert two sparse COO tensors have the same sparsity pattern and close values """
#     a, b = a.coalesce(), b.coalesce()
#     assert torch.equal(a.indices(), b.indices()), "Sparse indices mismatch"
#     assert torch.allclose(a.values(), b.values(), rtol=rtol, atol=atol), "Sparse values mismatch"





# # TODO: add functions for timing comparisons between implementations, device performance, etc.

# def test_forward_timing_cpu():
#     """ new implementation should be faster than old on CPU (baseline) """
#     # TODO: add warmup runs and average multiple runs for better timing approximation
#     img = torch.rand((1, 3, *IMG_SIZE), dtype=torch.float32)
#     loss_old = MLL_old(eps=EPSILON, win_rad=WIN_RAD)
#     loss_new = MLL_new(eps=EPSILON, win_rad=WIN_RAD)
#     t0 = time.perf_counter()
#     _ = loss_old(img, img)
#     old_dur = time.perf_counter() - t0
#     t0 = time.perf_counter()
#     _ = loss_new(img, img)
#     new_dur = time.perf_counter() - t0
#     assert new_dur < old_dur, f"New impl should be faster (old={old_dur:.3f}s, new={new_dur:.3f}s)"




# #& removed `timing_test_devices` but it should probably be compared again later - before, CUDA showed great speedup

# # TODO: add functions for comparisons between implementations' device performance

# def test_device_parity(single_image):
#     """ test whether outcomes are the same on CPU and CUDA for the Laplacian computation using sequential vs parallel construction in matting_laplacian.py """
#     #img = load_and_resize_torch(content_path)  # Shape: (C, H, W)
#     _, img = single_image
#     # instantiate the loss module
#     loss_module = MLL_old(eps=EPSILON, win_rad=WIN_RAD)
#     img_cpu = img.clone().to(device="cpu")  # ensure CPU tensor
#     img_cuda = img.clone().to(device="cuda")  # ensure CUDA tensor
#     # comparing CPU and CUDA implementations of the Laplacian
#     laplacian_cpu = loss_module.compute_laplacian_response(img_cpu)[0]
#     print("type of laplacian_cpu: ", type(laplacian_cpu))
#     print("laplacian_cpu device: ", laplacian_cpu.device)
#     # TODO: returns a list currently - need to add a stacking step in the MattingLaplacianLoss class itself
#     laplacian_cuda = loss_module.compute_laplacian_response(img_cuda)[0]
#     print("type of laplacian_cuda: ", type(laplacian_cuda))
#     # construct laplacian matrices sequentially and in parallel (with implementation in matting_laplacian.py)
#     #result_cpu = loss_module.construct_laplacian_sequential(laplacian_cpu, indices_cpu, img_shape)
#     #result_cuda = loss_module.construct_laplacian_parallel(laplacian_cuda, indices_cuda, img_shape)
#     # move results back to CPU for comparison
#     compare_sparse(laplacian_cpu, laplacian_cuda.to(device="cpu"), rtol=RTOL, atol=ATOL)
#     #compare_sparse(laplacian_cpu[0], laplacian_cuda.to(device="cpu")[0], rtol=RTOL, atol=ATOL)



# @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
# def test_cuda_peak_memory():
#     content = torch.rand((1, 3, *IMG_SIZE), device="cuda")
#     style = torch.rand((1, 3, *IMG_SIZE), device="cuda")
#     loss_old = MLL_old(eps=EPSILON, win_rad=WIN_RAD).cuda()
#     loss_new = MLL_new(eps=EPSILON, win_rad=WIN_RAD).cuda()
#     torch.cuda.reset_peak_memory_stats()
#     with torch.no_grad():
#         _ = loss_old(content, style)
#     mem_old = torch.cuda.max_memory_allocated()
#     torch.cuda.reset_peak_memory_stats()
#     with torch.no_grad():
#         _ = loss_new(content, style)
#     mem_new = torch.cuda.max_memory_allocated()
#     assert mem_new <= mem_old, f"New implementation should use ≤ CUDA memory: old={mem_old}, new={mem_new}"