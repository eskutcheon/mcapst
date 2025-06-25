import os, sys
import torch
import torchvision.transforms.v2 as TT
import torchvision.io as IO
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from mcapst.training.losses import MattingLaplacianLoss  # Ensure this is the latest PyTorch implementation
from mcapst.loss.matting_laplacian import MattingLaplacianLoss
from scripts.MattingLaplacian import compute_laplacian, get_local_statistics  # Ensure this is the original NumPy implementation

# Paths to test images (Update these to match your actual test images)
IMAGE_PATHS = [
    "data/content/01.jpg",
    "data/style/02.png"
]

# Hyperparameters
EPSILON = 1e-6
WIN_RAD = 1
TOLERANCE = 1e-3  # Allowable numerical deviation
IMG_SIZE = [512, 512]

torch_preprocessor = TT.Compose([
    TT.Resize(IMG_SIZE, interpolation=TT.InterpolationMode.BILINEAR),
    TT.ToDtype(torch.float32, scale=True),
])


def load_and_resize_numpy(image_path):
    """Loads and resizes images using OpenCV for NumPy-based processing."""
    # img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read image
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    # img = cv2.resize(img, IMG_SIZE[::-1], interpolation=cv2.INTER_AREA)  # Resize
    img = IO.read_image(image_path, mode=IO.ImageReadMode.RGB)
    img = torch_preprocessor(img)  # Resize
    # loading this with pytorch then converting to numpy because I otherwise can't ensure the same interpolation when resizing
    #img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    #return img.astype(np.float32) / 255.0  # Shape: (H, W, C)
    return img.permute(1, 2, 0).numpy()  # Shape: (H, W, C)

def load_and_resize_torch(image_path):
    """Loads and resizes images using torchvision for PyTorch-based processing."""
    img = IO.read_image(image_path, mode=IO.ImageReadMode.RGB)
    img = torch_preprocessor(img)  # Resize
    return img  # Load and normalize  # Shape: (C, H, W)

def numpy_laplacian_to_sparse_tensor(M):
    """Converts NumPy sparse matrix (COO format) to PyTorch sparse tensor."""
    indices = torch.tensor([M.row, M.col], dtype=torch.long)
    values = torch.tensor(M.data, dtype=torch.float32)
    shape = torch.Size(M.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def apply_laplacian(image_tensor, laplacian_tensor):
    """Applies a sparse Laplacian matrix to an image and returns the transformed result."""
    C, H, W = image_tensor.shape
    reshaped_image = image_tensor.view(C, H * W, 1)  # Flatten image to (C, H*W, 1)
    transformed_channels = []
    for i in range(C):
        transformed = torch.sparse.mm(laplacian_tensor, reshaped_image[i])  # Shape: (H*W, 1)
        transformed_channels.append(transformed.view(H, W))  # Reshape back to (H, W)
    return torch.stack(transformed_channels, dim=0)  # Stack to shape (C, H, W)

def compute_numpy_laplacian(image_np):
    """Computes the Matting Laplacian using the original NumPy-based implementation and converts to a sparse tensor."""
    laplacian_coo = compute_laplacian(image_np, eps=EPSILON, win_rad=WIN_RAD)
    return numpy_laplacian_to_sparse_tensor(laplacian_coo)

def compute_torch_laplacian(image_torch):
    """Computes the Matting Laplacian using the PyTorch-based implementation."""
    loss_module = MattingLaplacianLoss(eps=EPSILON, win_rad=WIN_RAD)
    laplacian = loss_module.compute_laplacian(image_torch.unsqueeze(0))
    return laplacian[0]  # Shape: (H, W)

def compare_sparse_laplacians(laplacian_np, laplacian_torch, rtol=1e-4, atol=1e-6):
    """ Efficiently compares two PyTorch sparse Laplacian tensors without converting to dense format. """
    # coalesce the sparse tensors to ensure they are in the same format
    laplacian_np = laplacian_np.coalesce()
    laplacian_torch = laplacian_torch.coalesce()
    # Extract nonzero elements from both sparse tensors
    nz_indices_np = laplacian_np.indices()  # Shape: (2, nnz)
    nz_values_np = laplacian_np.values()  # Shape: (nnz,)
    nz_indices_torch = laplacian_torch.indices()  # Shape: (2, nnz)
    nz_values_torch = laplacian_torch.values()  # Shape: (nnz,)
    # Ensure both have the same number of nonzero elements
    if nz_indices_np.shape != nz_indices_torch.shape or nz_values_np.shape != nz_values_torch.shape:
        print(f"❌ Mismatch in nonzero element shapes: np ({nz_indices_np.shape}, {nz_values_np.shape}) vs. torch ({nz_indices_torch.shape}, {nz_values_torch.shape})")
        return False
    # Compare nonzero indices
    indices_match = torch.equal(nz_indices_np, nz_indices_torch)
    if not indices_match:
        print("❌ Nonzero indices do not match!")
    # Compare nonzero values within tolerance
    values_match = torch.allclose(nz_values_np, nz_values_torch, rtol=rtol, atol=atol)
    if not values_match:
        print("❌ Nonzero values do not match within tolerance!")
        print("testing average magnitudes of nonzero values: ", nz_values_np.abs().mean(), nz_values_torch.abs().mean())
    if indices_match and values_match:
        print("✅ Laplacians match within tolerance!")
    return indices_match and values_match




def plot_heatmap_difference(mu_np, mu_torch, cov_np, cov_torch, H, W, title="Local Statistics Difference Heatmap"):
    """ Plots heatmaps to visualize spatial differences between NumPy and PyTorch local statistics.
        Args:
            mu_np (ndarray): NumPy mean values of shape (H*W, C).
            mu_torch (ndarray): PyTorch mean values of shape (H*W, C).
            cov_np (ndarray): NumPy covariance values of shape (H*W, C, C).
            cov_torch (ndarray): PyTorch covariance values of shape (H*W, C, C).
            H (int): Height of the image.
            W (int): Width of the image.
            title (str): Title for the figure.
    """
    # Compute absolute differences for visualization
    mu_diff = np.abs(mu_np - mu_torch).mean(axis=-1)  # Shape: (H*W,)
    cov_diff = np.abs(cov_np - cov_torch).mean(axis=(-1, -2))  # Shape: (H*W,)
    # Reshape back to spatial domain
    mu_diff_map = mu_diff.reshape(H, W)
    cov_diff_map = cov_diff.reshape(H, W)
    # Normalize for better visualization
    #mu_diff_map = (mu_diff_map - mu_diff_map.min()) / (mu_diff_map.max() - mu_diff_map.min() + 1e-8)
    #cov_diff_map = (cov_diff_map - cov_diff_map.min()) / (cov_diff_map.max() - cov_diff_map.min() + 1e-8)
    # Apply logarithmic scaling for better contrast (helps visualize small differences)
    #mu_diff_map = np.log1p(mu_diff_map)
    #cov_diff_map = np.log1p(cov_diff_map)
    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = axes[0].imshow(mu_diff_map, cmap='jet', interpolation='nearest')
    axes[0].set_title("Mean Difference Heatmap")
    axes[0].axis("off")
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Difference Intensity")
    im2 = axes[1].imshow(cov_diff_map, cmap='jet', interpolation='nearest')
    axes[1].set_title("Covariance Difference Heatmap")
    axes[1].axis("off")
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("Difference Intensity")
    plt.suptitle(title)
    plt.show()


def plot_img_heatmap_difference(img_np, img_torch, H, W, title="Image Difference Heatmap"):
    img_diff = np.abs(img_np - img_torch).mean(axis=-1)  # Shape: (H*W,)
    img_diff_map = img_diff.reshape(H, W)
    img_diff_map = (img_diff_map - img_diff_map.min()) / (img_diff_map.max() - img_diff_map.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(img_diff_map, cmap='jet', interpolation='nearest')
    ax.set_title("Image Difference Heatmap")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Difference Intensity")
    plt.suptitle(title)
    plt.show()



def compare_local_statistics(loss_module, img_np, img_torch):
    mu_np, cov_np, _, _ = get_local_statistics(img_np, win_rad=WIN_RAD)
    print("shape of mu_np: ", mu_np.shape)
    print("shape of cov_np: ", cov_np.shape)
    print()
    patches = loss_module.extract_patches(img_torch.unsqueeze(0))
    mu_torch, cov_torch = loss_module.compute_local_statistics(patches)
    print()
    print("shape of mu_torch: ", mu_torch.shape)
    print("shape of cov_torch: ", cov_torch.shape)
    mu_torch = mu_torch.squeeze(0).flatten(start_dim=-2).permute(2, 1, 0).numpy()
    #mu_torch = mu_torch.squeeze(0).permute(2, 1, 0).numpy()
    cov_torch = cov_torch.squeeze(0).flatten(start_dim=-2).permute(2, 1, 0).numpy()
    #cov_torch = cov_torch.squeeze(0).permute(2, 1, 0).numpy()
    print("shape of mu_torch: ", mu_torch.shape)
    print("shape of cov_torch: ", cov_torch.shape)
    print("testing if mu_np and mu_torch are close: ", np.allclose(mu_np, mu_torch, rtol=1e-5, atol=1e-8))
    print("testing if cov_np and cov_torch are close: ", np.allclose(cov_np, cov_torch, rtol=1e-4, atol=1e-6))
    print("average difference between mu_np and mu_torch: ", np.mean(np.abs(mu_np - mu_torch)))
    print("average difference between cov_np and cov_torch: ", np.mean(np.abs(cov_np - cov_torch)))
    print("mu_np and mu_torch avg magnitudes for reference: ", np.mean(np.abs(mu_np)), np.mean(np.abs(mu_torch)))
    print("cov_np and cov_torch avg magnitudes for reference: ", np.mean(np.abs(cov_np)), np.mean(np.abs(cov_torch)))
    plot_heatmap_difference(mu_np, mu_torch, cov_np, cov_torch, img_np.shape[0] - 2, img_np.shape[1] - 2)


def quadratic_timing_test():
    """ Compares the time taken for quadratic term calculation using einsum vs direct matrix multiplication """
    #patches = torch.load("np_winI_scaled.pt", weights_only=True) # shape: (B, H' * W', win_size, C,)
    #loss_module = MattingLaplacianLoss(eps=EPSILON, win_rad=WIN_RAD)
    diff = torch.load("np_winI_scaled.pt", weights_only=True) # shape: (B, H' * W', win_size, C,)
    inv_cov = torch.load("np_inv.pt", weights_only=True)      # shape: (B, H_new*W_new, win_diam, C)
    print("diff shape: ", diff.shape)
    print("inv_cov shape: ", inv_cov.shape)
    einsum_times = []
    matmul_times = []
    # letting R = win_diam, S = win_size and N = H'*W' for simplicity:
    # shape multiplication: (..., S, C) x (..., R, C) x (..., C, S) = (..., S, S)
    # warmup runs:
    for _ in range(3):
        #X = torch.einsum('...ij,...jk->...ik', diff, inv_cov)
        #quadratic1 = torch.einsum('...ij,...kj->...ik', X, diff)
        quadratic1 = torch.einsum('... i j, ... j k, ... n m -> ... i n', diff, inv_cov, diff)
        quadratic2 = diff @ inv_cov @ diff.transpose(2,3)
    for i in range(10):
        # measure einsum
        start_time = time.time()
        # XX = torch.einsum('...ij,...jk->...ik', diff, inv_cov)
        # quadratic1 = torch.einsum('...ij,...kj->...ik', XX, diff)
        quadratic1 = torch.einsum('... s c, ... r d, ... t e -> ... s t', diff, inv_cov, diff.transpose(2,3))
        einsum_times.append(time.time() - start_time)
        # measure matmul
        start_time = time.time()
        quadratic2 = diff @ inv_cov @ diff.transpose(2,3)
        print("quadratic1 shape: ", quadratic1.shape)
        print("quadratic2 shape: ", quadratic2.shape)
        matmul_times.append(time.time() - start_time)
        print(f"run {i} einsum time: {einsum_times[-1]}, matmul time: {matmul_times[-1]}")
        print("quadratic term comparison: ", torch.allclose(quadratic1, quadratic2, rtol=1e-3, atol=1e-5))
        del quadratic1, quadratic2
    print("average einsum time: ", np.mean(einsum_times))
    print("average matmul time: ", np.mean(matmul_times))
    sys.exit(0)





def regression_test():
    """Runs the full comparison test between NumPy and PyTorch implementations."""
    loss_module = MattingLaplacianLoss(eps=EPSILON, win_rad=WIN_RAD)
    #quadratic_timing_test()
    for image_path in IMAGE_PATHS:
        print(f"\n📌 Testing {image_path}")
        img_np = load_and_resize_numpy(image_path)  # Shape: (H, W, C)
        print("shape of img_np: ", img_np.shape)
        img_torch = load_and_resize_torch(image_path)  # Shape: (C, H, W)
        print("shape of img_torch: ", img_torch.shape, "\n")
        #plot_img_heatmap_difference(img_np, img_torch.permute(1, 2, 0).numpy(), *img_torch.shape[-2:])
        #compare_local_statistics(loss_module, img_np, img_torch)
        #sys.exit(0)
        # Compute Matting Laplacian
        laplacian_np = compute_numpy_laplacian(img_np)          # Sparse Tensor (H*W, H*W)
        print()
        laplacian_torch = compute_torch_laplacian(img_torch)    # Dense Tensor (H, W)
        print("shape of laplacian_np: ", laplacian_np.shape)
        print("shape of laplacian_torch: ", laplacian_torch.shape)
        print("Comparing Laplacian Matrices...")
        match = compare_sparse_laplacians(laplacian_np, laplacian_torch)
        if match:
            print("✅ Laplacians match within tolerance!")
        else:
            print("❌ Laplacians differ!")
        sys.exit(0)
        # Apply Laplacian transformation to the image
        # transformed_img_np = apply_laplacian(img_torch, laplacian_np).detach().cpu().numpy()
        # print("shape of transformed_img_np: ", transformed_img_np.shape)
        # transformed_img_torch = apply_laplacian(img_torch, laplacian_torch).detach().cpu().numpy()
        # print("shape of transformed_img_torch: ", transformed_img_torch.shape)

def timing_test_devices():
    """Runs a unit test for the Matting Laplacian."""
    image_path = IMAGE_PATHS[0]
    cpu_times = []
    cuda_times = []
    print(f"\n📌 Testing {image_path}")
    img = load_and_resize_torch(image_path)  # Shape: (C, H, W)
    print("shape of img: ", img.shape, "\n")
    loss_module = MattingLaplacianLoss(eps=EPSILON, win_rad=WIN_RAD)
    img_cpu = img.unsqueeze(0).to(device="cpu")
    img_cuda = img.unsqueeze(0).to(device="cuda")
    # warmup CPU for timing:
    for _ in range(5):
        laplacian_cpu = loss_module.compute_laplacian(img_cpu)
    for _ in range(10):
        start_time = time.time()
        laplacian_cpu = loss_module.compute_laplacian(img_cpu)
        cpu_times.append(time.time() - start_time)
        print("CPU time: ", cpu_times[-1])
    # warmup GPU for timing:
    for _ in range(5):
        laplacian_cuda = loss_module.compute_laplacian(img_cuda)
    for _ in range(10):
        start_time = time.time()
        laplacian_cuda = loss_module.compute_laplacian(img_cuda)
        cuda_times.append(time.time() - start_time)
        print("GPU time: ", cuda_times[-1])
    print("average CPU time: ", np.mean(cpu_times))
    print("average GPU time: ", np.mean(cuda_times))
    #compare_sparse_laplacians(laplacian_cpu[0], laplacian_cuda[0].to(device="cpu"), rtol=5e-1, atol=1e-2)
    #print("comparing Laplacian Matrices: ", torch.allclose(laplacian_cpu[0], laplacian_cuda[0].cpu(), rtol=1e-4, atol=1e-6))

def test_device_parity():
    image_path = IMAGE_PATHS[0]
    print(f"\n📌 Testing {image_path}")
    img = load_and_resize_torch(image_path)  # Shape: (C, H, W)
    print("shape of img: ", img.shape, "\n")
    loss_module = MattingLaplacianLoss(eps=EPSILON, win_rad=WIN_RAD)
    img = img.unsqueeze(0)
    img_shape = tuple(img.shape)
    laplacian_cpu, indices_cpu = loss_module.compute_laplacian(img)
    laplacian_cuda = laplacian_cpu.to(device="cuda")
    indices_cuda = indices_cpu.to(device="cuda")
    result_cpu = loss_module.construct_laplacian_sequential(laplacian_cpu, indices_cpu, img_shape)
    result_cuda = loss_module.construct_laplacian_parallel(laplacian_cuda, indices_cuda, img_shape)
    compare_sparse_laplacians(result_cpu[0], result_cuda[0].to(device="cpu"), rtol=1e-5, atol=1e-8)


def unit_test_matting_laplacian():
    """Runs a unit test for the Matting Laplacian."""
    img = torch.stack([load_and_resize_torch(p) for p in IMAGE_PATHS])  # Shape: (C, H, W)
    print("shape of img: ", img.shape)
    loss_module = MattingLaplacianLoss(eps=EPSILON, win_rad=WIN_RAD)
    laplacian = loss_module.compute_laplacian(img)
    print("shape of laplacian: ", torch.stack(laplacian).shape)


if __name__ == "__main__":
    #regression_test()
    #timing_test_devices()
    #test_device_parity()
    unit_test_matting_laplacian()
