
raise DeprecationWarning("this file is deprecated and will be removed in a future version; use mcapst.train.loss.matting_laplacian instead")

from typing import Union, List, Optional
import torch
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse


# original had no comments and only had the 3 functions `_rolling_block`, `compute_laplacian`, and `laplacian_loss_grad`



def save_win_inds(win_inds, filename="win_inds.pt"):
    """ Saves win_inds as a PyTorch tensor to disk.
        Args:
            win_inds (np.ndarray): The extracted window indices.
            filename (str): Path to save the tensor.
    """
    # Convert NumPy array to PyTorch tensor
    win_inds_tensor = torch.from_numpy(win_inds)
    # Save using PyTorch format
    torch.save(win_inds_tensor.unsqueeze(0), filename)
    print(f"[X] Saved win_inds to {filename}")



def _rolling_block(A, block=(3, 3)):
    """ Applies sliding window to given matrix.
        Args:
            A (ndarray): Input 2D array.
            block (tuple): Size of the sliding window.
        Returns:
            ndarray: A view of the input array with the sliding window applied.
    """
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def extract_patches(img: np.ndarray, mask: Optional[np.ndarray] = None, win_rad: int = 1):
    win_diam = win_rad * 2 + 1  # diameter of the window
    win_size = win_diam ** 2    # number of pixels in the window / window area
    h, w, d = img.shape         # height, width, depth of the image
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    # create an array of indices for the image
    indsM = np.arange(h * w).reshape((h, w))
    # flatten the image to a 2D array of shape (h * w, d)
    ravelImg = img.reshape(h * w, d)
    # apply a sliding window to the indices array
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam))
    win_inds = win_inds.reshape(c_h, c_w, win_size) # Shape: (c_h, c_w, win_size)
    if mask is not None:
        # dilate the mask to include neighboring pixels
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((win_diam, win_diam), np.uint8)).astype(np.bool)
        win_mask = np.sum(mask.ravel()[win_inds], axis=2) # sum mask values within the window
        win_inds = win_inds[win_mask > 0, :]              # keep only windows with non-zero mask values
        # ^ effect on shape above is the same as reshape(-1, win_size)
    else:
        win_inds = win_inds.reshape(-1, win_size)         # Shape: (num_windows, win_size)
    return ravelImg[win_inds], win_inds


def get_local_statistics(img: np.ndarray, mask: Optional[np.ndarray] = None, win_rad: int = 1):
    win_diam = win_rad * 2 + 1  # diameter of the window
    win_size = win_diam ** 2    # number of pixels in the window / window area
    # Number of window centre indices in h, w axes
    winI, win_inds = extract_patches(img, mask, win_rad) # Shape: (num_windows, win_size, d), (num_windows, win_size)
    # compute the mean of each patch
    win_mu = np.mean(winI, axis=1, keepdims=True)         # Shape: (num_windows, 1, d)
    # compute the covariance matrix of each patch (Shape: (num_windows, d, d))
    #! this was previously `'...ji,...jk ->...ik'` before I reorganized things for testing
    #patch_sq_sum = np.einsum('...ij,...ik->...jk', winI, winI) / win_size
    #! didn't make a difference to the underlying problem apparently
    patch_sq_sum = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size
    mean_sq = np.einsum('...ji,...jk ->...ik', win_mu, win_mu)
    win_var = patch_sq_sum - mean_sq
    return win_mu, win_var, winI, win_inds


def compute_laplacian(img: Union[str, np.ndarray], mask: Optional[np.ndarray] = None,
                      eps: float = 10 ** (-7), win_rad: int = 1):
    """Computes Matting Laplacian for a given image.
        Args:
            img: 3-dimensional image array or path to the image file.
            mask: mask of pixels for which Laplacian will be computed. If not set Laplacian will be computed for all pixels.
            eps: regularization parameter controlling alpha smoothness from Eq. 12 of the original paper. Defaults to 1e-7.
            win_rad: radius of window used to build Matting Laplacian (i.e. radius of omega_k in Eq. 12).
        Returns:
            scipy.sparse.coo_matrix: sparse matrix holding Matting Laplacian.
    """
    if type(img) == str:
        # read image and convert to RGB format
        img = cv2.imread(img, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.asarray(img)
    # normalize image to range [0, 1]
    if img.max() > 1.0:
        img = 1.0 * img / 255.0
    win_diam = win_rad * 2 + 1  # diameter of the window
    win_size = win_diam ** 2    # number of pixels in the window / window area
    h, w, d = img.shape         # height, width, depth of the image
    win_mu, win_var, winI, win_inds = get_local_statistics(img, mask, win_rad) # Shape: (num_windows, 1, d), (num_windows, d, d)
    # compute the inverse of the covariance matrix with regularization
    win_var += (eps / win_size) * np.eye(3)
    inv = np.linalg.inv(win_var).astype(np.float32) # Shape: (num_windows, d, d)
    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)     # Shape: (num_windows, win_size, d)
    # compute the values for the sparse matrix (Shape: (num_windows, win_size, win_size))
    vals = (1.0 / win_size) * (1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))
    # create the row and column indices for the sparse matrix
    nz_indsCol = np.tile(win_inds, win_size).ravel()    # Shape: (num_windows * win_size * win_size,)
    nz_indsRow = np.repeat(win_inds, win_size).ravel()  # Shape: (num_windows * win_size * win_size,)
    # create sparse matrix - necessary since it has to pad the elements back to the original image shape using the indices
        # reference final instantiation form in https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        # basically, for coo_matrix((data, (i, j)) and return matrix A, [shape=(M, N)]),    A[i[k], j[k]] = data[k]
    L = scipy.sparse.coo_matrix((vals.ravel(), (nz_indsRow, nz_indsCol)), shape=(h * w, h * w)) # shape (hw, hw)
    # compute the sum of each row in the sparse matrix
    sum_L = L.sum(axis=1).T.tolist()[0]     # length of list: hw
    # subtract the sum from the diagonal elements
    # NOTE: this may or may not be right according to this, but it seems like a workaround to avoid more memory allocation
        # https://people.csail.mit.edu/kaiming/publications/cvpr10matting.pdf
    L = scipy.sparse.diags([sum_L], [0], shape=(h * w, h * w)) - L
    # convert the matrix to COO format and cast to float32
    L = L.tocoo().astype(np.float32)
    return L


def laplacian_loss_grad(image, M):
    """ Computes the Laplacian loss and its gradient for a given image.
        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).
            M (scipy.sparse.coo_matrix): Matting Laplacian matrix.
        Returns:
            tuple: Loss value and gradient tensor of shape (C, H, W).
    """
    laplacian_m = M
    img = image
    channel, height, width = img.shape
    loss = 0
    grads = list()
    for i in range(channel):
        # compute the gradient for the current channel
        grad = torch.mm(laplacian_m, img[i, :, :].reshape(-1, 1))/(height*width) # Shape: (height * width, 1)
        # accumulate loss for the curretn channel
        loss += torch.mm(img[i, :, :].reshape(1, -1), grad)
        grads.append(grad.reshape((height, width))) # Reshape the gradient to (height, width)
    gradient = torch.stack(grads, dim=0)            # Stack the gradients for each channel
    return loss, 2.0*gradient                       # Return the loss and the gradient
