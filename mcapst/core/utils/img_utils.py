from typing import Dict, List, Literal, Union, Iterable, Callable, Tuple, Any
import torchvision.transforms.v2 as TT
from torchvision.io import read_image, ImageReadMode
import torch
from pathlib import Path
# will hopefully replace this soon:
from .utils import ensure_file_list_format



def ensure_batch_tensor(img: torch.Tensor) -> torch.Tensor:
    """ ensure that the input tensor is a batch tensor with 4 dimensions by front-padding with a singleton dimension """
    while len(img.shape) < 4:
        img = img.unsqueeze(0)
    return img


def get_scaled_dims(img: torch.Tensor, max_size: int) -> Tuple[int, int]:
    """ get dimensions of input such that longest dimension <= max_size; scales the shorter dimension by the same factor """
    H, W = img.shape[-2:]
    long_dim = max(H, W)
    if long_dim > max_size:
        scale = max_size / float(long_dim)
        H = int(H * scale)
        W = int(W * scale)
    return H, W

def _scaling_resize(img: torch.Tensor, max_size: int, interp_mode: TT.InterpolationMode) -> torch.Tensor:
    """ scale input so that its longest dimension <= max_size """
    H, W = get_scaled_dims(img, max_size)
    use_antialiasing = interp_mode in [TT.InterpolationMode.BICUBIC, TT.InterpolationMode.BILINEAR]
    return TT.functional.resize(img, [H, W], interp_mode, antialias=use_antialiasing)


def downscaling_resize(img: torch.Tensor, max_size: int, downscale: int = 4) -> torch.Tensor:
    """ scale img so that its longest dimension <= max_size and again by a constant factor for input into the reversible network for multi-scale feature extraction """
    H, W = img.shape[-2:]
    H_new, W_new = get_scaled_dims(img, max_size)
    # Adjust to make dimensions multiples of down_scale - usually 4 for nStrides = [1, 2, 2]
    if (H_new % downscale != 0 or W_new % downscale != 0): # only bother resizing if they're not already multiples of downscale
        H_new = (H_new//downscale)*downscale # same as H -= (H % down_scale) essentially
        W_new = (W_new//downscale)*downscale
    # if the dimensions have changed, resize the img tensor
    if H != H_new or W != W_new:
        return TT.functional.resize(img, [H_new, W_new], TT.InterpolationMode.BICUBIC, antialias=True)
    return img



def _resize_to_batch_tensor(tensor_list, max_size, interp_mode=TT.InterpolationMode.BICUBIC) -> List[torch.Tensor]:
    # Step 1: Check if all shapes are equivalent
    shapes = [tensor.shape for tensor in tensor_list]
    if all(shape == shapes[0] for shape in shapes):
        return tensor_list  # All shapes are the same, no need to resize
    # Step 2: Resize all tensors to the maximum size along their largest dimension
    resized_tensors = [_scaling_resize(tensor, max_size, interp_mode) for tensor in tensor_list]
    # Step 3: Compute the average size along the last two dimensions
    mean_dims = torch.mean(torch.Tensor([A.shape[-2:] for A in resized_tensors]), dim=1).to(dtype=torch.int32)
    # TODO: add try catch block here for ensuring mean_dims is the right size for this
    common_size = tuple(mean_dims.tolist())
    # Step 4: Interpolate all tensors to the average size
    use_antialiasing = interp_mode in [TT.InterpolationMode.BICUBIC, TT.InterpolationMode.BILINEAR]
    return [
        TT.functional.resize(tensor, size=common_size, interpolation=interp_mode, antialias=use_antialiasing)
        for tensor in resized_tensors
    ]

def iterable_to_tensor(
    tensor_list: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_resize: int = 512, is_mask: bool = False
) -> torch.Tensor:
    # ? NOTE: think I might really just need to deal with lists of tensors since I might be royally screwing up outcomes based on shapes
    interp_mode = TT.InterpolationMode.NEAREST if is_mask else TT.InterpolationMode.BICUBIC
    if not issubclass(type(tensor_list), torch.Tensor):
        tensor_list = _resize_to_batch_tensor(tensor_list, max_resize, interp_mode)
        # ? NOTE: just assuming they'll all have the same shape, but I'm checking them all rather than just the first element anyway
        collate_fn: Callable = torch.cat if all([len(A.shape) == 4 for A in tensor_list]) else torch.stack
        try:
            return ensure_batch_tensor(collate_fn(tensor_list, dim=0))
        except:
            dims_list = [list(tensor.shape) for tensor in tensor_list]
            # something went wrong if this runs, because this is just a redundant check after running resize_to_batch_tensor
            raise RuntimeError("The list of tensors to concatenate must have the same dimensions! Got shapes ", dims_list)
    return ensure_batch_tensor(tensor_list)


def post_transfer_blending(content_img, pastiche_img, c_weight=0.5, p_weight=0.5):
    """ implement the blending with specified weights to average the original content image and the pastiche image """
    # OR just have two different blending functions for photorealistic and artistic transfer, using an isotropic guided filter for the artistic transfer
    normalizer: callable = TT.ToDtype(torch.float32, scale=True)
    weight_sum = c_weight + p_weight # removed use of the walrus operator for earlier Python versions
    if weight_sum != 1:
        c_weight /= weight_sum
        p_weight /= weight_sum
    return c_weight * normalizer(content_img) + p_weight * normalizer(pastiche_img)



# commenting this out to remove the kornia dependency since the function is unused, but it may be worth adding back later

# import kornia.filters as KF

# def post_transfer_guided_filter(content_img, pastiche_img, blur_factor=1e-6):
#     """ a post-transfer blending function intended more for artistic style transfer to bring it back toward the content image structure """
#     content_img = ensure_batch_tensor(content_img)
#     pastiche_img = ensure_batch_tensor(pastiche_img)
#     # ? NOTE: should be an implementation of this in kornia
#     # ? kornia.filters.GuidedBlur or kornia.filters.guided_blur is almost what I intended - set content_img as guidance, kernel_size of 1, and very small eps
#     # ? might also make sense to use a bilateral blur based on Kaiming He's recommendation - https://arxiv.org/abs/1505.00996
#     # https://kornia.readthedocs.io/en/latest/filters.html
#     # TODO: change sampling size to speed it up
#     return KF.guided_blur(content_img, pastiche_img, kernel_size=11, eps=blur_factor, subsample=4)
