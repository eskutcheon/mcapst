from typing import Dict, List, Literal, Union, Iterable, Callable, Tuple, Any
import torchvision.transforms.v2 as TT
import torch
import kornia.filters as KF



def ensure_batch_tensor(input: Any) -> Any:
    while len(input.shape) < 4:
        input = input.unsqueeze(0)
    return input

def scaling_resize(input: torch.Tensor, max_size: int, interp_mode: TT.InterpolationMode):
    """ scale input so that its longest dimension <= max_size """
    # ? NOTE: maybe not totally necessary if I keep using exclusively square inputs with sides smaller than max
    H, W = input.shape[-2:]
    long_dim = max(H, W)
    # Resize if the larger dimension is greater than max_size
    if long_dim > max_size:
        scale = max_size / float(long_dim)
        H = int(H * scale)
        W = int(W * scale)
    use_antialiasing = interp_mode in [TT.InterpolationMode.BICUBIC, TT.InterpolationMode.BILINEAR]
    return TT.functional.resize(input, [H, W], interp_mode, antialias=use_antialiasing)

def resize_to_batch_tensor(tensor_list, max_size, interp_mode=TT.InterpolationMode.BICUBIC):
    # Step 1: Check if all shapes are equivalent
    shapes = [tensor.shape for tensor in tensor_list]
    if all(shape == shapes[0] for shape in shapes):
        return tensor_list  # All shapes are the same, no need to resize
    # Step 2: Resize all tensors to the maximum size along their largest dimension
    resized_tensors = [scaling_resize(tensor, max_size, interp_mode) for tensor in tensor_list]
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

def iterable_to_tensor(tensor_list: Union[torch.Tensor, Iterable[torch.Tensor]], max_resize: int = 512, is_mask: bool = False,):
    # ? NOTE: think I might really just need to deal with lists of tensors since I might be royally screwing up outcomes based on shapes
    interp_mode = (TT.InterpolationMode.NEAREST if is_mask else TT.InterpolationMode.BICUBIC)
    if not issubclass(type(tensor_list), torch.Tensor):
        # print(f"type of input tensor_list: {type(tensor_list)}")
        tensor_list = resize_to_batch_tensor(tensor_list, max_resize, interp_mode)
        # ? NOTE: just assuming they'll all have the same shape, but I'm checking them all rather than just the first element anyway
        collate_fn: Callable = torch.cat if all([len(A.shape) == 4 for A in tensor_list]) else torch.stack
        try:
            return ensure_batch_tensor(collate_fn(tensor_list, dim=0))
        except:
            dims_list = [list(tensor.shape) for tensor in tensor_list]
            raise RuntimeError("The list of tensors to concatenate must have the same dimensions! Got shapes ", dims_list)
    else:
        return ensure_batch_tensor(tensor_list)


def post_transfer_blending(content_img, pastiche_img, c_weight=0.5, p_weight=0.5):
    """ implement the blending with specified weights to average the original content image and the pastiche image """
    # OR just have two different blending functions for photorealistic and artistic transfer, using an isotropic guided filter for the artistic transfer
    normalizer: callable = TT.ToDtype(torch.float32, scale=True)
    if (weight_sum := c_weight + p_weight) != 1:
        c_weight /= weight_sum
        p_weight /= weight_sum
    return c_weight * normalizer(content_img) + p_weight * normalizer(pastiche_img)


def post_transfer_guided_filter(content_img, pastiche_img, blur_factor=1e-6):
    """ a post-transfer blending function intended more for artistic style transfer to bring it back toward the content image structure """
    content_img = ensure_batch_tensor(content_img)
    pastiche_img = ensure_batch_tensor(pastiche_img)
    # ? NOTE: should be an implementation of this in kornia
    # ? kornia.filters.GuidedBlur or kornia.filters.guided_blur is almost what I intended - set content_img as guidance, kernel_size of 1, and very small eps
    # ? might also make sense to use a bilateral blur based on Kaiming He's recommendation - https://arxiv.org/abs/1505.00996
    # https://kornia.readthedocs.io/en/latest/filters.html
    # TODO: change sampling size to speed it up
    return KF.guided_blur(content_img, pastiche_img, kernel_size=11, eps=blur_factor, subsample=4)