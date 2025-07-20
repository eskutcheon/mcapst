
import os.path
from typing import Literal, Union, Iterable, List, Sequence
# import torchvision.transforms.v2 as TT
from PIL import Image
import torch
import torchvision.utils as TVU


#^#################################################################################################
#^ ORIGINAL CAP-VSTNet CODE - will be reused until no longer applicable
#^#################################################################################################
#& keep next 2 image writing functions for regenerating the same 
def _write_images(image_outputs, display_image_num, file_name, normalize=False):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = TVU.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=normalize)
    TVU.save_image(image_grid, file_name, nrow=1)

def write_2images(image_outputs, display_image_num, image_directory, postfix, normalize=False):
    _write_images(image_outputs, display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix), normalize)


#& next 2 functions still needed for the HTML report generation by tensorboard after rewiring it or switching loggers (MLFlow, Neptune, etc.)
def _write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations, img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return

def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="60">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    _write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -image_save_iterations):
        if j % image_save_iterations == 0:
            _write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()

#& unused but kept for reference - might implement other learning later schedulers later
def adjust_learning_rate(optimizer, lr, lr_decay, iteration_count):
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def img_resize(img, max_size, down_scale=None):
    w, h = img.size
    if max(w, h) > max_size:
        w = int(1.0 * img.size[0] / max(img.size) * max_size)
        h = int(1.0 * img.size[1] / max(img.size) * max_size)
        img = img.resize((w, h), Image.BICUBIC)
    if down_scale is not None:
        w = w // down_scale * down_scale
        h = h // down_scale * down_scale
        img = img.resize((w, h), Image.BICUBIC)
    return img

#^#################################################################################################
#^ END OF ORIGINAL CAP-VSTNet CODE - BEGINNING OF NEW CODE
#^#################################################################################################



def ensure_list_format(str_list: Union[str, Iterable[str]]):
    if isinstance(str_list, str):
        str_list = [str_list]
    elif isinstance(str_list, (list, tuple, set)) and all([isinstance(s, str) for s in str_list]):
        str_list = list(str_list)
    else:
        raise ValueError("`paths` must be a string or an iterable of strings.")
    return str_list


def ensure_file_list_format(paths: Union[str, Iterable[str]]):
    paths = ensure_list_format(paths)
    if not all([os.path.isfile(p) for p in paths]):
        raise ValueError(f"ERROR: All style input paths must be existing files; got \n\t{paths}")
    return paths



def get_user_confirmation(prompt: str) -> bool:
    answers = {'y': True, 'n': False}
    response = input(f"{prompt} [Y/n] ").lower()
    while response not in answers:
        print("Invalid input. Please enter 'y' or 'n' (not case sensitive).")
        response = input(f"{prompt} [Y/n] ").lower()
    return answers[response]

#& unused but I just feel like there may be a need for it later
def _replace_dir_with_files(paths: List[str], supported_ext: Sequence[str]) -> List[str]:
    """ if any path is a directory, remove it from the list and add all its files to the list instead """
    if any(os.path.isdir(p) for p in paths):
        to_pop = [i for i, p in enumerate(paths) if os.path.isdir(p)]
        for i in reversed(to_pop):
            p = paths.pop(i)
            paths.extend(_get_full_path_list(p, supported_ext))
    return paths

def _get_full_path_list(dir_path: str, supported_ext: Sequence[str]) -> List[str]:
    file_list = []
    for p in os.listdir(dir_path):
        full_path = os.path.normpath(os.path.join(dir_path, p))
        if os.path.isfile(full_path) and os.path.splitext(full_path)[1].lower() in supported_ext:
            file_list.append(full_path)
    return file_list

def validate_path_arg(paths: Union[str, List[str]], arg_name: str, supported_ext: Sequence[str]) -> Union[str, List[str]]:
    """ Validates a path argument, ensuring it exists and is a file or directory as expected. """
    if not paths:
        raise ValueError(f"{arg_name} cannot be empty or None.")
    assert isinstance(paths, (str, list)), f"{arg_name} must be a string or a list of strings."
    if isinstance(paths, list):
        if not all(isinstance(p, str) for p in paths):
            raise ValueError(f"{arg_name} must be a list of strings.")
        if any(not os.path.exists(p) for p in paths):
            raise FileNotFoundError(f"One or more paths in {arg_name} do not exist.")
        # filter the list to only include files (not directories) with the correct extensions
        paths = [p for p in paths if os.path.isfile(p) and os.path.splitext(p)[1].lower() in supported_ext]
    else:
        if not os.path.exists(paths):
            raise FileNotFoundError(f"{arg_name} '{paths}' does not exist.")
        if os.path.isdir(paths):
            # if it's a directory, list all files in it
            paths = _get_full_path_list(paths, supported_ext)
    return paths



# originally written and applied within another repo to avoid GPU memory issues for certain operations - may be useful later
def cpu_wrapper(compute_on_cpu):
    def decorator(func):
        def wrapper(tensor, *args, **kwargs):
            if compute_on_cpu:
                try:
                    device_init = tensor.device
                    # ? NOTE: doing this because for some reason, a torvision.tv_tensors.Mask type gets converted to a torch.Tensor type when moved to the CPU
                    # apparently this works but not tensor.cpu(), so I'm guessing only the .to method is defined for tv_tensors
                    tensor = tensor.to(device="cpu")
                except AttributeError:
                    raise ValueError("The first argument to the function must be a tensor!")
                tensor = func(tensor, *args, **kwargs)
                # move back to original device
                return tensor.to(device=device_init)
            else:
                # if compute_on_cpu = False, just call the function directly
                return func(tensor, *args, **kwargs)
        return wrapper
    return decorator



def target_equals_benchmark(target: torch.Tensor, filename: str, exact = True, rtol=1e-3, atol=1e-5, names = ["SOURCE", "TARGET"]) -> bool:
    """ Loads saved tensors and compares with the target tensor
        Args:
            target (torch.Tensor): Extracted indices from PyTorch
            filename (str): Path to saved NumPy-based `source.pt`
    """
    # TODO: should probably separate loading the source tensor and comparing into separate functions
    # load the saved indices
    source: torch.Tensor = torch.load(filename, weights_only=True) # TODO: add check for loaded type
    # ensure shapes match
    if source.ndim != target.ndim or source.shape != target.shape:
        print(f"Shape Mismatch: {names[0]} {source.shape}, {names[1]} {target.shape}")
        return False
    # check if values match
    is_equal = False
    try:
        if exact:
            is_equal = torch.equal(source, target)
            out_str = f"{names[1]} {'matches' if is_equal else 'does NOT match'} {names[0]}!"
        else:
            is_equal = torch.allclose(source, target, rtol=rtol, atol=atol)
            out_str = f"{names[1]} {'is close to' if is_equal else 'is NOT close to'} {names[0]} within rtol={rtol}, atol={atol}!"
    except Exception as e:
        out_str = f"Error comparing tensors: {e}"
    print(out_str)
    return is_equal
