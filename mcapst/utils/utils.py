import os.path
from typing import Dict, List, Literal, Union, Iterable, Callable, Tuple, Any
import torchvision.transforms.v2 as TT
from PIL import Image
import torch
import torchvision.utils as vutils
import numpy as np

#^#####################################################################
#^ ORIGINAL CAP-VSTNet CODE - will be reused until no longer applicable
#^#####################################################################

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def __write_images(image_outputs, display_image_num, file_name, normalize=False):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=normalize)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix, normalize=False):
    __write_images(image_outputs, display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix), normalize)


def write_one_row_html(html_file, iterations, img_filename, all_size):
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
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -image_save_iterations):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()



def print_params(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue
    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')


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


def load_segment(image_path, size=None):
    def change_seg(seg):
        color_dict = {
            (0, 0, 255): 3,  # blue
            (0, 255, 0): 2,  # green
            (0, 0, 0): 0,  # black
            (255, 255, 255): 1,  # white
            (255, 0, 0): 4,  # red
            (255, 255, 0): 5,  # yellow
            (128, 128, 128): 6,  # grey
            (0, 255, 255): 7,  # lightblue
            (255, 0, 255): 8  # purple
        }
        arr_seg = np.array(seg)
        new_seg = np.zeros(arr_seg.shape[:-1])
        for x in range(arr_seg.shape[0]):
            for y in range(arr_seg.shape[1]):
                if tuple(arr_seg[x, y, :]) in color_dict:
                    new_seg[x, y] = color_dict[tuple(arr_seg[x, y, :])]
                else:
                    min_dist_index = 0
                    min_dist = 99999
                    for key in color_dict:
                        dist = np.sum(np.abs(np.asarray(key) - arr_seg[x, y, :]))
                        if dist < min_dist:
                            min_dist = dist
                            min_dist_index = color_dict[key]
                        elif dist == min_dist:
                            try:
                                min_dist_index = new_seg[x, y - 1, :]
                            except Exception:
                                pass
                    new_seg[x, y] = min_dist_index
        return new_seg.astype(np.uint8)
    if not os.path.exists(image_path):
        print("Can not find image path: %s " % image_path)
        return None
    image = Image.open(image_path).convert("RGB")
    if size is not None:
        w, h = size
        transform = TT.Resize((h, w), interpolation=Image.NEAREST)
        image = transform(image)
    image = np.array(image)
    if len(image.shape) == 3:
        image = change_seg(image)
    return image

#^#####################################################################
#^ END OF ORIGINAL CAP-VSTNet CODE - BEGINNING OF NEW CODE
#^#####################################################################

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


def is_given_filetype(paths, file_type: Literal["image", "encoding"]):
    file_extensions = {"image": {".png", ".jpg", ".jpeg"}, "encoding": {".pt", ".pth"}}
    paths = ensure_file_list_format(paths)
    if all(os.path.splitext(p.lower())[1] in file_extensions[file_type] for p in paths):
        return True


# originally written and applied within src.dataset.transforms.AugmentationFunctionalWrapper
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