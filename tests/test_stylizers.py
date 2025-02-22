import torch
import torchvision.transforms.v2 as TT
import torchvision.io as IO
# from torchvision.utils import make_grid
import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import time
from time import perf_counter
from tqdm import tqdm
import yaml

from mcapst.data.managers import BaseImageStylizer, BaseVideoStylizer


# TODO: define this in the YAML later
project_root = os.path.abspath(os.path.join(__file__, "..", ".."))
ckpt_dir = os.path.join(project_root, "checkpoints")
style_img_dir = os.path.join(project_root, "data", "style")
content_img_dir = os.path.join(project_root, "data", "content")



preprocessing = TT.Compose([
    TT.ToDtype(torch.float32, scale=True),
    TT.Lambda(lambda x: x.unsqueeze(0)),
    TT.Lambda(lambda x: x.to("cuda"))]
)

def time_unit_test(func: callable, type="photo"):
    start = perf_counter()
    for _ in tqdm(range(10)):
        func(type)
    print(f"average (n=10) time to load style encoding and perform style transfer: {(perf_counter() - start)/10}")


def get_file_basename(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def test_img_inference(transfer_type="photo"):
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{transfer_type}_image.pt")
    content_img_paths = [os.path.join(content_img_dir, p) for p in os.listdir(content_img_dir) if p.endswith(".jpg")]
    alpha_tests = [None, 0.1] if transfer_type == "photo" else [None, 0.1]
    style_filenames = sorted([p for p in os.listdir(style_img_dir) if os.path.isfile(os.path.join(style_img_dir, p))])
    # create style transfer manager with basic settings
    manager = BaseImageStylizer(transfer_type, ckpt_path, 960)
    for idx, style_filename in enumerate(style_filenames[:3]):
        style_img_path = os.path.join(style_img_dir, style_filename)
        if not os.path.isfile(style_img_path):
            print(f"file '{style_img_path}' not found; skipping...")
            continue
        for content_path in content_img_paths:
            content_basename = get_file_basename(content_path)
            style_basename = get_file_basename(style_img_path)
            content_img = IO.read_image(content_path, mode=IO.ImageReadMode.RGB)
            content_img = preprocessing(content_img)
            for idx, alpha in enumerate(alpha_tests):
                alpha_name = 0.0 if alpha is None else alpha
                output_filename = f"{transfer_type}_s-{style_basename}_c-{content_basename}_alpha-{alpha_name}.png"
                output_path = os.path.join(output_dir, output_filename)
                #style_alpha = 1 - alpha if alpha is not None else None
                style_alpha = 1
                pastiche = manager.transform(content_img, style_img_path, alpha_c=alpha, alpha_s=style_alpha)
                pastiche = pastiche.mul(255).clamp(0,255).byte().cpu()
                IO.write_png(pastiche, output_path, compression_level=2)
                time.sleep(1.0)


def test_multistyle_img_inference(transfer_type="photo"):
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{transfer_type}_image.pt")
    content_img_paths = [os.path.join(content_img_dir, p) for p in os.listdir(content_img_dir) if p.endswith(".jpg")]
    # NOTE: tested various alpha values and it seemingly catches invalid input fine
    alpha_tests = [None, 0.1] if transfer_type == "photo" else [None, 0.1]
    style_filenames = sorted([p for p in os.listdir(style_img_dir) if os.path.isfile(os.path.join(style_img_dir, p))])
    # create style transfer manager with basic settings
    manager = BaseImageStylizer(transfer_type, ckpt_path, 960)
    style_filenames = style_filenames[:2]
    style_img_paths = [os.path.join(style_img_dir, p) for p in style_filenames]
    for content_path in content_img_paths:
        content_basename = get_file_basename(content_path)
        style_basenames = [get_file_basename(p) for p in style_img_paths]
        content_img = IO.read_image(content_path, mode=IO.ImageReadMode.RGB)
        content_img = preprocessing(content_img)
        for idx, alpha in enumerate(alpha_tests):
            alpha_name = 0.0 if alpha is None else alpha
            output_filename = f"{transfer_type}_s-{'+'.join(style_basenames)}_c-{content_basename}_alpha-{alpha_name}.png"
            output_path = os.path.join(output_dir, output_filename)
            #style_alpha = 1 - alpha if alpha is not None else None
            style_alpha = None
            pastiche = manager.transform(content_img, style_img_paths, alpha_c=alpha, alpha_s=style_alpha)
            pastiche = pastiche.mul(255).clamp(0,255).byte().cpu()
            IO.write_png(pastiche, output_path, compression_level=2)
            time.sleep(1.0)


def test_video_inference(transfer_type="photo"):
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{transfer_type}-stylized_video.mp4")
    ckpt_path = os.path.join(ckpt_dir, f"{transfer_type}_video.pt")
    content_img_paths = [os.path.join(content_img_dir, p) for p in os.listdir(content_img_dir) if p.endswith(".avi")]
    content_img_paths = [content_img_paths[-1]]
    alpha_tests = [None] if transfer_type == "photo" else [0.1]
    style_filenames = sorted([p for p in os.listdir(style_img_dir) if os.path.isfile(os.path.join(style_img_dir, p))])
    style_filenames = [style_filenames[2]]
    # create style transfer manager with basic settings
    manager = BaseVideoStylizer(transfer_type, ckpt_path, 720, fps=20)
    for idx, style_filename in enumerate(style_filenames):
        style_img_path = os.path.join(style_img_dir, style_filename)
        for content_path in content_img_paths:
            content_basename = get_file_basename(content_path)
            style_basename = get_file_basename(style_img_path)
            for idx, alpha in enumerate(alpha_tests):
                alpha_name = 0.0 if alpha is None else alpha
                style_alpha = 1
                pastiche = manager.transform(content_path, style_img_path, alpha_c=alpha, alpha_s=style_alpha, output_path=output_path)
                time.sleep(1.0)


def test_major_refactor_inference():
    from mcapst.pipelines.infer import stage_inference_pipeline
    # Define inference config
    config_path = "temp_inference.yaml"
    config = {
        "base_name": "test_infer",
        "transfer_mode": "photorealistic",
        "input_path": "data/content/01.jpg",
        "output_path": "data/test_output",
        "alpha_c": 0.5,
        "alpha_s": 0.5,
        "use_segmentation": False,
        "mask_path": None
    }
    # Save configuration
    with open(config_path, "w") as file:
        yaml.dump(config, file)
    # Run inference
    stage_inference_pipeline(mode="inference", config_path=config_path)



#test_img_inference("photo")
#test_multistyle_img_inference("photo")
#test_video_inference("photo")
test_major_refactor_inference()