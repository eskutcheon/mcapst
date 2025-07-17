# CAP-VSTNet: Content Affinity Preserved Versatile Style Transfer (CVPR 2023)

### [**Paper**](https://arxiv.org/abs/2303.17867) | [**Poster**](https://cvpr2023.thecvf.com/media/PosterPDFs/CVPR%202023/22374.png?t=1685361737.0440488) | [**Video 1**](https://youtu.be/Mks9_xQNE_8) | [**Video 2**](https://youtu.be/OTJ1wEe29Hc)

<!-- teaser image placeholder -->
![](assets/image_stylization.webp)

## Project Overview
This repository reorganizes the original CAP-VSTNet implementation into a pip‑installable Python package named `mcapst`. Major additions include a dataclass-based configuration system, modular training and inference pipelines, and utility classes for processing images or videos. Both training and inference can be invoked from the command line or used as a Python API. Note that this is the first time I've packaged a repository and there may be mistakes along the way.


## Installation
Install PyTorch first. Visit the [official instructions](https://pytorch.org/get-started/locally/) and choose either a CUDA build (based on your individual `nvcc` version) or the CPU wheels. Example commands:
```bash
# GPU build (replace cu118 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# or CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```



<b><font color="red">WARNING: NOT YET IMPLEMENTED WITH BUILD BACKEND</font></b>


<b><font color="red">IN THE FUTURE:</font></b> With PyTorch available, install `mcapst` from source:
```bash
pip install mcapst            # minimal install
pip install mcapst[train]     # with training extras
pip install mcapst[infer]     # with inference extras
```

### Getting Pretrained Checkpoints
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/19xlQVprXdPJ9bhfnVEJ1ruVST-NuIlIE?usp=share_link) and place them in a local `checkpoints/` directory.

**COMING SOON**: setup scripts to download and organize pre-trained checkpoints and outside dependencies automatically.



## Usage Examples
The package exposes command line interfaces for training and inference as well as Python classes for programmatic use.

### CLI
Run inference on an image:
```bash
python -m mcapst.infer \
  --modality image \
  --input-path data/content/01.jpg \
  --output-path results \
  --transfer-mode photorealistic \
  --ckpt-path checkpoints/photo_image.pt \
  --alpha-s 0.7
```

Train on a pair of content and style image datasets:
```bash
python -m mcapst.train \
  --modality image \
  --transfer-mode artistic \
  --data-cfg.train_content path/to/content \
  --data-cfg.train_style path/to/style \
  --logs-directory logs/run1
```

Important flags:
* `--transfer-mode`: `photorealistic` or `artistic`.
* `--modality`: `image` or `video`.
* `--input-path` / `--data-cfg.train_content`: paths to content data.
* `--data-cfg.train_style`: path to style data (training only).
* `--ckpt-path`: location of a pretrained checkpoint.
* `--output-path`: directory for results.
* `--alpha-s` / `--alpha-c`: blending weights for style and content.
* `--max-size`: maximum spatial resolution.


### Python API
```python
from mcapst.infer import ImageInferenceOrchestrator, InferenceConfig

cfg = InferenceConfig(modality="image",
                      input_path="data/content/01.jpg",
                      output_path="results",
                      transfer_mode="photorealistic")
runner = ImageInferenceOrchestrator(cfg)
runner.run_inference()
```

Training can be launched in a similar manner using `ImageTrainer`, `VideoTrainer` and the associated `TrainingConfig`.


## Training
Download the VGG19 weights ([Google Drive](https://drive.google.com/drive/folders/19xlQVprXdPJ9bhfnVEJ1ruVST-NuIlIE?usp=share_link)) and place `vgg_normalised.pth` under `checkpoints/`. This will likely be replaced with setup scripts to do this automatically in the near future.

The original CAP-VSTNet implementation trained model checkpoints (e.g. the `photo_image.pt` and `art_video.pt`) using the MS-COCO dataset for content images of both "photorealistic" and "artistic" modes, as well as the style images for "photorealistic" models, and the WikiArt dataset for style images in "artistic" models.

`mcapst` provides a simpler approach to training using remote datasets streamed from Hugging Face via its `datasets` API, but still allows for your own local datasets specified by CLI/config parameters `--train_content` and `--train_style` respectively. The two folders may be the same.
If you would like to download these datasets locally anyway, the following were used by the original CAP-VSTNet authors:
  - [MS_COCO](http://images.cocodataset.org/zips/train2014.zip)
  - [WikiArt](https://www.wikiart.org/)


After initial setup, launch training with the CLI shown in [[###CLI]] or build the configuration in Python:
```python
from mcapst.train import ImageTrainer, TrainingConfig

cfg = TrainingConfig(modality="image",
                     data_cfg={"train_content": "path/to/content",
                               "train_style": "path/to/style"})
trainer = ImageTrainer(cfg)
trainer.train()
```

Training logs are saved inside the directory specified by `logs_directory` in the configuration (default: `logs/`). This is subject to change in the future, as the Tensorboard logging hasn't been updated from the original CAP-VSTNet repo.

New trained model checkpoints are saved in the `checkpoints/` directory as `.pt` files with time-stamped filenames by default.


### Inference

**COMING SOON**



## Results
### Video Style Transfer
* Photorealistic video stylization and temporal error heatmap

<div align="center">
<img src=assets/photorealistic_video.webp/>
</div>

* Artistic video stylization and temporal error heatmap

<div align="center">
<img src=assets/artistic_video.webp/>
</div>


### Style Interpolation
* Photorealistic style interpolation

![](assets/photo_interpolation.png)

* Artistic style interpolation

![](assets/art_interpolation.png)


## Remaining Issues
The original repo mentions remaining issues that were never completely addressed in the new implementation, partially because my own motivation in re-implementing much of this project was to use (specifically content-affinity preserving) style transfer as a training-time augmentation while training my semantic segmentation networks.

### Issues Inherited from CAP-VSTNet

1. Flow, Temporal Loss and Heatmap
   - See [issues#11](https://github.com/linfengWen98/CAP-VSTNet/issues/11#issuecomment-1749932696)
   - In the future, I hoped to integrate a small [RAFT](https://docs.pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html) model to predict optical flow






## Citation
```
@inproceedings{wen2023cap,
  title={CAP-VSTNet: Content Affinity Preserved Versatile Style Transfer},
  author={Wen, Linfeng and Gao, Chengying and Zou, Changqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18300--18309},
  year={2023}
}
```
