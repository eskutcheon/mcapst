# MCAPST: Modular Content Affinity-Preserved Style Transfer

### **A modular re-implementation of** [CAP-VSTNet: Content Affinity Preserved Versatile Style Transfer (CVPR 2023)]((https://arxiv.org/abs/2303.17867))

## Project Overview
This repository reorganizes the original CAP-VSTNet implementation into a pip‑installable Python package named `mcapst`. Major additions include a dataclass-based configuration system, modular training and inference pipelines, and utility classes for processing images or videos. Both training and inference can be invoked from the command line or used as a Python API.

**DISCLAIMER:** This is the first time I've packaged a repository and I'm still learning that process. I also may be more careless with releases until I know anyone besides myself is using it.


![](assets/image_stylization.webp)


### Features (Current + Planned)
- Perform simple style transfer using any provided content and style images
- Train new style transfer networks specialized for artistic or photorealistic style transfer
- Perform style transfer on video inputs with smooth temporal consistency, using any style image(s)
- Apply multiple styles to a single content image with seamless blending
- Use content/style segmentation masks during inference to apply region-based stylization
- Simple data loading pipelines for training and parallelized dispatchers for inference
- [ ] TODO: On-the-fly segmentation mask generation and guidance during style transfer inference
- [ ] TODO: Train video-based models using real optical flow inputs computed by a RAFT model


## Installation
**Install PyTorch first.** Visit the [official instructions](https://pytorch.org/get-started/locally/) and choose either a CUDA build (based on your individual version shown by running `nvcc --version`) or the CPU wheels. Example commands:
```bash
# CPU only
pip install torch torchvision
# or GPU build (replace cu118 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### CURRENTLY
The easiest way to install dependencies at the moment (until I finish packaging) is the typical
```bash
pip install -r requirements.txt # all dependencies for both train and infer modes besides PyTorch
```


---

#### LATER
<font color="red"><b>WARNING: NOT YET IMPLEMENTED WITH BUILD BACKEND</b></font>

With PyTorch available, install `mcapst` from source:
```bash
pip install mcapst            # minimal install
pip install mcapst[train]     # with training extras
pip install mcapst[infer]     # with inference extras
```

---

### Getting Pretrained Checkpoints
Download the pre-trained model weights from [Google Drive](https://drive.google.com/drive/folders/19xlQVprXdPJ9bhfnVEJ1ruVST-NuIlIE?usp=share_link) and place them in a top-level (i.e. `tests` should be at the same level) `checkpoints/` directory. This drive also includes a trained VGG checkpoint used during training to compute the style and content losses.

You will also be prompted to automatically download (if you haven't already) only the necessary default checkpoints before training or inference begins if you specified the default checkpoint paths (or don't specify any path).

>[!NOTE] Planned:
> In the future, I'll most likely be adding the option to default to loading models from the HuggingFace Hub as well.


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
* `--transfer-mode`: (both modes) Either `photorealistic` or `artistic`
* `--modality`: (both) Either `image` or `video` - the data modality during inference or the training approach
* `--use_local_data`: (training) flag specifying whether `train_content` or `train_style` are local paths or not
* `--data-cfg.train_content`: (training) paths to content data; expects a path to a directory of images or a HF link
* `--data-cfg.train_style`: (training) path to style image data; expects a path to a directory of images or a HF link
* `--train_iter`: (training) number of batches to train on (like the original, epochs aren't explicitly used)
* `--batch_size`: (training) integer specifying batch size during training
* `--new_size`: (training) integer specifying the longest dimension allowed during training
* `--ckpt-path`: (both) location of a pre-trained CAP-VSTNet checkpoint or path to save a new one
* `--input-path`: (inference) paths to content data - accepts single file paths, directory paths, or a list of paths
* `--style_paths`: (inference) paths to style images - accepts the same inputs as `input-path`
* `--output-path`: (inference) directory for saving stylized results
* `--alpha-s` / `--alpha-c`: (inference) blending weights for style and content, respectively



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

- Training can be launched in a similar manner using `ImageTrainer`, `VideoTrainer` and the associated `TrainingConfig`.
- Note that this hasn't been fully tested yet and the CLI options are still preferred at the moment.



## Training
Download the VGG19 weights ([Google Drive](https://drive.google.com/drive/folders/19xlQVprXdPJ9bhfnVEJ1ruVST-NuIlIE?usp=share_link)) and place `vgg_normalised.pth` under `checkpoints/`. You can also wait to be prompted to automatically download `vgg_normalised.pth` before training begins in the same way as the pre-trained models for inference.

The original CAP-VSTNet implementation trained model checkpoints (e.g. `photo_image.pt` and `art_video.pt`) using the MS-COCO dataset for content images of both "photorealistic" and "artistic" modes, as well as the style images for "photorealistic" models, and the WikiArt dataset for style images in "artistic" models.

`mcapst` provides a simpler approach to training using remote datasets streamed from Hugging Face via its `datasets` API, but still allows for your own local datasets specified by CLI/config parameters `--train_content` and `--train_style` respectively. The two directories may be the same.

If you would like to download these datasets locally anyway, the following were used by the original CAP-VSTNet authors:
  - [MS_COCO](http://images.cocodataset.org/zips/train2014.zip)
  - [WikiArt](https://www.wikiart.org/)


After initial setup, launch training as shown in [[### CLI]] or build the configuration in Python:
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


## Acknowledgements
- **Credit to [linfengWen98](https://github.com/linfengWen98)** for all image assets used in this README. They'll eventually be replaced by real examples after tracking down source images, re-running style transfer inference, and creating new figures.
- **Credit to the original [CAP-VSTNet](https://github.com/linfengWen98/CAP-VSTNet)** for being the starting point for this new repository. The citation from their original paper is below in [[## Citation]]

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
