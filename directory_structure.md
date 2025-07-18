mcapst/
├── .env/
├── .hypothesis/
├── .pytest_cache/
├── .vscode/
├── assets/
├── checkpoints/
|   ├── segmentation/
|   |   ├── ade20k_palette.npy
|   |   └── ade20k_semantic_rel.npy
|   ├── 2025-02-24_15-39-21.pt
|   ├── art_image.pt
|   ├── art_video.pt
|   ├── photo_image.pt
|   ├── photo_video.pt
|   └── vgg_normalised.pth
├── data/
|   ├── content/
|   |   ├── 01.jpg
|   |   ├── 02.jpg
|   |   ├── 03.avi
|   |   ├── 04.avi
|   |   ├── 05.jpg
|   |   └── 06.avi
|   ├── style/
|   |   ├── 01.jpg
|   |   ├── 02.png
|   |   ├── 03.jpeg
|   |   ├── 04.jpg
|   |   ├── 05.jpg
|   |   ├── 06.png
|   |   └── starrynightcity.jpg
|   ├── style_masks/
|   └── test_output/
├── examples/
├── logs/
├── mcapst/
|   ├── config/
|   |   ├── configure.py
|   |   ├── default_config.yaml
|   |   └── types.py
|   ├── datasets/
|   |   ├── datasets.py
|   |   ├── orchestrator.py
|   |   └── video_processor.py
|   ├── loss/
|   |   ├── manager.py
|   |   ├── matting_laplacian.py
|   |   └── temporal_loss.py
|   ├── models/
|   |   ├── segmentation/
|   |   |   └── SegReMapping.py
|   |   ├── __init__.py
|   |   ├── CAPVSTNet.py
|   |   ├── containers.py
|   |   ├── cWCT.py
|   |   ├── RevResNet.py
|   |   └── VGG.py
|   ├── pipelines/
|   |   ├── __init__.py
|   |   ├── dispatcher.py
|   |   ├── infer.py
|   |   └── train.py
|   ├── stylizers/
|   |   ├── base_stylizers.py
|   |   ├── image_stylizers.py
|   |   └── video_stylizers.py
|   ├── utils/
|   |   ├── img_utils.py
|   |   ├── label_remapping.py
|   |   ├── MattingLaplacian.py
|   |   ├── TemporalLoss.py
|   └── __init__.py
├── results/
├── scripts/
├── tests/
|   ├── test_inference_pipeline.py
|   ├── test_laplacian_regression.py
|   ├── test_stylizers.py
|   ├── test_train_pipeline_manual.py
|   ├── test_training.py
|   └── test_training_pipeline.py
├── .gitignore
├── directory_structure.md
├── LICENSE
├── README.md
├── setup.py