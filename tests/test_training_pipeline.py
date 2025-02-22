import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import tempfile
import pytest
import torch
# local imports
from mcapst.config.configure import TrainingConfig
from mcapst.pipelines.train import stage_training_pipeline, ImageTrainer, VideoTrainer


@pytest.fixture
def dummy_train_content():
    # In your real project, you'd have a small folder with a couple of images
    # for content, or set up a "mock" dataset. We assume "tests/assets/train_content/" exists.
    return r"data/content"

@pytest.fixture
def dummy_train_style():
    # Similarly, a small folder for style images, e.g. "tests/assets/train_style/"
    return r"data/style"  # or "tests/assets/train_style" for a have a mock dataset


# def test_image_training_pipeline(dummy_train_content, dummy_train_style):
#     """
#     Basic test that runs the ImageTrainer for a few iterations
#     and ensures no crash + a checkpoint file is produced.
#     """
#     if not os.path.isdir(dummy_train_content) or not os.path.isdir(dummy_train_style):
#         pytest.skip("No dummy train content/style data found; skipping image training test.")

#     with tempfile.TemporaryDirectory() as tmpdir:
#         config = TrainingConfig(
#             modality="image",
#             transfer_mode="photorealistic",
#             logs_directory=os.path.join(tmpdir, "logs"),
#             data_cfg={
#                 "train_content": dummy_train_content,
#                 "train_style": dummy_train_style,
#                 "batch_size": 1,
#                 "new_size": 256,
#                 "use_local_datasets": False,
#             },
#             loss_cfg={
#                 "style_weight": 1.0,
#                 "content_weight": 1.0,
#                 "temporal_weight": 0.0,
#                 "vgg_ckpt": "checkpoints/vgg_normalised.pth"
#             },
#             training_iterations=2,  # keep it small for test
#             model_save_interval=1,
#         )
#         print("Config created by image transfer version: ", config)
#         # We can call stage_training_pipeline if we want to mimic CLI usage:
#         # or directly instantiate ImageTrainer
#         trainer = ImageTrainer(config)
#         trainer.train()
#         # Check that a checkpoint file was produced
#         ckpt_path = trainer.config.ckpt_path
#         assert os.path.isfile(ckpt_path), f"Expected a checkpoint at {ckpt_path}"


# def test_video_training_pipeline(dummy_train_content, dummy_train_style):
#     """
#     Tests the 'fake flow' approach for video, but uses normal images 
#     for content/style. We just confirm that we can run a few steps 
#     without error, using temporal_weight > 0. 
#     """
#     if not os.path.isdir(dummy_train_content) or not os.path.isdir(dummy_train_style):
#         pytest.skip("No dummy train content/style data found; skipping video training test.")

#     with tempfile.TemporaryDirectory() as tmpdir:
#         config = TrainingConfig(
#             modality="video",
#             transfer_mode="photorealistic",
#             logs_directory=os.path.join(tmpdir, "logs"),
#             data_cfg={
#                 "train_content": dummy_train_content,
#                 "train_style": dummy_train_style,
#                 "batch_size": 1,
#                 "new_size": 256,
#                 "use_local_datasets": False,
#             },
#             loss_cfg={
#                 "style_weight": 1.0,
#                 "content_weight": 1.0,
#                 "temporal_weight": 0.5,  # non-zero => triggers fake flow
#                 "vgg_ckpt": "checkpoints/vgg_normalised.pth"
#             },
#             training_iterations=2,
#             model_save_interval=1,
#         )
#         print("Config created by video transfer version: ", config)
#         trainer = VideoTrainer(config)
#         trainer.train()
#         # Check that a checkpoint file was produced
#         ckpt_path = trainer.config.ckpt_path
#         assert os.path.isfile(ckpt_path), "Expected checkpoint for video trainer."


def test_stage_training_pipeline_dict_override(dummy_train_content, dummy_train_style):
    """
    Example of calling stage_training_pipeline with no direct CLI, 
    but building config in code. This is similar to a user passing a YAML file.
    """
    if not os.path.isdir(dummy_train_content) or not os.path.isdir(dummy_train_style):
        pytest.skip("No dummy data found.")

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = {
            "modality": "image",
            "transfer_mode": "artistic",
            "logs_directory": os.path.join(tmpdir, "logs"),
            "data_cfg": {
                "train_content": dummy_train_content,
                "train_style": dummy_train_style,
                "batch_size": 1,
                "new_size": 256,
                "use_local_datasets": False,
            },
            "loss_cfg": {
                "style_weight": 1.0,
                "content_weight": 1.0,
                "temporal_weight": 0.0,
                "vgg_ckpt": "checkpoints/vgg_normalised.pth"
            },
            "training_iterations": 2,
            "model_save_interval": 1,
        }
        # Construct a TrainingConfig from our dict
        config = TrainingConfig(**config_dict)
        print("Config created by dict override version: ", config)

        # Now run the pipeline. This is analogous to a CLI call with a config file.
        trainer = ImageTrainer(config)
        trainer.train()

        ckpt_path = trainer.config.ckpt_path
        assert os.path.isfile(ckpt_path), "No checkpoint file found after training."
