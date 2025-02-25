import os, sys
from typing import Dict, Type
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


@pytest.fixture
def dummy_hf_content():
    return r"bitmind/MS-COCO-unique-256"

@pytest.fixture
def dummy_hf_style():
    return r"huggan/wikiart"  # or "bitmind/MS-COCO-unique-256" for photorealistic transfer


def test_image_training_pipeline(dummy_train_content, dummy_train_style):
    """
    Basic test that runs the ImageTrainer for a few iterations
    and ensures no crash + a checkpoint file is produced.
    """
    if not os.path.isdir(dummy_train_content) or not os.path.isdir(dummy_train_style):
        pytest.skip("No dummy train content/style data found; skipping image training test.")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            modality="image",
            transfer_mode="photorealistic",
            logs_directory=os.path.join(tmpdir, "logs"),
            data_cfg={
                "train_content": dummy_train_content,
                "train_style": dummy_train_style,
                "batch_size": 2,
                "new_size": 256,
                "use_local_datasets": True,
            },
            loss_cfg={
                "style_weight": 1.0,
                "content_weight": 1.0,
                "temporal_weight": 0.0,
                "vgg_ckpt": "checkpoints/vgg_normalised.pth"
            },
            training_iterations=2,  # keep it small for test
            model_save_interval=1,
        )
        print("Config created by image transfer version: ", config)
        # We can call stage_training_pipeline if we want to mimic CLI usage:
        # or directly instantiate ImageTrainer
        trainer = ImageTrainer(config)
        trainer.train()
        # Check that a checkpoint file was produced
        ckpt_path = trainer.config.ckpt_path
        assert os.path.isfile(ckpt_path), f"Expected a checkpoint at {ckpt_path}"


def test_video_training_pipeline(dummy_train_content, dummy_train_style):
    """
    Tests the 'fake flow' approach for video, but uses normal images 
    for content/style. We just confirm that we can run a few steps 
    without error, using temporal_weight > 0. 
    """
    if not os.path.isdir(dummy_train_content) or not os.path.isdir(dummy_train_style):
        pytest.skip("No dummy train content/style data found; skipping video training test.")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            modality="video",
            transfer_mode="photorealistic",
            logs_directory=os.path.join(tmpdir, "logs"),
            data_cfg={
                "train_content": dummy_train_content,
                "train_style": dummy_train_style,
                "batch_size": 3, # 3 is the highest I can use with the current number of content images available
                "new_size": 256,
                "use_local_datasets": True,
            },
            loss_cfg={
                "style_weight": 1.0,
                "content_weight": 1.0,
                "temporal_weight": 0.5,  # non-zero => triggers fake flow
                "vgg_ckpt": "checkpoints/vgg_normalised.pth"
            },
            training_iterations=2,
            model_save_interval=1,
        )
        trainer = VideoTrainer(config)
        trainer.train()
        # Check that a checkpoint file was produced
        ckpt_path = trainer.config.ckpt_path
        assert os.path.isfile(ckpt_path), "Expected checkpoint for video trainer."


def test_image_training_pipeline_dict_override(dummy_hf_content, dummy_hf_style):
    """
    Example of calling stage_training_pipeline with no direct CLI, 
    but building config in code. This is similar to a user passing a YAML file.
    """
    # if not os.path.isdir(dummy_train_content) or not os.path.isdir(dummy_train_style):
    #     pytest.skip("No dummy data found.")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = {
            "modality": "image",
            "transfer_mode": "photo",
            "logs_directory": os.path.join(tmpdir, "logs"),
            "data_cfg": {
                "train_content": dummy_hf_content,
                "train_style": dummy_hf_style,
                "batch_size": 4,
                "new_size": 256,
                "use_local_datasets": False,
                "streaming": True,
            },
            # "bitmind/MS-COCO-unique-256", "train_style": "huggan/wikiart"
            "loss_cfg": {
                "style_weight": 1.0,
                "content_weight": 1.0,
                "temporal_weight": 0.0,
                "vgg_ckpt": "checkpoints/vgg_normalised.pth"
            },
            "training_iterations": 1,
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



def error_assertions(config_dict, error_type: Dict[str, Type[Exception]]):
    try:
        # Construct a TrainingConfig from our dict
        config = TrainingConfig(**config_dict)
        print("Config created by dict override version: ", config)
        # Now run the pipeline. This is analogous to a CLI call with a config file.
        trainer = ImageTrainer(config)
        trainer.train()
        ckpt_path = trainer.config.ckpt_path
        assert os.path.isfile(ckpt_path), "No checkpoint file found after training."
    except Exception as e:
        for key, val in error_type.items():
            assert isinstance(e, val), f"Expected {key} but got {type(e)}: {e}"

def test_image_training_pipeline_errors(dummy_hf_content, dummy_hf_style):
    """ Example of calling stage_training_pipeline with invalid config values """
    # if not os.path.isdir(dummy_train_content) or not os.path.isdir(dummy_train_style):
    #     pytest.skip("No dummy data found.")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = {
            "modality": "image",
            "transfer_mode": "photorealistic",
            "logs_directory": os.path.join(tmpdir, "logs"),
            "data_cfg": {
                "train_content": dummy_hf_content,
                "train_style": dummy_hf_style,
                "batch_size": 1,
                "new_size": 256,
                "use_local_datasets": True,
                "streaming": True,
            },
            # "bitmind/MS-COCO-unique-256", "train_style": "huggan/wikiart"
            "loss_cfg": {
                "style_weight": 1.0,
                "content_weight": 1.0,
                "temporal_weight": 0.0,
                "vgg_ckpt": "checkpoints/vgg_normalised.pth"
            },
            "training_iterations": 2,
            "model_save_interval": 1,
        }
        # expected NotImplementedError from using local datasets with streaming
        error_assertions(config_dict, {"NotImplementedError": NotImplementedError})
        config_dict["data_cfg"]["use_local_datasets"] = False
        config_dict["data_cfg"]["streaming"] = False
        # expected out of memory error from not using streaming with Wikiart
        error_assertions(config_dict, {"OSError": OSError})
        config_dict["data_cfg"]["use_local_datasets"] = True
        config_dict["data_cfg"]["train_content"] = "fake_path/fake_content"
        # expected FileNotFoundError from using a fake local dataset path
        error_assertions(config_dict, {"FileNotFoundError": FileNotFoundError})
        # expected ValueError from using a HuggingFace path with use_local_datasets=True
        config_dict["data_cfg"]["use_local_datasets"] = False
        error_assertions(config_dict, {"ValueError": ValueError})