import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
# import shutil
import tempfile
import pytest
# local imports
from mcapst.infer.config.config import InferenceConfig
from mcapst.infer.infer import stage_inference_pipeline, ImageInferenceOrchestrator, VideoInferenceOrchestrator


@pytest.fixture
def sample_image_path():
    return r"data/content/01.jpg"

@pytest.fixture
def sample_video_path():
    return r"data/content/03.avi"


def test_image_inference_pipeline_cli(sample_image_path):
    """ tests entire pipeline in image mode """
    if not os.path.isfile(sample_image_path):
        pytest.skip("No sample image file found; skipping image inference test.")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = InferenceConfig(
            modality="image",
            input_path=sample_image_path,
            output_path=tmpdir,
            alpha_s=0.7,
            transfer_mode="photorealistic",  # or "artistic"
        )
        # pass directly to stage_inference_pipeline as a dict or call stage_inference_pipeline() with a YAML file
        results = ImageInferenceOrchestrator(config).run_inference()
        # check that results is a list of stylized images
        assert len(results) > 0, "No images returned by inference."
        # check that a file was written
        out_files = os.listdir(tmpdir)
        assert len(out_files) > 0, "No output files created in the temp directory."


def test_video_inference_pipeline_cli(sample_video_path):
    """ tests entire pipeline in 'video' mode """
    if not os.path.isfile(sample_video_path):
        pytest.skip("No sample video file found; skipping video inference test.")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = InferenceConfig(
            modality="video",
            input_path=sample_video_path,
            output_path=tmpdir,
            alpha_s=0.5,
            transfer_mode="artistic",
        )
        results = VideoInferenceOrchestrator(config).run_inference()
        out_files = os.listdir(tmpdir)
        assert len(out_files) > 0, "No output files created for video inference test."
        # If `save_output=False`, you'd check that `results` is non-empty.


def test_stage_inference_pipeline_dict_override(sample_image_path):
    """ Uses stage_inference_pipeline() directly like CLI, but we pass the config at runtime rather than from a YAML """
    if not os.path.isfile(sample_image_path):
        pytest.skip("No sample image file found.")
    with tempfile.TemporaryDirectory() as tmpdir:
        # can feed a dict via InferenceConfig's constructor:
        config_dict = {
            "modality": "image",
            "input_path": sample_image_path,
            "output_path": tmpdir,
            "transfer_mode": "artistic",
            "alpha_s": 0.9
        }
        config = InferenceConfig(**config_dict)
        runner = ImageInferenceOrchestrator(config)
        results = runner.run_inference()
        assert len(results) > 0
        assert os.listdir(tmpdir), "Expected stylized image output."
