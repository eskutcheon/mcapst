import os, sys
from typing import Dict, Type
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import tempfile
import cProfile
import pstats
# local imports
from mcapst.config.configure import TrainingConfig
from mcapst.pipelines.train import stage_training_pipeline, ImageTrainer, VideoTrainer



def test_image_training_pipeline_dict_override(hf_content_path=r"bitmind/MS-COCO-unique-256", hf_style_path=r"huggan/wikiart"):
    """
    Example of calling stage_training_pipeline with no direct CLI, 
    but building config in code. This is similar to a user passing a YAML file.
    """
    # if not os.path.isdir(dummy_train_content) or not os.path.isdir(dummy_train_style):
    #     pytest.skip("No dummy data found.")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = {
            "modality": "image",
            "transfer_mode": "art",
            "logs_directory": os.path.join(tmpdir, "logs"),
            "data_cfg": {
                "train_content": hf_content_path,
                "train_style": hf_style_path,
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


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    test_image_training_pipeline_dict_override()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)  # Print the top 10 functions by cumulative time