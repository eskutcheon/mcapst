# mcapst/__main__.py
import argparse

from mcapst.core.utils.config_utils import ConfigManager, BaseConfigModel, attach_to_parser, BASE_FIELDS
from mcapst.train.config.config import TrainingConfig
from mcapst.infer.config.config import InferenceConfig
from mcapst.train.train import stage_training_pipeline
from mcapst.infer.infer import stage_inference_pipeline

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mcapst",
        description="MCAPST: unified entrypoint for training or inference."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        required=True,
        help="Whether to run in training or inference mode."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional YAML config file."
    )
    # common options inherited from BaseConfigModel
    common_group = parser.add_argument_group("Universal options")
    attach_to_parser(common_group, BaseConfigModel)
    # training options
    train_group = parser.add_argument_group("Training options")
    attach_to_parser(train_group, TrainingConfig, skip_fields=BASE_FIELDS)
    # inference options
    infer_group = parser.add_argument_group("Inference options")
    attach_to_parser(infer_group, InferenceConfig, skip_fields=BASE_FIELDS)
    args = parser.parse_args(argv)
    # now dispatch
    if args.mode == "train":
        cfg_mgr = ConfigManager(TrainingConfig, config_path=args.config_path)
        cfg = cfg_mgr.config_model
        return stage_training_pipeline(config=cfg)
    else:
        cfg_mgr = ConfigManager(InferenceConfig, config_path=args.config_path)
        cfg = cfg_mgr.config_model
        return stage_inference_pipeline(config=cfg)

if __name__ == "__main__":
    _ = main()
