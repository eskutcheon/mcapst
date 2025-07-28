# mcapst/train/__main__.py

# import argparse
# imported from `train` submodule
from .train import stage_training_pipeline, TrainingConfig
# from ..core.utils.config_utils import parser_from_cfg_factory


def main(argv=None):
    # parser = argparse.ArgumentParser(
    #     prog="mcapst.train",
    #     description="MCAPST training entrypoint"
    # )
    # parser.add_argument(
    #     "--config_path",
    #     type=str,
    #     default=None,
    #     required=False,
    #     help="(Optional) Path to YAML configuration file",
    # )
    # args: argparse.Namespace = parser_from_cfg_factory(
    #     TrainingConfig,
    #     desc="Training configuration options",
    #     argv=argv,
    #     parser=parser,
    # )
    # # load the Pydantic config (merges YAML + CLI) and hand off to the training pipeline
    # return stage_training_pipeline(args.config_path)
    cfg = TrainingConfig()
    return stage_training_pipeline(config=cfg)


if __name__ == "__main__":
    _ = main()