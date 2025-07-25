# mcapst/infer/__main__.py
import argparse
# import from `infer` submodule
from .infer import stage_inference_pipeline, InferenceConfig
from ..core.utils.config_utils import parser_from_cfg_factory

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mcapst-infer",
        description="MCAPST inference entrypoint")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        required=False,
        help="(Optional) Path to YAML configuration file for inference",
    )
    args: argparse.Namespace = parser_from_cfg_factory(
        InferenceConfig,
        desc="Inference configuration options",
        argv=argv,
        parser=parser,
    )
    return stage_inference_pipeline(config_path=args.config_path)


if __name__ == "__main__":
    _ = main()