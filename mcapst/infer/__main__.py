import argparse
import sys
# import from `infer` submodule
from .infer import stage_inference_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="MCAPST inference entrypoint")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    args, unknown = parser.parse_known_args(argv)
    # remove parsed args so InferenceConfigManager sees only remaining options
    sys.argv = [sys.argv[0]] + unknown
    stage_inference_pipeline(config_path=args.config_path)


if __name__ == "__main__":
    main()