

import argparse
import sys
# imported from `train` submodule
from .train import stage_training_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="MCAPST training entrypoint")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    args, unknown = parser.parse_known_args(argv)
    # remove parsed args so TrainingConfigManager sees only the remaining options
    sys.argv = [sys.argv[0]] + unknown
    stage_training_pipeline(config_path=args.config_path)


if __name__ == "__main__":
    main()