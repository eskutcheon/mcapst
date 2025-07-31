# mcapst/__main__.py

import sys
from argparse import ArgumentParser
from pydantic import ValidationError
from pydantic_settings import CliApp, CliSettingsSource
# local imports
from mcapst.train.config.config import TrainingConfig
from mcapst.infer.config.config import InferenceConfig

def main(argv=None):
    parser = ArgumentParser(
        prog="mcapst",
        description="MCAPST: unified entrypoint for training or inference."
    )
    # inference options' subparser
    subparsers = parser.add_subparsers(dest="command", required=True)
    # training options' subparser
    train_parser = subparsers.add_parser(
        "train", prog="mcapst", usage="%(prog)s {train|.train} [OPTIONS]",
        help="Run training", description=TrainingConfig.__doc__
    )
    # basically does train_cli_src.add_argument(...) for every field in TrainingConfig
    train_cli_src = CliSettingsSource(
        TrainingConfig, root_parser=train_parser,
        cli_parse_args=False, cli_hide_none_type=True, cli_prog_name=train_parser.usage, cli_avoid_json=True
    )
    infer_parser = subparsers.add_parser(
        "infer", prog="mcapst", usage="%(prog)s {infer|.infer} [OPTIONS]",
        help="Run inference", description=InferenceConfig.__doc__
    )
    #? NOTE: ensure cli_parse_args=False for both of these so that parser.parse_args() is called only once
    infer_cli_src = CliSettingsSource(
        InferenceConfig, root_parser=infer_parser,
        cli_parse_args=False, cli_hide_none_type=True, cli_prog_name=train_parser.usage, cli_avoid_json=True
    )
    # parse the one level of args
    args = parser.parse_args(argv)
    cmd = args.command
    try:
        if cmd == "train":
            from mcapst.train.train import stage_training_pipeline
            cfg = CliApp.run(
                TrainingConfig, cli_args=args, cli_settings_source=train_cli_src
            )
            stage_training_pipeline(config=cfg)
        else:  # cmd == "infer"
            from mcapst.infer.infer import stage_inference_pipeline
            cfg = CliApp.run(
                InferenceConfig, cli_args=args, cli_settings_source=infer_cli_src
            )
            _ = stage_inference_pipeline(config=cfg)
    except ValidationError as e:
        print("Configuration error:\n", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
