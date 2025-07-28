# mcapst/train/__main__.py

# imported from `train` submodule
from .train import stage_training_pipeline, TrainingConfig


def main(argv=None):
    cfg = TrainingConfig()
    stage_training_pipeline(config=cfg)


if __name__ == "__main__":
    main()