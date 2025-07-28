# mcapst/infer/__main__.py

from .infer import stage_inference_pipeline, InferenceConfig

def main(argv=None):
    cfg = InferenceConfig()
    _ = stage_inference_pipeline(config=cfg)


if __name__ == "__main__":
    main()