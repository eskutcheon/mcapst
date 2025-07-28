# tests/test_configs.py
import os
import sys
import yaml
import pytest
from pathlib import Path
# from tempfile import TemporaryDirectory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mcapst.train.config.config import TrainingConfig #, get_training_config_manager
from mcapst.infer.config.config import InferenceConfig #, get_inference_config_manager
# from mcapst.core.utils.config_utils import ConfigManager

# Helpers to make dummy image files
def make_dummy_images(dirpath: Path, count: int, ext: str = ".jpg"):
    dirpath.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (dirpath / f"img_{i}{ext}").write_bytes(b"")

###############################################################################
# 1) Scripted / dict creation
###############################################################################

def test_training_config_from_dict(tmp_path):
    content = tmp_path / "content"; make_dummy_images(content, 100)
    style   = tmp_path / "style";   make_dummy_images(style, 20, ext=".png")
    cfg_dict = {
        "transfer_mode": "artistic",  # tests aliasing
        "modality": "video",
        "data_cfg": {
            "use_local_data": True,
            "train_content": str(content),
            "train_style":   str(style),
            "batch_size":  2,
            "new_size":    128,
        },
        "loss_cfg": {
            "style_weight":   0.5,
            "content_weight": 0.2,
        },
        "train_iter": 1000,
        "ckpt_interval": 0,  # should warn, not fail
    }
    cfg = TrainingConfig(**cfg_dict)
    # top‑level aliasing worked:
    assert cfg.transfer_mode == "art"
    # nested models are correct types:
    assert cfg.data_cfg.use_local_data is True
    assert isinstance(cfg.loss_cfg, type(cfg.loss_cfg))
    # warnings for ckpt_interval=0 and video+temporal:
    assert cfg.loss_cfg.temporal_weight == 20.0


def test_inference_config_from_dict(tmp_path):
    inp    = tmp_path / "in";  make_dummy_images(inp, 3, ext=".png")
    styles = tmp_path / "sty"; make_dummy_images(styles, 2, ext=".jpg")
    cfg = InferenceConfig(
        transfer_mode="photorealistic",
        modality="image",
        input_paths=str(inp),
        style_paths=[str(styles / f"img_{i}.jpg") for i in range(2)],
        alpha_s=[0.3, 0.7],
    )
    # defaults applied:
    assert cfg.transfer_mode == "photo"
    assert len(cfg.input_paths) == 3
    assert len(cfg.style_paths) == 2
    # alphas normalized:
    assert sum(cfg.alpha_s) == pytest.approx(1.0)


###############################################################################
# 2) YAML loading + CLI overrides
###############################################################################

def test_training_config_via_yaml_and_cli(tmp_path, monkeypatch):
    # prepare YAML
    cfg_yaml = {
        #! FIXME: seems like it's not properly setting nested arguments from YAML
        "data_cfg": {
            "use_local_data": True,
            "train_content":  str(tmp_path/"c"),
            "train_style":    str(tmp_path/"s"),
            "batch_size":  1,
            "new_size": 256
        },
        "loss_cfg": {"lap_weight": 500.0},
    }
    # make dirs + files
    content = tmp_path/"c"; make_dummy_images(content, 100)
    style   = tmp_path/"s"; make_dummy_images(style, 20)
    yml = tmp_path/"cfg.yml"
    yml.write_text(yaml.safe_dump(cfg_yaml))
    # override lr and train_iter via CLI
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "dummy")  # avoid pytest argv pollution
    monkeypatch.setattr(sys, "argv", [
        __file__,
        f"--lr=2e-3",
        f"--train-iter=500"
    ])
    #mgr = get_training_config_manager(str(yml))
    #cfg = mgr.config_model
    cfg = TrainingConfig(config_path=yml) #, cli_args=sys.argv[1:])
    print("final instantiated config: ", cfg.model_dump())
    assert cfg.lr == pytest.approx(2e-3)
    assert cfg.train_iter == 500
    # preserves YAML defaults
    assert cfg.data_cfg.batch_size == 1
    assert cfg.loss_cfg.lap_weight == pytest.approx(500.0)


def test_inference_config_via_yaml_and_cli(tmp_path, monkeypatch):
    inp    = tmp_path/"in";  make_dummy_images(inp, 4, ext=".mp4")
    sty    = tmp_path/"sty"; make_dummy_images(sty, 3)
    yml = tmp_path/"inf.yml"
    yml.write_text(yaml.safe_dump({
        "input_paths": str(inp),
        "style_paths": str(sty),
        "alpha_s": [0.2, 0.56, 0.24],
    }))
    # override modality→video and max_size
    monkeypatch.setattr(sys, "argv", [
        __file__,
        "--modality=video",
        "--max-size=512"
    ])
    # mgr = get_inference_config_manager(str(yml))
    # cfg = mgr.config_model
    cfg = InferenceConfig(config_path=yml) #, cli_args=sys.argv[1:])
    # print(cfg.model_dump())
    assert cfg.modality == "video"
    assert cfg.max_size == 512

###############################################################################
# 3) CLI‑only (no YAML)
###############################################################################

def test_training_config_cli_only(tmp_path, monkeypatch):
    content = tmp_path/"c"; make_dummy_images(content, 100)
    style   = tmp_path/"s"; make_dummy_images(style, 20)
    monkeypatch.setattr(sys, "argv", [
        __file__,
        f"--data-cfg.use-local-data={True}",
        f"--data-cfg.train-content={content}",
        f"--data-cfg.train-style={style}",
        "--loss-cfg.style-weight=0.9",
        "--lr=1e-5"
    ])
    # mgr = get_training_config_manager(None)
    # cfg = mgr.config_model
    cfg = TrainingConfig() #cli_args=sys.argv[1:])
    assert cfg.data_cfg.use_local_data
    assert cfg.loss_cfg.style_weight == pytest.approx(0.9)
    assert cfg.lr == pytest.approx(1e-5)


def test_inference_config_cli_only(tmp_path, monkeypatch):
    inp  = tmp_path/"in"; make_dummy_images(inp, 2)
    sty  = tmp_path/"sty"; make_dummy_images(sty, 1)
    monkeypatch.setattr(sys, "argv", [
        __file__,
        f"--input-paths={inp}",
        f"--style-paths={sty}",
        "--alpha-s=0.5",
    ])
    # mgr = get_inference_config_manager(None)
    # cfg = mgr.config_model
    cfg = InferenceConfig() #cli_args=sys.argv[1:])
    assert len(cfg.input_paths) == 2
    assert len(cfg.style_paths) == 1
    assert cfg.alpha_s == [0.5]
