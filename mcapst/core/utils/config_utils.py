# mcapst/core/utils/config_utils.py
from typing import Literal, Optional # Dict, Any, Union, List, Type, Set, get_origin, get_args
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings, SettingsConfigDict,
    PydanticBaseSettingsSource, CliSettingsSource, YamlConfigSettingsSource
)


# mcapst {train|.train} [OPTIONS] OR mcapst {infer|.infer} [OPTIONS]
# essentially acts as both the `prog` and `usage` arguments for the CLI
prog_name_dict = {
    "TrainingConfig": "mcapst {train|.train}",
    "InferenceConfig": "mcapst {infer|.infer}",
}


class BaseConfigModel(BaseSettings):
    """Common fields for all configs, with built-in CLI + YAML support."""
    config_path: Optional[Path] = Field(
        None, description="Optional YAML config file for training/inference run settings."
    )
    #& still not used anywhere, but it could be used to make subdirectories later
    run_name: Optional[str] = Field(None, description="Name of the run, used for naming output logs and directories.")
    transfer_mode: Literal["photo","art","photorealistic","artistic"] = Field(
        "photo",
        description="Transfer mode for the run, either 'photorealistic'/'photo' or 'artistic'/'art'."
    )
    modality: Literal["image","video"] = Field(
        "image",
        description="Input data modality (Either 'image' or 'video')."
    )
    reg_method: Literal["ridge", "diagonal_load", "softmax_temp", "log_space"] = Field(
        "ridge",
        description="Regularization method to use in cWCT with default='ridge'."
    )
    # settings config for BaseSettings with CLI-friendly mode
    model_config = SettingsConfigDict(
        validate_default=True,
        env_prefix="",
        # avoid JSON blobs on complex fields (lists, nested models, etc.)
        cli_avoid_json=True,           # avoid --input-paths [JSON] problem
        # treat bools as flags, use kebab-case, ignore unknown args
        cli_implicit_flags=True,       #! MIGHT REMOVE: --use-segmentation / --no-use-segmentation
        cli_kebab_case=True,           # convert to kebab case: e.g. rename run_name -> --run-name
        cli_ignore_unknown_args=True,
        # standardize prefixing
        #cli_prefix="",
        #cli_flag_prefix_char="-",
        # still parse sys.argv by default
        cli_parse_args=True,
    )


    @field_validator("transfer_mode", mode="before")
    def _alias_transfer_mode(cls, v: str) -> Literal["photo", "art"]:
        return {"photorealistic": "photo", "artistic": "art"}.get(v, v)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # load YAML if provided -> override with CLI flags -> then env/.env/secrets:
        yaml_path = init_settings.init_kwargs.get("config_path", None)
        yaml_path = Path(yaml_path) if yaml_path else None
        cfg = init_settings.config
        try:
            prog_name = prog_name_dict[settings_cls.__name__]
        except KeyError:
            prog_name = "mcapst"
        return (
            # optional YAML config file - explicit arguments have higher priority
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path),  # only if not None
            init_settings, # explicit init settings
            # CLI flags—now with a real prefix instead of None
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                cli_prog_name=prog_name,
                cli_hide_none_type=True,
                cli_avoid_json=cfg["cli_avoid_json"],
                cli_implicit_flags=cfg["cli_implicit_flags"],
                cli_kebab_case=cfg["cli_kebab_case"],
                cli_ignore_unknown_args=cfg["cli_ignore_unknown_args"],
            ),
            # 4) env / .env / secrets
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
