# mcapst/core/utils/config_utils.py
from typing import Literal, Optional, Any, List, Sequence, Union, Callable, Type, Annotated # get_origin, get_args
from pathlib import Path
# from pydantic_core import core_schema
from pydantic import (
    Field, field_validator, ValidationError, AfterValidator, TypeAdapter, #PlainSerializer, #, WithJsonSchema
    FilePath, DirectoryPath, JsonValue
)
# from pydantic_core import PydanticCustomError
from pydantic_settings import (
    BaseSettings, SettingsConfigDict,
    PydanticBaseSettingsSource, CliSettingsSource, YamlConfigSettingsSource
)


def _coerce_paths_validator(value: Union[FilePath, DirectoryPath, List[FilePath], List[DirectoryPath]]) -> List[Path]:
    """ Helper function for custom type `PathList`, coercing inputs:
        - single str/Path -> List[Path(...)],
        - JSON-like '["a","b"]' str wrappers,
        - a dir -> all files inside,
        - an iterable of str/Path -> flattened file list.
            - if it contains a single directory path -> list of all files within
            - if an iterable of both files and directories, ignore those subdirectories and only include files
        Raises a validation error on anything else.
        Returns a list of Path objects.
    """
    def _unpack_dir(p: Path) -> List[Path]:
        dir_paths = [f for f in p.iterdir() if f.is_file()]
        if not dir_paths:
            raise ValidationError("path_list_empty_dir", f"Directory '{p}' is empty or contains no files.")
        return dir_paths
    # handle a single string or Path
    if isinstance(value, (str, Path)):
        # TODO: remove JSON string handling after confirming it doesn't break the config tests
        s = str(value).strip()
        # strip JSON-like wrapper
        if s.startswith('["') and s.endswith('"]'):
            s = s[2:-2]
        # coerce to pathlib.Path and deal with files or directories
        p = Path(s)
        if p.is_file():
            return [p]
        if p.is_dir():
            return _unpack_dir(p)
    # handle iterable of strings or Paths
    elif isinstance(value, Sequence):
        if not all(isinstance(x, (str, Path)) for x in value):
            raise ValidationError(
                "path_list_type_error",
                f"each item in path list must be a string or Path; got {value!r}"
            )
        # convert all to Path and filter out non-files; if it's a single directory, unpack it
        if len(value) == 1 and Path(value[0]).is_dir():
            return _unpack_dir(Path(value[0]))
        return [Path(x) for x in value if Path(x).is_file()]
    raise ValidationError("path_list_type_error", f"Expected a path, dir or sequence thereof, got: {value!r}")


PathList = Annotated[
    Union[FilePath, DirectoryPath, JsonValue, List[FilePath], List[DirectoryPath]],
    AfterValidator(_coerce_paths_validator) # return_type=List[Path],
]



def _normalize_alpha_weights(value: Union[float, Sequence[float]]) -> List[float]:
    """ Helper function for custom type `AlphaWeights`
        For a list of floats in range [0,1], it replaces the use of the old StyleWeights class, coercing:
            - single float/int -> [float],
            - any iterable of float/int -> List[float],
        Then normalizes them if more than one weight is present (so they sum to 1.0)
    """
    # wrap single scalar
    if isinstance(value, (int, float)):
        w = float(value)
        weights = [w]
    # elif iterable of scalars, cast all to float
    elif isinstance(value, Sequence):
        try:
            weights = [float(x) for x in value]
        except Exception:
            raise ValidationError(
                "alpha_type_error",
                f"each alpha must be convertible to float; got {value!r}"
            )
    else:
        raise ValidationError(
            "alpha_type_error",
            f"alpha must be a float or iterable of floats; got {value!r}"
        )
    # enforce that all weights are in [0,1] range
    for w in weights:
        if not 0.0 <= w <= 1.0:
            raise ValidationError("alpha_value_error", f"alpha weights must lie in [0,1]; got {weights!r}")
    # only normalize with multiple styles since multi-style interpolation blends them by their individual weights
    #! honestly need to rethink normalizing these at all since it could introduce unexpected behavior
    if len(weights) > 1:
        total = sum(weights)
        assert total > 1e-6, f"sum of alpha weights must be greater than tolerance={1e-6}"
        weights = [round(w / total, 8) for w in weights]
    return weights


AlphaWeights = Annotated[
    Union[float, List[float]],
    AfterValidator(_normalize_alpha_weights)
]
AlphaWeightsAdaptor = TypeAdapter(AlphaWeights)


def get_default_alpha_weights(
    v: Any,
    num_items: int = 1,
    weight_type: Literal['style', 'content'] = "style"
) -> List[float]:
    """ TEMPORARY utility function that returns a list of default weights for the given number of items """
    assert weight_type in ('style', 'content'), "Provided `weight_type` must be one of ('style', 'content')."
    if v is None:
        return [0.0] if weight_type == "content" else [1.0 / num_items] * num_items
    # if input is scalar, add to a new list
    if v is not None and not isinstance(v, (list, tuple)):
        v = [v]
    # if input is a sequence of the incorrect length, fill the remainder then normalize
    if isinstance(v, (list, tuple)) and len(v) != num_items:
        num_missing = num_items - len(v)
        if num_missing < 0:
            print(f"\x1b[33mWARNING: More alpha values were found than is required. Truncating to first {num_items}...\x1b[0m")
            v = v[:num_items]
        else:
            print(f"\x1b[33mWARNING: Not enough alpha values found; Defaulting to {num_items} equal weights\x1b[0m")
            return [1.0 / num_items] * num_items
    return AlphaWeightsAdaptor.validate_python(v)



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
            # env / .env / secrets
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
