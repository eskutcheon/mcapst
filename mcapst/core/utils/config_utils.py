
# import sys
# import argparse
# import yaml
# from dataclasses import is_dataclass, asdict, fields
from typing import Literal, Optional # Dict, Any, Union, List, Type, Set, get_origin, get_args
from pathlib import Path
from pydantic import (
    Field, field_validator #, model_validator,
    # BaseModel, RootModel, DirectoryPath, FilePath
)
from pydantic_settings import (
    BaseSettings, SettingsConfigDict,
    PydanticBaseSettingsSource, CliSettingsSource, YamlConfigSettingsSource
)


# TODO: add prog arguments to the CLI settings as well


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

    # @field_validator("model_config", mode="after", check_fields=False)
    # def _set_yaml_source(cls, v: SettingsConfigDict) -> SettingsConfigDict:
    #     """ Set the YAML source for BaseSettings to load from config_path if provided """
    #     print(f"attempting to access value {v.__name__}: ", v)
    #     if isinstance(v, SettingsConfigDict):
    #         v["yaml_file"] = cls.config_path if cls.config_path and Path(cls.config_path).exists() else None
    #     return v

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
        # 1) get `config_path` from __init__‐kwargs,
        # 2) load YAML if provided,
        # 3) then override with CLI flags,
        # 4) then env/.env/secrets:
        yaml_path = init_settings.init_kwargs.get("config_path", None)
        yaml_path = Path(yaml_path) if yaml_path else None
        cfg = init_settings.config
        return (
            # optional YAML config file - explicit arguments have higher priority
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path),  # only if not None
            init_settings, # explicit init settings
            # CLI flags—now with a real prefix instead of None
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                #cli_prefix=cfg["cli_prefix"],
                #cli_flag_prefix_char=cfg["cli_flag_prefix_char"],
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







# def _deep_merge(orig: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
#     """ Recursively merge two dicts, with `over` taking precedence """
#     result = dict(orig)
#     for k, v in over.items():
#         if k in result and isinstance(result[k], dict) and isinstance(v, dict):
#             result[k] = _deep_merge(result[k], v)
#         else:
#             result[k] = v
#     return result

# # TODO: use `_iterate_model_fields` to inspect Pydantic models' field descriptions to populate help messages for --help calls

# class ConfigManager:
#     """ ConfigManager for loading and managing configuration models from user inputs
#         - Load a Pydantic `config_model` from (in order of precedence):
#             1. YAML file (if provided)
#             2. CLI flags
#         - Supports nested models via the `.` delimiter in CLI inputs (e.g. --data-cfg.train-content)
#         - Usage:
#             mgr = ConfigManager(MyConfigModel, config_path="cfg.yml", description="…")
#             cfg = mgr.config_model
#         - Calling the module with `--help` should now print all flags and descriptions.
#     """
#     def __init__(
#         self,
#         config_model: Type[BaseModel],
#         config_path: Optional[str] = None,
#         description: str = "",
#     ):
#         # 1. load YAML if given or create an empty dict
#         yaml_data = yaml.safe_load(open(config_path)) if config_path else {}
#         # 2. build parser from the *class* fields
#         parser = self.build_parser(config_model, desc=description)
#         # 3. parse CLI flags (will exit on --help)
#         args, _ = parser.parse_known_args()
#         flat_overrides = {k: v for k, v in vars(args).items() if v is not None}
#         # 4. nest overrides by splitting on '.'
#         nested_overrides = self.nest_overrides(flat_overrides)
#         # 5. merge YAML + CLI, then final Pydantic validation
#         combined = _deep_merge(yaml_data, nested_overrides)
#         self.config = config_model(**combined)

#     @property
#     def config_model(self) -> BaseModel:
#         """ Return the fully validated Pydantic model instance """
#         return self.config

#     @staticmethod
#     def nest_overrides(flat_overrides: Dict[str, Any]) -> Dict[str, Any]:
#         """ Convert flat overrides with '.' delimiters to nested dicts """
#         nested_overrides: Dict[str, Any] = {}
#         for dest, value in flat_overrides.items():
#             parts = dest.split('.')
#             cur = nested_overrides
#             for p in parts[:-1]:
#                 cur = cur.setdefault(p, {})
#             cur[parts[-1]] = value
#         return nested_overrides


#     # TODO: add another version of this function that prioritizes building usage/help messages without flattening and using `add_argument_group`
#     @staticmethod
#     def build_parser(config_model: Type[BaseModel], desc: str = "", parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
#         """ Build an argument parser from the Pydantic model fields """
#         if parser is None:
#             parser = argparse.ArgumentParser(description=desc)
#         # TODO: might want to call get_default to set it in the parser and show in the help messages
#         for full_name, field_info in ConfigManager.iterate_model_fields(config_model):
#             # build flag with dotted nested names and hyphens replacing underscores
#             parts = full_name.split('.')
#             flag = "--" + ".".join(p.replace('_', '-') for p in parts)
#             # determine type/action/nargs
#             arg_type = field_info.annotation #.outer_type_
#             arg_params = ConfigManager.build_arg_params(arg_type, full_name, field_info)
#             #print("Adding argument:", flag, "with arg_type:", arg_type, "and params:", arg_params)
#             parser.add_argument(flag, **arg_params)
#         return parser

#     @staticmethod
#     def build_arg_params(arg_type: Any, arg_name: str, field_info: Any) -> Dict[str, Any]:
#         """ Helper to construct argument parameters for argparse based on Pydantic field annotations
#             Includes
#                 dest: dotted field name
#                 default: the Pydantic default
#                 help: field description + "[default: ...]"
#                 plus type/action/nargs as needed
#         """
#         origin = get_origin(arg_type)
#         args_ = get_args(arg_type)
#         # pull the Pydantic default, whether a literal or via default_factory
#         try:
#             default = field_info.get_default()
#         except Exception:
#             default = None
#         help_text = field_info.description or ""
#         arg_params: Dict[str, Any] = {'dest': arg_name, 'default': default, 'help': help_text}
#         if arg_type is bool or (origin is Literal and all(isinstance(a, bool) for a in args_)):
#             arg_params['action'] = 'store_true'
#             arg_params.pop('type', None) # no explicit type needed
#         elif arg_type is int:
#             arg_params['type'] = int
#         elif arg_type is float:
#             arg_params['type'] = float
#         elif origin is list:
#             arg_params['nargs'] = '+'
#             arg_params['type'] = str
#         else:
#             # for things like Literal types, we can just use str and let Pydantic validate
#             arg_params['type'] = str
#         return arg_params

#     @staticmethod
#     def iterate_model_fields(
#         model: Union[Type[BaseModel], BaseModel],
#         prefix: str = "",
#     ) -> List[Any]:
#         """ Yield (full_name, field_info) for all "leaf" fields, recursing into nested BaseModel types.
#             `full_name` uses dot notation, e.g. 'data_cfg.train_content'
#         """
#         # pick model class to inspect defaults and description
#         cls = model if isinstance(model, type) else type(model)
#         items = []
#         #! FIXME: shouldn't keep using model_fields, but every single BaseModel would need their own method otherwise
#         for name, field_info in cls.model_fields.items():
#             full = prefix + name
#             arg_type = field_info.annotation #.outer_type_
#             #print("field_info.annotation:", dir(field_info.annotation))
#             origin = get_origin(arg_type)
#             args = get_args(arg_type)
#             #print("name:", full, "arg_type:", arg_type, "origin:", origin, "args:", args)
#             # detect a nested BaseModel type or union thereof
#             nested = None
#             #? NOTE: have to ensure it's not a RootModel, or we'd recurse into it and mess with `root`
#             if isinstance(arg_type, type) and issubclass(arg_type, BaseModel) and not issubclass(arg_type, RootModel):
#                 nested = arg_type
#             elif origin in (list, Union, tuple, ):  # check union/annotated
#                 for a in args:
#                     if isinstance(a, type) and issubclass(a, BaseModel) and not issubclass(a, RootModel):
#                         nested = a
#                         break
#             if nested is not None:
#                 # recurse into nested model
#                 items += ConfigManager.iterate_model_fields(nested, prefix=full + '.')
#             else:
#                 items.append((full, field_info))
#         return items



# def parser_from_cfg_factory(
#     config_model: Type[BaseModel],
#     desc: str = "",
#     argv = None,
#     parser: Optional[argparse.ArgumentParser] = None
# ) -> argparse.ArgumentParser:
#     """ small (probably temporary) factory function to build a parser from a Pydantic model for CLI output """
#     if parser is None:
#         parser = argparse.ArgumentParser(description=desc)
#     # add a named group for all InferenceConfig fields
#     group_desc = "Inference options" if "InferenceConfig" in config_model.__name__ else "Training options"
#     group = parser.add_argument_group(group_desc)
#     group = ConfigManager.build_parser(config_model, desc=desc, parser=group)
#     args = parser.parse_args(argv)
#     # rebuild sys.argv so ConfigManager only sees its flags (strip off `--config-path` and its value)
#     rem = []
#     skip = False
#     for tok in sys.argv[1:]:
#         if skip:
#             skip = False
#             continue
#         if tok.startswith("--config-path"):
#             skip = "=" not in tok  # if `--config-path=foo.yml` no skip
#             continue
#         rem.append(tok)
#     sys.argv = [sys.argv[0]] + rem
#     return args


# def attach_to_parser(
#     parser_group: argparse._ArgumentGroup,
#     config_model: Type[BaseModel],
#     *,
#     prefix: str = "--",
#     skip_fields: Optional[Set[str]] = None,
# ):
#     """ Attach a --flag per leaf field of `config_model` into `parser`.
#         Args:
#             prefix: optionally prepend to each flag (unused here)
#             skip_fields: full_name(s) to omit (e.g. base fields)
#     """
#     if skip_fields is None:
#         skip_fields = set()
#     for full_name, field_info in ConfigManager.iterate_model_fields(config_model):
#         # if this field (or any sub-field) is in skip_fields, skip it
#         if any(full_name == f or full_name.startswith(f + ".") for f in skip_fields):
#             continue
#         # build flag with dotted nested names and hyphens replacing underscores
#         arg_type = field_info.annotation
#         parts = full_name.split('.')
#         flag = prefix + ".".join(p.replace('_', '-') for p in parts)
#         params = ConfigManager.build_arg_params(arg_type, full_name, field_info)
#         parser_group.add_argument(flag, **params)