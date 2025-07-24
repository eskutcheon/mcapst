
import argparse
import yaml
# from dataclasses import is_dataclass, asdict, fields
from typing import Dict, Literal, Any, Optional, Union, List, Type, get_origin, get_args
from pathlib import Path
from pydantic import (
    BaseModel, RootModel, Field, ConfigDict, model_validator, field_validator,
    DirectoryPath, FilePath
)


class PathList(RootModel[List[Path]]):

    @model_validator(mode="before")
    def coerce(cls, v: Union[FilePath, DirectoryPath, List[FilePath], Path, str]) -> List[Path]:
        if isinstance(v, (str, Path)):
            if Path(v).is_file():
                return [Path(v)]
            elif Path(v).is_dir():
                p = Path(v)
                return [f for f in p.iterdir() if f.is_file()]
        elif isinstance(v, list):
            # take only valid files from the list, not directories
            return [Path(x) for x in v if Path(x).is_file()]
        raise ValueError(f"Expected a path to a file, directory, or list of files; got {v!r}")

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)

    def __repr__(self) -> str:
        return f"PathList({self.root})"



class BaseConfigModel(BaseModel):
    """ Base class of type pydantic.BaseModel for all configuration models with common fields and methods
        - easily extensible for future use cases
    """
    #& still not used anywhere, but it could be used to make subdirectories later
    run_name: Optional[str] = Field(None, description="Name of the run, used for logging and checkpoints.")
    # pydantic now implicitly handles the validation of these string literals
    transfer_mode: Literal["photo", "art", "photorealistic", "artistic"] = Field(
        "photo",
        description="Transfer mode for the run, either 'photorealistic'/'photo' or 'artistic'/'art'."
    )
    modality: Literal["image", "video"] = Field(
        "image",
        description="Input data modality (Either 'image' or 'video')."
    )
    reg_method: Literal["ridge", "diagonal_load", "softmax_temp", "log_space"] = Field(
        "ridge",
        description="Regularization method to use in cWCT with default='ridge'. Options: ('ridge', 'diagonal_load', 'softmax_temp', 'log_space').",
    )
    model_config = ConfigDict(populate_by_name=True)

    @field_validator("transfer_mode", mode="before")
    def _alias_transfer_mode(cls, v: str) -> Literal["photo", "art"]:
        return {"photorealistic": "photo", "artistic": "art"}.get(v, v)

    @classmethod
    def get_model_fields(cls) -> Dict[str, Any]:
        # pydantic v2 stores __fields__ here
        return cls.model_fields



def _deep_merge(orig: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """ Recursively merge two dicts, with `over` taking precedence """
    result = dict(orig)
    for k, v in over.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

# TODO: use `_iterate_model_fields` to inspect Pydantic models' field descriptions to populate help messages for --help calls

class ConfigManager:
    """ ConfigManager for loading and managing configuration models from user inputs
        - Load a Pydantic `config_model` from (in order of precedence):
            1. YAML file (if provided)
            2. CLI flags
        - Supports nested models via the `.` delimiter in CLI inputs (e.g. --data-cfg.train-content)
        - Usage:
            mgr = ConfigManager(MyConfigModel, config_path="cfg.yml", description="…")
            cfg = mgr.config_model
        - Calling the module with `--help` should now print all flags and descriptions.
    """
    def __init__(
        self,
        config_model: Type[BaseModel],
        config_path: Optional[str] = None,
        description: str = "",
    ):
        # 1. load YAML if given or create an empty dict
        yaml_data = yaml.safe_load(open(config_path)) if config_path else {}
        # 2. build parser from the *class* fields
        parser = self._build_parser(config_model, desc=description)
        # 3. parse CLI flags (will exit on --help)
        args, _ = parser.parse_known_args()
        flat_overrides = {k: v for k, v in vars(args).items() if v is not None}
        # 4. nest overrides by splitting on '.'
        nested_overrides = self._nest_overrides(flat_overrides)
        # 5. merge YAML + CLI, then final Pydantic validation
        combined = _deep_merge(yaml_data, nested_overrides)
        self.config = config_model(**combined)

    def _build_parser(self, config_model: Type[BaseModel], desc: str = "") -> argparse.ArgumentParser:
        """ Build an argument parser from the Pydantic model fields """
        parser = argparse.ArgumentParser(description=desc)
        # TODO: might want to call get_default to set it in the parser and show in the help messages
        for full_name, field_info in self._iterate_model_fields(config_model):
            # build flag with dotted nested names and hyphens replacing underscores
            parts = full_name.split('.')
            flag = "--" + ".".join(p.replace('_', '-') for p in parts)
            # determine type/action/nargs
            arg_type = field_info.annotation #.outer_type_
            arg_params = self.build_arg_params(arg_type, full_name, field_info)
            #print("Adding argument:", flag, "with arg_type:", arg_type, "and params:", arg_params)
            parser.add_argument(flag, **arg_params)
        return parser

    def _nest_overrides(self, flat_overrides: Dict[str, Any]) -> Dict[str, Any]:
        """ Convert flat overrides with '.' delimiters to nested dicts """
        nested_overrides: Dict[str, Any] = {}
        for dest, value in flat_overrides.items():
            parts = dest.split('.')
            cur = nested_overrides
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = value
        return nested_overrides

    @staticmethod
    def build_arg_params(arg_type: Any, arg_name: str, field_info: Any) -> Dict[str, Any]:
        """ Helper to construct argument parameters for argparse based on Pydantic field annotations """
        origin = get_origin(arg_type)
        args_ = get_args(arg_type)
        arg_params: Dict[str, Any] = {'dest': arg_name, 'default': None, 'help': field_info.description or ""}
        if arg_type is bool or (origin is Literal and all(isinstance(a, bool) for a in args_)):
            arg_params['action'] = 'store_true'
            arg_params.pop('type', None) # no explicit type needed
        elif arg_type is int:
            arg_params['type'] = int
        elif arg_type is float:
            arg_params['type'] = float
        elif origin is list:
            arg_params['nargs'] = '+'
            arg_params['type'] = str
        else:
            # for things like Literal types, we can just use str and let Pydantic validate
            arg_params['type'] = str
        return arg_params

    @staticmethod
    def _iterate_model_fields(
        model: Union[Type[BaseModel], BaseModel],
        prefix: str = "",
    ) -> List[Any]:
        """ Yield (full_name, field_info) for all "leaf" fields, recursing into nested BaseModel types.
            `full_name` uses dot notation, e.g. 'data_cfg.train_content'
        """
        # pick model class to inspect defaults and description
        cls = model if isinstance(model, type) else type(model)
        items = []
        #! FIXME: shouldn't keep using model_fields, but every single BaseModel would need their own method otherwise
        for name, field_info in cls.model_fields.items():
            full = prefix + name
            arg_type = field_info.annotation #.outer_type_
            #print("field_info.annotation:", dir(field_info.annotation))
            origin = get_origin(arg_type)
            args = get_args(arg_type)
            #print("name:", full, "arg_type:", arg_type, "origin:", origin, "args:", args)
            # detect a nested BaseModel type or union thereof
            nested = None
            #? NOTE: have to ensure it's not a RootModel, or we'd recurse into it and mess with `root`
            if isinstance(arg_type, type) and issubclass(arg_type, BaseModel) and not issubclass(arg_type, RootModel):
                nested = arg_type
            elif origin in (list, Union, tuple, ):  # check union/annotated
                for a in args:
                    if isinstance(a, type) and issubclass(a, BaseModel) and not issubclass(a, RootModel):
                        nested = a
                        break
            if nested is not None:
                # recurse into nested model
                items += ConfigManager._iterate_model_fields(nested, prefix=full + '.')
            else:
                items.append((full, field_info))
        return items

    @property
    def config_model(self) -> BaseModel:
        """ Return the fully validated Pydantic model instance """
        return self.config
