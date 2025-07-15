
import argparse
import dataclasses
from dataclasses import fields
from typing import Any, Dict, Optional, Type, Union
import yaml


#? NOTE: below is partially AI-generated from the original ConfigManager, which used more
    #? conditional branching for training and inference
class BaseConfigManager:
    """ utility class for loading YAML configs and applying CLI overrides """
    def __init__(self,
                 config_cls: Type[Any],
                 config_path: Optional[str] = None,
                 help_messages: Optional[Dict[str, str]] = None,
                 description: str = ""):
        self.config_cls = config_cls
        self.help_messages = help_messages or {}
        yaml_args: Dict[str, Any] = {}
        if config_path:
            yaml_args = self._load_yaml_config(config_path)
        self.config = config_cls(**yaml_args)
        self.parser = argparse.ArgumentParser(description=description or f"Configuration for {config_cls.__name__}")
        # first, add arguments for top-level config and possible nested dataclasses
        self._add_dataclass_args(self.config)
        # actually parse input collected so far
        args, _ = self.parser.parse_known_args()
        # apply CLI overrides to the top-level config arguments
        self._apply_overrides(self.config, args)
        self._rerun_post_init()

    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """ Loads settings from a YAML configuration file and updates defaults """
        with open(config_path, "r") as file:
            # merge YAML config into the current config object
            return yaml.safe_load(file)

    def _add_dataclass_args(self, dc_instance: Any, prefix: str = "") -> None:
        for f in fields(dc_instance):
            field_name = f.name
            full_name = f"{prefix}{field_name}"
            value = getattr(dc_instance, field_name)
            if dataclasses.is_dataclass(value):
                self._add_dataclass_args(value, prefix=f"{full_name}.")
            else:
                help_msg = self.help_messages.get(full_name, self.help_messages.get(field_name, full_name))
                cli_flag = f"--{full_name.replace('_', '-')}"
                if f.type == bool:
                    self.parser.add_argument(cli_flag, action="store_true", default=None, help=help_msg)
                else:
                    self.parser.add_argument(cli_flag, type=str, default=None, help=help_msg)

    def _apply_overrides(self, dc_instance: Any, args_namespace: argparse.Namespace, prefix: str = "") -> None:
        for f in fields(dc_instance):
            field_name = f.name
            full_name = f"{prefix}{field_name}"
            value = getattr(dc_instance, field_name)
            if dataclasses.is_dataclass(value):
                self._apply_overrides(value, args_namespace, prefix=f"{full_name}.")
            else:
                cli_key = full_name.replace('.', '_').replace('-', '_')
                if hasattr(args_namespace, cli_key):
                    raw_val = getattr(args_namespace, cli_key)
                    if raw_val is not None:
                        setattr(dc_instance, field_name, self._cast_value(f.type, raw_val))

    def _cast_value(self, field_type: Any, raw_val: Any) -> Any:
        """ helper function to cast raw strings from CLI to the correct type if needed
            e.g. if field_type = int, we do int(raw_val), if field_type is bool, we do (raw_val.lower() in ['true','1','yes']), etc.
            for nested or complex types, I might need to just raise a NotImplementedError for now
        """
        origin_type = getattr(field_type, "__origin__", None)
        if field_type == int:
            return int(raw_val)
        elif field_type == float:
            return float(raw_val)
        elif field_type == bool:
            if isinstance(raw_val, bool):
                return raw_val
            # if action="store_true", then if present => True, else => False or just parse manually:
            return str(raw_val).lower() in ["true", "1", "yes"]
        elif field_type == str:
            return str(raw_val)
        elif origin_type is Union:
            # Union type => pick the first that doesn't fail
            for t in field_type.__args__:
                if t == type(None):
                    continue
                try:
                    return self._cast_value(t, raw_val)
                except Exception:
                    pass
            # if all types fail, return the raw value
            return raw_val
        else: # for other types, assume they can be directly assigned
            return raw_val

    def _rerun_post_init(self) -> None:
        """ re-run the dataclass's __post_init__ to ensure correct validation """
        self.config = self.config_cls(**dataclasses.asdict(self.config))

    def get_config(self) -> Any:
        """ returns the final configuration object """
        return self.config
