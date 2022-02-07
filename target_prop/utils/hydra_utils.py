from collections import OrderedDict
import dataclasses
from numpy import safe_eval
from omegaconf import DictConfig, OmegaConf
import omegaconf
from dataclasses import dataclass
from typing import Callable, ClassVar, Any, Dict, Optional, Set, Type, TypeVar, Union
from simple_parsing.helpers.serialization.serializable import Serializable
from pathlib import Path
import importlib

from hydra.core.config_store import ConfigStore

L = TypeVar("L", bound="LoadableFromHydra")


# @dataclass
class LoadableFromHydra(Serializable):

    _target_: ClassVar[Union[Type, Callable]]
    _group: ClassVar[str]
    _name: ClassVar[str]

    @classmethod
    def get_target(cls) -> str:
        cls = cls
        if hasattr(cls, "_target_"):
            return get_full_name(cls._target_)
        else:
            # Find a good '_target_'.
            return get_full_name(cls)

    @classmethod
    def hydra_extra_dict(cls) -> Dict:
        return {
            # "_target_": cls.get_target(),
            # "_convert_": "all",
            # "defaults": ["base_" + cls._name],
        }

    @classmethod
    def from_dictconfig(cls: Type[L], config: DictConfig) -> L:
        actual_type = config._metadata.object_type
        assert inspect.isclass(actual_type) and issubclass(actual_type, Serializable), actual_type
        obj = OmegaConf.to_object(config)
        assert isinstance(obj, cls)
        return obj

    def to_dict(self, dict_factory: Type[Dict] = dict, recurse: bool = True) -> Dict:
        d = dict()
        d.update(self.hydra_extra_dict())
        d.update(super().to_dict(dict_factory=dict_factory, recurse=recurse))
        return d

    @classmethod
    def from_dict(cls, obj: Dict, drop_extra_fields: bool = None):
        target = obj.pop("_target_", cls)
        obj.pop("_convert_", "")
        if target is cls:
            # Use the from_dict of Serializable.
            return super().from_dict(obj, drop_extra_fields=drop_extra_fields)
        return target.from_dict(obj, drop_extra_fields=drop_extra_fields)

    @classmethod
    def cs_store(cls: Type["L"], group: str, name: str, default: L = None):
        cls._group = group
        cls._name = name
        default = cls()
        # assert False, name_without_base
        group_dir = Path("conf") / group
        cs = ConfigStore.instance()
        # name_without_base = name.split("base_", maxsplit=1)[1] if name.startswith("base_") else name
        # name_with_base = "base_" + name_without_base
        cs.store(group=group, name=name, node=default)

        # with open(group_dir / f"{name_without_base}.yaml", "w") as f:
        #     yaml.dump({"defaults": [name_with_base]})

        group_dir.mkdir(exist_ok=True, parents=True)
        # default.save_yaml(group_dir / f"{name_with_base}.yaml")


from pathlib import Path
import yaml


class Bob:
    @dataclass
    class Foo:
        bar: int = 123

        @classmethod
        def parent_class(cls) -> Type:
            full_name = cls.__qualname__
            # self_name = type(self).__name__
            parent_name, self_name = full_name.rsplit(".", maxsplit=1)
            parent_class: Type = eval(parent_name)
            return parent_class


def is_inner_class(object_type: Type) -> bool:
    return "." in object_type.__qualname__


def get_class_from_full_name(object_type: Type) -> Type:
    full_name = get_full_name(object_type)
    container_class_module, _, container_class_name = full_name.rpartition(".")
    # assert False, (container_class_module, class_name, child_name)
    mod = importlib.import_module(container_class_module)
    return getattr(mod, container_class_name)
    # assert False, mod
    # return eval(container_full_name)


def get_full_name(object_type: Type) -> str:
    return object_type.__module__ + "." + object_type.__qualname__


def get_all_subclasses(t: Type) -> Set[Type]:
    subclasses = t.__subclasses__()
    result = set()
    for sub in subclasses:
        result.add(sub)
        result.update(get_all_subclasses(sub))
    return result


def get_outer_class(inner_class: Type) -> Type:
    inner_full_name = get_full_name(inner_class)
    parent_full_name, _, _ = inner_full_name.rpartition(".")
    outer_class_module, _, outer_class_name = parent_full_name.rpartition(".")
    # assert False, (container_class_module, class_name, child_name)
    mod = importlib.import_module(outer_class_module)
    return getattr(mod, outer_class_name)


import re


def camel_case(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


# names_to_subclass = {camel_case(t.__qualname__): t for t in get_all_subclasses(Model.HParams)}
# names_to_subclass = {name.rpartition(".h_params")[0]: t for name, t in names_to_subclass.items()}
# names_to_subclass["backprop"] = names_to_subclass.pop("baseline_model")


def _validate_get(self: DictConfig, key: Any, value: Any = None) -> None:
    is_typed = self._is_typed()

    is_struct = self._get_flag("struct") is True
    if key not in self.__dict__["_content"]:
        if is_typed:
            # do not raise an exception if struct is explicitly set to False
            if self._get_node_flag("struct") is False:
                return
        if is_typed or is_struct:
            if is_typed:
                assert self._metadata.object_type is not None
                msg = f"Key '{key}' not in '{self._metadata.object_type.__qualname__}'"  # note: changed this to make things a bit clearer.
            else:
                msg = f"Key '{key}' is not in struct"
            self._format_and_raise(key=key, value=value, cause=ConfigAttributeError(msg))


from omegaconf.errors import ConfigAttributeError

# Hacky, but it works.
omegaconf.DictConfig._validate_get = _validate_get
# setattr(omegaconf.DictConfig, "_validate_get", _validate_get)
import omegaconf._utils

from omegaconf._utils import type_str as _type_str, _resolve_optional
import sys
import inspect


def type_str(t: Any) -> str:
    # Little 'patch', so the names are a bit more useful for inner classes. Otherwise we get
    # "HParams is not a subclass of HParams", which isn't very helpful.
    their_value = _type_str(t)
    if (
        hasattr(t, "__name__")
        and hasattr(t, "__qualname__")
        and their_value == t.__name__
        and t.__qualname__ != t.__name__
    ):
        print(f"Returning {t.__qualname__} (qualname) instead of {t.__name__} (name).")
        return t.__qualname__
    return their_value


omegaconf._utils.type_str = type_str
