from omegaconf import DictConfig
import omegaconf
from typing import Any, Type
import importlib


def is_inner_class(object_type: Type) -> bool:
    return "." in object_type.__qualname__


def get_full_name(object_type: Type) -> str:
    return object_type.__module__ + "." + object_type.__qualname__


def get_outer_class(inner_class: Type) -> Type:
    inner_full_name = get_full_name(inner_class)
    parent_full_name, _, _ = inner_full_name.rpartition(".")
    outer_class_module, _, outer_class_name = parent_full_name.rpartition(".")
    # assert False, (container_class_module, class_name, child_name)
    mod = importlib.import_module(outer_class_module)
    return getattr(mod, outer_class_name)


# "patch" for some of the Hydra logging methods, so they are more informative with inner classes.


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

from omegaconf._utils import type_str as _type_str


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
