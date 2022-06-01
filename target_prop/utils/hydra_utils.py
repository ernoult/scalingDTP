import importlib
from typing import Type


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
