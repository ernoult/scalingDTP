from __future__ import annotations

import importlib
import inspect

from hydra_zen import builds as _builds


def is_inner_class(object_type: type) -> bool:
    return "." in object_type.__qualname__


def get_full_name(object_type: type) -> str:
    return object_type.__module__ + "." + object_type.__qualname__


def get_outer_class(inner_class: type) -> type:
    inner_full_name = get_full_name(inner_class)
    parent_full_name, _, _ = inner_full_name.rpartition(".")
    outer_class_module, _, outer_class_name = parent_full_name.rpartition(".")
    # assert False, (container_class_module, class_name, child_name)
    mod = importlib.import_module(outer_class_module)
    return getattr(mod, outer_class_name)


class_to_config_class: dict[type, type] = {}


def builds(thing, *args, **kwargs):
    kwargs.setdefault("dataclass_name", thing.__qualname__ + "Config")
    # kwargs.setdefault("populate_full_signature", True)

    builds_bases = list(kwargs.pop("builds_bases", []))
    if inspect.isclass(thing):
        for base in thing.mro():
            # Add the config classes for the parent to the builds_bases.
            if base in class_to_config_class and class_to_config_class[base] not in builds_bases:
                builds_bases.append(class_to_config_class[base])
                # TODO: Should we do this only for the first base? or all the bases?
    kwargs["builds_bases"] = tuple(builds_bases)

    config_dataclass = _builds(thing, *args, **kwargs)
    if thing not in class_to_config_class:
        class_to_config_class[thing] = config_dataclass
    return config_dataclass
