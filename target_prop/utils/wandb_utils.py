""" Wandb utilities. """
from typing import Dict, List, Optional, Type, TypeVar

import wandb
from simple_parsing.helpers.serialization.serializable import Serializable
from wandb.apis.public import Run

try:
    from simple_parsing.helpers.serialization.serializable import FrozenSerializable
except ImportError:
    FrozenSerializable = Serializable

S = TypeVar("S", bound=Serializable)

_api: Optional[wandb.Api] = None


def load_from_run(
    cls: Type[S],
    run_path: str,
    key: str = None,
    renamed_keys: Dict[str, str] = None,
    removed_keys: List[str] = None,
) -> S:
    global _api
    if key is None:
        key = getattr(cls, "_stored_at_key", None)
    if key is None:
        raise RuntimeError(
            f"Don't know which key to look at in the wandb config to create entries of type "
            f"{cls}, since `key` wasn't passed, and the class doesn't have a '_stored_at_key' "
            f"attribute set."
        )

    if _api is None:
        _api = wandb.Api()
    run: Run = _api.run(run_path)

    config = dict(run.config)
    # if renamed_keys is not None:

    #     if not any(old_key.startswith(("/", key)) for old_key in renamed_keys):
    #         # Add the prefix, for convenience.
    #         renamed_keys = {
    #             key + "/" + old_key: key + "/" + new_key
    #             for old_key, new_key in renamed_keys.items()
    #         }

    # if removed_keys is not None and not any(
    #     removed_key.startswith(("/", key)) for removed_key in removed_keys
    # ):
    #     # Add the prefix, for convenience.
    #     removed_keys = [key + "/" + removed_key for removed_key in removed_keys]

    if renamed_keys:
        for source, dest in renamed_keys.items():
            if source not in config:
                raise RuntimeError(
                    f"can't find old key '{source}' in config with keys {config.keys()}"
                )
            if dest in config:
                raise RuntimeError(
                    f"Can't rename key '{source}' to '{dest}', there's already an entry at that "
                    f"value: {config[dest]}"
                )
            config[dest] = config.pop(source)
    if removed_keys:
        for k in removed_keys:
            if k not in config:
                raise RuntimeError(
                    f"Can't remove key '{k}' from the config, since it isn't there!\n"
                    f"keys in config: {list(config.keys())}"
                )
            config.pop(k)

    config = unflatten(config, sep="/")

    # print(json.dumps(config, indent="\t"))
    hparams_dict = config[key]
    return cls.from_dict(hparams_dict, drop_extra_fields=False)


def unflatten(dictionary, sep="/"):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(sep)
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict
