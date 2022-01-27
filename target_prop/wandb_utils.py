""" Wandb utilities. """
from typing import ClassVar, List, TypeVar
import json
import wandb
from typing import Type, TypeVar, Dict, Optional
from simple_parsing.helpers.serialization.serializable import Serializable
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from wandb.apis.public import Run
from abc import ABC
from pathlib import Path

try:
    from simple_parsing.helpers.serialization.serializable import FrozenSerializable
except ImportError:
    FrozenSerializable = Serializable

S = TypeVar("S", bound=Serializable)

_api: Optional[wandb.Api] = None


class LoggedToWandb:
    # Class attribute that indicates where objects of this type are to be found in the wandb config.
    # NOTE: Either this value needs to be set, or the key need to be passed every time to
    # `from_run`.
    _stored_at_key: ClassVar[Optional[str]] = None

    @classmethod
    def from_run(
        cls,
        run_path: str,
        key: str = None,
        renamed_keys: Dict[str, str] = None,
        removed_keys: List[str] = None,
        cache_dir: Path = None,
    ):
        if key is None:
            key = cls._stored_at_key
        if key is None:
            raise RuntimeError(
                f"Don't know which key to look at in the wandb config to create entries of type "
                f"{cls}, since `key` wasn't passed, and {cls._stored_at_key=}."
            )
        if not issubclass(cls, (Serializable, FrozenSerializable)):
            raise NotImplementedError(
                f"LoggedToWandb is supposed to be used on a subclass of either Serializable or "
                f"FrozenSerializable."
            )
        if cache_dir is not None:
            cached_file = cache_dir / Path(run_path.replace("/", "_")) / f"{cls.__name__}.json"
            if cached_file.exists():
                print(f"Loading from cached file at {cached_file}")
                return cls.load_json(cached_file)

        try:
            instance = load_from_run(
                cls, run_path, key=key, renamed_keys=renamed_keys, removed_keys=removed_keys
            )
        except RuntimeError as err:
            raise RuntimeError(f"Unable to instantiate class {cls} from run {run_path}: {err}")

        if cache_dir is not None:
            cached_file = cache_dir / Path(run_path.replace("/", "_")) / f"{cls.__name__}.json"
            cached_file.parent.mkdir(exist_ok=True, parents=True)
            instance.save_json(cached_file)
        return instance


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
