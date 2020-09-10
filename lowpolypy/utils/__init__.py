from typing import Type
from pathlib import Path
from omegaconf import OmegaConf


class registry:
    registries = {}

    @classmethod
    def register(cls, registry_name: str, key: str):
        """
        Decorator for adding an entry to a registry.

        Args:
            registry_name: Name of the registry to add the entry to
            key: Name to file the entry under
            value: The value of the entry
        """

        def inner(obj):
            registry = cls.registries.setdefault(registry_name, {})
            registry[key] = obj
            return obj

        return inner

    @classmethod
    def get(cls, registry_name: str, key: str, allow_passthrough=True):
        """
        Get an element from a registry

        Args:
            registry_name: Name of the registry.
            key: Entry key in the specified registry to retrieve.
            allow_passthrough: If True, then if `key` is not a key in the specified registry but is present as a value in the registry, `key` is returned.
        """
        try:
            registry = cls.registries[registry_name]
        except KeyError as e:
            raise KeyError(f"No such registry: '{registry_name}'") from e
        try:
            return registry[key]
        except KeyError as e:
            if key in registry.values() and allow_passthrough:
                return key
            raise KeyError(
                f"Couldn't find '{key}' in registry '{registry_name}'"
            ) from e

    # TODO: Implement custom registration the right way
    # @classmethod
    # def register_model(cls, name: str, model_class: Type):
    #     return cls.register("model", name)(model_class)
    #
    # @classmethod
    # def register_dataset(cls, name: str, dataset_class: Type):
    #     return cls.register("dataset", name)(dataset_class)


def load_config(config_path="config.yaml", default=True, cli=True):
    yaml_config = OmegaConf.load(Path(__file__).parents[1] / config_path)
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(yaml_config, cli_config)
    return config
