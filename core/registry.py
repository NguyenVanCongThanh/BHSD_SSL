"""
Extensible Component Registry for MedSSL-BHSD Framework.
Allows dynamic registration and instantiation of Models, SSL Methods, Datasets, and Losses.
Eliminates brittle if-else chains.
"""

import importlib
from typing import Any, Callable, Dict, Optional, Sequence, Type


class Registry:
    """
    Registry maps string identifiers to Python classes or factory callables.
    """
    def __init__(self, name: str, bootstrap_modules: Sequence[str] = ()):
        self._name = name
        self._module_dict: Dict[str, Any] = {}
        self._bootstrap_modules = tuple(bootstrap_modules)
        self._bootstrapped = False

    def _ensure_bootstrapped(self) -> None:
        if self._bootstrapped or not self._bootstrap_modules:
            return
        self._bootstrapped = True
        for mod_name in self._bootstrap_modules:
            importlib.import_module(mod_name)

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a class or function.
        If name is None, class.__name__.lower() will be used.
        """
        def _register(cls_or_func: Any) -> Any:
            key = name if name is not None else cls_or_func.__name__.lower()
            key_lower = key.lower()
            if key_lower in self._module_dict:
                pass  # Allow overwrite / re-registration gracefully
            self._module_dict[key_lower] = cls_or_func
            return cls_or_func

        return _register

    def get(self, name: str) -> Any:
        """Retrieve registered object by name (case-insensitive)."""
        key = name.lower()
        if key not in self._module_dict:
            self._ensure_bootstrapped()
        if key not in self._module_dict:
            available = ", ".join(sorted(self._module_dict.keys()))
            raise KeyError(
                f"'{name}' is not registered in {self._name} registry! Available: [{available}]"
            )
        return self._module_dict[key]

    def build(self, name: str, *args, **kwargs) -> Any:
        """Retrieve registered object and instantiate with given arguments."""
        obj_cls = self.get(name)
        return obj_cls(*args, **kwargs)

    def contains(self, name: str) -> bool:
        key = name.lower()
        if key not in self._module_dict:
            self._ensure_bootstrapped()
        return key in self._module_dict

    def list_keys(self) -> list:
        if not self._module_dict:
            self._ensure_bootstrapped()
        return sorted(list(self._module_dict.keys()))


# Global registries
MODELS = Registry("models")
METHODS = Registry("methods")
DATASETS = Registry("datasets", bootstrap_modules=("data.dataset_bhsd",))
LOSSES = Registry("losses")
