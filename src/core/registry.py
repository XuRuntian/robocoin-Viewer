from typing import Dict, Type
from src.core.interface import BaseDatasetReader

class AdapterRegistry:
    _adapters: Dict[str, Type[BaseDatasetReader]] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：将 Adapter 注册到全局字典中"""
        def wrapper(adapter_cls: Type[BaseDatasetReader]):
            cls._adapters[name] = adapter_cls
            return adapter_cls
        return wrapper

    @classmethod
    def get_class(cls, name: str) -> Type[BaseDatasetReader]:
        if name not in cls._adapters:
            raise ValueError(f"未注册的 Adapter 类型: {name}")
        return cls._adapters[name]