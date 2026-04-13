from typing import Type, Dict, Any

class AdapterRegistry:
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(adapter_cls):
            cls._registry[name] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def get_class(cls, name: str) -> Type:
        if name not in cls._registry:
            raise ValueError(f"未注册的 Adapter 类型: {name}")
        return cls._registry[name]