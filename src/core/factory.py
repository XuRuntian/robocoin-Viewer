# src/core/factory.py
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import src.adapters
from src.core.registry import AdapterRegistry
from src.core.interface import BaseDatasetReader, AdapterConfig

class ReaderFactory:
    _rules_cache = None

    @classmethod
    def load_rules(cls) -> dict:
        if cls._rules_cache is None:
            rules_path = Path("configs/adapter_rules.json")
            if rules_path.exists():
                with open(rules_path, 'r', encoding='utf-8') as f:
                    cls._rules_cache = json.load(f)
            else:
                cls._rules_cache = {}
        return cls._rules_cache

    @staticmethod
    def _evaluate_rules(path: Path, match_rules: dict) -> bool:
        """轻量级业务规则匹配器 (如匹配智平方)"""
        if not match_rules: return False
        if "file_extensions" in match_rules and path.suffix.lower() not in match_rules["file_extensions"]:
            return False
        if "path_keywords" in match_rules:
            path_str = str(path.absolute())
            if not any(kw in path_str for kw in match_rules["path_keywords"]):
                return False
        return True

    @staticmethod
    def detect_type(path: Path) -> str:
        """
        🚀 完美还原你原本的物理层级探测逻辑！
        这个方法供 src/core/inspector.py 扫描目录时使用。
        """
        if path.is_dir():
            if (path / "data.json").exists(): return "Unitree"
            if (path / "meta" / "info.json").exists(): return "LeRobot"
            if (path / "metadata.yaml").exists(): return "ROS"
            
            has_loose_files = list(path.glob("*.hdf5")) or list(path.glob("*.h5")) or list(path.glob("*.bag")) or list(path.glob("*.mcap"))
            if not has_loose_files:
                if list(path.glob("*/meta/info.json")): return "LeRobot"
                if list(path.glob("*/data.json")): return "Unitree"
                if (path / "data").is_dir():
                    try:
                        next((path / "data").rglob("*.parquet"))
                        return "LeRobot"
                    except StopIteration: pass
                    
            if list(path.glob("*.hdf5")) or list(path.glob("*.h5")): return "HDF5"
            if list(path.glob("*.bag")) or list(path.glob("*.mcap")): return "ROS"
            if list(path.glob("*.jpg")) or list(path.glob("*.png")) or list(path.glob("colors/*.jpg")): return "RawFolder"
        else:
            ext = path.suffix.lower()
            if ext in ['.h5', '.hdf5']: return "HDF5"
            if ext in ['.bag', '.mcap']: return "ROS"
            if ext == '.parquet': return "LeRobot"
        
        return "Unknown"

    @staticmethod
    def get_reader(file_path: str) -> BaseDatasetReader:
        """
        实例化入口：结合探测器和规则引擎
        """
        path = Path(file_path)
        
        # 1. 物理探测 (识别它是 HDF5, ROS 还是 LeRobot)
        base_type = ReaderFactory.detect_type(path)
        if base_type == "Unknown":
            raise ValueError(f"无法识别的数据格式: {path.name} (绝对路径: {path.absolute()})")

        # 2. 匹配业务规则 Config
        all_rules = ReaderFactory.load_rules()
        config_dict = {}
        
        # 先尝试匹配具体业务 (如 ZhiPingFang)
        for profile_name, data in all_rules.items():
            if ReaderFactory._evaluate_rules(path, data.get("match_rules", {})):
                config_dict = data
                break
                
        # 如果没匹配到具体业务，就用基础格式的默认规则 (如 "ROS" 或 "HDF5")
        if not config_dict:
            config_dict = all_rules.get(base_type, {})
            
        # 3. 组装 Config
        adapter_config = AdapterConfig(
            length_reference_key=config_dict.get("length_reference_key", ""),
            image_keys_map=config_dict.get("image_keys_map", {}),
            state_keys_map=config_dict.get("state_keys_map", {}),
            extra_options=config_dict.get("extra_options", {})
        )
        
        # 4. 从注册表动态获取类并实例化，注入 Config
        adapter_class = AdapterRegistry.get_class(base_type)
        return adapter_class(config=adapter_config)