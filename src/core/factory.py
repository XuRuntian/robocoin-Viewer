# src/core/factory.py
import json
from typing import Optional
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

    @classmethod
    def get_reader(cls, file_path: str, rule_name: Optional[str] = None) -> BaseDatasetReader:
        path = Path(file_path)
        all_rules = cls.load_rules()  # 加载 adapter_rules.json
        
        config_dict = {}
        base_type = None

        # --- 第一优先级：CLI 显式指定 rule_name ---
        if rule_name is not None:
            if rule_name not in all_rules:
                raise ValueError(f"❌ 指定的规则 '{rule_name}' 不存在于 adapter_rules.json 中！")
            
            config_dict = all_rules[rule_name]
            # 防御：确保配置是字典
            if not isinstance(config_dict, dict):
                raise ValueError(f"❌ 规则 '{rule_name}' 的内容格式错误，应为 JSON 对象。")
                
            base_type = config_dict.get("base_type")
            if not base_type:
                raise ValueError(f"❌ 规则 '{rule_name}' 缺少核心字段 'base_type'！")
                
            print(f"🎯 [Factory] 强行使用指定规则: {rule_name} -> 基类 {base_type}")
            
        # --- 第二优先级：自动匹配逻辑 (指纹匹配 + 物理探测) ---
        else:
            # 1. 物理层级探测 (获取基础格式，如 ROS, HDF5, Unitree)
            base_type = cls.detect_type(path)
            if base_type == "Unknown":
                raise ValueError(f"无法识别的数据物理格式: {path.name}")

            # 2. 尝试匹配具体的业务规则 (指纹识别)
            # 💡 修复重点：跳过非字典项，防止 'str' object has no attribute 'get'
            for profile_name, data in all_rules.items():
                if not isinstance(data, dict):
                    continue  # 比如跳过 "_instruction": "..."
                
                # 如果业务规则匹配上了 (通过路径、后缀等)
                if cls._evaluate_rules(path, data.get("match_rules", {})):
                    config_dict = data
                    rule_name = profile_name
                    base_type = data.get("base_type") or base_type
                    print(f"🔍 [Factory] 业务匹配成功: {rule_name}")
                    break
                    
            # 3. 如果指纹没匹配上，回退到基础格式的默认配置
            if not config_dict:
                config_dict = all_rules.get(base_type, {})
                # 再次防御：如果基础格式在 rules 里也是个字符串或者不存在
                if not isinstance(config_dict, dict):
                    config_dict = {}
                print(f"✨ [Factory] 无特定业务规则，使用基类默认配置: {base_type}")

        # --- 后续实例化逻辑保持不变 ---
        extra_opts = config_dict.get("extra_options", {}).copy() if isinstance(config_dict, dict) else {}
        
        adapter_config = AdapterConfig(
            length_reference_key=config_dict.get("length_reference_key", ""),
            image_keys_map=config_dict.get("cameras", {}),
            arm_groups=config_dict.get("arm_groups", {}), 
            state_keys_map=config_dict.get("base", {}),
            extra_options=extra_opts
        )
        
        adapter_class = AdapterRegistry.get_class(base_type)
        return adapter_class(config=adapter_config)