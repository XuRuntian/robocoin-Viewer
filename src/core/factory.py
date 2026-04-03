# src/core/factory.py
from pathlib import Path
import json
from typing import Optional
from src.core.interface import BaseDatasetReader
from src.adapters.hdf5_adapter import HDF5Adapter
from src.adapters.ros_adapter import RosAdapter
from src.adapters.unitree_adapter import UnitreeAdapter
from src.adapters.folder_adapter import FolderAdapter
from src.adapters.lerobot_adapter import LeRobotAdapter

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
    def detect_type(path: Path) -> str:
        """只检测类型，不返回 Reader 实例（轻量级）"""
        if path.is_dir():
            # 1. 严格匹配结构化数据集根目录
            if (path / "data.json").exists(): return "Unitree"
            if (path / "meta" / "info.json").exists(): return "LeRobot"
            if (path / "metadata.yaml").exists(): return "ROS"
            
            # 2. 嵌套多轨迹探测 (仅当同级没有混合的独立数据文件时，才视为专属容器)
            has_loose_files = list(path.glob("*.hdf5")) or list(path.glob("*.h5")) or list(path.glob("*.bag")) or list(path.glob("*.mcap"))
            
            if not has_loose_files:
                if list(path.glob("*/meta/info.json")): return "LeRobot"
                if list(path.glob("*/data.json")): return "Unitree"
                if (path / "data").is_dir():
                    try:
                        next((path / "data").rglob("*.parquet"))
                        return "LeRobot"
                    except StopIteration: pass
                    
            # 3. 松散文件聚合模式
            if list(path.glob("*.hdf5")) or list(path.glob("*.h5")): return "HDF5"
            if list(path.glob("*.bag")) or list(path.glob("*.mcap")): return "ROS"
            if list(path.glob("*.jpg")) or list(path.glob("*.png")) or list(path.glob("colors/*.jpg")): return "RawFolder"
        else:
            # 4. 单文件精确匹配
            ext = path.suffix.lower()
            if ext in ['.h5', '.hdf5']: return "HDF5"
            if ext in ['.bag', '.mcap']: return "ROS"
            if ext == '.parquet': return "LeRobot"
        
        return "Unknown"

    @staticmethod
    def get_reader(file_path: str) -> BaseDatasetReader:
        path = Path(file_path)
        dtype = ReaderFactory.detect_type(path)
        
        if dtype == "Unitree": return UnitreeAdapter()
        if dtype == "LeRobot": return LeRobotAdapter()
        if dtype == "RawFolder": return FolderAdapter()
        if dtype == "HDF5": return HDF5Adapter()
        if dtype == "ROS": return RosAdapter()
        
        raise ValueError(f"无法识别的数据格式: {path.name} (绝对路径: {path.absolute()})")