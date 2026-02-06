# src/core/factory.py
from pathlib import Path
from typing import Optional
from src.core.interface import BaseDatasetReader
from src.adapters.hdf5_adapter import HDF5Adapter
from src.adapters.ros_adapter import RosAdapter
from src.adapters.unitree_adapter import UnitreeAdapter
from src.adapters.folder_adapter import FolderAdapter
from src.adapters.lerobot_adapter import LeRobotAdapter # <--- 确保导入

class ReaderFactory:
    @staticmethod
    def detect_type(path: Path) -> str:
        """
        只检测类型，不返回 Reader 实例（轻量级）
        """
        if path.is_dir():
            # Unitree 特征
            if (path / "data.json").exists():
                return "Unitree"
            # LeRobot 特征 (包含 parquet 且有 meta 或 data 目录)
            if list(path.glob("*.parquet")) or list(path.glob("data/*.parquet")):
                return "LeRobot"
            # Folder 特征 (包含图片)
            if list(path.glob("*.jpg")) or list(path.glob("*.png")) or \
               list(path.glob("colors/*.jpg")):
                return "RawFolder"
        else:
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
        
        raise ValueError(f"无法识别的数据格式: {path.name}")