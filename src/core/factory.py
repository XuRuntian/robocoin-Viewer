# src/core/factory.py
from pathlib import Path
from src.adapters.hdf5_adapter import HDF5Adapter
from src.adapters.ros_adapter import RosAdapter
from src.adapters.unitree_adapter import UnitreeAdapter
from src.adapters.lerobot_adapter import LeRobotAdapter # <--- 新增
from src.adapters.folder_adapter import FolderAdapter

class ReaderFactory:
    @staticmethod
    def get_reader(file_path: str):
        path = Path(file_path)
        
        # 1. 文件夹判断
        if path.is_dir():
            # Unitree 检查
            if (path / "data.json").exists():
                return UnitreeAdapter()
            # LeRobot 检查: 有 data/*.parquet 或 meta/info.json
            has_meta_info = (path / "meta" / "info.json").exists()
            has_parquet = any(path.rglob("data/**/*.parquet")) or any(path.glob("*.parquet"))
            if has_meta_info or has_parquet:
                return LeRobotAdapter()
            # 默认
            return FolderAdapter()

        # 2. 文件判断
        ext = path.suffix.lower()
        if ext == '.parquet':
            return LeRobotAdapter() # 直接选 parquet 文件
        elif ext in ['.h5', '.hdf5']:
            return HDF5Adapter()
        elif ext in ['.bag', '.mcap']:
            return RosAdapter()
        elif ext == '.json':
            return UnitreeAdapter()
        else:
            raise ValueError(f"暂不支持该格式: {ext}")