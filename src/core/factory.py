# src/core/factory.py
import os
from src.adapters.hdf5_adapter import HDF5Adapter
from src.adapters.ros_adapter import RosAdapter # <--- 新增导入

class ReaderFactory:
    @staticmethod
    def get_reader(file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.h5', '.hdf5']:
            return HDF5Adapter()
        # 增加对 .bag 和 .mcap 的支持
        elif ext in ['.bag', '.mcap']:
            return RosAdapter()
        else:
            raise ValueError(f"暂不支持该格式: {ext}")