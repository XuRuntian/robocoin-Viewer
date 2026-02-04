# src/core/factory.py
from pathlib import Path
from src.adapters.hdf5_adapter import HDF5Adapter
from src.adapters.ros_adapter import RosAdapter
from src.adapters.unitree_adapter import UnitreeAdapter # <--- 新增
from src.adapters.folder_adapter import FolderAdapter

class ReaderFactory:
    @staticmethod
    def get_reader(file_path: str):
        path = Path(file_path)
        
        # 1. 如果是文件夹
        if path.is_dir():
            # 优先检查有没有 data.json，且内容包含 "unitree"
            json_path = path / "data.json"
            if json_path.exists():
                # 简单读一下头，确认是不是 Unitree 格式
                try:
                    import json
                    with open(json_path, 'r') as f:
                        header = json.load(f)
                        # 检查特征
                        if "info" in header and header.get("info", {}).get("author") == "unitree":
                            return UnitreeAdapter()
                except:
                    pass
            
            # 如果不是 Unitree，回退到暴力扫描模式
            return FolderAdapter()

        # 2. 如果是文件
        ext = path.suffix.lower()
        if ext == '.json':
             return UnitreeAdapter() # 直接选了 data.json 文件
        elif ext in ['.h5', '.hdf5']:
            return HDF5Adapter()
        elif ext in ['.bag', '.mcap']:
            return RosAdapter()
        else:
            raise ValueError(f"暂不支持该格式: {ext}")