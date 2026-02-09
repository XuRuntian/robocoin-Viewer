import shutil
from pathlib import Path
from datetime import datetime
import os

class DatasetOrganizer:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
    
    def sort_by_type(self, grouped_datasets: dict, target_root: str) -> dict:
        """
        将数据集按类型分类并移动到目标目录下的相应文件夹中
        
        Args:
            grouped_datasets: 包含不同类型数据集路径的字典
            target_root: 目标根目录路径
            
        Returns:
            包含移动后新路径的字典
        """
        target_root = Path(target_root)
        new_paths = {}
        
        for dtype, paths in grouped_datasets.items():
            if not paths:
                continue
                
            # 为每种类型创建子文件夹
            type_folder = target_root / f"grouped_{dtype}"
            type_folder.mkdir(exist_ok=True)
            
            new_type_paths = []
            for src_path in paths:
                src_path = Path(src_path)
                dst_path = type_folder / src_path.name
                
                # 如果目标位置已存在同名文件夹，先删除
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                
                # 移动文件夹并记录新路径
                print(f"Moving {src_path} -> {dst_path}")
                shutil.move(str(src_path), str(dst_path))
                new_type_paths.append(str(dst_path))
            
            new_paths[dtype] = new_type_paths
        
        return new_paths
    
    def quarantine_bad_data(self, bad_paths: list, root_dir: str) -> str:
        """
        将无效数据集移动到隔离区并生成清单文件
        
        Args:
            bad_paths: 需要隔离的文件路径列表
            root_dir: 根目录路径
            
        Returns:
            隔离区路径
        """
        root_dir = Path(root_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_dir = root_dir / f"_QUARANTINE_{timestamp}"
        quarantine_dir.mkdir(exist_ok=True)
        
        manifest_path = quarantine_dir / "manifest.txt"
        
        with open(manifest_path, "w") as f:
            for path in bad_paths:
                path = Path(path)
                dst_path = quarantine_dir / path.name
                
                # 如果目标位置已存在同名文件夹，先删除
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                
                # 移动文件夹并记录信息
                print(f"Moving {path} -> {dst_path}")
                shutil.move(str(path), str(dst_path))
                
                # 记录原始信息
                f.write(f"Original Path: {path}\n")
                f.write(f"Moved to: {dst_path}\n")
                f.write("-" * 50 + "\n")
        
        return str(quarantine_dir)
