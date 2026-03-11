import shutil
from pathlib import Path
from datetime import datetime
import os

class DatasetOrganizer:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
    
    def sort_by_type(self, grouped_datasets: dict, target_root: str) -> dict:
        target_root = Path(target_root).resolve()
        base_name = target_root.name
        new_paths = {}

        for dtype, paths in grouped_datasets.items():
            if not paths:
                continue

            type_folder = target_root / f"{base_name}_{dtype.lower()}"
            type_folder.mkdir(exist_ok=True)

            new_type_paths = []
            for src_path in paths:
                src_path = Path(src_path).resolve()

                if src_path == target_root:
                    print(f"⚠️ 无法移动根目录自身，跳过: {src_path}")
                    new_type_paths.append(str(src_path))
                    continue

                if src_path.parent == type_folder:
                    new_type_paths.append(str(src_path))
                    continue

                dst_path = type_folder / src_path.name

                # 💡 修复点：安全地覆盖同名的文件或文件夹
                if dst_path.exists():
                    if dst_path.is_dir():
                        shutil.rmtree(dst_path)
                    else:
                        dst_path.unlink()

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
