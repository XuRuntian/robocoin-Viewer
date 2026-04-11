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

                # [安全修复] 不再暴力删除旧数据，而是通过时间戳保证名字唯一
                if dst_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # stem 取文件名主体，suffix 取后缀
                    new_name = f"{src_path.stem}_{timestamp}{src_path.suffix}"
                    dst_path = type_folder / new_name
                    print(f"⚠️ 目标路径存在重名，已重命名为: {new_name}")

                print(f"Moving {src_path} -> {dst_path}")
                shutil.move(str(src_path), str(dst_path))
                new_type_paths.append(str(dst_path))

            new_paths[dtype] = new_type_paths

        return new_paths
    
    def quarantine_bad_data(self, bad_paths: list, root_dir: str) -> str:
        root_dir = Path(root_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_dir = root_dir / f"_QUARANTINE_{timestamp}"
        quarantine_dir.mkdir(exist_ok=True)
        
        manifest_path = quarantine_dir / "manifest.txt"
        
        with open(manifest_path, "w") as f:
            for path in bad_paths:
                path = Path(path)
                dst_path = quarantine_dir / path.name
                
                # [安全修复] 如果隔离区发生极其罕见的内部重名，也同样加后缀
                if dst_path.exists():
                    time_suffix = datetime.now().strftime("%H%M%S")
                    new_name = f"{path.stem}_{time_suffix}{path.suffix}"
                    dst_path = quarantine_dir / new_name
                
                print(f"Moving {path} -> {dst_path}")
                shutil.move(str(path), str(dst_path))
                
                f.write(f"Original Path: {path}\n")
                f.write(f"Moved to: {dst_path}\n")
                f.write("-" * 50 + "\n")
        
        return str(quarantine_dir)