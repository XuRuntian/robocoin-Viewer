# src/core/organizer.py
import shutil
from pathlib import Path
from datetime import datetime
import os
from src.core.factory import ReaderFactory

class DatasetOrganizer:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
    
    def auto_organize(self):
        """
        全自动整理：扫描根目录，判断类型，并移动到 grouped_XXX 文件夹中。
        返回整理后的统计信息。
        """
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"无效的根目录: {self.root}")

        print(f"\n🔍 开始扫描并整理目录: {self.root}")
        
        # 1. 扫描并分组
        grouped_datasets = {
            "HDF5": [], "LeRobot": [], "ROS": [], 
            "Unitree": [], "RawFolder": [], "Unknown": []
        }
        
        for item in self.root.iterdir():
            # 跳过系统隐藏文件或已经分好类的文件夹
            if item.name.startswith('.') or item.name.startswith('grouped_') or item.name.startswith('_QUARANTINE_'):
                continue
                
            # 利用强大的 Factory 自动嗅探格式
            dtype = ReaderFactory.detect_type(item)
            if dtype in grouped_datasets:
                grouped_datasets[dtype].append(item)
            else:
                grouped_datasets["Unknown"].append(item)

        # 2. 移动分类好的数据
        result_summary = {}
        for dtype, paths in grouped_datasets.items():
            if not paths:
                continue
                
            if dtype == "Unknown":
                # 无法识别的数据送入隔离区
                target_dir = self.quarantine_bad_data(paths, self.root)
                result_summary["隔离区 (Unknown)"] = len(paths)
            else:
                # 正常数据放入 grouped_XXX
                target_dir = self.root / f"grouped_{dtype}"
                target_dir.mkdir(exist_ok=True)
                
                for src_path in paths:
                    dst_path = target_dir / src_path.name
                    # 如果目标存在，先删掉，防止 shutil.move 报错
                    if dst_path.exists():
                        if dst_path.is_dir():
                            shutil.rmtree(dst_path)
                        else:
                            dst_path.unlink()
                            
                    print(f"📦 移动 [{dtype}]: {src_path.name} -> {target_dir.name}/")
                    shutil.move(str(src_path), str(dst_path))
                
                result_summary[dtype] = len(paths)
                
        print("\n✅ 整理完成！统计结果：")
        for k, v in result_summary.items():
            print(f" - {k}: {v} 个项")
            
        return result_summary

    def quarantine_bad_data(self, bad_paths: list, root_dir: Path) -> str:
        """将无法识别或损坏的数据移动到隔离区"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_dir = root_dir / f"_QUARANTINE_{timestamp}"
        quarantine_dir.mkdir(exist_ok=True)
        
        manifest_path = quarantine_dir / "manifest.txt"
        with open(manifest_path, "w", encoding='utf-8') as f:
            for path in bad_paths:
                dst_path = quarantine_dir / path.name
                if dst_path.exists():
                    if dst_path.is_dir(): shutil.rmtree(dst_path)
                    else: dst_path.unlink()
                    
                shutil.move(str(path), str(dst_path))
                
                f.write(f"原始路径: {path}\n")
                f.write(f"移动至: {dst_path}\n")
                f.write("-" * 50 + "\n")
                print(f"🚨 隔离未知数据: {path.name}")
                
        return str(quarantine_dir)