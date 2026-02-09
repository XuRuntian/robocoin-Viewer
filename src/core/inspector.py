# src/core/inspector.py
import os
from pathlib import Path
from collections import defaultdict
from src.core.factory import ReaderFactory
import pandas as pd

class DatasetInspector:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.report = []
        self.stats = defaultdict(int)
        self.grouped_datasets = defaultdict(list)  # 按类型存储有效数据集
        self.dominant_type = None

    def scan(self):
        print(f"🕵️‍♂️ 正在扫描目录: {self.root}")
        
        # 使用 os.walk 进行递归扫描
        for root_path, dirs, files in os.walk(self.root):
            current_path = Path(root_path)
            
            # 跳过隐藏目录
            if any(part.startswith('.') for part in current_path.parts):
                continue
                
            # 检查当前目录是否为有效数据集
            dtype = ReaderFactory.detect_type(current_path)
            
            # 如果是有效数据集（非 Unknown 且非 RawFolder）
            if dtype not in ("Unknown", "RawFolder"):
                self.stats[dtype] += 1
                
                info = {
                    "name": current_path.name,
                    "path": str(current_path),
                    "type": dtype,
                    "status": "OK"
                }
                
                self.report.append(info)
                self.grouped_datasets[dtype].append(str(current_path))
                
                # 跳过对该目录内容的进一步递归
                dirs[:] = []
            else:
                # 否则继续处理子目录和文件
                pass

    def check_consistency(self) -> bool:
        """
        放宽的检查逻辑 - 允许混合类型
        """
        print("\n" + "="*40)
        print("🔍 阶段一：格式一致性检查")
        print("="*40)
        
        # 1. 检查是否有 Unknown
        if self.stats["Unknown"] > 0:
            print(f"❌ 失败: 包含 {self.stats['Unknown']} 个未知格式的文件/文件夹。")
            self._print_problems()
            return False

        valid_types = [t for t in self.stats.keys() if t != "Unknown"]
        if len(valid_types) == 0:
            print("❌ 失败: 目录下没有有效数据。")
            return False

        print(f"✅ 通过: 目录下共 {sum(len(v) for v in self.grouped_datasets.values())} 个数据，包含类型: {valid_types}")
        return True

    def _print_problems(self):
        df = pd.DataFrame(self.report)
        problems = df[df['status'].str.contains("Unknown|Corrupt|❌|⚠️")]
        if not problems.empty:
            print("\n🚨 问题数据清单:")
            print(problems[['name', 'type', 'status']].to_markdown(index=False))

    def get_all_valid_paths(self):
        all_paths = []
        for paths in self.grouped_datasets.values():
            all_paths.extend(paths)
        return sorted(all_paths)
