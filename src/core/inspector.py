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
        self.valid_datasets = [] # 存储所有通过检查的数据集路径
        self.dominant_type = None

    def scan(self):
        print(f"🕵️‍♂️ 正在扫描目录: {self.root}")
        items = sorted([p for p in self.root.iterdir()])
        
        for p in items:
            if p.name.startswith("."): continue
            
            dtype = ReaderFactory.detect_type(p)
            self.stats[dtype] += 1
            
            info = {
                "name": p.name,
                "path": str(p),
                "type": dtype,
                "status": "OK" if dtype != "Unknown" else "⚠️ Unknown"
            }
            
            # 简单的文件完整性检查
            if dtype == "Unitree" and not (p / "data.json").exists():
                info["status"] = "❌ Missing data.json"
            
            self.report.append(info)
            if info["status"] == "OK":
                self.valid_datasets.append(str(p))

    def check_consistency(self) -> bool:
        """
        严厉的检查逻辑
        """
        print("\n" + "="*40)
        print("🔍 阶段一：格式一致性检查")
        print("="*40)
        
        # 1. 检查是否有 Unknown
        if self.stats["Unknown"] > 0:
            print(f"❌ 失败: 包含 {self.stats['Unknown']} 个未知格式的文件/文件夹。")
            self._print_problems()
            return False

        # 2. 检查是否只有一种类型
        valid_types = [t for t in self.stats.keys() if t != "Unknown"]
        if len(valid_types) > 1:
            print(f"❌ 失败: 检测到多种数据格式混合: {dict(self.stats)}")
            self._print_problems()
            return False
        
        if len(valid_types) == 0:
            print("❌ 失败: 目录下没有有效数据。")
            return False

        self.dominant_type = valid_types[0]
        print(f"✅ 通过: 目录下共 {len(self.valid_datasets)} 个数据，格式统一为 [{self.dominant_type}]")
        return True

    def _print_problems(self):
        df = pd.DataFrame(self.report)
        problems = df[df['status'].str.contains("Unknown|Corrupt|❌|⚠️")]
        if not problems.empty:
            print("\n🚨 问题数据清单:")
            print(problems[['name', 'type', 'status']].to_markdown(index=False))

    def get_all_valid_paths(self):
        return sorted(self.valid_datasets)