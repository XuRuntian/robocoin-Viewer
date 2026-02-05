# src/adapters/lerobot_adapter.py
import json
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from src.core.interface import BaseDatasetReader, FrameData

class LeRobotAdapter(BaseDatasetReader):
    def __init__(self):
        self.root_path = None
        self.df = None
        self.fps = 30.0
        self.image_keys = []      # 存储 UI 显示用的短名称: image_left
        self.full_feature_keys = {} # 映射短名称到完整特征名: {"image_left": "observation.images.image_left"}
        self.image_path_tpl = ""
        self.cap_cache = {} 

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        
        # 1. 递归搜索 .parquet 文件
        parquet_files = sorted(list(self.root_path.rglob("data/**/*.parquet")))
        if not parquet_files:
            parquet_files = sorted(list(self.root_path.glob("*.parquet")))

        if not parquet_files:
            print(f"❌ [LeRobot] 找不到 parquet 数据")
            return False

        try:
            self.df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
            
            # 2. 解析 info.json
            meta_path = self.root_path / "meta" / "info.json"
            if not meta_path.exists():
                print("❌ [LeRobot] 缺少 meta/info.json")
                return False

            with open(meta_path, 'r') as f:
                info = json.load(f)
                self.fps = info.get("fps", 30.0)
                self.image_path_tpl = info.get("image_path", "")
                
                features = info.get("features", {})
                self.image_keys = []
                self.full_feature_keys = {}
                
                for key, val in features.items():
                    if isinstance(val, dict) and val.get("dtype") == "image":
                        # 短名称用于 UI (如 image_left)
                        short_name = key.split(".")[-1]
                        self.image_keys.append(short_name)
                        # 完整名称用于对应目录名 (如 observation.images.image_left)
                        self.full_feature_keys[short_name] = key
                
            print(f"✅ [LeRobot] 加载成功: {len(self.df)} 帧, 相机: {self.image_keys}")
            return True

        except Exception as e:
            print(f"❌ [LeRobot] 加载异常: {e}")
            return False

    def get_length(self) -> int:
        return len(self.df) if self.df is not None else 0

    def get_all_sensors(self) -> List[str]:
        return self.image_keys

    def get_frame(self, index: int) -> FrameData:
        if self.df is None or index >= len(self.df): return None
        
        row = self.df.iloc[index]
        images = {}
        
        # 获取索引信息
        ep_idx = int(row["episode_index"])
        frame_idx = int(row["frame_index"])
        
        for short_name in self.image_keys:
            # 获取对应的完整目录名 (如 observation.images.image_left)
            full_key = self.full_feature_keys[short_name]
            
            # 填充模板
            # 你的路径是: images/observation.images.image_left/episode_000000/frame_000000.jpg
            # 模板是: images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg
            if self.image_path_tpl:
                rel_path = self.image_path_tpl.format(
                    image_key=full_key,  # 这里传入完整的 observation.images.xxx
                    episode_index=ep_idx,
                    frame_index=frame_idx
                )
                full_path = self.root_path / rel_path
                
                if full_path.exists():
                    # 使用 cv2 读取并转换颜色
                    img = cv2.imread(str(full_path))
                    if img is not None:
                        images[short_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    # 打印一次调试信息，确认路径拼接是否正确
                    if index == 0:
                        print(f"⚠️ [LeRobot] 图片未找到: {full_path}")

        # 状态数据
        state = {
            "action": np.array(row["action"]) if "action" in row else None,
            "qpos": np.array(row["observation.state"]) if "observation.state" in row else None
        }

        return FrameData(
            timestamp=float(row.get("timestamp", index / self.fps)),
            images=images,
            state={k: v for k, v in state.items() if v is not None}
        )

    def close(self):
        pass