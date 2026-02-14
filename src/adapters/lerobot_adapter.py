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
                ep_idx = int(row["episode_index"])
                frame_idx = int(row["frame_index"])
                
                # 优先尝试静态图片路径
                if self.image_path_tpl:
                    rel_path = self.image_path_tpl.format(
                        image_key=full_key,
                        episode_index=ep_idx,
                        frame_index=frame_idx
                    )
                    full_path = self.root_path / rel_path
                    
                    if full_path.exists():
                        # 使用 cv2 读取并转换颜色
                        img = cv2.imread(str(full_path))
                        if img is not None:
                            images[short_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            continue  # 成功读取图片，跳过视频逻辑
                
                # 静态图片不存在或没有image_path模板时尝试视频路径
                if self.video_path_tpl or not self.image_path_tpl:
                    # 构建视频路径模板
                    if self.video_path_tpl:
                        video_path_str = self.video_path_tpl.format(
                            image_key=full_key,
                            ep_idx=ep_idx
                        )
                    else:
                        # 默认视频路径结构包含chunk目录
                        video_path_str = f"videos/chunk-000/{full_key}/episode_{ep_idx:06d}.mp4"
                    
                    # 查找匹配的视频文件（考虑chunk目录结构）
                    video_files = list(self.root_path.rglob(f"**/{full_key}/*episode_{ep_idx:06d}.mp4"))
                    
                    if video_files:
                        video_path = video_files[0]
                        local_frame_index = frame_idx
                        
                        # 缓存处理
                        cap = self.cap_cache.get(str(video_path))
                        if not cap:
                            cap = cv2.VideoCapture(str(video_path))
                            self.cap_cache[str(video_path)] = cap
                        
                        # 设置帧位置
                        cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame_index)
                        ret, frame = cap.read()
                        
                    if ret:
                        images[short_name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        print(f"❌ [LeRobot] 视频读取失败: {video_path} 帧 {local_frame_index}")

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
