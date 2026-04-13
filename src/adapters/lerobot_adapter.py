# src/adapters/lerobot_adapter.py
import json
import pandas as pd
import cv2
import numpy as np
import imageio
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.core.interface import BaseDatasetReader, FrameData, AdapterConfig
from src.core.registry import AdapterRegistry

@AdapterRegistry.register("LeRobot")
class LeRobotAdapter(BaseDatasetReader):
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.root_path = None
        self.current_dataset_root = None
        self.df = None
        
        # 1. 基础配置标准化
        self.camera_map = getattr(self.config, 'image_keys_map', {}) or {}
        self.arm_groups = getattr(self.config, 'arm_groups', {}) or {}
        self.base_map = getattr(self.config, 'state_keys_map', {}) or {}
        
        extra_opts = getattr(self.config, 'extra_options', {}) or {}
        
        self.fps = 30.0
        self.image_keys = []
        self.full_feature_keys = {}
        self.image_path_tpl = ""
        self.video_path_tpl = ""  
        self.cap_cache = {} 
        self.version = ""
        self.dorobot_version = "" 
        
        self.episodes_meta = [] 
        self.current_episode_idx = 0

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        self.episodes_meta = []
        
        meta_paths = []
        if (self.root_path / "meta" / "info.json").exists():
            meta_paths.append(self.root_path / "meta" / "info.json")
        else:
            meta_paths = sorted(list(self.root_path.glob("*/meta/info.json")))
            
        if not meta_paths:
            print(f"❌ [LeRobot] 找不到任何 meta/info.json")
            return False

        for meta_path in meta_paths:
            dataset_root = meta_path.parent.parent
            try:
                with open(meta_path, 'r') as f: info = json.load(f)
                parquet_files = sorted(list(dataset_root.rglob("data/**/*.parquet")))
                if not parquet_files: parquet_files = sorted(list(dataset_root.glob("*.parquet")))
                for pq in parquet_files:
                    self.episodes_meta.append({"root": dataset_root, "parquet": pq, "info": info})
            except Exception as e:
                print(f"⚠️ [LeRobot] 读取 {meta_path} 失败: {e}")

        if not self.episodes_meta: return False
            
        self.set_episode(0)
        return True

    def set_episode(self, episode_idx: int):
        if episode_idx < 0 or episode_idx >= len(self.episodes_meta): return
        self.current_episode_idx = episode_idx
        self.close() 
        
        ep_meta = self.episodes_meta[episode_idx]
        self.current_dataset_root = ep_meta["root"]
        parquet_path = ep_meta["parquet"]
        info = ep_meta["info"]
        
        self.version = info.get("codebase_version", "v2.1")
        self.dorobot_version = info.get("dorobot_dataset_version", "")
        self.fps = info.get("fps", 30.0)
        self.image_path_tpl = info.get("image_path", "")
        
        features = info.get("features", {})
        self.image_keys = []
        self.full_feature_keys = {}
        
        # 兼容自动探测与手动 Config 映射
        if self.camera_map:
            self.image_keys = list(self.camera_map.keys())
            self.full_feature_keys = self.camera_map
        else:
            for key, val in features.items():
                if isinstance(val, dict) and val.get("dtype") in ["image", "video"]:
                    short_name = key.split(".")[-1]
                    self.image_keys.append(short_name)
                    self.full_feature_keys[short_name] = key
                
        self.df = pd.read_parquet(parquet_path)
        print(f"🔄 [LeRobot] 切换至 Episode {episode_idx}, 帧数: {len(self.df)}")

    def get_total_episodes(self) -> int: return len(self.episodes_meta)
    def get_length(self) -> int: return len(self.df) if self.df is not None else 0
    def get_all_sensors(self) -> List[str]: return self.image_keys

    def get_frame(self, index: int, specific_cameras: Optional[List[str]] = None) -> FrameData:
        if self.df is None or index >= len(self.df): return None
        row = self.df.iloc[index]
        images = {}
        
        ep_idx = int(row["episode_index"])
        frame_idx = int(row["frame_index"])
        keys_to_fetch = specific_cameras if specific_cameras else self.image_keys
        
        for short_name in keys_to_fetch:
            full_key = self.full_feature_keys.get(short_name)
            if not full_key: continue
            
            img_loaded = False
            
            # 策略1: 检查 Parquet 数据列中是否本身就存了 raw bytes（针对无图床的情况）
            if full_key in row and isinstance(row[full_key], bytes):
                img_data = cv2.imdecode(np.frombuffer(row[full_key], np.uint8), cv2.IMREAD_COLOR)
                if img_data is not None:
                    images[short_name] = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                    img_loaded = True

            # 策略2: 基于模版组装图片路径
            if not img_loaded and self.image_path_tpl:
                for key_variant in [short_name, full_key]:
                    rel_path = self.image_path_tpl.format(image_key=key_variant, episode_index=ep_idx, frame_index=frame_idx)
                    full_path = self.current_dataset_root / rel_path
                    if full_path.exists():
                        img = cv2.imread(str(full_path))
                        if img is not None:
                            images[short_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_loaded = True
                            break
            
            # 策略3: 加载本地压缩视频帧
            if not img_loaded:
                video_files = list(self.current_dataset_root.rglob(f"**/{full_key}/*episode_{ep_idx:06d}.mp4"))
                if video_files:
                    video_path = video_files[0]
                    try:
                        reader = self.cap_cache.get(str(video_path))
                        if not reader:
                            reader = imageio.get_reader(str(video_path), 'ffmpeg')
                            self.cap_cache[str(video_path)] = reader
                        images[short_name] = reader.get_data(frame_idx)
                    except Exception:
                        pass

        # 状态读取适配
        state_mapping = self.base_map if self.base_map else {"action": "action", "qpos": "observation.state"}
        state = {std_name: np.array(row[df_col]) for std_name, df_col in state_mapping.items() if df_col in row}

        return FrameData(timestamp=float(row.get("timestamp", index / self.fps)), images=images, state=state)
        
    def get_current_episode_path(self) -> str:
        return str(self.current_dataset_root) if self.dorobot_version and self.current_dataset_root else None
            
    def close(self):
        for reader in self.cap_cache.values():
            try: reader.close()
            except: pass
        self.cap_cache.clear()