# src/adapters/folder_adapter.py
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from src.core.interface import BaseDatasetReader, FrameData

class FolderAdapter(BaseDatasetReader):
    def __init__(self):
        self.root_path = None
        self.frames = [] 
        self.sensors = []

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        if not self.root_path.is_dir():
            return False

        # 1. 扫描图片
        # 兼容 Unitree 结构：图片可能在 colors/ 下，也可能在根目录
        search_dirs = [self.root_path, self.root_path / "colors"]
        img_files = []
        for d in search_dirs:
            if d.exists():
                img_files.extend(sorted(list(d.glob("*.jpg")) + list(d.glob("*.png"))))
        
        if not img_files:
            return False

        # 2. 建立索引
        frame_dict = {} 
        detected_sensors = set()

        for p in img_files:
            # 解析文件名: 000000_color_0.jpg -> idx=0, sensor=color_0
            match = re.match(r"(\d+)_+(.*)\.(jpg|png)", p.name)
            if match:
                idx = int(match.group(1))
                sensor_name = match.group(2)
                
                if idx not in frame_dict:
                    frame_dict[idx] = {'images': {}}
                
                frame_dict[idx]['images'][sensor_name] = str(p)
                detected_sensors.add(sensor_name)

        # 3. 排序
        sorted_indices = sorted(frame_dict.keys())
        self.frames = [frame_dict[i] for i in sorted_indices]
        self.sensors = list(detected_sensors)
        
        print(f"[FolderAdapter] 扫描完成: {len(self.frames)} 帧")
        return True

    def get_length(self) -> int:
        return len(self.frames)

    def get_all_sensors(self) -> List[str]:
        return self.sensors

    def get_frame(self, index: int) -> FrameData:
        frame_info = self.frames[index]
        images = {}
        for sensor, path in frame_info['images'].items():
            img = cv2.imread(path)
            if img is not None:
                images[sensor] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return FrameData(timestamp=float(index)/30.0, images=images, state={})

    def close(self):
        pass