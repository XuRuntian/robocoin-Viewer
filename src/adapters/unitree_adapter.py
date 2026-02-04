import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any
from src.core.interface import BaseDatasetReader, FrameData

class UnitreeAdapter(BaseDatasetReader):
    def __init__(self):
        self.root_path = None
        self.data_list = [] 
        self.fps = 30.0
        self.image_keys = []

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        
        # 路径修正：确保指向包含 data.json 的目录或文件
        if self.root_path.is_dir():
            json_file = self.root_path / "data.json"
        else:
            json_file = self.root_path
            self.root_path = self.root_path.parent

        if not json_file.exists():
            print(f"❌ [Unitree] 找不到 data.json: {json_file}")
            return False

        print(f"[Unitree] 正在解析: {json_file.name} ...")
        
        try:
            with open(json_file, 'r') as f:
                content = json.load(f)
            
            # 1. 解析 Info
            if isinstance(content, dict) and "info" in content:
                info = content["info"]
                if "image" in info:
                    self.fps = float(info["image"].get("fps", 30.0))
                    print(f"[Unitree] 检测到 FPS: {self.fps}")

            # 2. 核心：定位数据列表
            # 你的 JSON 明确使用 "data" 作为 key
            if isinstance(content, dict) and "data" in content and isinstance(content["data"], list):
                self.data_list = content["data"]
                print(f"[Unitree] ✅ 成功加载 'data' 列表，共 {len(self.data_list)} 帧")
            
            # 备用：如果没有 "data" key，尝试遍历寻找长得很像的列表
            elif isinstance(content, dict):
                for key, val in content.items():
                    if isinstance(val, list) and len(val) > 0:
                        first = val[0]
                        # 你的数据特征：有 "colors", "states", "idx"
                        if isinstance(first, dict) and ("colors" in first or "states" in first):
                            print(f"[Unitree] 推测 Key '{key}' 是数据列表")
                            self.data_list = val
                            break
            
            # 备用：如果根节点就是列表
            elif isinstance(content, list):
                self.data_list = content

            if not self.data_list:
                print("❌ [Unitree] JSON 中未找到有效的数据列表 (缺少 'data' key 或结构不匹配)")
                return False

            # 3. 自动探测相机名
            first_frame = self.data_list[0]
            if "colors" in first_frame:
                self.image_keys = list(first_frame["colors"].keys())
                print(f"[Unitree] 发现相机: {self.image_keys}")
            else:
                self.image_keys = ["color_0", "color_1"] # 兜底
                print(f"[Unitree] 使用默认相机名: {self.image_keys}")

            return True

        except Exception as e:
            print(f"❌ [Unitree] 解析异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_length(self) -> int:
        return len(self.data_list)

    def get_all_sensors(self) -> List[str]:
        return self.image_keys

    def get_frame(self, index: int) -> FrameData:
        if index >= len(self.data_list): return None
        frame_dict = self.data_list[index]
        
        images = {}
        # A. 读取图像 (从 "colors" 字段)
        if "colors" in frame_dict:
            for cam_name, rel_path in frame_dict["colors"].items():
                if rel_path:
                    # 你的路径可能是 "colors/000.jpg"，root_path 已经是 episode_0000 目录
                    full_path = self.root_path / rel_path
                    if full_path.exists():
                        img = cv2.imread(str(full_path))
                        if img is not None: 
                            images[cam_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # B. 读取状态 (从 "states" 字段)
        # 你的结构: "states" -> "left_arm" -> "qpos"
        state = {}
        try:
            if "states" in frame_dict:
                source = frame_dict["states"]
                qpos_list = []
                
                # 按固定顺序拼接 QPOS，方便在图表中看
                # 通常顺序: left_arm, right_arm, left_ee, right_ee
                parts_order = ["left_arm", "right_arm", "left_ee", "right_ee", "head", "body"]
                
                for part in parts_order:
                    if part in source and "qpos" in source[part]:
                        vals = source[part]["qpos"]
                        if vals:
                            qpos_list.extend(vals)
                
                if qpos_list:
                    state['qpos'] = np.array(qpos_list)

            # C. 读取触觉 (从 "tactiles" 字段)
            if "tactiles" in frame_dict:
                for t_name, t_path in frame_dict["tactiles"].items():
                    if t_path:
                        tp = self.root_path / t_path
                        if tp.exists():
                            # NPY 读取
                            state[t_name] = np.load(tp)

        except Exception as e:
            # print(f"State error: {e}")
            pass

        # D. 时间戳
        # 你的数据里没有 explicit time，用 idx / fps 计算
        idx = frame_dict.get("idx", index)
        timestamp = idx / self.fps

        return FrameData(
            timestamp=timestamp,
            images=images,
            state=state
        )

    def close(self):
        pass