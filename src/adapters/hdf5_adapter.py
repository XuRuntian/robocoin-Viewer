# src/adapters/hdf5_adapter.py
import h5py
import numpy as np
from typing import List, Dict, Any
from src.core.interface import BaseDatasetReader, FrameData
import cv2
class HDF5Adapter(BaseDatasetReader):
    def __init__(self):
        self.file_path = None
        self.file = None
        self.image_keys = [] # 存储类似 'observations/images/cam_high' 的路径
        self._length = 0

    def load(self, file_path: str) -> bool:
        """
        打开 HDF5 文件并解析其内部结构，找到哪里存着图片。
        """
        try:
            self.file_path = file_path
            self.file = h5py.File(file_path, 'r')
            
            # 1. 确定数据集长度
            # 通常寻找 'action' 或 'qpos' 来确定总帧数
            # 这里做一个简单的探测
            if 'action' in self.file:
                self._length = self.file['action'].shape[0]
            elif 'qpos' in self.file:
                self._length = self.file['qpos'].shape[0]
            else:
                # 兜底：随便找个 key 看看长度
                first_key = list(self.file.keys())[0]
                self._length = self.file[first_key].shape[0]

            # 2. 自动寻找图片存放的路径
            # 常见结构: observations -> images -> cam_name
            self.image_keys = []
            if 'observations' in self.file and 'images' in self.file['observations']:
                img_grp = self.file['observations']['images']
                for cam_name in img_grp.keys():
                    self.image_keys.append(f"observations/images/{cam_name}")
            
            # 如果没找到，打印个警告 (实际项目中可能需要更复杂的搜索逻辑)
            if not self.image_keys:
                print(f"Warning: 在文件 {file_path} 中未检测到标准路径 'observations/images'。")
                
            print(f"[HDF5] Loaded: {self._length} frames, Found images: {self.image_keys}")
            return True

        except Exception as e:
            print(f"Error loading HDF5: {e}")
            return False

    def get_length(self) -> int:
        return self._length

    def get_all_sensors(self) -> List[str]:
        # 返回相机名字，例如从 "observations/images/top_cam" 提取 "top_cam"
        return [k.split('/')[-1] for k in self.image_keys]

    def get_frame(self, index: int) -> FrameData:
        if self.file is None:
            raise RuntimeError("File not loaded")
        
        # 越界检查
        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} out of bounds (0-{self._length-1})")

        images = {}
        # 1. 读取图像
        for key in self.image_keys:
            cam_name = key.split('/')[-1]
            # HDF5 支持切片读取，不会把整个数组读入内存
            raw_data = self.file[key][index] 
            if raw_data.ndim == 1:
                img_data = cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_COLOR)
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            else:
                img_data = raw_data
                if img_data.shape[0] == 3 and img_data.ndim == 3:
                    img_data = np.transpose(img_data, (1, 2, 0))
                
            images[cam_name] = img_data

        # 2. 读取状态 (示例读取 qpos)
        state_data = {}
        if 'qpos' in self.file:
            state_data['qpos'] = self.file['qpos'][index]

        # 3. 时间戳 (有些数据集没有 timestamp，暂用 index 代替)
        timestamp = float(index) 

        return FrameData(
            timestamp=timestamp,
            images=images,
            state=state_data
        )

    def close(self):
        if self.file:
            self.file.close()
            print("[HDF5] File closed.")