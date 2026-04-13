# src/adapters/hdf5_adapter.py
import h5py
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.core.interface import BaseDatasetReader, FrameData, AdapterConfig
from src.core.registry import AdapterRegistry

@AdapterRegistry.register("HDF5")
class HDF5Adapter(BaseDatasetReader):
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.root_path = None
        self.file = None
        self._length = 0
        
        # 1. 基础配置标准化
        self.camera_map = getattr(self.config, 'image_keys_map', {}) or {}
        self.arm_groups = getattr(self.config, 'arm_groups', {}) or {}
        self.base_map = getattr(self.config, 'state_keys_map', {}) or {}
        
        extra_opts = getattr(self.config, 'extra_options', {}) or {}
        self.length_reference_key = getattr(self.config, 'length_reference_key', None)
        
        self.episode_files = [] 
        self.current_episode_idx = 0
        self.image_keys = []

    def _find_dataset_length(self, h5_node) -> int:
        if isinstance(h5_node, h5py.Dataset):
            # 💡 修复点：增加对标量(SCALAR)的判断，只有 ndim > 0 才有 shape[0]
            if h5_node.ndim > 0:
                return h5_node.shape[0]
            return 0 # 标量数据不作为长度参考
        
        if isinstance(h5_node, h5py.Group):
            # 如果配置了参考 key，优先使用
            if self.length_reference_key and self.length_reference_key in h5_node:
                node = h5_node[self.length_reference_key]
                if isinstance(node, h5py.Dataset) and node.ndim > 0:
                    return node.shape[0]
            
            # 递归查找组内第一个有长度的数据集
            for key in h5_node.keys():
                length = self._find_dataset_length(h5_node[key])
                if length > 0:
                    return length
        return 0
    
    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        self.episode_files = []
        
        if self.root_path.is_file() and self.root_path.suffix.lower() in ['.h5', '.hdf5']:
            self.episode_files.append(self.root_path)
        elif self.root_path.is_dir():
            files = list(self.root_path.glob("*.hdf5")) + list(self.root_path.glob("*.h5"))
            self.episode_files = sorted(files, key=lambda p: p.name)

        if not self.episode_files:
            print(f"❌ [HDF5] 路径 {file_path} 下未找到任何 HDF5 文件。")
            return False

        print(f"✅ [HDF5] 扫描到 {len(self.episode_files)} 条轨迹。")
        try:
            self.set_episode(0)
            return True
        except Exception as e:
            print(f"❌ [HDF5] 初始化第一条轨迹失败: {e}")
            return False

    def set_episode(self, episode_idx: int):
        if episode_idx < 0 or episode_idx >= len(self.episode_files):
            raise IndexError(f"轨迹索引 {episode_idx} 越界")
            
        self.current_episode_idx = episode_idx
        self.close()
        
        target_file = self.episode_files[episode_idx]
        self.file = h5py.File(target_file, 'r')
        self._length = self._find_dataset_length(self.file)
        
        self.image_keys = list(self.camera_map.keys())
        if not self.image_keys:
            if 'observations' in self.file and 'images' in self.file['observations']:
                img_grp = self.file['observations']['images']
                for cam_name in img_grp.keys():
                    self.camera_map[cam_name] = f"observations/images/{cam_name}"
            self.image_keys = list(self.camera_map.keys())
            
        print(f"🔄 [HDF5] 切换至 Episode {episode_idx}: {self._length} 帧")

    def get_total_episodes(self) -> int: return len(self.episode_files)
    
    def get_length(self) -> int: return self._length

    def get_all_sensors(self) -> List[str]: return self.image_keys

    def get_frame(self, index: int, specific_cameras: Optional[List[str]] = None) -> FrameData:
        if self.file is None: raise RuntimeError("File not loaded")
        if index < 0 or index >= self._length: raise IndexError(f"Index {index} out of bounds")

        images = {}
        keys_to_fetch = specific_cameras if specific_cameras else self.image_keys
        
        # 1. 加载图像 (保持不变)
        for std_cam_name in keys_to_fetch:
            h5_path = self.camera_map.get(std_cam_name)
            if h5_path and h5_path in self.file:
                dataset = self.file[h5_path]
                raw_data = dataset[index] 
                if dataset.ndim == 1:
                    buffer = np.frombuffer(raw_data, dtype=np.uint8)
                    img_data = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                    if img_data is not None:
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                else:
                    img_data = raw_data
                    if img_data.ndim == 3 and img_data.shape[0] == 3:
                        img_data = np.transpose(img_data, (1, 2, 0))
                if img_data is not None:
                    images[std_cam_name] = img_data

        # 2. 构建状态数据
        state_data = {}
        
        # --- 修复点 A: 处理 base_map (对应 JSON 中的 "base" 字段) ---
        for std_state_name, h5_path in self.base_map.items():
            if h5_path in self.file:
                 state_data[std_state_name] = self.file[h5_path][index]

        # --- 修复点 B: 处理 arm_groups (对应 JSON 中的 "arm_groups" 字段) ---
        for arm_name, group_cfg in self.arm_groups.items():
            # 遍历 group 里的 key，比如 qpos, action 等
            for attr_name, h5_path in group_cfg.items():
                if h5_path in self.file:
                    # 组合 key 名，例如 "left_qpos"
                    combined_key = f"{arm_name}_{attr_name}"
                    state_data[combined_key] = self.file[h5_path][index]

        return FrameData(timestamp=float(index), images=images, state=state_data)
    def get_current_episode_path(self) -> str:
        if self.episode_files and 0 <= self.current_episode_idx < len(self.episode_files):
            return str(self.episode_files[self.current_episode_idx])
        return None

    def close(self):
        if self.file:
            self.file.close()
            self.file = None