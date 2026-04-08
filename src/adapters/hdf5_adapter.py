# src/adapters/hdf5_adapter.py
import h5py
import numpy as np
from typing import List, Dict, Any,Optional
from pathlib import Path
from src.core.interface import BaseDatasetReader, FrameData, AdapterConfig
import cv2
from src.core.registry import AdapterRegistry

@AdapterRegistry.register("HDF5")
class HDF5Adapter(BaseDatasetReader):
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config) # 注入配置
        self.root_path = None
        self.file = None
        self._length = 0
        
        self.episode_files = [] 
        self.current_episode_idx = 0

    def _find_dataset_length(self, h5_node) -> int:
        """递归寻找有效的 Dataset，避免对 Group 调用 .shape"""
        if isinstance(h5_node, h5py.Dataset):
            return h5_node.shape[0]
        
        if isinstance(h5_node, h5py.Group):
            # 优先使用配置中指定的 Key
            ref_key = self.config.length_reference_key
            if ref_key and ref_key in h5_node and isinstance(h5_node[ref_key], h5py.Dataset):
                return h5_node[ref_key].shape[0]
            
            # 如果没配置或找不到，遍历子节点找第一个 Dataset 兜底
            for key, val in h5_node.items():
                length = self._find_dataset_length(val)
                if length > 0:
                    return length
        return 0
    
    def load(self, file_path: str) -> bool:
        """
        加载 HDF5 数据集。
        支持传入单个 .hdf5 文件，也支持传入包含多个 .hdf5 文件的文件夹。
        """
        self.root_path = Path(file_path)
        self.episode_files = []
        
        if self.root_path.is_file():
            # 传入的是单文件
            if self.root_path.suffix.lower() in ['.h5', '.hdf5']:
                self.episode_files.append(self.root_path)
        elif self.root_path.is_dir():
            # 传入的是文件夹，扫描所有 HDF5
            files = list(self.root_path.glob("*.hdf5")) + list(self.root_path.glob("*.h5"))
            # 尝试按照文件名中的数字排序 (例如 episode_0, episode_1)
            self.episode_files = sorted(files, key=lambda p: p.name)

        if not self.episode_files:
            print(f"❌ [HDF5] 路径 {file_path} 下未找到任何 HDF5 文件。")
            return False

        print(f"✅ [HDF5] 扫描到 {len(self.episode_files)} 条轨迹。")
        
        # 默认加载第一条轨迹，验证并初始化内部状态
        try:
            self.set_episode(0)
            return True
        except Exception as e:
            print(f"❌ [HDF5] 初始化第一条轨迹失败: {e}")
            return False

    def set_episode(self, episode_idx: int):
        """[新增] 实现抽象方法：切换并加载指定索引的 HDF5 文件"""
        if episode_idx < 0 or episode_idx >= len(self.episode_files):
            raise IndexError(f"轨迹索引 {episode_idx} 越界 (0-{len(self.episode_files)-1})")
            
        self.current_episode_idx = episode_idx
        
        # 关闭之前打开的文件，释放内存
        self.close()
        
        target_file = self.episode_files[episode_idx]

        self.file = h5py.File(target_file, 'r')
        self._length = self._find_dataset_length(self.file)
        self.image_keys = list(self.config.image_keys_map.values())
        if not self.image_keys:
            if 'observations' in self.file and 'images' in self.file['observations']:
                img_grp = self.file['observations']['images']
                for cam_name in img_grp.keys():
                    # 动态构建配置映射
                    self.config.image_keys_map[cam_name] = f"observations/images/{cam_name}"
            self.image_keys = list(self.config.image_keys_map.values())
            
        print(f"🔄 [HDF5] 切换至 Episode {episode_idx}: {self._length} 帧, 相机映射: {self.config.image_keys_map}")

    def get_total_episodes(self) -> int:
        """[新增] 实现抽象方法：获取总轨迹数"""
        return len(self.episode_files)
    
    def get_length(self) -> int:
        return self._length

    def get_all_sensors(self) -> List[str]:
        return list(self.config.image_keys_map.keys())

    def get_frame(self, index: int) -> FrameData:
        if self.file is None:
            raise RuntimeError("File not loaded")
        
        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} out of bounds (0-{self._length-1})")

        images = {}
        for std_cam_name, h5_path in self.config.image_keys_map.items():
            if h5_path in self.file:
                raw_data = self.file[h5_path][index] 
            
            # 处理不同格式的图片存储
            if raw_data.ndim == 1:
                # JPG/PNG 压缩字节流
                img_data = cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_COLOR)
                if img_data is not None:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            else:
                # 原始矩阵 (H,W,C) 或 (C,H,W)
                img_data = raw_data
                if img_data.shape[0] == 3 and img_data.ndim == 3:
                    img_data = np.transpose(img_data, (1, 2, 0))
            
            # 防止 decode 失败导致报错
            if img_data is not None:
                images[std_cam_name] = img_data

        # 读取状态
        state_data = {}
        for std_state_name, h5_path in self.config.state_keys_map.items():
            if h5_path in self.file:
                 state_data[std_state_name] = self.file[h5_path][index]

        return FrameData(
            timestamp=float(index),
            images=images,
            state=state_data
        )

    def get_current_episode_path(self) -> str:
        """[新增] 返回当前正在播放的 .hdf5 文件路径"""
        if self.episode_files and 0 <= self.current_episode_idx < len(self.episode_files):
            return str(self.episode_files[self.current_episode_idx])
        return None

    def close(self):
        if self.file:
            self.file.close()
            self.file = None