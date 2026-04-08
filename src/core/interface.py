# src/core/interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AdapterConfig:
    """
    通用数据源解析配置协议。
    统一管理不同数据集中 key 的映射关系。
    """
    # 获取轨迹总帧数的基准参考 Key (如 "action", "qpos" 或 "observations/state")
    length_reference_key: str = "" 
    
    # 图像数据的提取路径映射 { "统一相机命名": "底层HDF5/ROS中的真实路径" }
    # 例如: {"cam_front": "observations/images/cam_high"}
    image_keys_map: Dict[str, str] = field(default_factory=dict)
    
    # 状态数据的提取路径映射 { "统一状态命名": "底层HDF5/ROS中的真实路径" }
    # 例如: {"qpos": "observations/qpos", "action": "action"}
    state_keys_map: Dict[str, str] = field(default_factory=dict)
    
    # 预留扩展字典，用于特定 Adapter 的特殊配置（如 ROS 的 ignore_compressed）
    extra_options: Dict[str, Any] = field(default_factory=dict)
@dataclass
class FrameData:
    """
    统一的数据帧结构。
    无论底层是 ROS Message 还是 HDF5 Group，
    传给 UI 的必须是这个标准结构。
    """
    timestamp: float
    # 图像数据: key=摄像头名, value=图像矩阵(H,W,C) RGB
    images: Dict[str, np.ndarray] 
    # 机器人状态: 比如关节角、末端位姿 (根据需要拓展)
    state: Optional[Dict[str, Any]] = None
    # 摄像头内参等信息
    camera_info: Optional[Dict[str, Any]] = None  

class BaseDatasetReader(ABC):
    """
    数据读取器的抽象基类 (Interface)
    """
    def __init__(self, config: AdapterConfig): # 新增构造函数
        self.config = config
    @abstractmethod
    def load(self, file_path: str) -> bool:
        """
        加载文件元数据/建立索引。
        注意：不要在这里把所有图片读入内存！
        """
        pass

    @abstractmethod
    def get_length(self) -> int:
        """返回数据集的总帧数"""
        pass

    @abstractmethod
    def get_all_sensors(self) -> List[str]:
        """返回所有可用的传感器名称列表"""
        pass

    @abstractmethod
    def get_frame(self, index: int) -> FrameData:
        """
        根据索引随机读取一帧数据。
        实现懒加载：在这里才真正去磁盘读图片/解码。
        """
        pass
    
    @abstractmethod
    def get_total_episodes(self) -> int:
        """
        返回数据集包含的总轨迹数。
        默认返回 1（适用于单文件数据集，如单个 ROS bag 或单条 HDF5）。
        支持多轨迹的数据集（如 LeRobot）需重写此方法。
        """
        return 1
    
    @abstractmethod
    def set_episode(self, episode_idx: int):
        """
        切换当前读取的轨迹。
        对于单轨迹数据集，此方法可不做任何事。
        """
        pass
    
    def get_current_episode_path(self) -> str:
        """
        获取当前正在读取的 Episode 的物理绝对路径。
        对于 HDF5，返回的是 .hdf5 文件路径；
        对于 Dorobot/LeRobot，返回的是 Episode 所在的子文件夹路径。
        """
        pass
    @abstractmethod
    def close(self):
        """释放文件句柄"""
        pass