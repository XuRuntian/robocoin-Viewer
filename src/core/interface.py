# src/core/interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AdapterConfig:
    # 基础探测与匹配
    base_type: str = ""
    match_rules: Dict[str, Any] = field(default_factory=dict)
    
    # 视觉桶 (Cameras)
    image_keys_map: Dict[str, str] = field(default_factory=dict)
    
    # 手臂桶 (Arms & Grippers) - 💡 新增这个核心字段
    arm_groups: Dict[str, Any] = field(default_factory=dict)
    
    # 状态桶/底座桶 (Base / Other states)
    state_keys_map: Dict[str, str] = field(default_factory=dict)
    
    # 长度参考
    length_reference_key: str = ""
    
    # 额外配置 (如 enable_undistort 等开关)
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
    camera_info: Optional[Dict[str, Any]] = None  # 摄像头内参等信息

class BaseDatasetReader(ABC):
    """
    数据读取器的抽象基类 (Interface)
    """
    def __init__(self, config: Optional[AdapterConfig] = None):
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
    
    @abstractmethod
    def close(self):
        """释放文件句柄"""
        pass

    @abstractmethod
    def get_current_episode_path(self) -> str:
        """返回当前轨迹隔离的物理目录/文件路径"""
        pass